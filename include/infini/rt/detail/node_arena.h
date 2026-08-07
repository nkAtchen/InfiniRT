#ifndef INFINI_RT_DETAIL_NODE_ARENA_H_
#define INFINI_RT_DETAIL_NODE_ARENA_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <new>
#include <vector>

namespace infini::rt::detail {

/// ## Recycling node storage for a node-based container.
///
/// A pool that keeps its free extents in an ordered container (`std::set`) hits
/// the same problem the pools themselves exist to solve: the container calls
/// `operator new` once per insert and `operator delete` once per erase. In a
/// steady-state alloc/free loop that is one host heap round trip per pooled
/// allocation, which is most of the overhead a pool is meant to remove.
///
/// `NodeArena` hands out fixed-size nodes from bulk-allocated blocks and keeps
/// released nodes on an intrusive free list, so after warm-up a container
/// backed by it performs no host allocation at all. Blocks are never returned
/// individually; the whole arena is freed at destruction.
///
/// The arena specializes itself to the first node size it sees, which is the
/// only size a given container ever asks for. Requests of any other size (or a
/// stricter alignment) fall through to the global allocation functions, so the
/// arena stays correct even if it is shared or reused.
///
/// Not thread-safe: callers serialize access with their own lock.
class NodeArena {
 public:
  NodeArena() = default;

  NodeArena(const NodeArena&) = delete;
  NodeArena& operator=(const NodeArena&) = delete;

  ~NodeArena() {
    for (void* block : blocks_) {
      ::operator delete(block, std::align_val_t{node_align_});
    }
  }

  void* Allocate(std::size_t bytes, std::size_t alignment) {
    if (node_size_ == 0) {
      // First request fixes the pooled geometry. A node must be able to hold
      // the free-list link while it is unused.
      node_size_ = std::max(bytes, sizeof(void*));
      node_align_ = std::max(alignment, alignof(void*));
    }

    if (bytes > node_size_ || alignment > node_align_) {
      return ::operator new(bytes, std::align_val_t{alignment});
    }

    if (free_ == nullptr) {
      Grow();
    }
    void* node = free_;
    // The link lives in the node's own storage; `memcpy` reads it back without
    // assuming anything about the object that used to be there.
    std::memcpy(&free_, node, sizeof(void*));
    return node;
  }

  void Deallocate(void* node, std::size_t bytes, std::size_t alignment) {
    if (node == nullptr) {
      return;
    }
    if (bytes > node_size_ || alignment > node_align_) {
      ::operator delete(node, std::align_val_t{alignment});
      return;
    }
    std::memcpy(node, &free_, sizeof(void*));
    free_ = node;
  }

 private:
  // Blocks grow geometrically so a large live set costs a bounded number of
  // host allocations, then capped so one huge burst does not reserve an
  // unreasonable block.
  static constexpr std::size_t kInitialNodes = 32;
  static constexpr std::size_t kMaxNodesPerBlock = 4096;

  void Grow() {
    const std::size_t count = next_count_;
    next_count_ = std::min(next_count_ * 2, kMaxNodesPerBlock);

    const std::size_t stride =
        (node_size_ + node_align_ - 1) / node_align_ * node_align_;
    void* block =
        ::operator new(stride * count, std::align_val_t{node_align_});
    blocks_.push_back(block);

    char* cursor = static_cast<char*>(block);
    for (std::size_t i = 0; i < count; ++i) {
      Deallocate(cursor + i * stride, node_size_, node_align_);
    }
  }

  std::vector<void*> blocks_;
  void* free_ = nullptr;
  std::size_t node_size_ = 0;
  std::size_t node_align_ = alignof(std::max_align_t);
  std::size_t next_count_ = kInitialNodes;
};

/// Standard-library allocator adaptor over a `NodeArena`. The arena is not
/// owned: it must outlive every container using it, which callers arrange by
/// declaring the arena before the container it backs.
template <typename T>
class ArenaAllocator {
 public:
  using value_type = T;

  explicit ArenaAllocator(NodeArena* arena) : arena_(arena) {}

  template <typename U>
  ArenaAllocator(const ArenaAllocator<U>& other)  // NOLINT: allocator rebind
      : arena_(other.arena()) {}

  T* allocate(std::size_t count) {
    if (count != 1) {
      return static_cast<T*>(
          ::operator new(count * sizeof(T), std::align_val_t{alignof(T)}));
    }
    return static_cast<T*>(arena_->Allocate(sizeof(T), alignof(T)));
  }

  void deallocate(T* ptr, std::size_t count) {
    if (count != 1) {
      ::operator delete(ptr, std::align_val_t{alignof(T)});
      return;
    }
    arena_->Deallocate(ptr, sizeof(T), alignof(T));
  }

  NodeArena* arena() const { return arena_; }

  template <typename U>
  bool operator==(const ArenaAllocator<U>& other) const {
    return arena_ == other.arena();
  }

  template <typename U>
  bool operator!=(const ArenaAllocator<U>& other) const {
    return arena_ != other.arena();
  }

 private:
  NodeArena* arena_;
};

}  // namespace infini::rt::detail

#endif
