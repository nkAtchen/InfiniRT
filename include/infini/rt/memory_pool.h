#ifndef INFINI_RT_MEMORY_POOL_H_
#define INFINI_RT_MEMORY_POOL_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infini::rt {

/// ## Backend-agnostic caching allocator.
///
/// `cudaMalloc`/`cudaFree` (and other device allocators) are synchronous and
/// expensive, so runtimes typically layer a caching allocator on top. A
/// `MemoryPool` keeps freed blocks in per-size-class free lists and hands them
/// back on the next matching request, so hot allocation loops pay the upstream
/// allocator only on a cache miss.
///
/// The pool is a pure composition over an `Upstream` allocator: any type that
/// provides `Malloc(void**, size_t)`, `Free(void*)`, an `Error` type alias, and
/// a `static constexpr Error kSuccess` satisfies the contract. Every
/// `runtime::Runtime<...>` device specialization (CPU, NVIDIA, ...) qualifies,
/// and the same interface serves CPU aligned allocations. Tests can inject a
/// mock upstream to exercise the pool without any device.
///
/// Blocks are reused only when both the rounded size and the requested
/// alignment match, so a reused block is always geometrically identical to the
/// request; the pool never splits or coalesces, which keeps reuse free of
/// fragmentation hazards at the cost of some retained-but-unused memory (call
/// `ReleaseCached` to hand that back to the upstream allocator).
///
/// The pool is thread-safe: every public method takes an internal mutex. It is
/// neither copyable nor movable.
template <typename Upstream>
class MemoryPool {
 public:
  using Error = typename Upstream::Error;

  /// Runtime statistics. Byte counters are cumulative live totals; `peak_*`
  /// track high-water marks. The remaining counters are monotonic tallies.
  struct Stats {
    /// Bytes currently handed out to callers (sum of rounded block sizes).
    std::size_t bytes_in_use = 0;

    /// Bytes currently held from the upstream allocator (in use + cached).
    std::size_t bytes_reserved = 0;

    /// High-water mark of `bytes_in_use`.
    std::size_t peak_bytes_in_use = 0;

    /// High-water mark of `bytes_reserved`.
    std::size_t peak_bytes_reserved = 0;

    /// Number of `Allocate` calls that returned a non-null pointer.
    std::size_t alloc_count = 0;

    /// Number of `Deallocate` calls that released a live block.
    std::size_t free_count = 0;

    /// Allocations served from a cached free block.
    std::size_t cache_hit_count = 0;

    /// Allocations that required a fresh upstream allocation.
    std::size_t cache_miss_count = 0;

    /// Calls into `Upstream::Malloc`.
    std::size_t upstream_alloc_count = 0;

    /// Calls into `Upstream::Free`.
    std::size_t upstream_free_count = 0;
  };

  MemoryPool() = default;

  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;

  /// Frees every block still held from the upstream allocator, including
  /// blocks that were never handed back via `Deallocate`. Any outstanding
  /// pointer from `Allocate` dangles after destruction.
  ~MemoryPool() {
    for (auto& [key, blocks] : free_lists_) {
      for (const Block& block : blocks) {
        Upstream::Free(block.base);
      }
    }
    allocated_.ForEach([](const Block& block) { Upstream::Free(block.base); });
  }

  /// Allocates at least `size` bytes, reusing a cached block when one with a
  /// matching size class and alignment is available. `alignment` of `0` uses
  /// the upstream allocator's natural alignment; otherwise the returned pointer
  /// is aligned up to `alignment` (which must be a power of two).
  ///
  /// On success writes the pointer to `*ptr` and returns `kSuccess`. A `size`
  /// of `0` succeeds with `*ptr == nullptr`. On upstream failure the upstream
  /// error is returned and `*ptr` is set to `nullptr`.
  Error Allocate(void** ptr, std::size_t size, std::size_t alignment = 0) {
    if (ptr == nullptr) {
      return InvalidValue();
    }

    *ptr = nullptr;
    if (size == 0) {
      return Upstream::kSuccess;
    }

    const std::size_t rounded = RoundSize(size);
    const BucketKey key{rounded, alignment};

    {
      std::lock_guard<std::mutex> lock(mutex_);
      Block cached{};
      if (TakeCached(key, &cached)) {
        ++stats_.cache_hit_count;
        *ptr = Register(cached);
        return Upstream::kSuccess;
      }
    }

    // Cache miss. The upstream allocator is a synchronous device call costing
    // hundreds of microseconds, so it runs with the lock released: holding it
    // here would stall every other thread -- including ones that only need a
    // cache hit -- for the duration of one `cudaMalloc`.
    const std::size_t upstream_size =
        alignment == 0 ? rounded : rounded + alignment;
    void* base = nullptr;
    const Error status = Upstream::Malloc(&base, upstream_size);

    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.upstream_alloc_count;

    if (status != Upstream::kSuccess) {
      // Another thread may have returned a matching block while the lock was
      // released, which turns an upstream failure into a hit.
      Block cached{};
      if (TakeCached(key, &cached)) {
        ++stats_.cache_hit_count;
        *ptr = Register(cached);
        return Upstream::kSuccess;
      }
      return status;
    }

    Block block{};
    block.base = base;
    block.aligned = alignment == 0 ? base : AlignUp(base, alignment);
    block.rounded_size = rounded;
    block.upstream_size = upstream_size;
    block.alignment = alignment;

    stats_.bytes_reserved += upstream_size;
    if (stats_.bytes_reserved > stats_.peak_bytes_reserved) {
      stats_.peak_bytes_reserved = stats_.bytes_reserved;
    }
    ++stats_.cache_miss_count;

    *ptr = Register(block);
    return Upstream::kSuccess;
  }

  /// Returns a block from `Allocate` to the pool's free list for reuse. The
  /// block is not handed back to the upstream allocator until `ReleaseCached`
  /// or destruction. `nullptr` is a no-op. Returns an invalid-value error if
  /// `ptr` was not produced by this pool (or was already freed).
  Error Deallocate(void* ptr) {
    if (ptr == nullptr) {
      return Upstream::kSuccess;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    Block block{};
    if (!allocated_.Take(ptr, &block)) {
      return InvalidValue();
    }

    stats_.bytes_in_use -= block.rounded_size;
    ++stats_.free_count;

    free_lists_[BucketKey{block.rounded_size, block.alignment}].push_back(
        block);
    return Upstream::kSuccess;
  }

  /// Hands every cached (freed but not-yet-returned) block back to the upstream
  /// allocator. Blocks currently in use are untouched. This is the pool's
  /// defragmentation / trim knob: call it to release retained memory back to
  /// the device.
  void ReleaseCached() {
    // Detach the free lists under the lock, then call upstream without it. As
    // in `Allocate`, an upstream call is far more expensive than the
    // bookkeeping, so it must not block other threads. Stats are settled while
    // the lock is held: once detached the blocks are no longer the pool's, so a
    // concurrent `GetStats` sees them gone even though the frees are in flight.
    std::unordered_map<BucketKey, std::vector<Block>, BucketKeyHash> detached;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      detached.swap(free_lists_);
      for (const auto& [key, blocks] : detached) {
        for (const Block& block : blocks) {
          ++stats_.upstream_free_count;
          stats_.bytes_reserved -= block.upstream_size;
        }
      }
    }

    for (const auto& [key, blocks] : detached) {
      for (const Block& block : blocks) {
        Upstream::Free(block.base);
      }
    }
  }

  /// Returns a snapshot of the pool's statistics.
  Stats GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

 private:
  static_assert(
      std::is_invocable_v<decltype(Upstream::Malloc), void**, std::size_t>,
      "`Upstream::Malloc` must be callable with `(void**, size_t)`.");
  static_assert(std::is_invocable_v<decltype(Upstream::Free), void*>,
                "`Upstream::Free` must be callable with `(void*)`.");
  static_assert(
      std::is_same_v<std::remove_cv_t<decltype(Upstream::kSuccess)>, Error>,
      "`Upstream` must define `static constexpr Error kSuccess`.");

  // A single upstream allocation tracked by the pool. `base` is the pointer
  // owned by the upstream allocator; `aligned` is what the caller sees.
  struct Block {
    void* base = nullptr;
    void* aligned = nullptr;
    std::size_t rounded_size = 0;
    std::size_t upstream_size = 0;
    std::size_t alignment = 0;
  };

  // Open-addressing table mapping a live pointer to its `Block`.
  //
  // `std::unordered_map` is node-based, so it would call `operator new` on
  // every insert and `operator delete` on every erase -- meaning each pooled
  // allocation performs a host heap allocation of its own, which is most of
  // what the pool is trying to avoid. This table stores blocks inline in one
  // vector and only allocates when it grows, so a steady-state alloc/free loop
  // performs no host allocation at all.
  //
  // Linear probing with tombstones; the load factor is held at 1/2 so probe
  // sequences stay short and an empty slot always terminates a probe.
  class BlockTable {
   public:
    void Insert(void* key, const Block& block) {
      // Tombstones count toward the load factor: they still sit on probe paths,
      // and a table saturated with them would break the empty-slot terminator.
      if ((occupied_ + 1) * 2 > slots_.size()) {
        Rehash();
      }

      const std::size_t mask = slots_.size() - 1;
      std::size_t index = Hash(key) & mask;
      std::size_t tombstone = kNoSlot;

      for (;; index = (index + 1) & mask) {
        Slot& slot = slots_[index];
        if (slot.state == State::kOccupied) {
          if (slot.key == key) {  // Overwrite an existing entry.
            slot.block = block;
            return;
          }
          continue;
        }
        if (slot.state == State::kTombstone) {
          if (tombstone == kNoSlot) {
            tombstone = index;
          }
          continue;
        }
        break;  // Empty: the key is absent.
      }

      if (tombstone != kNoSlot) {
        index = tombstone;  // Reuse a tombstone ahead of the empty slot.
        --tombstones_;
      } else {
        ++occupied_;
      }

      slots_[index] = Slot{key, block, State::kOccupied};
      ++live_;
    }

    // Removes `key` and writes its block to `*out`. Returns false if `key` is
    // not present, which is how `Deallocate` detects a foreign pointer.
    bool Take(void* key, Block* out) {
      if (live_ == 0) {
        return false;
      }

      const std::size_t mask = slots_.size() - 1;
      for (std::size_t index = Hash(key) & mask;; index = (index + 1) & mask) {
        Slot& slot = slots_[index];
        if (slot.state == State::kEmpty) {
          return false;
        }
        if (slot.state == State::kOccupied && slot.key == key) {
          *out = slot.block;
          slot.state = State::kTombstone;
          slot.key = nullptr;
          ++tombstones_;
          --live_;
          return true;
        }
      }
    }

    template <typename Visitor>
    void ForEach(Visitor&& visitor) const {
      for (const Slot& slot : slots_) {
        if (slot.state == State::kOccupied) {
          visitor(slot.block);
        }
      }
    }

   private:
    enum class State : std::uint8_t { kEmpty, kOccupied, kTombstone };

    struct Slot {
      void* key = nullptr;
      Block block{};
      State state = State::kEmpty;
    };

    static constexpr std::size_t kInitialSlots = 16;
    static constexpr std::size_t kNoSlot = static_cast<std::size_t>(-1);

    // Pointers from an allocator are aligned, so their low bits are mostly
    // zero; a multiply-shift spreads the informative high bits down.
    static std::size_t Hash(void* key) {
      auto value =
          static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(key));
      value *= 0x9e3779b97f4a7c15ULL;
      return static_cast<std::size_t>(value >> 29);
    }

    // Rebuilds the table, dropping tombstones. Capacity is sized for the live
    // entries, not for `occupied_`: the tombstones counted there are discarded
    // by this very rebuild, so sizing for them would buy room for what is about
    // to be thrown away. Capacity stays a power of two.
    //
    // A churn-heavy workload reaches the load factor via tombstones rather than
    // live entries, so this usually rebuilds at the same capacity instead of
    // growing -- hence the name.
    void Rehash() {
      std::size_t capacity = kInitialSlots;
      while (capacity <= (live_ + 1) * 2) {
        capacity *= 2;
      }

      std::vector<Slot> old_slots(capacity);
      old_slots.swap(slots_);
      occupied_ = 0;
      tombstones_ = 0;
      live_ = 0;

      // Reusing `Insert` cannot recurse: `capacity` was chosen above the load
      // factor for exactly this many entries, so the check in `Insert` stays
      // false throughout.
      for (const Slot& slot : old_slots) {
        if (slot.state == State::kOccupied) {
          Insert(slot.key, slot.block);
        }
      }
    }

    std::vector<Slot> slots_;
    std::size_t occupied_ = 0;  // live + tombstones, for the load factor
    std::size_t tombstones_ = 0;
    std::size_t live_ = 0;
  };

  // Free lists are keyed by rounded size and alignment so a reused block is
  // always geometrically identical to the request.
  struct BucketKey {
    std::size_t size = 0;
    std::size_t alignment = 0;

    bool operator==(const BucketKey& other) const {
      return size == other.size && alignment == other.alignment;
    }
  };

  struct BucketKeyHash {
    std::size_t operator()(const BucketKey& key) const {
      // Mix the two fields; alignment is small so a shift keeps it out of the
      // low bits that size dominates.
      return key.size ^ (key.alignment << 1);
    }
  };

  // Small allocations round to 512 B; large ones to 2 MB. This keeps the number
  // of distinct size classes bounded so freed blocks are likely to be reused.
  static constexpr std::size_t kSmallThreshold = 1u << 20;  // 1 MB
  static constexpr std::size_t kSmallGranularity = 512;
  static constexpr std::size_t kLargeGranularity = 1u << 21;  // 2 MB

  static std::size_t RoundUp(std::size_t size, std::size_t granularity) {
    return (size + granularity - 1) / granularity * granularity;
  }

  static std::size_t RoundSize(std::size_t size) {
    return size <= kSmallThreshold ? RoundUp(size, kSmallGranularity)
                                   : RoundUp(size, kLargeGranularity);
  }

  static void* AlignUp(void* ptr, std::size_t alignment) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    const auto aligned = (address + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned);
  }

  static Error InvalidValue() { return static_cast<Error>(1); }

  // Pops a cached block for `key`. Caller must hold `mutex_`.
  bool TakeCached(const BucketKey& key, Block* out) {
    auto it = free_lists_.find(key);
    if (it == free_lists_.end() || it->second.empty()) {
      return false;
    }
    *out = it->second.back();
    it->second.pop_back();
    return true;
  }

  // Marks `block` live and returns the pointer the caller sees. Caller must
  // hold `mutex_`.
  void* Register(const Block& block) {
    allocated_.Insert(block.aligned, block);
    stats_.bytes_in_use += block.rounded_size;
    if (stats_.bytes_in_use > stats_.peak_bytes_in_use) {
      stats_.peak_bytes_in_use = stats_.bytes_in_use;
    }
    ++stats_.alloc_count;
    return block.aligned;
  }

  mutable std::mutex mutex_;
  BlockTable allocated_;
  std::unordered_map<BucketKey, std::vector<Block>, BucketKeyHash> free_lists_;
  Stats stats_;
};

}  // namespace infini::rt

#endif
