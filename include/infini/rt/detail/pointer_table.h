#ifndef INFINI_RT_DETAIL_POINTER_TABLE_H_
#define INFINI_RT_DETAIL_POINTER_TABLE_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace infini::rt::detail {

/// ## Open-addressing table mapping a live pointer to a `Value`.
///
/// Every pool in this directory needs the same structure: given the pointer a
/// caller hands back, find the bookkeeping record for it. `std::unordered_map`
/// is node-based, so it would call `operator new` on every insert and
/// `operator delete` on every erase -- meaning each pooled allocation performs
/// a host heap allocation of its own, which is most of what a pool is trying to
/// avoid. This table stores values inline in one vector and only allocates when
/// it grows, so a steady-state alloc/free loop performs no host allocation at
/// all.
///
/// Linear probing with tombstones; the load factor is held at 1/2 so probe
/// sequences stay short and an empty slot always terminates a probe.
///
/// Not thread-safe: callers serialize access with their own lock.
template <typename Value>
class PointerTable {
 public:
  void Insert(void* key, const Value& value) {
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
          slot.value = value;
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

    slots_[index] = Slot{key, value, State::kOccupied};
    ++live_;
  }

  /// Writes `key`'s value to `*out` without removing it. Returns false if `key`
  /// is not present. Lets a caller inspect a record before deciding whether the
  /// entry should come out, which `Take` alone cannot do -- it has already
  /// tombstoned the slot by the time the value is available.
  bool Find(void* key, Value* out) const {
    if (live_ == 0) {
      return false;
    }

    const std::size_t mask = slots_.size() - 1;
    for (std::size_t index = Hash(key) & mask;; index = (index + 1) & mask) {
      const Slot& slot = slots_[index];
      if (slot.state == State::kEmpty) {
        return false;
      }
      if (slot.state == State::kOccupied && slot.key == key) {
        *out = slot.value;
        return true;
      }
    }
  }

  /// Removes `key` and writes its value to `*out`. Returns false if `key` is
  /// not present, which is how a pool's `Deallocate` detects a foreign pointer.
  bool Take(void* key, Value* out) {
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
        *out = slot.value;
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
        visitor(slot.value);
      }
    }
  }

  /// Number of live entries.
  std::size_t Size() const { return live_; }

 private:
  enum class State : std::uint8_t { kEmpty, kOccupied, kTombstone };

  struct Slot {
    void* key = nullptr;
    Value value{};
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
        Insert(slot.key, slot.value);
      }
    }
  }

  std::vector<Slot> slots_;
  std::size_t occupied_ = 0;  // live + tombstones, for the load factor
  std::size_t tombstones_ = 0;
  std::size_t live_ = 0;
};

}  // namespace infini::rt::detail

#endif
