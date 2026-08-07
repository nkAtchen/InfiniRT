#ifndef INFINI_RT_ARENA_MEMORY_POOL_H_
#define INFINI_RT_ARENA_MEMORY_POOL_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

#include "infini/rt/detail/node_arena.h"
#include "infini/rt/detail/pointer_table.h"

namespace infini::rt {

/// Tunables for `ArenaMemoryPool`. Every threshold is a template parameter
/// rather than a hard-coded constant because the interesting behaviors --
/// growth past the cap, oversize backings, automatic shrink -- are only
/// reachable at multi-hundred-megabyte scale with the production values, which
/// no CI machine (least of all a GPU-less one) can exercise. Tests instantiate
/// the pool with a kilobyte-scale config and drive the same code paths.
struct DefaultArenaConfig {
  /// Capacity of the first backing store, and the base of the doubling ramp.
  static constexpr std::size_t kInitialCapacity = 64ull << 20;  // 64 MB

  /// Ceiling on the doubling ramp. Once reached, further growth adds more
  /// backings of this size rather than larger ones.
  static constexpr std::size_t kMaxCapacity = 512ull << 20;  // 512 MB

  /// Requests at or below this size count as "small" for the shrink heuristic.
  static constexpr std::size_t kSmallThreshold = 1ull << 20;  // 1 MB

  /// Every slice starts at a multiple of this, whatever the caller asked for.
  /// Device allocators guarantee a healthy natural alignment (256 B for
  /// `cudaMalloc`) and callers -- plus vectorized kernels -- rely on it, so
  /// slicing must not hand back a worse-aligned pointer than `Upstream::Malloc`
  /// would have.
  static constexpr std::size_t kMinSliceAlignment = 512;

  /// A split leaving less than this behind is not performed; the remainder goes
  /// to the caller as internal waste instead of becoming an unusable sliver.
  static constexpr std::size_t kMinSplitRemainder = 512;

  /// Consecutive small allocations that mark the end of a burst. Reaching this
  /// count triggers one idle scan.
  static constexpr std::size_t kShrinkThreshold = 16;

  /// Consecutive idle scans a non-resident backing must be found empty for
  /// before it is destroyed. This is the hysteresis that keeps a "one big op
  /// plus twenty small ops" loop from destroying and re-creating a backing
  /// every iteration -- `cudaFree` implicitly synchronizes the whole device, so
  /// thrashing it from the allocation path is far worse than holding the memory
  /// for one more round.
  static constexpr std::uint32_t kEmptyScansToDestroy = 2;

  /// Blocks one thread may hold per exact size in its front cache.
  ///
  /// One, because depth beyond one only pays off when several blocks of the same
  /// size are live at once, and the loops the cache exists for -- allocate, use,
  /// free, allocate the same size again -- hit at depth one already. Depth is
  /// not free: a parked block stays in `allocated_`, so it is invisible to
  /// coalescing and cannot be merged into a larger request. Deeper caches
  /// scatter more such blocks through the backings, which on a workload whose
  /// footprint grows monotonically shows up as extra backing growth. Depth one
  /// keeps the hit and pays the least for it.
  static constexpr std::size_t kThreadCacheDepth = 1;

  /// Total bytes one thread may retain across every one of its cache lists.
  /// This, not the depth, is what bounds the cache on large sizes: eight 512 MB
  /// blocks per size class would be absurd, and this cap is what stops it.
  static constexpr std::size_t kThreadCacheBytes = 8ull << 20;  // 8 MB
};

namespace detail {

// The front cache arrived after `Config` had several implementations in this
// tree, so its two knobs are detected rather than required: a `Config` that
// predates them keeps working. Every other threshold is mandatory, because
// omitting one is almost certainly a mistake; these two are the exception only
// because their absence has a safe reading.
//
// The byte fallback is proportional rather than absolute. An arena configured at
// kilobyte scale (the tests) and one configured at production scale differ by
// five orders of magnitude, and a fixed default would let one thread privatize
// an entire backing in the former case.
template <typename Config, typename = void>
struct ArenaCacheDepth
    : std::integral_constant<std::size_t,
                             DefaultArenaConfig::kThreadCacheDepth> {};

template <typename Config>
struct ArenaCacheDepth<Config, std::void_t<decltype(Config::kThreadCacheDepth)>>
    : std::integral_constant<std::size_t, Config::kThreadCacheDepth> {};

template <typename Config, typename = void>
struct ArenaCacheBytes
    : std::integral_constant<std::size_t, Config::kInitialCapacity / 8> {};

template <typename Config>
struct ArenaCacheBytes<Config, std::void_t<decltype(Config::kThreadCacheBytes)>>
    : std::integral_constant<std::size_t, Config::kThreadCacheBytes> {};

}  // namespace detail

/// ## Backend-agnostic arena (sub-allocating) allocator.
///
/// `MemoryPool` calls `Upstream::Malloc` on every cache miss. Device allocators
/// are synchronous and cost hundreds of microseconds, so a workload whose sizes
/// keep missing pays that price over and over. `ArenaMemoryPool` instead
/// requests large *backing stores* from the upstream allocator and satisfies
/// requests by slicing them, so upstream calls scale with the pool's high-water
/// footprint rather than with the number of allocations.
///
/// Like `MemoryPool`, this is a pure composition over an `Upstream` allocator:
/// any type providing `Malloc(void**, size_t)`, `Free(void*)`, an `Error` type
/// alias, and a `static constexpr Error kSuccess` satisfies the contract, so
/// every `runtime::Runtime<...>` device specialization qualifies and tests can
/// inject a mock upstream.
///
/// ### Structure
///
/// Each backing store is modeled as one doubly-linked list of `Chunk`s sorted
/// by address, tiling its usable span with no gaps. A chunk is either free or
/// handed out; allocation splits a free chunk, and `Deallocate` marks a chunk
/// free. Adjacent free chunks are coalescible, so a drained backing can always
/// be reassembled into one chunk spanning its whole span and serve a contiguous
/// request of its full size again. (A bump-pointer design cannot express this:
/// memory released below the bump cursor is physically adjacent to the
/// untouched tail yet unreachable from it, so a drained 512 MB backing could
/// not produce a 512 MB block.)
///
/// That merge is *deferred* rather than performed on every free. Coalescing
/// touches both physical neighbors and re-inserts into an ordered container --
/// several cold cache lines inside the lock -- and a workload that cycles
/// through a handful of sizes never needs it, because the next request for a
/// size pops back exactly the chunk that was just released. So `Deallocate`
/// files a released chunk straight into its exact-size bin and returns, and the
/// merging happens in one pass (`CoalesceAll`) only when a request cannot be
/// served from what is already indexed. The cost is that "this region is free"
/// has more than one representation, so emptiness is a per-backing live-chunk
/// count rather than a list-length test, and `Stats::largest_free_chunk`
/// reports the largest coalescible *run* rather than the largest single chunk.
///
/// Free chunks across all backings are indexed together, so allocation is a
/// best-fit lookup rather than an exact size-class match: splitting and
/// coalescing mean a released block is reusable at any size, not only at the
/// size class it was allocated with. The index is in two parts. Sizes that are
/// a small multiple of `Config::kMinSliceAlignment` go in *fast bins* -- one
/// intrusive list per exact size, with a bitmap over their occupancy -- so a
/// request that matches an occupied bin is served by popping a list head, in
/// O(1) and with no node allocation. Everything else lives in a tree ordered by
/// (size, address), searched in O(log n). Since the bins are exact, a hit there
/// is already the best fit and the tree is not consulted at all; workloads that
/// cycle through a handful of sizes stay entirely on that path.
///
/// ### Front cache
///
/// Everything above happens under one mutex, so throughput is bounded by how
/// long each thread holds it -- and best fit plus splitting is a longer hold
/// than a size-class pool's list pop. In front of it sits a small per-thread
/// cache of exact-size blocks: a thread that cycles through a handful of sizes
/// serves its own allocations by popping one of its own lists, touching no
/// shared state and taking no lock at all.
///
/// `Deallocate` still takes the mutex, because rejecting a foreign pointer or a
/// double free requires the live-block table, which is shared. What it does
/// while holding it is file the block into the *releasing* thread's cache rather
/// than into the global free index, so the next allocation on that thread finds
/// it without the lock. In an alloc/free loop that halves lock acquisitions and
/// removes the expensive half of the work -- the best-fit lookup and the split.
///
/// A cached block is still counted live by its backing store, which is what
/// keeps automatic shrink from handing that backing upstream while a cache
/// points into it. The retention is bounded per thread, and any path that needs
/// the memory back -- a request nothing indexed can serve, `ReleaseCached`,
/// destruction -- reclaims every cache first. The cost is that `Stats` cannot be
/// maintained on the lock-free path, so `GetStats` folds each cache's own
/// tallies in as it reads them.
///
/// ### Growth and shrink
///
/// Backing capacity follows `Config::kInitialCapacity` doubling up to
/// `Config::kMaxCapacity`; past the cap, additional backings of the cap size
/// are added. A single request larger than the cap gets its own exactly sized
/// *oversize* backing -- multiple capped backings cannot serve it, since
/// slicing never spans two upstream allocations.
///
/// The first non-oversize backing is *resident*: never destroyed except by
/// `ReleaseCached` or destruction, so a steady small workload keeps a warm
/// arena and never thrashes. Every other backing is subject to automatic
/// shrink: once `Config::kShrinkThreshold` consecutive small allocations
/// indicate the burst is over, drained non-resident backings are returned
/// upstream (after `Config::kEmptyScansToDestroy` scans of hysteresis; oversize
/// backings are exempt and go back on the first scan, since holding gigabytes
/// idle is far more expensive than one extra upstream call).
///
/// ### Caller responsibilities
///
/// **Streams.** A block is reusable the instant `Deallocate` returns, and
/// coalescing means the memory may come back as a *differently sized* block at
/// a *different offset* handed to an unrelated caller. If a previously launched
/// kernel still reads the old address, it corrupts a live allocation whose size
/// and bounds bear no relation to the original. The pool has no notion of
/// streams: callers must ensure device-side access to a block has completed
/// before calling `Deallocate` (e.g. by synchronizing the stream, or recording
/// an event and waiting on it).
///
/// **Devices.** A backing is bound to whichever device was current when it was
/// created. The pool does not track device ids, so multi-device use needs one
/// pool instance per device.
///
/// The pool is thread-safe: every public method takes an internal mutex, and no
/// upstream call is ever made while it is held. It is neither copyable nor
/// movable.
template <typename Upstream, typename Config = DefaultArenaConfig>
class ArenaMemoryPool {
 public:
  using Error = typename Upstream::Error;

  /// Runtime statistics. Byte counters are live totals; `peak_*` track
  /// high-water marks. The remaining counters are monotonic tallies.
  struct Stats {
    /// Bytes currently handed out to callers (sum of served chunk sizes,
    /// including any remainder folded in by the split threshold).
    std::size_t bytes_in_use = 0;

    /// Bytes currently held from the upstream allocator: the sum of every
    /// backing store's capacity.
    std::size_t bytes_reserved = 0;

    /// High-water mark of `bytes_in_use`.
    std::size_t peak_bytes_in_use = 0;

    /// High-water mark of `bytes_reserved`.
    std::size_t peak_bytes_reserved = 0;

    /// Number of `Allocate` calls that returned a non-null pointer.
    std::size_t alloc_count = 0;

    /// Number of `Deallocate` calls that released a live block.
    std::size_t free_count = 0;

    /// Allocations served from existing backing stores.
    std::size_t cache_hit_count = 0;

    /// Allocations that required a new backing store.
    std::size_t cache_miss_count = 0;

    /// Calls into `Upstream::Malloc` that produced a backing store.
    std::size_t upstream_alloc_count = 0;

    /// Calls into `Upstream::Free` (one per backing store).
    std::size_t upstream_free_count = 0;

    /// Number of live backing stores.
    std::size_t backing_count = 0;

    /// Bytes sitting free inside backing stores. This is the pool's
    /// fragmentation: reserved but not in use, and not returnable upstream
    /// unless a whole backing drains.
    std::size_t bytes_free_in_backings = 0;

    /// Size of the largest request the pool could serve without going upstream:
    /// the longest run of adjacent free chunks, since coalescing is deferred and
    /// any such run can be merged on demand. Together with
    /// `bytes_free_in_backings` this shows whether free space is usable or
    /// shattered.
    std::size_t largest_free_chunk = 0;

    /// Bytes inside served chunks beyond what the caller asked for: remainders
    /// too small to split off. A subset of `bytes_in_use`, counted while the
    /// chunks carrying them are live.
    std::size_t bytes_internal_waste = 0;

    /// Reserved bytes that are in no chunk at all: the head of each backing
    /// store trimmed off to bring the first slice up to
    /// `Config::kMinSliceAlignment`. Zero whenever the upstream allocator
    /// already returns suitably aligned pointers, which device allocators do.
    ///
    /// Together these close the books:
    /// `bytes_reserved == bytes_in_use + bytes_free_in_backings +
    /// bytes_unusable`.
    std::size_t bytes_unusable = 0;

    /// Backing stores destroyed by automatic shrink.
    std::size_t shrink_count = 0;
  };

  ArenaMemoryPool()
      : registry_(std::make_shared<Registry>()), mutex_(registry_->mutex) {
    // No lock: nothing else can reach this registry yet.
    registry_->pool = this;
  }

  ArenaMemoryPool(const ArenaMemoryPool&) = delete;
  ArenaMemoryPool& operator=(const ArenaMemoryPool&) = delete;

  /// Frees every backing store, exactly once each. Blocks still outstanding are
  /// slices of those backings, so they must not be freed individually -- only
  /// the upstream base pointers are valid arguments to `Upstream::Free`. Any
  /// pointer from `Allocate` dangles after destruction.
  ~ArenaMemoryPool() {
    {
      // Clearing `pool` under the lock is the handshake with every thread still
      // holding a cache: from here on their exit handlers see a dead pool and
      // touch nothing but their own node. Reclaiming first keeps the chunk
      // arena's bookkeeping consistent while `DestroyChunks` walks it.
      std::lock_guard<std::mutex> lock(registry_->mutex);
      for (ThreadCache* cache = registry_->head; cache != nullptr;
           cache = cache->next) {
        ReclaimCacheLocked(cache);
      }
      registry_->pool = nullptr;
    }

    for (const auto& backing : backings_) {
      DestroyChunks(backing.get());
      Upstream::Free(backing->base);
    }
  }

  /// Allocates at least `size` bytes by slicing a backing store, requesting a
  /// new one from the upstream allocator only when no existing backing has
  /// room. `alignment` of `0` uses the pool's natural slice alignment
  /// (`Config::kMinSliceAlignment`); otherwise the returned pointer is aligned
  /// up to `alignment`, which must be a power of two.
  ///
  /// On success writes the pointer to `*ptr` and returns `kSuccess`. A `size` of
  /// `0` succeeds with `*ptr == nullptr`. On upstream failure the upstream error
  /// is returned and `*ptr` is set to `nullptr`.
  Error Allocate(void** ptr, std::size_t size, std::size_t alignment = 0) {
    if (ptr == nullptr) {
      return InvalidValue();
    }

    *ptr = nullptr;
    if (size == 0) {
      return Upstream::kSuccess;
    }

    const std::size_t align = SliceAlignment(alignment);
    const std::size_t rounded = RoundSize(size);

    // Front cache first, for the ordinary case: natural alignment and a size the
    // bins cover. A hit takes no lock and touches nothing another thread reads.
    //
    // Over-aligned requests skip it. A cached block is only known to start at
    // `kMinSliceAlignment`, so serving one would mean re-checking alignment and
    // falling through on failure -- work on the hot path for a rare request.
    if (align == Config::kMinSliceAlignment) {
      if (void* cached = TryCacheAllocate(rounded); cached != nullptr) {
        *ptr = cached;
        return Upstream::kSuccess;
      }
    }

    // Worst case a chunk must cover. Every chunk starts at a multiple of
    // `kMinSliceAlignment` (see `AdoptBacking`), so aligning up to `align`
    // costs at most the difference between the two.
    const std::size_t needed = rounded + align - Config::kMinSliceAlignment;

    std::unique_lock<std::mutex> lock(mutex_);

    {
      // Backings the idle scan selects are freed with the lock dropped:
      // `cudaFree` implicitly synchronizes the whole device, so it must never
      // run inside the critical section, let alone on the allocation path.
      std::vector<std::unique_ptr<BackingStore>> doomed;
      UpdateShrinkState(size, &doomed);
      if (!doomed.empty()) {
        lock.unlock();
        FreeBackings(doomed);
        doomed.clear();
        lock.lock();
      }
    }

    if (Chunk* chunk = FindFit(needed); chunk != nullptr) {
      ++stats_.cache_hit_count;
      *ptr = Serve(chunk, rounded, align);
      return Upstream::kSuccess;
    }

    // No room anywhere: a new backing store is needed. The upstream allocator
    // is a synchronous device call costing hundreds of microseconds, so it runs
    // with the lock released -- holding it here would stall every other thread,
    // including ones that only need to slice an existing backing.
    std::size_t capacity = NextCapacity(needed);
    lock.unlock();

    void* base = nullptr;
    Error status = Upstream::Malloc(&base, capacity);
    if (status != Upstream::kSuccess) {
      status = AllocateFallback(&base, &capacity, needed, status);
    }

    lock.lock();

    if (status != Upstream::kSuccess) {
      // Another thread may have released space while the lock was down, which
      // turns an upstream failure into a hit.
      if (Chunk* chunk = FindFit(needed); chunk != nullptr) {
        ++stats_.cache_hit_count;
        *ptr = Serve(chunk, rounded, align);
        return Upstream::kSuccess;
      }
      return status;
    }

    ++stats_.upstream_alloc_count;
    Chunk* chunk = AdoptBacking(base, capacity);
    ++stats_.cache_miss_count;
    *ptr = Serve(chunk, rounded, align);
    return Upstream::kSuccess;
  }

  /// Returns a block from `Allocate` to its backing store, where it becomes
  /// available at its own size immediately and at any larger size once merging
  /// runs (deferred to the next request that needs it -- see the class comment).
  /// The memory is not handed back to the upstream allocator here; that happens
  /// on automatic shrink, `ReleaseCached`, or destruction. `nullptr` is a no-op.
  /// Returns an invalid-value error if `ptr` was not produced by this pool (or
  /// was already freed).
  Error Deallocate(void* ptr) {
    if (ptr == nullptr) {
      return Upstream::kSuccess;
    }

    // The lock is unavoidable here: rejecting a foreign pointer or a double free
    // means consulting the live-block table, which is shared. What the front
    // cache saves is not this acquisition but the *next* allocation's -- and the
    // best-fit lookup and split that would have come with it.
    std::lock_guard<std::mutex> lock(mutex_);

    // `Find` rather than `Take`: a chunk already parked in some thread's cache is
    // still in the table, and telling that case apart from a live block is what
    // makes a double free of a cached block detectable.
    Chunk* chunk = nullptr;
    if (!allocated_.Find(ptr, &chunk) || chunk->cached) {
      return InvalidValue();
    }

    ++stats_.free_count;

    // File it into the releasing thread's cache, so the next allocation of this
    // size on this thread needs no lock at all. The table entry deliberately
    // stays: the block never became free, it changed hands from the caller to
    // the cache.
    //
    // Resolve the bin before touching thread-local state. Sizes the bins do not
    // cover cannot be parked at all, and on a workload built from large blocks
    // that is every free -- looking the cache up first would spend a TLS access
    // and a scan on every one of them only to be turned away. `IfPresent`
    // because registering a cache takes `mutex_`, which is already held: a
    // thread that has never allocated a binnable size has no cache here, and a
    // free is not a reason to give it one.
    const std::size_t bin = BinIndex(chunk->size);
    if (bin != kNoBin) {
      if (ThreadCache* cache = LocalCacheIfPresent();
          cache != nullptr && TryParkCached(cache, bin, chunk)) {
        return Upstream::kSuccess;
      }
    }

    allocated_.Take(ptr, &chunk);
    ReleaseChunk(chunk);
    return Upstream::kSuccess;
  }

  /// Immediately returns every fully drained backing store to the upstream
  /// allocator, including the resident one and ignoring shrink hysteresis.
  /// Backings with live blocks are untouched.
  ///
  /// Note this is a coarser knob than `MemoryPool::ReleaseCached`: free space
  /// *inside* a backing that still holds live blocks cannot be handed back, so
  /// a fragmented pool may release nothing. `Stats::bytes_free_in_backings`
  /// reports how much is retained.
  void ReleaseCached() {
    // Decide under the lock, free outside it. Stats are settled while the lock
    // is held: once detached the backings are no longer the pool's, so a
    // concurrent `GetStats` sees them gone even though the frees are in flight.
    std::vector<std::unique_ptr<BackingStore>> doomed;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Blocks parked in a cache count live, so a backing holding nothing but
      // cached blocks would look occupied. Draining the caches first is what
      // makes this release everything a caller has actually returned.
      ReclaimAllCaches();
      for (std::size_t i = backings_.size(); i-- > 0;) {
        if (IsDrained(backings_[i].get())) {
          doomed.push_back(Detach(i));
        }
      }
      if (backings_.empty()) {
        // Nothing is warm any more, so the next allocation should restart the
        // growth ramp rather than resume at the burst-time capacity.
        next_capacity_ = Config::kInitialCapacity;
      }
    }

    FreeBackings(doomed);
  }

  /// Returns a snapshot of the pool's statistics.
  Stats GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats stats = stats_;
    stats.backing_count = backings_.size();
    stats.largest_free_chunk = LargestFreeChunk();

    // Fold in what the lock-free path tallied. Hits are counted per cache
    // because counting them in `stats_` would mean a shared write on exactly the
    // path that exists to avoid one.
    std::size_t cached_bytes = 0;
    SumCaches(&cached_bytes, &stats.alloc_count, &stats.cache_hit_count);

    // A parked block is charged to `bytes_in_use` internally -- that is how
    // shrink knows not to reclaim the backing under it -- but no caller holds it,
    // so reporting it as in use would be wrong: `bytes_in_use` would never reach
    // zero after a balanced run. Reported as fragmentation instead, which is what
    // it is: reserved, not handed out, not returnable upstream.
    stats.bytes_in_use -= cached_bytes;
    stats.bytes_free_in_backings += cached_bytes;
    return stats;
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
  static_assert(
      Config::kMinSliceAlignment != 0 &&
          (Config::kMinSliceAlignment & (Config::kMinSliceAlignment - 1)) == 0,
      "`Config::kMinSliceAlignment` must be a power of two.");
  static_assert(Config::kInitialCapacity <= Config::kMaxCapacity,
                "`Config::kInitialCapacity` must not exceed "
                "`Config::kMaxCapacity`.");

  struct BackingStore;

  // A free chunk is published in exactly one of two containers; a served chunk,
  // or one mid-coalesce, is in neither.
  enum class Location : std::uint8_t { kNone, kFastBin, kTree };

  // One extent of a backing store, either free or handed out. Chunks tile their
  // backing's usable span in address order with no gaps, so `prev`/`next` are
  // exactly the physical neighbors and coalescing is a local operation.
  struct Chunk {
    BackingStore* owner = nullptr;
    Chunk* prev = nullptr;
    Chunk* next = nullptr;
    void* ptr = nullptr;
    std::size_t size = 0;
    // Bytes the caller asked for (rounded). Smaller than `size` when a
    // remainder was too small to split off; the difference is internal waste.
    std::size_t requested = 0;
    bool is_free = true;
    // Parked in a thread's front cache: not free (its backing still counts it
    // live, which is what stops shrink from reclaiming the backing underneath
    // the cache) but not held by a caller either, so `Deallocate` must reject it
    // as a double free.
    bool cached = false;
    // Which free container currently holds this chunk, so `EraseFree` can
    // unlink it without searching -- and can skip the work entirely for a chunk
    // that is in neither, which is every chunk arriving on the `Deallocate`
    // coalescing path.
    Location location = Location::kNone;
    // Links in the exact-size fast bin holding this chunk, when one does. The
    // list is doubly linked because coalescing removes a neighbor from the
    // middle of a bin, which must stay O(1).
    //
    // A cached chunk is in no fast bin (`location` is `kNone`, and `EraseFree`
    // returns early on that), so `fast_next` does double duty as the front
    // cache's list link. The cache is LIFO and never unlinks from the middle, so
    // it needs only the forward link.
    Chunk* fast_prev = nullptr;
    Chunk* fast_next = nullptr;
  };

  // `Chunk`s are owned by their backing store and referenced by raw pointer
  // from the live-block table, the free set, and their neighbors. They are
  // allocated individually and never moved, so those references stay valid for
  // a chunk's whole life.
  //
  // Backings themselves are held by `unique_ptr` in a vector so that erasing
  // one -- which shrink does routinely -- does not move the others. `Chunk`
  // stores a raw `owner` pointer, which an index into the vector could not do
  // safely: every live chunk's index would shift on erase, and those indices
  // live inside the live-block table where they cannot be fixed up.
  struct BackingStore {
    void* base = nullptr;
    std::size_t capacity = 0;
    Chunk* head = nullptr;
    // Chunks currently handed out. Deferred coalescing means a drained backing
    // is no longer recognizable from its list length, so emptiness is this
    // counter reaching zero. Maintained by `Serve` and `Deallocate`, the only
    // two places a chunk changes hands.
    std::size_t live_chunks = 0;
    // The resident backing is exempt from automatic shrink so a steady small
    // workload keeps a warm arena.
    bool resident = false;
    // Larger than `Config::kMaxCapacity`, created for one request no capped
    // backing could serve. Never resident, and exempt from shrink hysteresis.
    bool oversize = false;
    std::uint32_t empty_scans = 0;
  };

  // Orders free chunks by size, then by address to break ties. Best fit is
  // `lower_bound` on the needed size; the largest free chunk is `rbegin`.
  struct BySizeThenAddress {
    bool operator()(const Chunk* lhs, const Chunk* rhs) const {
      if (lhs->size != rhs->size) {
        return lhs->size < rhs->size;
      }
      return reinterpret_cast<std::uintptr_t>(lhs->ptr) <
             reinterpret_cast<std::uintptr_t>(rhs->ptr);
    }
  };

  using FreeSet =
      std::set<Chunk*, BySizeThenAddress, detail::ArenaAllocator<Chunk*>>;

  // Number of exact-size fast bins. Bin `i` holds free chunks of exactly
  // `(i + 1) * Config::kMinSliceAlignment` bytes, so the bins cover every
  // aligned size up to `kFastBinCount * Config::kMinSliceAlignment` (64 KB
  // under the default config). Everything else -- oversize chunks, and the
  // non-aligned tails a backing's leading trim leaves behind -- stays in the
  // tree.
  static constexpr std::size_t kFastBinCount = 128;
  static constexpr std::size_t kFastBinWords = (kFastBinCount + 63) / 64;

  // "No such bin": either the size is not bin-eligible, or no bin in the
  // scanned range is occupied.
  static constexpr std::size_t kNoBin = static_cast<std::size_t>(-1);

  static Error InvalidValue() { return static_cast<Error>(1); }

  static std::size_t CountTrailingZeros(std::uint64_t word) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<std::size_t>(__builtin_ctzll(word));
#else
    std::size_t count = 0;
    while ((word & 1ull) == 0) {
      word >>= 1;
      ++count;
    }
    return count;
#endif
  }

  // The bin holding chunks of exactly `size`, or `kNoBin` if no bin does.
  static std::size_t BinIndex(std::size_t size) {
    if (size == 0 ||
        (size & (Config::kMinSliceAlignment - 1)) != 0) {
      return kNoBin;
    }
    const std::size_t multiples = size / Config::kMinSliceAlignment;
    return multiples <= kFastBinCount ? multiples - 1 : kNoBin;
  }

  // The lowest-indexed bin whose chunks are large enough for `needed`, or
  // `kNoBin` when `needed` outruns the bins entirely.
  static std::size_t FirstEligibleBin(std::size_t needed) {
    const std::size_t multiples =
        (needed + Config::kMinSliceAlignment - 1) / Config::kMinSliceAlignment;
    if (multiples == 0) {
      return 0;
    }
    return multiples <= kFastBinCount ? multiples - 1 : kNoBin;
  }

  static constexpr std::size_t kCacheDepth =
      detail::ArenaCacheDepth<Config>::value;
  static constexpr std::size_t kCacheBytes =
      detail::ArenaCacheBytes<Config>::value;

  // Exact-size lists a single thread owns outright. Sizes are indexed the same
  // way the fast bins are -- `size / kMinSliceAlignment - 1` -- so a request
  // whose rounded size lands in range checks one list head and is done, with no
  // lock and no shared line touched.
  //
  // Only the owning thread reads or writes its lists, so nothing here is atomic.
  // What *is* shared is `pool`, `next`, and `orphaned`: the pool reaches a live
  // cache through the registry to reclaim it, and a cache reaches the pool on
  // thread exit to hand its blocks back. Both directions take `mutex_`.
  struct ThreadCache {
    // Same bins as the global fast bins, so a size is cacheable exactly when it
    // is binnable and the two indexes agree.
    Chunk* lists[kFastBinCount] = {};
    std::size_t depths[kFastBinCount] = {};
    std::size_t bytes = 0;

    // Tallies this thread accumulated without the pool's lock. `GetStats` folds
    // them in rather than having the hot path write shared counters. Only hits
    // are tallied here: every miss and every free already holds `mutex_` and
    // writes `stats_` directly.
    std::size_t alloc_count = 0;
    std::size_t cache_hit_count = 0;

    // Held by the owning thread across a pop or a park, and by any other thread
    // reclaiming this cache. Uncontended in the common case -- two atomic RMWs,
    // an order of magnitude cheaper than the global mutex it stands in front of.
    // A reclaimer always takes it *after* `mutex_` and the owner's fast path
    // never holds `mutex_`, so there is one lock order and no cycle.
    std::atomic<bool> busy{false};

    // Registry link, guarded by `mutex_`.
    ThreadCache* next = nullptr;
  };

  // Shared rendezvous between the pool and the threads holding caches into it.
  // Either side may die first: a worker can outlive the pool, and the main
  // thread's cache is destroyed at process exit, long after any pool on the
  // stack. So the lock lives here rather than in the pool, held alive by a
  // `shared_ptr` from each side. A thread exiting locks it and reads `pool`; a
  // null `pool` means the pool is gone and took its memory upstream with it.
  //
  // One lock does both jobs -- guarding pool state and guarding this handshake --
  // precisely so there is no second lock to order against the first.
  struct Registry {
    std::mutex mutex;
    ArenaMemoryPool* pool = nullptr;
    ThreadCache* head = nullptr;
  };

  // Owns one thread's `ThreadCache` for one pool. The destructor runs on thread
  // exit -- or at process exit for the main thread -- and is what hands a
  // departing thread's blocks back, so memory is never stranded in the cache of
  // a thread that has gone away.
  //
  // Holding the registry by `shared_ptr` is what makes the destructor safe in
  // either order: if the pool went first it cleared `pool` and already reclaimed
  // this cache, and all that is left to do is free the node.
  class ThreadCacheHandle {
   public:
    explicit ThreadCacheHandle(std::shared_ptr<Registry> registry)
        : registry_(std::move(registry)), cache_(new ThreadCache) {
      std::lock_guard<std::mutex> lock(registry_->mutex);
      cache_->next = registry_->head;
      registry_->head = cache_;
    }

    ~ThreadCacheHandle() {
      {
        std::lock_guard<std::mutex> lock(registry_->mutex);
        if (registry_->pool != nullptr) {
          registry_->pool->RetireCacheLocked(cache_);
        }
        Unlink(cache_);
      }
      delete cache_;
    }

    ThreadCacheHandle(const ThreadCacheHandle&) = delete;
    ThreadCacheHandle& operator=(const ThreadCacheHandle&) = delete;

    ThreadCache* get() const { return cache_; }

   private:
    // Caller must hold `registry_->mutex`.
    void Unlink(ThreadCache* cache) {
      ThreadCache** link = &registry_->head;
      while (*link != nullptr && *link != cache) {
        link = &(*link)->next;
      }
      if (*link != nullptr) {
        *link = cache->next;
      }
    }

    std::shared_ptr<Registry> registry_;
    ThreadCache* cache_;
  };

  // Spin lock over one `ThreadCache`. Uncontended in the common case -- only a
  // reclaim landing on a cache whose owner is mid-operation contends -- and the
  // critical sections are a handful of loads, so spinning beats parking. The
  // owner's fast path holds this and nothing else; a reclaimer holds `mutex_`
  // first, so the one lock order is `mutex_` then `busy`.
  class CacheGuard {
   public:
    explicit CacheGuard(ThreadCache* cache) : cache_(cache) {
      while (cache_->busy.exchange(true, std::memory_order_acquire)) {
      }
    }
    ~CacheGuard() { cache_->busy.store(false, std::memory_order_release); }

    CacheGuard(const CacheGuard&) = delete;
    CacheGuard& operator=(const CacheGuard&) = delete;

   private:
    ThreadCache* cache_;
  };

  // One cache per (thread, pool instance) pair, keyed by registry address. That
  // pairing matters because a process may run several pools -- one per device --
  // and a block from one is not servable from another.
  //
  // A small vector rather than a hash: a thread touches one or two pools, so a
  // linear scan over a handful of pointers beats a hash lookup.
  struct CacheEntry {
    const Registry* key;
    std::unique_ptr<ThreadCacheHandle> handle;
  };

  // Destroyed at thread exit in reverse order, which is what returns this
  // thread's blocks. Function-local `thread_local` in a template gives one
  // instance per `ArenaMemoryPool` specialization, so different `Upstream` types
  // never share a vector.
  static std::vector<CacheEntry>& CacheMap() {
    static thread_local std::vector<CacheEntry> caches;
    return caches;
  }

  ThreadCache* LocalCache() {
    if (ThreadCache* cache = LocalCacheIfPresent(); cache != nullptr) {
      return cache;
    }
    // Takes `registry_->mutex` -- which is `mutex_` -- to link the new cache in,
    // so this must not run with the pool's lock held.
    CacheMap().push_back(
        CacheEntry{registry_.get(),
                   std::make_unique<ThreadCacheHandle>(registry_)});
    return CacheMap().back().handle->get();
  }

  // This thread's cache for this pool if it already has one, else `nullptr`.
  // Reads thread-local state and takes no lock, so unlike `LocalCache` it is
  // safe to call while holding `mutex_`.
  ThreadCache* LocalCacheIfPresent() {
    for (const CacheEntry& entry : CacheMap()) {
      if (entry.key == registry_.get()) {
        return entry.handle->get();
      }
    }
    return nullptr;
  }

  static std::size_t RoundUp(std::size_t size, std::size_t granularity) {
    return (size + granularity - 1) / granularity * granularity;
  }

  // Requests round to the slice alignment and nothing coarser. The 2 MB
  // rounding `MemoryPool` applies to large requests exists to make exact
  // size-class matching hit; with splitting and coalescing that motivation is
  // gone, and coarse rounding would waste up to 2 MB per large allocation.
  static std::size_t RoundSize(std::size_t size) {
    return RoundUp(size, Config::kMinSliceAlignment);
  }

  static std::size_t SliceAlignment(std::size_t alignment) {
    return alignment > Config::kMinSliceAlignment ? alignment
                                                  : Config::kMinSliceAlignment;
  }

  static void* Offset(void* ptr, std::size_t bytes) {
    return static_cast<char*>(ptr) + bytes;
  }

  static std::size_t AlignPadding(void* ptr, std::size_t alignment) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    const auto aligned = (address + alignment - 1) &
                         ~static_cast<std::uintptr_t>(alignment - 1);
    return static_cast<std::size_t>(aligned - address);
  }

  // A backing is drained when nothing it contains is handed out. With deferred
  // coalescing its chunk list may still hold many adjacent free chunks, so this
  // does *not* imply the span is available as one extent -- `Detach` runs
  // `CoalesceBacking` to restore that before handing the memory back.
  static bool IsDrained(const BackingStore* backing) {
    return backing->live_chunks == 0;
  }

  Chunk* NewChunk() {
    void* storage = chunk_arena_.Allocate(sizeof(Chunk), alignof(Chunk));
    return new (storage) Chunk();
  }

  void DeleteChunk(Chunk* chunk) {
    chunk->~Chunk();
    chunk_arena_.Deallocate(chunk, sizeof(Chunk), alignof(Chunk));
  }

  void DestroyChunks(BackingStore* backing) {
    Chunk* chunk = backing->head;
    while (chunk != nullptr) {
      Chunk* next = chunk->next;
      DeleteChunk(chunk);
      chunk = next;
    }
    backing->head = nullptr;
  }

  // Publishes `chunk` as free, in its exact-size bin when one exists and in the
  // tree otherwise. Caller must hold `mutex_`.
  void InsertFree(Chunk* chunk) {
    const std::size_t bin = BinIndex(chunk->size);
    if (bin == kNoBin) {
      chunk->location = Location::kTree;
      free_chunks_.insert(chunk);
      return;
    }

    chunk->location = Location::kFastBin;
    chunk->fast_prev = nullptr;
    chunk->fast_next = fast_bins_[bin];
    if (chunk->fast_next != nullptr) {
      chunk->fast_next->fast_prev = chunk;
    }
    fast_bins_[bin] = chunk;
    fast_bitmap_[bin / 64] |= 1ull << (bin % 64);
  }

  // Removes `chunk` from whichever container holds it. Must be called before
  // any change to `chunk->size`, since the size selects the bin. Caller must
  // hold `mutex_`.
  void EraseFree(Chunk* chunk) {
    if (chunk->location == Location::kNone) {
      return;
    }
    if (chunk->location == Location::kTree) {
      chunk->location = Location::kNone;
      free_chunks_.erase(chunk);
      return;
    }

    const std::size_t bin = BinIndex(chunk->size);
    if (chunk->fast_prev != nullptr) {
      chunk->fast_prev->fast_next = chunk->fast_next;
    } else {
      fast_bins_[bin] = chunk->fast_next;
    }
    if (chunk->fast_next != nullptr) {
      chunk->fast_next->fast_prev = chunk->fast_prev;
    }
    chunk->fast_prev = nullptr;
    chunk->fast_next = nullptr;
    chunk->location = Location::kNone;

    if (fast_bins_[bin] == nullptr) {
      fast_bitmap_[bin / 64] &= ~(1ull << (bin % 64));
    }
  }

  // Pops an exact-size block from this thread's cache, or returns `nullptr`.
  // Takes no pool lock and touches no shared state, which is the whole point.
  //
  // `bin` selects by exact size, so a hit means the block is exactly `rounded`
  // bytes: no split, no alignment padding, and no internal waste to account
  // for. Alignment beyond `kMinSliceAlignment` is not served from here at all
  // (see `Allocate`), so the chunk's own start alignment is sufficient.
  void* TryCacheAllocate(std::size_t rounded) {
    const std::size_t bin = BinIndex(rounded);
    if (bin == kNoBin) {
      return nullptr;
    }

    ThreadCache* cache = LocalCache();
    CacheGuard guard(cache);
    Chunk* chunk = cache->lists[bin];
    if (chunk == nullptr) {
      return nullptr;
    }

    cache->lists[bin] = chunk->fast_next;
    --cache->depths[bin];
    cache->bytes -= chunk->size;
    ++cache->alloc_count;
    ++cache->cache_hit_count;

    chunk->fast_next = nullptr;
    chunk->cached = false;
    // Still in `allocated_` and still counted live by its backing -- parking
    // never removed either -- so handing it back needs no shared write at all.
    chunk->requested = rounded;
    return chunk->ptr;
  }

  // Parks `chunk` in `cache`, or returns false if the cache has no room for it.
  // Caller must hold `mutex_` and must have established that `chunk` is live.
  //
  // A parked chunk stays in `allocated_` and stays counted in its backing's
  // `live_chunks`, so `stats_.bytes_in_use` still covers it -- which is what
  // keeps automatic shrink from handing the backing upstream while the cache
  // points into it. `GetStats` reclassifies those bytes as free, since no caller
  // holds them.
  //
  // `bin` is the caller's already-resolved `BinIndex(chunk->size)`, never
  // `kNoBin`: the caller has to test that anyway to decide whether looking up a
  // cache is worth it, so recomputing it here would be the second time.
  bool TryParkCached(ThreadCache* cache, std::size_t bin, Chunk* chunk) {
    CacheGuard guard(cache);
    if (cache->depths[bin] >= kCacheDepth ||
        cache->bytes + chunk->size > kCacheBytes) {
      return false;
    }

    // `requested` becomes the whole extent: a cache hit is an exact-size match,
    // so nothing parked here carries internal waste, and the pop can leave the
    // waste tally alone.
    stats_.bytes_internal_waste -= chunk->size - chunk->requested;
    chunk->requested = chunk->size;
    chunk->cached = true;

    chunk->fast_next = cache->lists[bin];
    cache->lists[bin] = chunk;
    ++cache->depths[bin];
    cache->bytes += chunk->size;
    return true;
  }

  // Hands every block in `cache` back to the global free index. Caller must hold
  // `mutex_`; the cache may belong to another thread, which `CacheGuard` covers.
  //
  // Leaves the tallies in place: they are monotonic and `GetStats` folds them, so
  // clearing them here would lose allocations from the totals.
  void ReclaimCacheLocked(ThreadCache* cache) {
    CacheGuard guard(cache);
    if (cache->bytes == 0) {
      return;
    }

    for (std::size_t bin = 0; bin < kFastBinCount; ++bin) {
      Chunk* chunk = cache->lists[bin];
      cache->lists[bin] = nullptr;
      cache->depths[bin] = 0;
      while (chunk != nullptr) {
        Chunk* next = chunk->fast_next;
        chunk->fast_next = nullptr;
        chunk->cached = false;
        ReleaseChunk(chunk);
        chunk = next;
      }
    }
    cache->bytes = 0;
  }

  // Drains `cache` and absorbs its tallies, for a cache about to leave the
  // registry with its thread. `GetStats` folds live caches' tallies as it reads
  // them, so a departing one has to hand its own over or the allocations it
  // served would vanish from the totals. Caller must hold `mutex_`.
  void RetireCacheLocked(ThreadCache* cache) {
    ReclaimCacheLocked(cache);
    stats_.alloc_count += cache->alloc_count;
    stats_.cache_hit_count += cache->cache_hit_count;
    cache->alloc_count = 0;
    cache->cache_hit_count = 0;
  }

  // Drains every registered cache. Caller must hold `mutex_`.
  void ReclaimAllCaches() {
    for (ThreadCache* cache = registry_->head; cache != nullptr;
         cache = cache->next) {
      ReclaimCacheLocked(cache);
    }
  }

  // Total bytes parked across every cache, and the tallies to fold into `Stats`.
  // Caller must hold `mutex_`.
  void SumCaches(std::size_t* bytes, std::size_t* allocs,
                 std::size_t* hits) const {
    for (ThreadCache* cache = registry_->head; cache != nullptr;
         cache = cache->next) {
      CacheGuard guard(cache);
      *bytes += cache->bytes;
      *allocs += cache->alloc_count;
      *hits += cache->cache_hit_count;
    }
  }

  // Returns a chunk that is no longer held by anyone to the global free index.
  // `chunk` must already be out of `allocated_` and must not be cached. Caller
  // must hold `mutex_`.
  //
  // Shared by `Deallocate` and by cache reclaim, which differ only in who was
  // holding the block; the accounting from here down is identical.
  void ReleaseChunk(Chunk* chunk) {
    stats_.bytes_in_use -= chunk->size;
    stats_.bytes_internal_waste -= chunk->size - chunk->requested;

    chunk->is_free = true;
    chunk->requested = 0;
    stats_.bytes_free_in_backings += chunk->size;

    // No coalescing here -- see the class comment. Publishing the chunk at its
    // own size is the whole critical section, and for the common case (the next
    // request for this size pops this very chunk back out of its bin) merging
    // would be pure overhead: two neighbor loads, a tree rebalance, and a second
    // rebalance to split the result apart again.
    BackingStore* owner = chunk->owner;
    // Conservative: this free may have created an adjacent free pair, so the
    // next `FindFit` miss has to try coalescing before growing.
    coalesce_dirty_ = true;
    InsertFree(chunk);

    if (--owner->live_chunks == 0 && !owner->resident) {
      // This backing just drained and is shrinkable. Merge it now, even though
      // merging is otherwise deferred: the leftover slivers are *bait*. A 64 B
      // tail sitting in an exact-size bin is the best possible fit for the next
      // 64 B request, so it would pull a fresh live block into the one backing
      // that was about to be handed back, and shrink could never reclaim it.
      // Merged, the backing presents a single large chunk that best fit passes
      // over while any smaller one exists.
      //
      // The resident backing is deliberately excluded: it is shrink-exempt, so
      // it has no bait problem, and it is where a near-empty alloc/free loop
      // lives -- coalescing it on every drain would put a list walk and a tree
      // round trip back on exactly the hot path this deferral exists to clear.
      CoalesceBacking(owner);
      ++drained_candidates_;
    }
  }

  // First chunk in the lowest occupied bin at or above `from`, or `nullptr`.
  // The bitmap makes this a couple of word scans rather than a walk over 128
  // list heads. Caller must hold `mutex_`.
  Chunk* ScanBins(std::size_t from) const {
    if (from >= kFastBinCount) {
      return nullptr;
    }
    std::size_t word = from / 64;
    std::uint64_t bits = fast_bitmap_[word] & (~0ull << (from % 64));
    while (bits == 0) {
      if (++word >= kFastBinWords) {
        return nullptr;
      }
      bits = fast_bitmap_[word];
    }
    return fast_bins_[word * 64 + CountTrailingZeros(bits)];
  }

  // Best fit over what is currently indexed, coalescing and retrying once if
  // nothing fits. Deferred merging means a failure here does not mean the pool
  // is out of room -- adjacent free chunks may add up to a fit -- so growing a
  // new backing must never be decided on `FindFitIndexed` alone.
  //
  // Caller must hold `mutex_`.
  Chunk* FindFit(std::size_t needed) {
    if (Chunk* chunk = FindFitIndexed(needed); chunk != nullptr) {
      return chunk;
    }
    if (coalesce_dirty_) {
      CoalesceAll();
      if (Chunk* chunk = FindFitIndexed(needed); chunk != nullptr) {
        return chunk;
      }
    }

    // Last resort before growing: the memory may be parked in some thread's
    // cache. Reclaiming is what keeps the caches from turning retention into an
    // upstream call -- a thread must never hold blocks another thread's request
    // cannot get back.
    if (!AnyCached()) {
      return nullptr;
    }
    ReclaimAllCaches();
    if (Chunk* chunk = FindFitIndexed(needed); chunk != nullptr) {
      return chunk;
    }
    // Reclaim published chunks at their own sizes, which may have created
    // adjacent free pairs of its own.
    if (!coalesce_dirty_) {
      return nullptr;
    }
    CoalesceAll();
    return FindFitIndexed(needed);
  }

  // Whether any registered cache holds anything. Caller must hold `mutex_`.
  bool AnyCached() const {
    for (ThreadCache* cache = registry_->head; cache != nullptr;
         cache = cache->next) {
      CacheGuard guard(cache);
      if (cache->bytes != 0) {
        return true;
      }
    }
    return false;
  }

  // Best fit: the smallest indexed free chunk that can hold `needed`. Caller
  // must hold `mutex_`.
  Chunk* FindFitIndexed(std::size_t needed) {
    const std::size_t first = FirstEligibleBin(needed);

    // An occupied exact-fit bin ends the search: no chunk anywhere can be a
    // better fit, so the common case -- a workload cycling through a handful of
    // sizes -- never touches the tree at all. This is the whole point of the
    // bins; a best-fit tree lookup on every allocation was the pool's dominant
    // per-call cost.
    if (first != kNoBin && fast_bins_[first] != nullptr) {
      return fast_bins_[first];
    }

    // Otherwise both containers are candidates. Bins beyond the exact one hold
    // aligned sizes; the tree holds everything too large to bin plus the
    // non-aligned tails a backing's leading trim leaves behind, either of which
    // may be the tighter fit.
    Chunk* binned = first == kNoBin ? nullptr : ScanBins(first + 1);

    Chunk probe{};
    probe.size = needed;
    probe.ptr = nullptr;  // Sorts before any real chunk of the same size.
    auto it = free_chunks_.lower_bound(&probe);
    Chunk* treed = it == free_chunks_.end() ? nullptr : *it;

    if (binned == nullptr) {
      return treed;
    }
    if (treed == nullptr) {
      return binned;
    }
    return treed->size < binned->size ? treed : binned;
  }

  // Largest request servable without going upstream, for `Stats`. Because
  // coalescing is deferred, this is the longest *run* of adjacent free chunks,
  // not the largest indexed one -- a drained backing sitting as fifty separate
  // free chunks can still serve its full span, and reporting the largest single
  // chunk would understate the pool's capability by the deferral.
  //
  // Walks the chunk lists rather than the free index, so it is linear in the
  // chunk count. `GetStats` is a diagnostic call, not on the hot path.
  //
  // Caller must hold `mutex_`.
  std::size_t LargestFreeChunk() const {
    std::size_t largest = 0;
    for (const auto& backing : backings_) {
      std::size_t run = 0;
      for (const Chunk* chunk = backing->head; chunk != nullptr;
           chunk = chunk->next) {
        // A cached chunk counts toward the run: no caller holds it, so any
        // request that needs the span reclaims the caches and gets it. Excluding
        // it would report a drained backing as shattered.
        if (chunk->is_free || chunk->cached) {
          run += chunk->size;
          if (run > largest) {
            largest = run;
          }
        } else {
          run = 0;
        }
      }
    }
    return largest;
  }

  // Merges every adjacent free pair in `backing`, leaving one chunk per
  // contiguous free region. Runs in one address-order walk, so a full pass over
  // the pool is linear in the number of chunks rather than in merges performed.
  // Caller must hold `mutex_`.
  void CoalesceBacking(BackingStore* backing) {
    for (Chunk* chunk = backing->head; chunk != nullptr;) {
      if (!chunk->is_free) {
        chunk = chunk->next;
        continue;
      }
      // Absorb the whole free run to the right in one go. Only the survivor is
      // re-published, so a run of N chunks costs one insert, not N.
      Chunk* next = chunk->next;
      if (next == nullptr || !next->is_free) {
        chunk = next;
        continue;
      }
      EraseFree(chunk);
      while (next != nullptr && next->is_free) {
        EraseFree(next);
        chunk->size += next->size;
        Chunk* after = next->next;
        DeleteChunk(next);
        next = after;
      }
      chunk->next = next;
      if (next != nullptr) {
        next->prev = chunk;
      }
      InsertFree(chunk);
      chunk = next;
    }
  }

  // Coalesces every backing. Caller must hold `mutex_`.
  void CoalesceAll() {
    for (const auto& backing : backings_) {
      CoalesceBacking(backing.get());
    }
    coalesce_dirty_ = false;
  }

  // Splits `chunk` at `offset` and returns the tail. Purely structural: the
  // caller owns both halves' free-set membership and accounting, because the
  // two call sites want different outcomes for the left part.
  Chunk* SplitAt(Chunk* chunk, std::size_t offset) {
    Chunk* tail = NewChunk();
    tail->owner = chunk->owner;
    tail->ptr = Offset(chunk->ptr, offset);
    tail->size = chunk->size - offset;
    tail->is_free = chunk->is_free;
    tail->prev = chunk;
    tail->next = chunk->next;
    if (chunk->next != nullptr) {
      chunk->next->prev = tail;
    }
    chunk->next = tail;
    chunk->size = offset;
    return tail;
  }

  // Carves `rounded` bytes at `alignment` out of the free chunk `chunk`,
  // registers the result, and returns the pointer the caller sees. `chunk` must
  // be large enough for the request including alignment padding, which is what
  // `FindFit`'s `needed` guarantees. Caller must hold `mutex_`.
  void* Serve(Chunk* chunk, std::size_t rounded, std::size_t alignment) {
    EraseFree(chunk);
    BackingStore* owner = chunk->owner;
    if (owner->live_chunks++ == 0) {
      DropCandidate(owner);
    }
    stats_.bytes_free_in_backings -= chunk->size;
    // The whole extent is the pool's to carve now; the pieces handed back below
    // are re-marked individually.
    chunk->is_free = false;

    // Aligning inside the chunk leaves a leading gap. Split it off as its own
    // free chunk rather than folding it into the served block: with coalescing
    // it is genuinely reusable, which is also why this pool does not need
    // `MemoryPool`'s trick of over-requesting `rounded + alignment` upstream.
    const std::size_t padding = AlignPadding(chunk->ptr, alignment);
    if (padding != 0) {
      Chunk* body = SplitAt(chunk, padding);
      chunk->is_free = true;
      stats_.bytes_free_in_backings += chunk->size;
      InsertFree(chunk);
      chunk = body;
    }

    if (chunk->size - rounded >= Config::kMinSplitRemainder) {
      Chunk* tail = SplitAt(chunk, rounded);
      tail->is_free = true;
      stats_.bytes_free_in_backings += tail->size;
      InsertFree(tail);
    }
    // Otherwise the remainder stays with the served chunk: splitting it off
    // would only create a sliver too small to satisfy anything. It is counted
    // as internal waste below.

    chunk->requested = rounded;
    allocated_.Insert(chunk->ptr, chunk);

    stats_.bytes_in_use += chunk->size;
    stats_.bytes_internal_waste += chunk->size - chunk->requested;
    if (stats_.bytes_in_use > stats_.peak_bytes_in_use) {
      stats_.peak_bytes_in_use = stats_.bytes_in_use;
    }
    ++stats_.alloc_count;
    return chunk->ptr;
  }

  // Capacity to request for a new backing store. The floor carries one extra
  // slice alignment because the upstream base may not be `kMinSliceAlignment`
  // aligned and `AdoptBacking` trims up to that much off the front.
  //
  // Caller must hold `mutex_`.
  std::size_t NextCapacity(std::size_t needed) const {
    const std::size_t floor = needed + Config::kMinSliceAlignment;
    return next_capacity_ > floor ? next_capacity_ : floor;
  }

  // Takes ownership of a fresh upstream allocation and returns its sole free
  // chunk. Caller must hold `mutex_`.
  Chunk* AdoptBacking(void* base, std::size_t capacity) {
    auto backing = std::make_unique<BackingStore>();
    backing->base = base;
    backing->capacity = capacity;
    backing->oversize = capacity > Config::kMaxCapacity;
    // Residency follows the growth ramp, not arrival order: a first request
    // that happens to be huge gets an oversize backing, and pinning *that*
    // forever would be the opposite of what shrink is for.
    backing->resident = !backing->oversize && !HasResident();

    // Trim the front so every chunk in this backing -- and therefore every
    // pointer the pool hands out -- starts at a multiple of the slice
    // alignment. `base` itself is only guaranteed whatever the upstream
    // allocator promises. `capacity` still records what upstream gave us, since
    // that is what `Upstream::Free` releases.
    const std::size_t lead = AlignPadding(base, Config::kMinSliceAlignment);

    Chunk* chunk = NewChunk();
    chunk->owner = backing.get();
    chunk->ptr = Offset(base, lead);
    chunk->size = capacity - lead;
    chunk->is_free = true;
    backing->head = chunk;

    stats_.bytes_reserved += capacity;
    if (stats_.bytes_reserved > stats_.peak_bytes_reserved) {
      stats_.peak_bytes_reserved = stats_.bytes_reserved;
    }
    stats_.bytes_free_in_backings += chunk->size;
    stats_.bytes_unusable += lead;
    InsertFree(chunk);
    // A fresh backing is drained by definition, and the caller `Serve`s out of
    // it immediately -- which decrements. Counting it here keeps the pair
    // balanced.
    if (!backing->resident) {
      ++drained_candidates_;
    }

    if (!backing->oversize) {
      // Advance the ramp, capped. An oversize backing is a one-off exception
      // and must not advance it -- otherwise a single 2 GB request would make
      // the next ordinary backing 4 GB.
      next_capacity_ = capacity >= Config::kMaxCapacity / 2
                           ? Config::kMaxCapacity
                           : capacity * 2;
    }

    backings_.push_back(std::move(backing));
    return chunk;
  }

  // `drained_candidates_` counts drained, shrinkable (non-resident) backings:
  // exactly what an idle scan could find. Keeping it exact is what lets the
  // allocation path skip that scan on a single load in the steady state, so
  // every transition into or out of "drained and non-resident" must be paired.
  // The resident backing is never counted -- it is shrink-exempt, so counting it
  // would defeat the skip whenever the pool is idle.
  //
  // Caller must hold `mutex_`.
  void DropCandidate(const BackingStore* backing) {
    if (!backing->resident) {
      --drained_candidates_;
    }
  }

  // Caller must hold `mutex_`.
  bool HasResident() const {
    for (const auto& backing : backings_) {
      if (backing->resident) {
        return true;
      }
    }
    return false;
  }

  // Removes the backing at `index` from the pool, settling stats and dropping
  // its chunk from the free set. The returned owner keeps the upstream pointer
  // alive until the caller frees it outside the lock. Caller must hold
  // `mutex_`, and the backing must be drained.
  std::unique_ptr<BackingStore> Detach(std::size_t index) {
    std::unique_ptr<BackingStore> backing = std::move(backings_[index]);
    backings_.erase(backings_.begin() + static_cast<std::ptrdiff_t>(index));

    // Deferred coalescing may have left the drained span as many chunks. Merge
    // them so the accounting below (and the `head`-spans-everything assumption)
    // holds, exactly as it did when `Deallocate` merged eagerly.
    CoalesceBacking(backing.get());
    DropCandidate(backing.get());

    EraseFree(backing->head);
    stats_.bytes_free_in_backings -= backing->head->size;
    stats_.bytes_unusable -= backing->capacity - backing->head->size;
    stats_.bytes_reserved -= backing->capacity;
    ++stats_.upstream_free_count;
    DestroyChunks(backing.get());
    return backing;
  }

  static void FreeBackings(
      const std::vector<std::unique_ptr<BackingStore>>& doomed) {
    for (const auto& backing : doomed) {
      Upstream::Free(backing->base);
    }
  }

  // Updates the consecutive-small-allocation run and, when it indicates a burst
  // has ended, runs one idle scan. Selected backings are moved to `*doomed` for
  // the caller to free once the lock is dropped. Caller must hold `mutex_`.
  void UpdateShrinkState(std::size_t size,
                         std::vector<std::unique_ptr<BackingStore>>* doomed) {
    // Nothing is shrinkable, so there is nothing for a scan to find and no
    // reason to touch the run counter. This is the steady state -- one resident
    // backing, or every backing holding live blocks -- and skipping it here
    // keeps `Allocate` from dirtying a shared cache line on every call, which
    // under contention costs a remote-dirty miss for whichever thread holds the
    // lock next.
    if (drained_candidates_ == 0) {
      return;
    }

    if (size > Config::kSmallThreshold) {
      // A large request means a burst is starting or ongoing: keep everything.
      small_alloc_since_last_trim_ = 0;
      return;
    }

    if (++small_alloc_since_last_trim_ < Config::kShrinkThreshold) {
      return;
    }
    // Reset unconditionally, so a scan that frees nothing does not repeat on
    // every subsequent allocation.
    small_alloc_since_last_trim_ = 0;

    // The burst is over, so the blocks it left parked in caches are what is
    // standing between its backings and the upstream allocator. Reclaiming here
    // is what lets the scan below actually find them drained -- and it is also
    // where a thread that has gone quiet stops holding memory: the run of small
    // allocations that got us here is the signal that nothing needs it.
    ReclaimAllCaches();

    // Reverse iteration keeps the remaining indices valid across erases.
    for (std::size_t i = backings_.size(); i-- > 0;) {
      BackingStore* backing = backings_[i].get();
      if (backing->resident) {
        continue;
      }
      if (!IsDrained(backing)) {
        backing->empty_scans = 0;
        continue;
      }
      ++backing->empty_scans;
      // Oversize backings skip the hysteresis: holding gigabytes idle for
      // another round of small allocations costs far more than one upstream
      // call.
      if (!backing->oversize &&
          backing->empty_scans < Config::kEmptyScansToDestroy) {
        continue;
      }
      doomed->push_back(Detach(i));
      ++stats_.shrink_count;
    }
  }

  // OOM fallback chain, called with the lock released after `Upstream::Malloc`
  // failed. Frees drained backings (the upstream allocator may be exactly what
  // is out of memory), retries at the requested capacity, then retries at the
  // smallest capacity that can serve this one request. Returns `failure` -- the
  // original error -- if none of that helps.
  Error AllocateFallback(void** base, std::size_t* capacity,
                         std::size_t needed, Error failure) {
    std::vector<std::unique_ptr<BackingStore>> doomed;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Upstream is out of memory, so every retained byte is worth having back.
      ReclaimAllCaches();
      for (std::size_t i = backings_.size(); i-- > 0;) {
        if (IsDrained(backings_[i].get())) {
          doomed.push_back(Detach(i));
        }
      }
    }

    if (!doomed.empty()) {
      FreeBackings(doomed);
      doomed.clear();
      if (Upstream::Malloc(base, *capacity) == Upstream::kSuccess) {
        return Upstream::kSuccess;
      }
    }

    const std::size_t minimum = needed + Config::kMinSliceAlignment;
    if (*capacity > minimum &&
        Upstream::Malloc(base, minimum) == Upstream::kSuccess) {
      *capacity = minimum;
      return Upstream::kSuccess;
    }
    return failure;
  }

  // Outlives this pool when a thread holding a cache does. Declared first so it
  // is constructed before the reference below binds to it.
  std::shared_ptr<Registry> registry_;
  // The pool's one lock, living in the registry so a departing thread can take
  // it without having to know whether the pool is still there. Named as a member
  // because every critical section in this file locks it directly.
  std::mutex& mutex_;

  // Declared before `free_chunks_`: the set's nodes come from `free_set_arena_`,
  // so the arena must outlive it. Members are destroyed in reverse declaration
  // order, which puts the set first.
  detail::NodeArena chunk_arena_;
  detail::NodeArena free_set_arena_;

  detail::PointerTable<Chunk*> allocated_;
  FreeSet free_chunks_{BySizeThenAddress{},
                       detail::ArenaAllocator<Chunk*>{&free_set_arena_}};
  // Exact-size free lists and an occupancy bitmap over them. Together these
  // keep the allocation hot path off the tree: a request whose size matches an
  // occupied bin is served by popping a list head.
  Chunk* fast_bins_[kFastBinCount] = {};
  std::uint64_t fast_bitmap_[kFastBinWords] = {};

  std::vector<std::unique_ptr<BackingStore>> backings_;
  std::size_t next_capacity_ = Config::kInitialCapacity;
  std::size_t small_alloc_since_last_trim_ = 0;
  // Drained non-resident backings: the exact number of things an idle scan could
  // find. Zero is the steady state and lets `UpdateShrinkState` return on one
  // load.
  std::size_t drained_candidates_ = 0;
  // Set by any `Deallocate` that may have created an adjacent free pair, cleared
  // by `CoalesceAll`. Lets a `FindFit` miss skip the coalescing pass when
  // nothing has been released since the last one.
  bool coalesce_dirty_ = false;
  Stats stats_;
};

}  // namespace infini::rt

#endif
