// A/B benchmark of `MemoryPool` against `ArenaMemoryPool` in one process.
//
// This is a different axis from `pool_ab.cc`, which compares two git revisions
// of one header. Here both headers are the current ones and the question is
// which *design* wins: a size-class cache that calls upstream once per miss, or
// an arena that reserves a large backing and slices it.
//
// The comparison is meaningless on a fast upstream. A host `malloc` costs tens
// of nanoseconds, so the arena's whole advantage -- turning N upstream calls into
// one -- is worth less than the bookkeeping it adds, and the arena loses on every
// timing. That is a real result for the CPU backend and it is reported as such,
// but it says nothing about a device. So three upstreams are used:
//
//   `HostUpstream`   - `std::malloc`, tens of ns. The CPU backend. The arena's
//                      floor: pure overhead, no amortization to earn back.
//   `SlowUpstream`   - a calibrated busy-wait, tens of us. A synchronous
//                      `cudaMalloc`. This is where the arena is supposed to win,
//                      and by how much is the number this harness exists to
//                      produce.
//   Call counting    - both upstreams tally calls, so every timing is reported
//                      next to the exact upstream call count that produced it.
//                      A count is immune to machine noise and is the honest
//                      summary of what the arena changes.
//
// Output format matches `tests/performance/perf_common.h`: one JSON object per
// line on stdout, a human-readable table on stderr. Every benchmark emits two
// rows differing only in the `allocator` param (`pool` vs `arena`).
#include <infini/rt/arena_memory_pool.h>
#include <infini/rt/memory_pool.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "perf_common.h"

namespace {

namespace perf = infini::rt::perf;

// Counts calls into the global operator new so host heap traffic is an exact
// count rather than something inferred from a timing. Both pools claim to
// perform no host allocation in steady state -- the size-class pool via its
// inline `PointerTable`, the arena additionally via a `NodeArena` behind its
// ordered free set -- and this is what checks the claim.
std::atomic<std::size_t> g_operator_new_calls{0};

}  // namespace

// Replacing the global allocation functions is the only portable way to observe
// per-insert container allocation. Defined at global scope because the standard
// requires these to be replaced, not overloaded.
void* operator new(std::size_t size) {
  g_operator_new_calls.fetch_add(1, std::memory_order_relaxed);
  void* ptr = std::malloc(size);
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void* operator new[](std::size_t size) { return operator new(size); }

void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete[](void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::align_val_t) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::align_val_t) noexcept { std::free(ptr); }

namespace {

// --------------------------------------------------------------------------
// Upstream allocators
// --------------------------------------------------------------------------

// Fast path: `std::malloc` directly, as the CPU backend's `Runtime::Malloc`
// does. Aligned to 256 B to match what a device allocator guarantees, so the
// pools' own alignment logic is what the timings observe.
struct HostUpstream {
  using Error = int;
  static constexpr Error kSuccess = 0;

  static std::atomic<std::size_t> mallocs;
  static std::atomic<std::size_t> frees;

  static Error Malloc(void** ptr, std::size_t size) {
    mallocs.fetch_add(1, std::memory_order_relaxed);
    *ptr = std::aligned_alloc(256, (size + 255) / 256 * 256);
    return (size != 0 && *ptr == nullptr) ? 2 : 0;
  }

  static Error Free(void* ptr) {
    frees.fetch_add(1, std::memory_order_relaxed);
    std::free(ptr);
    return 0;
  }

  static void Reset() {
    mallocs.store(0, std::memory_order_relaxed);
    frees.store(0, std::memory_order_relaxed);
  }
};

std::atomic<std::size_t> HostUpstream::mallocs{0};
std::atomic<std::size_t> HostUpstream::frees{0};

// Slow path: models a synchronous device allocator. Busy-waits rather than
// sleeping, so the stall is CPU-bound like a real driver call and not a
// scheduler artifact that would let other threads run for free.
struct SlowUpstream {
  using Error = int;
  static constexpr Error kSuccess = 0;

  // Set from the command line; 50 us is the order of a `cudaMalloc`.
  static double stall_us;

  static std::atomic<std::size_t> mallocs;
  static std::atomic<std::size_t> frees;

  static void Stall() {
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - start)
               .count() < stall_us) {
    }
  }

  static Error Malloc(void** ptr, std::size_t size) {
    mallocs.fetch_add(1, std::memory_order_relaxed);
    Stall();
    *ptr = std::aligned_alloc(256, (size + 255) / 256 * 256);
    return (size != 0 && *ptr == nullptr) ? 2 : 0;
  }

  static Error Free(void* ptr) {
    frees.fetch_add(1, std::memory_order_relaxed);
    Stall();
    std::free(ptr);
    return 0;
  }

  static void Reset() {
    mallocs.store(0, std::memory_order_relaxed);
    frees.store(0, std::memory_order_relaxed);
  }
};

double SlowUpstream::stall_us = 50.0;
std::atomic<std::size_t> SlowUpstream::mallocs{0};
std::atomic<std::size_t> SlowUpstream::frees{0};

// --------------------------------------------------------------------------
// Configuration
// --------------------------------------------------------------------------

// A megabyte-scale arena config. The production default reserves 64 MB per
// backing and ramps to 512 MB; at that scale this harness would reserve
// gigabytes of host memory and measure the page allocator rather than the pool.
// The ratios that matter are preserved: an 8x doubling headroom to the cap, and
// a small/large threshold well below it.
struct AbArenaConfig {
  static constexpr std::size_t kInitialCapacity = 8ull << 20;  // 8 MB
  static constexpr std::size_t kMaxCapacity = 64ull << 20;     // 64 MB
  static constexpr std::size_t kSmallThreshold = 1ull << 20;   // 1 MB
  static constexpr std::size_t kMinSliceAlignment = 512;
  static constexpr std::size_t kMinSplitRemainder = 512;
  static constexpr std::size_t kShrinkThreshold = 16;
  static constexpr std::uint32_t kEmptyScansToDestroy = 2;
};

// Each arm is named by a tag carrying both the output label and the pool
// template, so every benchmark below is written once and instantiated twice.
struct SizeClassArm {
  static constexpr const char* kName = "pool";
  template <typename Upstream>
  using Pool = infini::rt::MemoryPool<Upstream>;
};

struct ArenaArm {
  static constexpr const char* kName = "arena";
  template <typename Upstream>
  using Pool = infini::rt::ArenaMemoryPool<Upstream, AbArenaConfig>;
};

// --------------------------------------------------------------------------
// Result collection
// --------------------------------------------------------------------------

// One comparison row. `higher_is_better` flips the ratio for metrics where a
// larger number is the good outcome; `pool_only` marks rows where the "no pool"
// notion does not apply.
struct Comparison {
  std::string workload;
  std::string params;
  std::string unit;
  double pool = 0.0;
  double arena = 0.0;
  bool higher_is_better = false;
};

std::vector<Comparison> g_comparisons;

void Record(std::string workload, std::string params, std::string unit,
            double pool, double arena, bool higher_is_better = false) {
  g_comparisons.push_back(Comparison{std::move(workload), std::move(params),
                                     std::move(unit), pool, arena,
                                     higher_is_better});
}

std::vector<perf::Param> WithArm(std::vector<perf::Param> params,
                                 const char* arm) {
  params.push_back(perf::StringParam("allocator", arm));
  return params;
}

std::string DescribeSize(std::size_t size) {
  if (size >= 1024 * 1024) {
    return std::to_string(size / (1024 * 1024)) + " MiB";
  }
  if (size >= 1024) {
    return std::to_string(size / 1024) + " KiB";
  }
  return std::to_string(size) + " B";
}

// --------------------------------------------------------------------------
// 1. Host heap traffic
// --------------------------------------------------------------------------

// Counts `operator new` calls across a fixed alloc/free loop, after a warmup so
// one-time table and node-block growth is excluded. This is the arena's biggest
// structural risk: it keeps free extents in a `std::set`, which without the
// `NodeArena` behind it would allocate once per insert and once per erase --
// two host allocations per pooled allocation, worse than no pool at all.
template <typename Arm>
double MeasureHostHeapTraffic(std::size_t iterations) {
  typename Arm::template Pool<HostUpstream> pool;

  for (std::size_t i = 0; i < 2000; ++i) {
    void* ptr = nullptr;
    pool.Allocate(&ptr, 4096);
    pool.Deallocate(ptr);
  }

  const auto before = g_operator_new_calls.load(std::memory_order_relaxed);
  for (std::size_t i = 0; i < iterations; ++i) {
    void* ptr = nullptr;
    pool.Allocate(&ptr, 4096);
    pool.Deallocate(ptr);
  }
  const auto calls = static_cast<double>(
      g_operator_new_calls.load(std::memory_order_relaxed) - before);

  perf::PrintResult(
      "arena_vs_pool.HostHeapTraffic",
      WithArm({perf::NumberParam("size_bytes", 4096)}, Arm::kName), iterations,
      "count", calls, calls);
  return calls;
}

// Same measurement under churn at mixed sizes, where the arena's free set is
// genuinely exercised: extents are split and coalesced on every operation, so
// the set sees inserts and erases rather than sitting on one entry.
template <typename Arm>
double MeasureHostHeapTrafficUnderChurn(std::size_t iterations) {
  constexpr std::size_t kLive = 64;
  constexpr std::size_t kClasses = 32;
  typename Arm::template Pool<HostUpstream> pool;

  std::mt19937 rng(4242);
  std::vector<void*> live;
  live.reserve(kLive);

  auto step = [&pool, &live, &rng](std::size_t index) {
    if (live.size() < kLive && ((rng() & 3) != 0 || live.empty())) {
      void* ptr = nullptr;
      if (pool.Allocate(&ptr, (index % kClasses + 1) * 512) == 0) {
        live.push_back(ptr);
      }
    } else {
      const std::size_t at = rng() % live.size();
      pool.Deallocate(live[at]);
      live.erase(live.begin() + static_cast<std::ptrdiff_t>(at));
    }
  };

  for (std::size_t i = 0; i < 20000; ++i) {  // warm the tables and node blocks
    step(i);
  }

  const auto before = g_operator_new_calls.load(std::memory_order_relaxed);
  for (std::size_t i = 0; i < iterations; ++i) {
    step(i);
  }
  const auto calls = static_cast<double>(
      g_operator_new_calls.load(std::memory_order_relaxed) - before);

  for (void* ptr : live) {
    pool.Deallocate(ptr);
  }

  perf::PrintResult("arena_vs_pool.HostHeapTrafficUnderChurn",
                    WithArm({perf::NumberParam("live_blocks", kLive),
                             perf::NumberParam("size_classes", kClasses)},
                            Arm::kName),
                    iterations, "count", calls, calls);
  return calls;
}

// --------------------------------------------------------------------------
// 2. Steady state on a fast upstream -- the arena's overhead floor
// --------------------------------------------------------------------------

// Allocate one block, free it, repeat. Every iteration after the first is a hit
// in both designs, so this isolates per-operation bookkeeping with no upstream
// cost to amortize. The arena is expected to lose here: it does strictly more
// work per hit (an ordered-set lookup, a split, a coalesce) than a free-list
// pop, and this prices exactly that.
template <typename Arm>
perf::Measurement BenchSteadyStateHit(std::size_t size,
                                      std::size_t iterations) {
  typename Arm::template Pool<HostUpstream> pool;
  return perf::RunBenchmarkMeasured(
      "arena_vs_pool.SteadyStateHit",
      WithArm(
          {perf::NumberParam("size_bytes", static_cast<std::uint64_t>(size))},
          Arm::kName),
      iterations, "ns", [&pool, size] {
        void* ptr = nullptr;
        auto status = pool.Allocate(&ptr, size);
        perf::DoNotOptimize(status);
        if (status == 0) {
          status = pool.Deallocate(ptr);
          perf::DoNotOptimize(status);
        }
      });
}

// A rolling window of live blocks: free the oldest, allocate a replacement --
// the shape a layer-by-layer inference loop produces. Keeps both pools' live
// tables genuinely populated, and for the arena keeps its free set fragmented
// rather than collapsed to one extent.
template <typename Arm>
perf::Measurement BenchLiveSetChurn(std::size_t live_blocks,
                                    std::size_t iterations) {
  constexpr std::size_t kSize = 4096;
  typename Arm::template Pool<HostUpstream> pool;

  std::vector<void*> blocks(live_blocks, nullptr);
  for (auto& block : blocks) {
    if (pool.Allocate(&block, kSize) != 0) {
      perf::SkipBenchmark("arena_vs_pool.LiveSetChurn", "prefill failed");
      return {};
    }
  }

  std::size_t cursor = 0;
  const auto measurement = perf::RunBenchmarkMeasured(
      "arena_vs_pool.LiveSetChurn",
      WithArm({perf::NumberParam("live_blocks", live_blocks),
               perf::NumberParam("size_bytes", kSize)},
              Arm::kName),
      iterations, "ns", [&pool, &blocks, &cursor, live_blocks] {
        void*& slot = blocks[cursor];
        cursor = (cursor + 1) % live_blocks;
        auto status = pool.Deallocate(slot);
        slot = nullptr;
        if (status == 0) {
          status = pool.Allocate(&slot, kSize);
        }
        perf::DoNotOptimize(status);
      });

  for (void* block : blocks) {
    if (block != nullptr) {
      pool.Deallocate(block);
    }
  }
  return measurement;
}

// --------------------------------------------------------------------------
// 3. The miss path on a slow upstream -- where the arena earns its keep
// --------------------------------------------------------------------------

// A growing set of live blocks at mixed sizes with nothing freed until the end.
// Neither pool can reuse anything, so every allocation is a miss. The size-class
// pool must call upstream once per block; the arena calls upstream once per
// backing and slices the rest. On a 50 us upstream that is the difference
// between N stalls and a handful.
//
// Reported per whole build-up rather than per block, since one sample is one
// cycle. The upstream call count is reported alongside: it is the cause, the
// timing is the effect.
template <typename Arm, typename Upstream>
perf::Measurement BenchFirstTouchGrowth(std::size_t blocks,
                                        std::size_t samples,
                                        std::size_t* upstream_calls) {
  constexpr std::size_t kClasses = 64;
  constexpr std::size_t kStride = 512;

  std::vector<void*> live;
  live.reserve(blocks);
  std::size_t last_calls = 0;

  const auto measurement = perf::RunBenchmarkMeasured(
      "arena_vs_pool.FirstTouchGrowth",
      WithArm({perf::NumberParam("blocks", blocks)}, Arm::kName), samples, "us",
      [&live, &last_calls, blocks] {
        // A fresh pool per sample: a warm one would serve the whole build-up
        // from cache and measure the opposite of what this benchmark is for.
        Upstream::Reset();
        typename Arm::template Pool<Upstream> pool;
        for (std::size_t i = 0; i < blocks; ++i) {
          void* ptr = nullptr;
          if (pool.Allocate(&ptr, (i % kClasses + 1) * kStride) == 0) {
            live.push_back(ptr);
          }
        }
        last_calls = Upstream::mallocs.load(std::memory_order_relaxed);
        for (void* ptr : live) {
          pool.Deallocate(ptr);
        }
        live.clear();
      });

  *upstream_calls = last_calls;
  return measurement;
}

// Trim the cache every iteration so nothing is ever reused: each allocation must
// go upstream. This is both pools' worst case, and it prices what a pool costs
// when its cache is useless -- the arena still amortizes, because one backing
// covers the whole iteration's slicing, but it also has a whole backing to
// release on every trim.
template <typename Arm, typename Upstream>
perf::Measurement BenchTrimmedMissPath(std::size_t size,
                                       std::size_t iterations,
                                       std::size_t* upstream_calls) {
  Upstream::Reset();
  typename Arm::template Pool<Upstream> pool;

  const auto measurement = perf::RunBenchmarkMeasured(
      "arena_vs_pool.TrimmedMissPath",
      WithArm(
          {perf::NumberParam("size_bytes", static_cast<std::uint64_t>(size))},
          Arm::kName),
      iterations, "us", [&pool, size] {
        void* ptr = nullptr;
        auto status = pool.Allocate(&ptr, size);
        if (status == 0) {
          status = pool.Deallocate(ptr);
        }
        perf::DoNotOptimize(status);
        pool.ReleaseCached();
      });

  *upstream_calls = Upstream::mallocs.load(std::memory_order_relaxed);
  return measurement;
}

// Warm a cold pool up across many size classes on a slow upstream. The
// size-class pool pays one upstream stall per class before it can start
// hitting; the arena pays once and then serves every class out of the same
// backing, because coalescing lets a freed block of one size feed a request of
// another.
//
// One sample is one whole cold-start rotation, not one allocation. It has to be:
// `RunBenchmarkMeasured` warms up before it times, so a long-lived pool would
// have paid every per-class stall outside the timed region and the timing would
// report a steady state where both designs only ever hit -- the opposite of what
// this benchmark is for. A fresh pool per sample is the only way the cold-start
// cost lands inside the measurement.
template <typename Arm, typename Upstream>
perf::Measurement BenchMixedClassesSlow(std::size_t classes,
                                        std::size_t samples,
                                        std::size_t* upstream_calls) {
  constexpr std::size_t kStride = 512;
  std::size_t last_calls = 0;

  const auto measurement = perf::RunBenchmarkMeasured(
      "arena_vs_pool.MixedClassesSlow",
      WithArm({perf::NumberParam("size_classes", classes),
               perf::NumberParam("stride_bytes", kStride)},
              Arm::kName),
      samples, "us", [&last_calls, classes] {
        Upstream::Reset();
        typename Arm::template Pool<Upstream> pool;
        // Two passes: the first is all misses, the second all hits. Both arms
        // reach a warm state, so what separates them is the cost of getting
        // there.
        for (std::size_t pass = 0; pass < 2; ++pass) {
          for (std::size_t i = 0; i < classes; ++i) {
            void* ptr = nullptr;
            auto status = pool.Allocate(&ptr, (i + 1) * kStride);
            perf::DoNotOptimize(status);
            if (status == 0) {
              status = pool.Deallocate(ptr);
              perf::DoNotOptimize(status);
            }
          }
        }
        last_calls = Upstream::mallocs.load(std::memory_order_relaxed);
      });

  *upstream_calls = last_calls;
  return measurement;
}

// --------------------------------------------------------------------------
// 4. Memory amplification
// --------------------------------------------------------------------------

// How much upstream memory each design holds for the same live demand. Neither
// answer is strictly better -- the arena trades retention for upstream calls --
// so this is reported rather than judged. It is a byte count, not a timing, so
// it is exact.
//
// The shape is deliberately adversarial to size classes: a long tail of distinct
// sizes, each seen once. The size-class pool retains a block per class forever;
// the arena's coalescing folds them back into reusable extents.
template <typename Arm>
double MeasureAmplification(std::size_t classes) {
  constexpr std::size_t kStride = 512;
  constexpr std::size_t kLive = 32;
  typename Arm::template Pool<HostUpstream> pool;

  std::mt19937 rng(1337);
  std::vector<void*> live;
  std::size_t peak_demand = 0;
  std::size_t demand = 0;

  for (std::size_t i = 0; i < 20000; ++i) {
    if (live.size() < kLive && ((rng() & 3) != 0 || live.empty())) {
      const std::size_t size = (i % classes + 1) * kStride;
      void* ptr = nullptr;
      if (pool.Allocate(&ptr, size) == 0) {
        live.push_back(ptr);
        demand += size;
        peak_demand = std::max(peak_demand, demand);
      }
    } else {
      const std::size_t at = rng() % live.size();
      // Demand is tracked approximately: the exact size of the block being
      // freed is not retained, so the mean class size stands in. Only the
      // order of magnitude matters for an amplification ratio.
      demand -= std::min(demand, (classes / 2 + 1) * kStride);
      pool.Deallocate(live[at]);
      live.erase(live.begin() + static_cast<std::ptrdiff_t>(at));
    }
  }

  const auto reserved = static_cast<double>(pool.GetStats().bytes_reserved);
  for (void* ptr : live) {
    pool.Deallocate(ptr);
  }

  perf::PrintResult("arena_vs_pool.BytesReserved",
                    WithArm({perf::NumberParam("size_classes", classes),
                             perf::NumberParam("live_blocks", kLive)},
                            Arm::kName),
                    20000, "bytes", reserved, reserved);
  return reserved;
}

// --------------------------------------------------------------------------
// 5. Concurrency
// --------------------------------------------------------------------------

// Runs `op(thread, index)` on `threads` threads and reports ns per operation.
// Threads park on `go` so thread creation stays out of the timed region.
template <typename Op>
perf::Measurement RunThreaded(const std::string& benchmark,
                              const std::vector<perf::Param>& params,
                              std::size_t threads, std::size_t ops_per_thread,
                              Op&& op) {
  constexpr std::size_t kSamples = 7;
  const auto total_ops = threads * ops_per_thread;
  std::vector<double> samples;
  samples.reserve(kSamples);

  for (std::size_t sample = 0; sample < kSamples + 1; ++sample) {
    std::atomic<bool> go{false};
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (std::size_t t = 0; t < threads; ++t) {
      workers.emplace_back([&go, &op, ops_per_thread, t] {
        while (!go.load(std::memory_order_acquire)) {
        }
        for (std::size_t i = 0; i < ops_per_thread; ++i) {
          op(t, i);
        }
      });
    }

    const auto start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto& worker : workers) {
      worker.join();
    }
    const auto end = std::chrono::steady_clock::now();

    if (sample == 0) {  // warmup
      continue;
    }
    samples.push_back(
        std::chrono::duration<double, std::nano>(end - start).count() /
        static_cast<double>(total_ops));
  }

  const perf::Measurement measurement{perf::Mean(samples),
                                      perf::Median(samples)};
  perf::PrintResult(benchmark, params, total_ops, "ns", measurement.mean,
                    measurement.median);
  return measurement;
}

// Threads start at different size classes and rotate. Both pools serialize on
// one mutex, so this measures how long each holds it: the arena's critical
// section is longer (set lookup plus split plus coalesce), which is the cost it
// pays for needing upstream less often.
template <typename Arm>
perf::Measurement BenchConcurrentMixedSizes(std::size_t threads,
                                            std::size_t ops_per_thread) {
  constexpr std::size_t kClasses = 16;
  constexpr std::size_t kStride = 512;
  typename Arm::template Pool<HostUpstream> pool;

  return RunThreaded(
      "arena_vs_pool.ConcurrentMixedSizes",
      WithArm({perf::NumberParam("threads", threads),
               perf::NumberParam("size_classes", kClasses)},
              Arm::kName),
      threads, ops_per_thread, [&pool](std::size_t thread, std::size_t op) {
        const std::size_t size = ((thread + op) % kClasses + 1) * kStride;
        void* ptr = nullptr;
        auto status = pool.Allocate(&ptr, size);
        if (status == 0) {
          status = pool.Deallocate(ptr);
        }
        perf::DoNotOptimize(status);
      });
}

// --------------------------------------------------------------------------
// 6. Tail latency under a slow upstream
// --------------------------------------------------------------------------

struct Percentiles {
  double p50 = 0.0;
  double p99 = 0.0;
  double max = 0.0;
  double count = 0.0;
};

Percentiles Summarize(std::vector<double>& samples) {
  if (samples.empty()) {
    return {};
  }
  std::sort(samples.begin(), samples.end());
  const auto p99_index = std::min(
      samples.size() - 1, static_cast<std::size_t>(samples.size() * 0.99));
  return {samples[samples.size() / 2], samples[p99_index], samples.back(),
          static_cast<double>(samples.size())};
}

// One interfering thread repeatedly takes the miss path on a slow upstream while
// this thread only ever hits. Both pools call upstream with the lock released, so
// neither should let a hitter wait out a full stall -- this is the check that the
// arena did not regress that property while adding its shrink scan, which also
// runs on the allocation path.
template <typename Arm>
Percentiles MeasureHitStallUnderMiss(double seconds) {
  constexpr std::size_t kHitSize = 4096;
  typename Arm::template Pool<SlowUpstream> pool;

  // Prime the hit path so the measured allocation never misses.
  void* warm = nullptr;
  pool.Allocate(&warm, kHitSize);
  pool.Deallocate(warm);

  std::atomic<bool> stop{false};
  std::thread misser([&pool, &stop] {
    std::size_t i = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      void* ptr = nullptr;
      // A large fresh request every call, so neither design can serve it from
      // what it already holds.
      if (pool.Allocate(&ptr, (1u << 20) + (++i % 64) * 4096) == 0) {
        pool.Deallocate(ptr);
      }
      pool.ReleaseCached();
    }
  });

  std::vector<double> stalls;
  stalls.reserve(1u << 20);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
  while (std::chrono::steady_clock::now() < deadline) {
    void* ptr = nullptr;
    const auto start = std::chrono::steady_clock::now();
    const auto status = pool.Allocate(&ptr, kHitSize);
    const auto end = std::chrono::steady_clock::now();
    if (status == 0) {
      pool.Deallocate(ptr);
    }
    stalls.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }

  stop.store(true, std::memory_order_relaxed);
  misser.join();
  return Summarize(stalls);
}

void ReportPercentiles(const std::string& benchmark,
                       std::vector<perf::Param> params, const char* arm,
                       const Percentiles& p) {
  for (const auto& [suffix, value] :
       {std::pair{"p50", p.p50}, std::pair{"p99", p.p99},
        std::pair{"max", p.max}}) {
    auto row = params;
    row.push_back(perf::StringParam("percentile", suffix));
    perf::PrintResult(benchmark, WithArm(std::move(row), arm),
                      static_cast<std::size_t>(p.count), "us", value, value);
  }
}

// --------------------------------------------------------------------------
// 7. Behavioral parity
// --------------------------------------------------------------------------

// A performance comparison between two allocators is only meaningful if both
// honor the same contract, so the shared parts of it are checked on each arm and
// any divergence is reported as a failure rather than left for the reader to
// infer from the timings.
//
// Only the *shared* contract is checked. The two designs deliberately differ on
// reuse geometry -- the size-class pool returns the same address for the same
// class, the arena returns whatever extent fits -- so pointer identity is not
// asserted here.
int g_parity_failures = 0;

void Check(bool ok, const std::string& what, const char* arm) {
  if (!ok) {
    std::cerr << "  PARITY FAIL [" << arm << "] " << what << "\n";
    ++g_parity_failures;
  }
}

template <typename Arm>
void CheckSharedContract() {
  const char* arm = Arm::kName;
  typename Arm::template Pool<HostUpstream> pool;

  void* zero = reinterpret_cast<void*>(0x1234);
  Check(pool.Allocate(&zero, 0) == 0 && zero == nullptr,
        "zero size succeeds with a null pointer", arm);
  Check(pool.Deallocate(nullptr) == 0, "freeing nullptr is a no-op", arm);
  Check(pool.Allocate(nullptr, 64) != 0, "a null out-pointer is rejected", arm);

  int not_from_pool = 0;
  Check(pool.Deallocate(&not_from_pool) != 0, "a foreign pointer is rejected",
        arm);

  void* live = nullptr;
  Check(pool.Allocate(&live, 8192) == 0, "an ordinary allocate succeeds", arm);
  Check(pool.Deallocate(live) == 0, "the first free succeeds", arm);
  Check(pool.Deallocate(live) != 0, "a double free is rejected", arm);

  void* aligned = nullptr;
  Check(pool.Allocate(&aligned, 4096, 4096) == 0, "aligned allocate succeeds",
        arm);
  Check(reinterpret_cast<std::uintptr_t>(aligned) % 4096 == 0,
        "requested alignment is honored", arm);
  pool.Deallocate(aligned);

  // Distinct live blocks must not alias, which a slicing bug in the arena would
  // violate in a way no counter would reveal.
  std::vector<std::pair<void*, std::size_t>> blocks;
  for (std::size_t i = 0; i < 400; ++i) {
    const std::size_t size = (i % 32 + 1) * 512;
    void* ptr = nullptr;
    if (pool.Allocate(&ptr, size) == 0) {
      std::memset(ptr, static_cast<int>(i & 0xff), size);
      blocks.emplace_back(ptr, size);
    }
  }
  bool distinct = true;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const auto* bytes = static_cast<const unsigned char*>(blocks[i].first);
    for (std::size_t j = 0; j < blocks[i].second; ++j) {
      if (bytes[j] != static_cast<unsigned char>(i & 0xff)) {
        distinct = false;
        break;
      }
    }
  }
  Check(distinct, "concurrently live blocks never alias", arm);
  Check(pool.GetStats().bytes_in_use > 0, "bytes_in_use is positive when live",
        arm);
  for (const auto& [ptr, size] : blocks) {
    pool.Deallocate(ptr);
  }

  const auto settled = pool.GetStats();
  Check(settled.bytes_in_use == 0, "bytes_in_use returns to zero", arm);
  Check(settled.alloc_count == settled.free_count,
        "alloc_count equals free_count", arm);
  Check(settled.cache_hit_count + settled.cache_miss_count ==
            settled.alloc_count,
        "hits plus misses equals allocs", arm);

  pool.ReleaseCached();
  Check(pool.GetStats().bytes_reserved == 0,
        "bytes_reserved is zero after a trim", arm);
}

// Every upstream `Malloc` must be matched by a `Free` once the pool dies,
// including blocks the caller never handed back.
template <typename Arm>
void CheckNoUpstreamLeak() {
  HostUpstream::Reset();
  {
    typename Arm::template Pool<HostUpstream> pool;
    std::vector<void*> live;
    for (std::size_t i = 0; i < 600; ++i) {
      void* ptr = nullptr;
      if (pool.Allocate(&ptr, 512 * (i % 24 + 1)) == 0) {
        live.push_back(ptr);
      }
    }
    for (std::size_t i = 0; i < live.size() / 2; ++i) {
      pool.Deallocate(live[i]);
    }
  }
  Check(HostUpstream::mallocs.load() == HostUpstream::frees.load(),
        "no upstream leak (mallocs == frees)", Arm::kName);
}

// Concurrent smoke test: state must not corrupt and accounting must settle.
template <typename Arm>
void CheckConcurrentIntegrity() {
  typename Arm::template Pool<HostUpstream> pool;
  std::atomic<std::size_t> errors{0};
  std::vector<std::thread> workers;

  for (std::size_t t = 0; t < 8; ++t) {
    workers.emplace_back([&pool, &errors, t] {
      for (std::size_t i = 0; i < 20000; ++i) {
        void* ptr = nullptr;
        if (pool.Allocate(&ptr, 512 * ((t + i) % 24 + 1)) != 0 ||
            ptr == nullptr) {
          errors.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        if (pool.Deallocate(ptr) != 0) {
          errors.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }

  Check(errors.load() == 0, "no errors under 8 concurrent threads", Arm::kName);
  Check(pool.GetStats().bytes_in_use == 0,
        "bytes_in_use settles to zero after concurrent churn", Arm::kName);
}

// --------------------------------------------------------------------------
// Summary
// --------------------------------------------------------------------------

void PrintSummary() {
  if (g_comparisons.empty()) {
    return;
  }

  std::cerr << "\n=== MemoryPool vs ArenaMemoryPool ===\n";
  std::cerr << std::left << std::setw(30) << "workload" << std::setw(16)
            << "params" << std::right << std::setw(16) << "pool"
            << std::setw(16) << "arena" << std::setw(12) << "arena win"
            << "\n";

  for (const Comparison& row : g_comparisons) {
    std::cerr << std::left << std::setw(30) << row.workload << std::setw(16)
              << row.params << std::right << std::fixed << std::setprecision(2)
              << std::setw(12) << row.pool << " " << std::setw(3) << row.unit
              << std::setw(12) << row.arena << " " << std::setw(3) << row.unit;

    // >1 means the arena won. Inverted for higher-is-better metrics so the
    // direction of "win" is the same in every row. A zero on either side is
    // spelled out rather than reported as a ratio: for a count metric zero is a
    // meaningful outcome, not missing data, and dividing by it would hide which
    // arm reached it.
    const double good = row.higher_is_better ? row.arena : row.pool;
    const double bad = row.higher_is_better ? row.pool : row.arena;
    if (good > 0.0 && bad > 0.0) {
      std::cerr << std::setw(11) << std::setprecision(2) << (good / bad) << "x";
    } else if (row.pool == 0.0 && row.arena == 0.0) {
      std::cerr << std::setw(12) << "both zero";
    } else if (row.arena == 0.0) {
      std::cerr << std::setw(12) << "arena zero";
    } else {
      std::cerr << std::setw(12) << "pool zero";
    }
    std::cerr << "\n";
  }

  std::cerr << "\narena win > 1 means ArenaMemoryPool is better on that row.\n";
  std::cerr
      << "Read the fast-upstream rows (SteadyStateHit, LiveSetChurn,\n"
         "ConcurrentMixedSizes) as the arena's overhead floor: with upstream\n"
         "at tens of ns there is nothing to amortize, so a ratio below 1 there\n"
         "is expected and is the CPU backend's real answer. The slow-upstream\n"
         "rows are the device case, and `calls` rows are the exact cause.\n";
  if (g_parity_failures == 0) {
    std::cerr << "shared contract: all checks passed on both allocators.\n";
  } else {
    std::cerr << "shared contract: " << g_parity_failures
              << " check(s) FAILED -- treat the timings above as suspect.\n";
  }
  std::cerr << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  double stall_seconds = 1.5;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--quick") {
      quick = true;
      stall_seconds = 0.5;
    } else if (arg.rfind("--stall-us=", 0) == 0) {
      SlowUpstream::stall_us = std::atof(arg.c_str() + 11);
    } else if (arg.rfind("--stall-seconds=", 0) == 0) {
      stall_seconds = std::atof(arg.c_str() + 16);
    } else if (arg == "--help" || arg == "-h") {
      std::cerr << "usage: arena_vs_pool [--quick] [--stall-us=N] "
                   "[--stall-seconds=N]\n"
                   "  --quick           shorter slow-upstream arms\n"
                   "  --stall-us=N      simulated upstream latency "
                   "(default 50, a cudaMalloc)\n"
                   "  --stall-seconds=N duration of each latency arm "
                   "(default 1.5)\n";
      return 0;
    }
  }

  std::cerr << "--- shared contract (must pass before timings mean "
               "anything) ---\n";
  CheckSharedContract<SizeClassArm>();
  CheckSharedContract<ArenaArm>();
  CheckNoUpstreamLeak<SizeClassArm>();
  CheckNoUpstreamLeak<ArenaArm>();
  CheckConcurrentIntegrity<SizeClassArm>();
  CheckConcurrentIntegrity<ArenaArm>();
  if (g_parity_failures == 0) {
    std::cerr << "  all contract checks passed.\n";
  }

  // 1. Host heap traffic -- exact counts, the least noisy signal here.
  const std::size_t traffic_ops = quick ? 20000 : 100000;
  {
    const double p = MeasureHostHeapTraffic<SizeClassArm>(traffic_ops);
    const double a = MeasureHostHeapTraffic<ArenaArm>(traffic_ops);
    Record("HostHeapTraffic", std::to_string(traffic_ops) + " ops", "calls", p,
           a);

    const double pc = MeasureHostHeapTrafficUnderChurn<SizeClassArm>(
        traffic_ops);
    const double ac = MeasureHostHeapTrafficUnderChurn<ArenaArm>(traffic_ops);
    Record("HostHeapTraffic/churn", std::to_string(traffic_ops) + " ops",
           "calls", pc, ac);
  }

  // 2. Fast upstream: the arena's overhead floor.
  const std::size_t iterations = quick ? 50000 : 200000;
  for (const std::size_t size : {4096u, 65536u, 1u << 20}) {
    const auto p = BenchSteadyStateHit<SizeClassArm>(size, iterations);
    const auto a = BenchSteadyStateHit<ArenaArm>(size, iterations);
    Record("SteadyStateHit", DescribeSize(size), "ns", p.median, a.median);
  }

  for (const std::size_t live : {1u, 8u, 64u, 512u}) {
    const auto p = BenchLiveSetChurn<SizeClassArm>(live, iterations);
    const auto a = BenchLiveSetChurn<ArenaArm>(live, iterations);
    Record("LiveSetChurn", std::to_string(live) + " live", "ns", p.median,
           a.median);
  }

  // 3. Memory amplification -- exact byte counts.
  {
    const double p = MeasureAmplification<SizeClassArm>(64);
    const double a = MeasureAmplification<ArenaArm>(64);
    Record("BytesReserved", "64 classes", "B", p, a);
  }

  // 4. Concurrency on a fast upstream.
  const std::size_t ops_per_thread = quick ? 20000 : 60000;
  for (const std::size_t threads : {1u, 2u, 4u, 8u}) {
    const auto p = BenchConcurrentMixedSizes<SizeClassArm>(threads,
                                                           ops_per_thread);
    const auto a = BenchConcurrentMixedSizes<ArenaArm>(threads, ops_per_thread);
    Record("ConcurrentMixedSizes", std::to_string(threads) + "T x16cls", "ns",
           p.median, a.median);
  }

  // 5. Slow upstream: the device case, and the reason the arena exists.
  {
    const std::size_t blocks = quick ? 200u : 800u;
    const std::size_t samples = quick ? 3u : 5u;
    std::size_t pool_calls = 0;
    std::size_t arena_calls = 0;

    const auto p = BenchFirstTouchGrowth<SizeClassArm, SlowUpstream>(
        blocks, samples, &pool_calls);
    const auto a = BenchFirstTouchGrowth<ArenaArm, SlowUpstream>(
        blocks, samples, &arena_calls);
    Record("FirstTouchGrowth", std::to_string(blocks) + " blocks", "us",
           p.median, a.median);

    for (const auto& [arm, calls] :
         {std::pair{"pool", pool_calls}, std::pair{"arena", arena_calls}}) {
      perf::PrintResult(
          "arena_vs_pool.FirstTouchGrowthUpstreamCalls",
          WithArm({perf::NumberParam("blocks", blocks)}, arm), blocks, "count",
          static_cast<double>(calls), static_cast<double>(calls));
    }
    Record("FirstTouchGrowth/upstream", std::to_string(blocks) + " blocks",
           "calls", static_cast<double>(pool_calls),
           static_cast<double>(arena_calls));
  }

  {
    const std::size_t classes = 64;
    // One sample is a whole cold-start rotation over every class, so a handful
    // of them is enough -- and on a 50 us upstream the size-class arm pays
    // `classes` stalls per sample.
    const std::size_t samples = quick ? 3u : 5u;
    std::size_t pool_calls = 0;
    std::size_t arena_calls = 0;

    const auto p = BenchMixedClassesSlow<SizeClassArm, SlowUpstream>(
        classes, samples, &pool_calls);
    const auto a = BenchMixedClassesSlow<ArenaArm, SlowUpstream>(
        classes, samples, &arena_calls);
    Record("MixedClassesSlow", std::to_string(classes) + " classes", "us",
           p.median, a.median);
    Record("MixedClassesSlow/upstream", std::to_string(classes) + " classes",
           "calls", static_cast<double>(pool_calls),
           static_cast<double>(arena_calls));
  }

  {
    const std::size_t iters = quick ? 100u : 300u;
    std::size_t pool_calls = 0;
    std::size_t arena_calls = 0;

    const auto p = BenchTrimmedMissPath<SizeClassArm, SlowUpstream>(
        65536, iters, &pool_calls);
    const auto a =
        BenchTrimmedMissPath<ArenaArm, SlowUpstream>(65536, iters, &arena_calls);
    Record("TrimmedMissPath", "64 KiB", "us", p.median, a.median);
  }

  // 6. Tail latency: does a hitter wait out an unrelated thread's stall?
  {
    const std::vector<perf::Param> params{perf::NumberParam(
        "stall_us", static_cast<std::uint64_t>(SlowUpstream::stall_us))};

    const auto p = MeasureHitStallUnderMiss<SizeClassArm>(stall_seconds);
    const auto a = MeasureHitStallUnderMiss<ArenaArm>(stall_seconds);
    ReportPercentiles("arena_vs_pool.HitStallUnderMiss", params, "pool", p);
    ReportPercentiles("arena_vs_pool.HitStallUnderMiss", params, "arena", a);
    Record("HitStall/Miss p50", "50us upstream", "us", p.p50, a.p50);
    Record("HitStall/Miss p99", "50us upstream", "us", p.p99, a.p99);
    Record("HitStall/Miss max", "50us upstream", "us", p.max, a.max);
  }

  PrintSummary();
  // A contract failure fails the run: a faster allocator that behaves
  // differently is not a win.
  return g_parity_failures == 0 ? 0 : 1;
}
