// Allocator matrix: the same workloads run across every allocation strategy
// available on the current backend, so the three comparisons the project cares
// about all come out of one binary.
//
//   `direct`      - the backend allocator itself (`malloc` / `cudaMalloc`).
//   `pool`        - `MemoryPool`, a size-class cache: one upstream call per miss.
//   `arena`       - `ArenaMemoryPool`, one upstream call per backing, then slices.
//   `cuda_async`  - `cudaMallocAsync`, CUDA's own stream-ordered pool. Present
//                   only where the backend supports it, and *not* semantically
//                   equivalent to the others: see `CudaAsyncArm`.
//
// Reading the output:
//   direct vs arena   - is the arena worth having at all on this backend?
//   pool vs arena     - which pool design wins, and on which shapes?
//   arena vs cuda_async - does a hand-written arena beat the vendor's pool?
//
// `perf_memory_pool.cc` already covers the general shapes (single block,
// working-set churn, mixed size classes, first-touch growth, thread scaling).
// This file deliberately does *not* repeat them. What it adds is the set of
// measurements that file cannot make:
//
//   - shapes that straddle `MemoryPool`'s 1 MB small/large boundary, where its
//     rounding granularity jumps from 512 B to 2 MB;
//   - large-block recycling at sizes that are *not* multiples of 2 MB, which is
//     the only way the size-class rounding waste becomes visible;
//   - growth to a gigabyte-scale high-water mark, far enough to walk the
//     production 64 MB -> 512 MB ramp;
//   - trim cost separated into its two components: bookkeeping traversal and
//     the number of upstream frees, which on a device are wildly different
//     costs because `cudaFree` implicitly synchronizes;
//   - thread counts past 8, where lock contention actually bends;
//   - device-only effects (implicit-sync cost, ledger accuracy against
//     `MemGetInfo`, multi-stream traffic);
//   - a synthetic layer-by-layer inference sequence, the only workload here
//     that resembles what the library is for.
//
// stdout is one JSON object per line for `scripts/compare_allocators.py`;
// stderr carries the human-readable tables.
#include <infini/rt.h>
#include <infini/rt/arena_memory_pool.h>
#include <infini/rt/memory_pool.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "perf_common.h"

namespace {

namespace perf = infini::rt::perf;
namespace runtime = infini::rt::runtime;

bool Success(runtime::Error status) { return status == runtime::kSuccess; }

// --------------------------------------------------------------------------
// Configuration
// --------------------------------------------------------------------------

// On a device the production values are the point: only at 64 MB -> 512 MB does
// the growth ramp, the oversize path, and the shrink heuristic behave the way
// they will in deployment. On the CPU backend the same values would reserve
// gigabytes of host memory and measure the page allocator, so the host build
// keeps the reduced scale and the difference is reported in the JSON so the two
// are never read as one series.
#if defined(INFINI_RT_PERF_LARGE_ARENA_CONFIG)
struct MatrixArenaConfig {
  static constexpr std::size_t kInitialCapacity = 64ull << 20;   // 64 MB
  static constexpr std::size_t kMaxCapacity = 512ull << 20;      // 512 MB
  static constexpr std::size_t kSmallThreshold = 1ull << 20;     // 1 MB
  static constexpr std::size_t kMinSliceAlignment = 512;
  static constexpr std::size_t kMinSplitRemainder = 512;
  static constexpr std::size_t kShrinkThreshold = 16;
  static constexpr std::uint32_t kEmptyScansToDestroy = 2;
};
constexpr const char* kConfigName = "production";
#else
struct MatrixArenaConfig {
  static constexpr std::size_t kInitialCapacity = 8ull << 20;  // 8 MB
  static constexpr std::size_t kMaxCapacity = 64ull << 20;     // 64 MB
  static constexpr std::size_t kSmallThreshold = 1ull << 20;   // 1 MB
  static constexpr std::size_t kMinSliceAlignment = 512;
  static constexpr std::size_t kMinSplitRemainder = 512;
  static constexpr std::size_t kShrinkThreshold = 16;
  static constexpr std::uint32_t kEmptyScansToDestroy = 2;
};
constexpr const char* kConfigName = "reduced";
#endif

// Scales every footprint in this file. The high-water and inference workloads
// are sized in gigabytes on a device; on a host build that would measure the
// page allocator, so they shrink by this divisor.
#if defined(INFINI_RT_PERF_LARGE_ARENA_CONFIG)
constexpr std::size_t kFootprintDivisor = 1;
constexpr bool kDeviceBackend = true;
#else
constexpr std::size_t kFootprintDivisor = 16;
constexpr bool kDeviceBackend = false;
#endif

// Forwards to the dispatch API, so one binary measures whichever backend the
// library was built with.
struct DispatchUpstream {
  using Error = runtime::Error;
  static constexpr Error kSuccess = runtime::kSuccess;

  static Error Malloc(void** ptr, std::size_t size) {
    return runtime::Malloc(ptr, size);
  }
  static Error Free(void* ptr) { return runtime::Free(ptr); }
};

// --------------------------------------------------------------------------
// Arms
// --------------------------------------------------------------------------
//
// Every arm exposes the same surface so each benchmark below is written once
// and instantiated per arm. `UpstreamAllocs`/`UpstreamFrees` are the exact
// counters that explain the timings; `Sync` is a no-op for the synchronous arms
// and the stream synchronization point for the asynchronous one.

class DirectArm {
 public:
  static constexpr const char* kName = "direct";
  static constexpr bool kStreamOrdered = false;
  // Nothing is retained past a `Deallocate`, so `ReleaseCached` is a no-op and
  // the benchmarks that exist to price a trim have nothing to price.
  static constexpr bool kHasCache = false;

  static bool Available() { return true; }

  runtime::Error Allocate(void** ptr, std::size_t size) {
    ++upstream_allocs_;
    return runtime::Malloc(ptr, size);
  }

  runtime::Error Deallocate(void* ptr) {
    if (ptr == nullptr) {
      return runtime::kSuccess;
    }
    ++upstream_frees_;
    return runtime::Free(ptr);
  }

  // No cache to trim, and nothing retained beyond what is live.
  void ReleaseCached() {}
  void Sync() {}

  std::size_t UpstreamAllocs() const { return upstream_allocs_; }
  std::size_t UpstreamFrees() const { return upstream_frees_; }
  // Direct allocation reserves exactly what is live, so retention is not a
  // meaningful axis for this arm and every byte metric is reported as absent
  // rather than as zero.
  bool TracksBytes() const { return false; }
  std::size_t BytesReserved() const { return 0; }

 private:
  std::size_t upstream_allocs_ = 0;
  std::size_t upstream_frees_ = 0;
};

class PoolArm {
 public:
  static constexpr const char* kName = "pool";
  static constexpr bool kStreamOrdered = false;
  static constexpr bool kHasCache = true;

  static bool Available() { return true; }

  runtime::Error Allocate(void** ptr, std::size_t size) {
    return pool_.Allocate(ptr, size);
  }
  runtime::Error Deallocate(void* ptr) { return pool_.Deallocate(ptr); }
  void ReleaseCached() { pool_.ReleaseCached(); }
  void Sync() {}

  std::size_t UpstreamAllocs() const {
    return pool_.GetStats().upstream_alloc_count;
  }
  std::size_t UpstreamFrees() const {
    return pool_.GetStats().upstream_free_count;
  }
  bool TracksBytes() const { return true; }
  std::size_t BytesReserved() const { return pool_.GetStats().bytes_reserved; }

 private:
  infini::rt::MemoryPool<DispatchUpstream> pool_;
};

class ArenaArm {
 public:
  static constexpr const char* kName = "arena";
  static constexpr bool kStreamOrdered = false;
  static constexpr bool kHasCache = true;

  static bool Available() { return true; }

  runtime::Error Allocate(void** ptr, std::size_t size) {
    return pool_.Allocate(ptr, size);
  }
  runtime::Error Deallocate(void* ptr) { return pool_.Deallocate(ptr); }
  void ReleaseCached() { pool_.ReleaseCached(); }
  void Sync() {}

  std::size_t UpstreamAllocs() const {
    return pool_.GetStats().upstream_alloc_count;
  }
  std::size_t UpstreamFrees() const {
    return pool_.GetStats().upstream_free_count;
  }
  bool TracksBytes() const { return true; }
  std::size_t BytesReserved() const { return pool_.GetStats().bytes_reserved; }

 private:
  infini::rt::ArenaMemoryPool<DispatchUpstream, MatrixArenaConfig> pool_;
};

// CUDA's own stream-ordered pool, present as the fairest available reference:
// the honest question is not whether an arena beats `cudaMalloc` -- of course it
// does -- but whether it beats the pool the vendor already ships.
//
// It is NOT semantically equivalent to the other three arms and its numbers must
// not be read as a drop-in speedup. `FreeAsync` only *orders* the release behind
// the stream's current work; it does not wait for it, and reuse is likewise
// stream-ordered. The synchronous arms return memory that is immediately safe
// for any consumer. So this arm gets to overlap release with compute in a way
// the others cannot, and every benchmark synchronizes it once at the sample
// boundary rather than per operation -- measuring what it actually offers, at
// the cost of a weaker guarantee.
class CudaAsyncArm {
 public:
  static constexpr const char* kName = "cuda_async";
  static constexpr bool kStreamOrdered = true;
  // The driver's pool has a cache, but exposes no way to release it on demand
  // short of destroying the pool, so `ReleaseCached` is a no-op here too and the
  // trim benchmarks have nothing to measure.
  static constexpr bool kHasCache = false;

  // Probed rather than assumed: the CPU backend's `MallocAsync` returns an
  // error, and not every device backend implements the stream-ordered API.
  static bool Available() {
    runtime::Stream stream{};
    if (!Success(runtime::StreamCreate(&stream))) {
      return false;
    }
    void* ptr = nullptr;
    const bool ok = Success(runtime::MallocAsync(&ptr, 4096, stream)) &&
                    Success(runtime::StreamSynchronize(stream));
    if (ok) {
      runtime::FreeAsync(ptr, stream);
      runtime::StreamSynchronize(stream);
    }
    runtime::StreamDestroy(stream);
    return ok;
  }

  CudaAsyncArm() { runtime::StreamCreate(&stream_); }
  ~CudaAsyncArm() {
    if (stream_ != runtime::Stream{}) {
      runtime::StreamSynchronize(stream_);
      runtime::StreamDestroy(stream_);
    }
  }

  CudaAsyncArm(const CudaAsyncArm&) = delete;
  CudaAsyncArm& operator=(const CudaAsyncArm&) = delete;

  runtime::Error Allocate(void** ptr, std::size_t size) {
    ++upstream_allocs_;
    return runtime::MallocAsync(ptr, size, stream_);
  }

  runtime::Error Deallocate(void* ptr) {
    if (ptr == nullptr) {
      return runtime::kSuccess;
    }
    ++upstream_frees_;
    return runtime::FreeAsync(ptr, stream_);
  }

  void ReleaseCached() {}
  void Sync() { runtime::StreamSynchronize(stream_); }

  // These count API calls, not driver allocations: the whole point of the
  // stream-ordered pool is that it services most of them from its own cache,
  // and it exposes no counter for how often it went to the driver. Reported
  // anyway so the column is not blank, but it is not comparable to the pools'
  // upstream counts.
  std::size_t UpstreamAllocs() const { return upstream_allocs_; }
  std::size_t UpstreamFrees() const { return upstream_frees_; }
  bool TracksBytes() const { return false; }
  std::size_t BytesReserved() const { return 0; }

 private:
  runtime::Stream stream_{};
  std::size_t upstream_allocs_ = 0;
  std::size_t upstream_frees_ = 0;
};

// --------------------------------------------------------------------------
// Result collection
// --------------------------------------------------------------------------

// One measured value for one arm. Kept as a flat list and pivoted at print
// time, so adding an arm needs no change to the table code.
struct Cell {
  std::string workload;
  std::string params;
  std::string arm;
  std::string unit;
  double value = 0.0;
  bool present = true;
};

std::vector<Cell> g_cells;
std::vector<std::string> g_arm_order;

void RecordCell(const std::string& workload, const std::string& params,
                const std::string& arm, const std::string& unit, double value,
                bool present = true) {
  g_cells.push_back(Cell{workload, params, arm, unit, value, present});
  if (std::find(g_arm_order.begin(), g_arm_order.end(), arm) ==
      g_arm_order.end()) {
    g_arm_order.push_back(arm);
  }
}

std::vector<perf::Param> WithArm(std::vector<perf::Param> params,
                                 const char* arm) {
  params.push_back(perf::StringParam("allocator", arm));
  params.push_back(perf::StringParam("arena_config", kConfigName));
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
// 1. Cross-threshold rotation
// --------------------------------------------------------------------------

// Rotates 1 KB / 2 MB / 4 KB. The sizes are chosen to straddle `MemoryPool`'s
// 1 MB small/large boundary, where its rounding granularity jumps from 512 B to
// 2 MB: the two small sizes land in distinct 512 B classes and the 2 MB one in
// its own large class, so no request can ever reuse another's block and the pool
// must hold one live block per class forever. The arena serves all three out of
// one backing, and coalescing means the 2 MB hole can be re-split into small
// requests.
//
// `perf_memory_pool.cc`'s `MixedSizeClasses` rotates 512 B-strided sizes that
// all stay on the small side of the boundary, so it never exercises the
// granularity jump this benchmark exists for.
template <typename Arm>
void BenchCrossThresholdRotation(std::size_t iterations) {
  const std::size_t sizes[] = {1024, 2ull << 20, 4096};
  constexpr std::size_t kCount = 3;

  Arm arm;
  std::size_t index = 0;
  const auto measurement = perf::RunBenchmarkMeasured(
      "allocator_matrix.CrossThresholdRotation",
      WithArm({perf::NumberParam("size_count", kCount)}, Arm::kName), iterations,
      "us", [&arm, &index, &sizes] {
        const std::size_t size = sizes[index];
        index = (index + 1) % kCount;
        void* ptr = nullptr;
        auto status = arm.Allocate(&ptr, size);
        perf::DoNotOptimize(status);
        if (Success(status)) {
          status = arm.Deallocate(ptr);
          perf::DoNotOptimize(status);
        }
      });
  arm.Sync();

  RecordCell("CrossThreshold", "1K/2M/4K", Arm::kName, "us",
             measurement.median);
  RecordCell("CrossThreshold/upstream", "1K/2M/4K", Arm::kName, "calls",
             static_cast<double>(arm.UpstreamAllocs()));
  RecordCell("CrossThreshold/reserved", "1K/2M/4K", Arm::kName, "B",
             static_cast<double>(arm.BytesReserved()), arm.TracksBytes());

  perf::PrintResult("allocator_matrix.CrossThresholdRotationUpstreamCalls",
                    WithArm({perf::NumberParam("size_count", kCount)},
                            Arm::kName),
                    iterations, "count",
                    static_cast<double>(arm.UpstreamAllocs()),
                    static_cast<double>(arm.UpstreamAllocs()));
}

// --------------------------------------------------------------------------
// 2. Large-block recycling
// --------------------------------------------------------------------------

// Rotates 9 / 11 / 13 MiB: deliberately *not* multiples of `MemoryPool`'s 2 MB
// large-size granularity. Each request is rounded up to 10 / 12 / 14 MiB, so the
// pool wastes 1 MB inside every block (about 8%) and, because the rounded sizes
// are distinct classes, retains one block of each forever. A rotation at 10 MiB
// would show none of this -- 10 is already a multiple of 2 -- which is why the
// sizes are odd.
//
// The arena rounds to its 512 B slice granularity instead, and coalescing lets
// one freed 13 MiB extent serve the next 9 MiB request.
template <typename Arm>
void BenchLargeBlockRecycle(std::size_t iterations) {
  const std::size_t sizes[] = {9ull << 20, 11ull << 20, 13ull << 20};
  constexpr std::size_t kCount = 3;

  // Probe first: three live blocks at once must fit, or the numbers would be an
  // OOM rather than a measurement.
  {
    std::vector<void*> probe;
    bool ok = true;
    for (std::size_t i = 0; i < kCount && ok; ++i) {
      void* ptr = nullptr;
      ok = Success(runtime::Malloc(&ptr, sizes[i])) && ptr != nullptr;
      if (ok) {
        probe.push_back(ptr);
      }
    }
    for (void* ptr : probe) {
      runtime::Free(ptr);
    }
    if (!ok) {
      perf::SkipBenchmark("allocator_matrix.LargeBlockRecycle",
                          "device cannot hold the working set");
      return;
    }
  }

  Arm arm;
  std::size_t index = 0;
  const auto measurement = perf::RunBenchmarkMeasured(
      "allocator_matrix.LargeBlockRecycle",
      WithArm({perf::NumberParam("size_count", kCount)}, Arm::kName), iterations,
      "us", [&arm, &index, &sizes] {
        const std::size_t size = sizes[index];
        index = (index + 1) % kCount;
        void* ptr = nullptr;
        auto status = arm.Allocate(&ptr, size);
        perf::DoNotOptimize(status);
        if (Success(status)) {
          status = arm.Deallocate(ptr);
          perf::DoNotOptimize(status);
        }
      });
  arm.Sync();

  // Demand is the largest single request, since only one block is live at a
  // time. Anything reserved beyond that is the design's retention.
  const std::size_t demand = sizes[kCount - 1];
  RecordCell("LargeRecycle", "9/11/13 MiB", Arm::kName, "us",
             measurement.median);
  RecordCell("LargeRecycle/reserved", "9/11/13 MiB", Arm::kName, "B",
             static_cast<double>(arm.BytesReserved()), arm.TracksBytes());
  RecordCell("LargeRecycle/amplif", "9/11/13 MiB", Arm::kName, "x",
             static_cast<double>(arm.BytesReserved()) /
                 static_cast<double>(demand),
             arm.TracksBytes());
  RecordCell("LargeRecycle/upstream", "9/11/13 MiB", Arm::kName, "calls",
             static_cast<double>(arm.UpstreamAllocs()));
}

// --------------------------------------------------------------------------
// 3. High-water growth
// --------------------------------------------------------------------------

// Allocates without ever freeing until a gigabyte-scale high-water mark, which
// on the production config is far enough to walk the whole 64 MB -> 512 MB ramp
// and then keep adding capped backings. Nothing can be reused, so every request
// is a miss: `MemoryPool` needs one upstream call per block, the arena one per
// backing.
//
// The call count is the honest summary. It is exact, immune to machine noise,
// and on a device it is also the timing: at hundreds of microseconds per
// `cudaMalloc`, thousands of calls versus a handful is the entire result.
template <typename Arm>
void BenchHighWaterGrowth() {
  const std::size_t target = (1ull << 30) / kFootprintDivisor;
  constexpr std::size_t kBlock = 1ull << 20;  // 1 MiB per block
  const std::size_t blocks = target / kBlock;

  Arm arm;
  std::vector<void*> live;
  live.reserve(blocks);

  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < blocks; ++i) {
    void* ptr = nullptr;
    if (!Success(arm.Allocate(&ptr, kBlock))) {
      break;
    }
    live.push_back(ptr);
  }
  arm.Sync();
  const auto end = std::chrono::steady_clock::now();

  const auto elapsed_us =
      std::chrono::duration<double, std::micro>(end - start).count();
  const auto reserved = arm.BytesReserved();
  const auto upstream = arm.UpstreamAllocs();
  const bool tracks = arm.TracksBytes();

  for (void* ptr : live) {
    arm.Deallocate(ptr);
  }
  arm.Sync();

  const std::string params = DescribeSize(live.size() * kBlock) + " live";
  RecordCell("HighWater", params, Arm::kName, "us", elapsed_us);
  RecordCell("HighWater/upstream", params, Arm::kName, "calls",
             static_cast<double>(upstream));
  RecordCell("HighWater/reserved", params, Arm::kName, "B",
             static_cast<double>(reserved), tracks);

  perf::PrintResult("allocator_matrix.HighWaterGrowth",
                    WithArm({perf::NumberParam(
                                "live_bytes", static_cast<std::uint64_t>(
                                                  live.size() * kBlock))},
                            Arm::kName),
                    live.size(), "us", elapsed_us, elapsed_us);
  perf::PrintResult("allocator_matrix.HighWaterGrowthUpstreamCalls",
                    WithArm({perf::NumberParam(
                                "live_bytes", static_cast<std::uint64_t>(
                                                  live.size() * kBlock))},
                            Arm::kName),
                    live.size(), "count", static_cast<double>(upstream),
                    static_cast<double>(upstream));
}

// --------------------------------------------------------------------------
// 4. Trim cost, split into its two components
// --------------------------------------------------------------------------

// Fills a cache, then trims it. The interesting part is not the wall time but
// its decomposition: `MemoryPool` walks its free lists and issues one upstream
// free *per cached block*, while the arena walks an ordered set plus its backing
// vector and issues one *per backing*. On a host those are similar; on a device
// they are not remotely, because every `cudaFree` implicitly synchronizes the
// whole device. Both numbers are reported so the cause is visible next to the
// effect.
template <typename Arm>
void BenchTrimCost(std::size_t cached_blocks) {
  constexpr std::size_t kSize = 64 * 1024;
  constexpr std::size_t kIterations = 100;

  // An arm with no releasable cache would report the cost of the alloc/free
  // loop with the trim removed, which is a different measurement wearing this
  // one's name. Skipped rather than printed, so the row cannot be read as
  // "trimming is free here".
  if (!Arm::kHasCache) {
    perf::SkipBenchmark("allocator_matrix.TrimCost",
                        std::string(Arm::kName) + " has no releasable cache");
    return;
  }

  Arm arm;
  std::vector<void*> blocks(cached_blocks, nullptr);

  auto fill_and_trim = [&arm, &blocks] {
    for (void*& block : blocks) {
      arm.Allocate(&block, kSize);
    }
    for (void*& block : blocks) {
      arm.Deallocate(block);
      block = nullptr;
    }
    arm.ReleaseCached();
  };

  const auto measurement = perf::RunBenchmarkMeasured(
      "allocator_matrix.TrimCost",
      WithArm({perf::NumberParam("cached_blocks", cached_blocks),
               perf::NumberParam("size_bytes", kSize)},
              Arm::kName),
      kIterations, "us", fill_and_trim);
  arm.Sync();

  // Counted in its own cycle rather than divided out of the timed run: the
  // runner's warmup and sample counts are its business, and dividing by an
  // assumed total would silently go wrong the moment either changes.
  const auto before_frees = arm.UpstreamFrees();
  fill_and_trim();
  arm.Sync();
  const auto frees_per_trim =
      static_cast<double>(arm.UpstreamFrees() - before_frees);

  const std::string params = std::to_string(cached_blocks) + " cached";
  RecordCell("TrimCost", params, Arm::kName, "us", measurement.median);
  RecordCell("TrimCost/frees", params, Arm::kName, "calls", frees_per_trim);
}

// --------------------------------------------------------------------------
// 5. Thread scaling past 8
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

// Both pools serialize on one mutex, so past a handful of threads the curve is
// about how long each design holds it. `perf_memory_pool.cc` stops at 8, which
// on a 160-core host is well before the knee; this goes to 64 to find where the
// arena's longer critical section (ordered-set lookup, split, coalesce) starts
// to dominate its fewer upstream calls.
template <typename Arm>
void BenchThreadScaling(std::size_t threads, std::size_t ops_per_thread) {
  constexpr std::size_t kClasses = 16;
  constexpr std::size_t kStride = 512;

  Arm arm;
  const auto measurement = RunThreaded(
      "allocator_matrix.ThreadScaling",
      WithArm({perf::NumberParam("threads", threads),
               perf::NumberParam("size_classes", kClasses)},
              Arm::kName),
      threads, ops_per_thread, [&arm](std::size_t thread, std::size_t op) {
        const std::size_t size = ((thread + op) % kClasses + 1) * kStride;
        void* ptr = nullptr;
        auto status = arm.Allocate(&ptr, size);
        if (Success(status)) {
          status = arm.Deallocate(ptr);
        }
        perf::DoNotOptimize(status);
      });
  arm.Sync();

  RecordCell("ThreadScaling", std::to_string(threads) + "T", Arm::kName, "ns",
             measurement.median);
}

// --------------------------------------------------------------------------
// 6. Device-only: implicit synchronization cost
// --------------------------------------------------------------------------

// `cudaFree` implicitly synchronizes the whole device: it waits for every
// previously enqueued operation on every stream. That is the entire reason the
// arena has shrink hysteresis, and until now nothing measured it.
//
// The shape: keep a stream loaded with asynchronous work, then trim the
// allocator and time it. Compared against the same trim against an idle stream,
// the delta is what the trim cost the pending work. An allocator that trims by
// issuing one upstream free per cached block pays this once per block; one that
// frees whole backings pays it once per backing.
//
// On a backend whose `MemsetAsync` is synchronous (the CPU one) there is no
// pending work to stall, and the delta is reported as approximately zero -- a
// real answer for that backend, not a missing measurement.
template <typename Arm>
void BenchImplicitSyncCost() {
  constexpr std::size_t kSize = 64 * 1024;
  constexpr std::size_t kCachedBlocks = 64;
  // Enough queued work that the stream is still busy when the trim lands.
  constexpr std::size_t kBusyOps = 200;
  const std::size_t busy_bytes = (32ull << 20) / kFootprintDivisor;

  // Same reasoning as `BenchTrimCost`: with no cache to release there is no
  // trim, so there is no stall to attribute to one.
  if (!Arm::kHasCache) {
    perf::SkipBenchmark("allocator_matrix.ImplicitSyncCost",
                        std::string(Arm::kName) + " has no releasable cache");
    return;
  }

  runtime::Stream stream{};
  if (!Success(runtime::StreamCreate(&stream))) {
    perf::SkipBenchmark("allocator_matrix.ImplicitSyncCost",
                        "stream creation failed");
    return;
  }

  void* busy_buffer = nullptr;
  if (!Success(runtime::Malloc(&busy_buffer, busy_bytes))) {
    runtime::StreamDestroy(stream);
    perf::SkipBenchmark("allocator_matrix.ImplicitSyncCost",
                        "could not reserve the interference buffer");
    return;
  }

  Arm arm;
  std::vector<void*> blocks(kCachedBlocks, nullptr);

  // Fills the cache and returns everything, leaving the allocator holding
  // memory that the next trim will release.
  auto fill_cache = [&arm, &blocks] {
    for (void*& block : blocks) {
      arm.Allocate(&block, kSize);
    }
    for (void*& block : blocks) {
      arm.Deallocate(block);
      block = nullptr;
    }
  };

  auto time_trim = [&](bool with_interference) {
    constexpr std::size_t kSamples = 5;
    std::vector<double> samples;
    samples.reserve(kSamples);

    for (std::size_t sample = 0; sample < kSamples + 1; ++sample) {
      fill_cache();
      if (with_interference) {
        for (std::size_t i = 0; i < kBusyOps; ++i) {
          runtime::MemsetAsync(busy_buffer, static_cast<int>(i & 0xff),
                               busy_bytes, stream);
        }
      }

      const auto start = std::chrono::steady_clock::now();
      arm.ReleaseCached();
      arm.Sync();
      const auto end = std::chrono::steady_clock::now();

      // Drain before the next sample so leftovers cannot bleed across.
      runtime::StreamSynchronize(stream);
      if (sample == 0) {  // warmup
        continue;
      }
      samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }
    return perf::Median(samples);
  };

  const double idle = time_trim(false);
  const double loaded = time_trim(true);

  runtime::Free(busy_buffer);
  runtime::StreamDestroy(stream);

  const auto params = WithArm(
      {perf::NumberParam("cached_blocks", kCachedBlocks),
       perf::NumberParam("queued_ops", kBusyOps)},
      Arm::kName);
  perf::PrintResult("allocator_matrix.TrimIdleStream", params, kCachedBlocks,
                    "us", idle, idle);
  perf::PrintResult("allocator_matrix.TrimBusyStream", params, kCachedBlocks,
                    "us", loaded, loaded);

  RecordCell("Trim/idle stream", "64 cached", Arm::kName, "us", idle);
  RecordCell("Trim/busy stream", "64 cached", Arm::kName, "us", loaded);
  // The stall the trim imposed on pending work. Negative values are noise on a
  // backend with no asynchrony to stall, and are clamped so the table reads as
  // "no measurable stall" rather than as a nonsensical negative cost.
  RecordCell("Trim/sync stall", "64 cached", Arm::kName, "us",
             std::max(0.0, loaded - idle));
}

// --------------------------------------------------------------------------
// 7. Device-only: ledger accuracy
// --------------------------------------------------------------------------

// Checks the pool's self-reported `bytes_reserved` against what the device says
// it lost. A pool that under-reports retention would look good on the
// amplification rows for the wrong reason, and no unit test can catch that
// because only the driver knows the truth.
//
// Every block is written to before the second reading. On a device that is
// merely belt-and-braces, but on a host it is required: `malloc` returns
// untouched pages that consume nothing until first touch, so without the write
// the ledger would look like a gross over-report of memory that genuinely had
// not been committed yet.
//
// The device figure also includes the driver's own per-allocation overhead (page
// rounding, internal metadata), so the two are not expected to match exactly.
// The ratio is what matters: near 1.0 means the ledger is honest.
//
// Device builds only. The CPU backend's `MemGetInfo` reports system-wide free
// memory from `/proc/meminfo`, so every other process on the machine moves the
// reading and the ratio would describe the machine rather than the allocator.
// Skipped there rather than printed with a caveat, because a number nobody
// should act on is worse than no number.
template <typename Arm>
void MeasureLedgerAccuracy() {
  constexpr std::size_t kBlock = 1ull << 20;
  const std::size_t blocks = (256ull << 20) / kFootprintDivisor / kBlock;

  if (!kDeviceBackend) {
    perf::SkipBenchmark("allocator_matrix.LedgerAccuracy",
                        "MemGetInfo is system-wide on the host backend");
    return;
  }

  std::size_t free_before = 0;
  std::size_t total = 0;
  if (!Success(runtime::MemGetInfo(&free_before, &total)) || free_before == 0) {
    perf::SkipBenchmark("allocator_matrix.LedgerAccuracy",
                        "the backend does not report device memory");
    return;
  }

  Arm arm;
  if (!arm.TracksBytes()) {
    return;  // Nothing to check against for the direct and async arms.
  }

  std::vector<void*> live;
  live.reserve(blocks);
  for (std::size_t i = 0; i < blocks; ++i) {
    void* ptr = nullptr;
    if (!Success(arm.Allocate(&ptr, kBlock))) {
      break;
    }
    // Commits the pages, so the reading below reflects what was reserved rather
    // than what happens to have been faulted in.
    runtime::Memset(ptr, 0, kBlock);
    live.push_back(ptr);
  }
  arm.Sync();
  runtime::DeviceSynchronize();

  std::size_t free_after = 0;
  runtime::MemGetInfo(&free_after, &total);
  const double device_consumed =
      free_before > free_after ? static_cast<double>(free_before - free_after)
                               : 0.0;
  const double reported = static_cast<double>(arm.BytesReserved());

  for (void* ptr : live) {
    arm.Deallocate(ptr);
  }
  arm.Sync();

  const std::string params = DescribeSize(live.size() * kBlock) + " live";
  RecordCell("Ledger/reported", params, Arm::kName, "B", reported);
  RecordCell("Ledger/device", params, Arm::kName, "B", device_consumed);
  RecordCell("Ledger/ratio", params, Arm::kName, "x",
             reported == 0.0 ? 0.0 : device_consumed / reported);

  perf::PrintResult(
      "allocator_matrix.LedgerAccuracy",
      WithArm({perf::NumberParam(
                  "live_bytes",
                  static_cast<std::uint64_t>(live.size() * kBlock))},
              Arm::kName),
      live.size(), "bytes", device_consumed, device_consumed);
}

// --------------------------------------------------------------------------
// 8. Device-only: multi-stream traffic
// --------------------------------------------------------------------------

// The other concurrency benchmark shares one allocator across threads that do
// nothing but allocate. A GPU server's real shape is different: each worker owns
// a stream, and allocation is interleaved with enqueued device work. That work
// is what an allocator can stall -- so this measures per-operation cost when
// every thread also has a stream to keep fed.
template <typename Arm>
void BenchMultiStream(std::size_t streams, std::size_t ops_per_stream) {
  constexpr std::size_t kSize = 256 * 1024;

  std::vector<runtime::Stream> handles(streams, runtime::Stream{});
  for (auto& stream : handles) {
    if (!Success(runtime::StreamCreate(&stream))) {
      for (auto& created : handles) {
        if (created != runtime::Stream{}) {
          runtime::StreamDestroy(created);
        }
      }
      perf::SkipBenchmark("allocator_matrix.MultiStream",
                          "stream creation failed");
      return;
    }
  }

  Arm arm;
  const auto measurement = RunThreaded(
      "allocator_matrix.MultiStream",
      WithArm({perf::NumberParam("streams", streams),
               perf::NumberParam("size_bytes", kSize)},
              Arm::kName),
      streams, ops_per_stream,
      [&arm, &handles](std::size_t index, std::size_t) {
        void* ptr = nullptr;
        if (!Success(arm.Allocate(&ptr, kSize))) {
          return;
        }
        // Enqueue work against the block, then wait for it before releasing:
        // the pools make no stream guarantees, so a caller must synchronize
        // before handing memory back. This is the cost of using them correctly.
        runtime::MemsetAsync(ptr, 0, kSize, handles[index]);
        runtime::StreamSynchronize(handles[index]);
        const auto status = arm.Deallocate(ptr);
        perf::DoNotOptimize(status);
      });
  arm.Sync();

  for (auto& stream : handles) {
    runtime::StreamDestroy(stream);
  }

  RecordCell("MultiStream", std::to_string(streams) + " streams", Arm::kName,
             "ns", measurement.median);
}

// --------------------------------------------------------------------------
// 9. Layer-by-layer inference
// --------------------------------------------------------------------------

// Everything above is a microbenchmark. This is the only workload here shaped
// like what the library is actually for, and the only one whose number answers
// "how much faster does inference get".
//
// The sequence mirrors a transformer forward pass:
//   - weights allocated once and held for the whole run, so the arena's
//     resident backing carries a permanent live block;
//   - prefill walks the layers, allocating this layer's activations before
//     releasing the previous layer's, which is what keeps two layers live at
//     once and prevents a trivially reusable single-block pattern;
//   - the KV cache grows monotonically, one allocation per layer per step, never
//     freed until the end -- the shape that defeats a size-class cache, since
//     each step's cache slab is a different size;
//   - then many decode steps, each a small activation per layer.
//
// Reported per whole sequence, since one sample is one inference run.
template <typename Arm>
void BenchLayerwiseInference(std::size_t decode_steps) {
  constexpr std::size_t kLayers = 32;
  const std::size_t weight_bytes = (4ull << 20) / kFootprintDivisor;
  const std::size_t activation_bytes = (2ull << 20) / kFootprintDivisor;
  const std::size_t kv_step_bytes = (128ull << 10) / kFootprintDivisor;
  const std::size_t decode_bytes = (64ull << 10) / kFootprintDivisor;

  Arm arm;

  // Weights: allocated up front, released only at the very end.
  std::vector<void*> weights;
  weights.reserve(kLayers);
  for (std::size_t layer = 0; layer < kLayers; ++layer) {
    void* ptr = nullptr;
    if (!Success(arm.Allocate(&ptr, weight_bytes))) {
      for (void* held : weights) {
        arm.Deallocate(held);
      }
      perf::SkipBenchmark("allocator_matrix.LayerwiseInference",
                          "device cannot hold the model weights");
      return;
    }
    weights.push_back(ptr);
  }
  arm.Sync();

  std::vector<void*> kv_cache;
  kv_cache.reserve(kLayers * (decode_steps + 1));

  const auto before_upstream = arm.UpstreamAllocs();
  const auto start = std::chrono::steady_clock::now();

  // Prefill: two layers' activations live at once.
  void* previous = nullptr;
  for (std::size_t layer = 0; layer < kLayers; ++layer) {
    void* activation = nullptr;
    if (Success(arm.Allocate(&activation, activation_bytes))) {
      if (previous != nullptr) {
        arm.Deallocate(previous);
      }
      previous = activation;
    }
    void* kv = nullptr;
    // Each layer's slab differs slightly in size, as a real cache's does with
    // sequence length -- and as a size-class cache cannot reuse.
    if (Success(arm.Allocate(&kv, kv_step_bytes + layer * 512))) {
      kv_cache.push_back(kv);
    }
  }
  if (previous != nullptr) {
    arm.Deallocate(previous);
    previous = nullptr;
  }

  // Decode: one small activation per layer per step, plus a growing cache.
  for (std::size_t step = 0; step < decode_steps; ++step) {
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
      void* activation = nullptr;
      if (Success(arm.Allocate(&activation, decode_bytes))) {
        arm.Deallocate(activation);
      }
      void* kv = nullptr;
      if (Success(arm.Allocate(&kv, kv_step_bytes + step * 256))) {
        kv_cache.push_back(kv);
      }
    }
  }
  arm.Sync();

  const auto end = std::chrono::steady_clock::now();
  const auto elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  const auto upstream = arm.UpstreamAllocs() - before_upstream;
  const auto reserved = arm.BytesReserved();
  const bool tracks = arm.TracksBytes();

  for (void* ptr : kv_cache) {
    arm.Deallocate(ptr);
  }
  for (void* ptr : weights) {
    arm.Deallocate(ptr);
  }
  arm.Sync();

  const std::string params = std::to_string(decode_steps) + " steps";
  RecordCell("Inference", params, Arm::kName, "ms", elapsed_ms);
  RecordCell("Inference/upstream", params, Arm::kName, "calls",
             static_cast<double>(upstream));
  RecordCell("Inference/reserved", params, Arm::kName, "B",
             static_cast<double>(reserved), tracks);

  const auto json_params =
      WithArm({perf::NumberParam("layers", kLayers),
               perf::NumberParam("decode_steps", decode_steps)},
              Arm::kName);
  perf::PrintResult("allocator_matrix.LayerwiseInference", json_params, 1, "ms",
                    elapsed_ms, elapsed_ms);
  perf::PrintResult("allocator_matrix.LayerwiseInferenceUpstreamCalls",
                    json_params, 1, "count", static_cast<double>(upstream),
                    static_cast<double>(upstream));
}

// --------------------------------------------------------------------------
// 10. Concurrent inference: the shape of a real server
// --------------------------------------------------------------------------

// The gap the rest of this file leaves. `ThreadScaling` is multi-threaded but
// its sizes (512 B - 8 KiB) all sit inside the arena's fast-bin range, so every
// thread serves itself out of its own front cache and the measurement is close
// to a best case for that cache. `LayerwiseInference` has the right shapes but
// runs on one thread. Neither answers "does the arena still win when several
// threads each drive a real forward pass", which is the question a serving
// deployment actually asks.
//
// The arena's fast bins top out at `kFastBinCount * kMinSliceAlignment` = 64 KiB.
// Of the shapes below only the decode activation is at or under that, so the
// front cache covers roughly one allocation in four and everything else
// serializes on the pool mutex with a best-fit lookup and a split. That is the
// point: the win here has to come from amortizing upstream calls, not from the
// cache, and this is where we find out whether it does.
//
// Each thread owns a stream and runs an independent sequence, so the threads
// contend for one allocator exactly as concurrent requests would. Latency
// percentiles rather than a mean: a serving system is bought on its tail, and a
// pool that occasionally stalls a thread behind a `cudaMalloc` shows up in p99
// while a mean hides it.
struct ConcurrentInferenceResult {
  double p50 = 0.0;
  double p99 = 0.0;
  double max = 0.0;
  double throughput = 0.0;  // sequences per second
  std::size_t upstream_allocs = 0;
  std::size_t bytes_reserved = 0;
  bool tracks_bytes = false;
};

template <typename Arm>
ConcurrentInferenceResult BenchConcurrentInference(std::size_t threads,
                                                   std::size_t sequences,
                                                   std::size_t decode_steps) {
  // Per-thread footprints, so `threads` of them are live at once. Divided by the
  // thread count rather than fixed: a 32-thread run at the single-threaded
  // sizes would need 32 models resident, which is an OOM rather than a
  // measurement. What stays constant is total pressure on the allocator.
  constexpr std::size_t kLayers = 8;
  const std::size_t weight_bytes = (4ull << 20) / kFootprintDivisor;
  const std::size_t activation_bytes = (2ull << 20) / kFootprintDivisor;
  const std::size_t kv_step_bytes = (128ull << 10) / kFootprintDivisor;
  const std::size_t decode_bytes = (64ull << 10) / kFootprintDivisor;

  std::vector<runtime::Stream> handles(threads, runtime::Stream{});
  for (auto& stream : handles) {
    if (!Success(runtime::StreamCreate(&stream))) {
      for (auto& created : handles) {
        if (created != runtime::Stream{}) {
          runtime::StreamDestroy(created);
        }
      }
      perf::SkipBenchmark("allocator_matrix.ConcurrentInference",
                          "stream creation failed");
      return {};
    }
  }

  Arm arm;

  // Weights are per-thread and held for the whole run, the way a replica's
  // parameters are. Allocated before the timed region so the ramp-up cost of
  // creating the first backings is not charged to a sequence's latency.
  std::vector<std::vector<void*>> weights(threads);
  bool weights_ok = true;
  for (std::size_t t = 0; t < threads && weights_ok; ++t) {
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
      void* ptr = nullptr;
      if (!Success(arm.Allocate(&ptr, weight_bytes))) {
        weights_ok = false;
        break;
      }
      weights[t].push_back(ptr);
    }
  }
  if (!weights_ok) {
    for (const auto& held : weights) {
      for (void* ptr : held) {
        arm.Deallocate(ptr);
      }
    }
    for (auto& stream : handles) {
      runtime::StreamDestroy(stream);
    }
    perf::SkipBenchmark("allocator_matrix.ConcurrentInference",
                        "device cannot hold one model per thread");
    return {};
  }
  arm.Sync();

  // One vector per thread, so recording a sample takes no lock and the
  // measurement does not add contention of its own on top of the allocator's.
  std::vector<std::vector<double>> samples(threads);
  for (auto& per_thread : samples) {
    per_thread.reserve(sequences);
  }

  const auto before_upstream = arm.UpstreamAllocs();
  std::atomic<bool> go{false};
  std::vector<std::thread> workers;
  workers.reserve(threads);

  for (std::size_t t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      const runtime::Stream stream = handles[t];
      std::vector<void*> kv_cache;
      kv_cache.reserve(kLayers * (decode_steps + 1));

      while (!go.load(std::memory_order_acquire)) {
      }

      for (std::size_t sequence = 0; sequence < sequences; ++sequence) {
        const auto start = std::chrono::steady_clock::now();

        // Prefill: two layers' activations live at once, and a KV slab per
        // layer whose size varies with the layer -- the shape a size-class
        // cache cannot reuse.
        void* previous = nullptr;
        for (std::size_t layer = 0; layer < kLayers; ++layer) {
          void* activation = nullptr;
          if (Success(arm.Allocate(&activation, activation_bytes))) {
            // Enqueued work against the block, then waited on before release:
            // the pools make no stream guarantees, so this is what using them
            // correctly costs.
            runtime::MemsetAsync(activation, 0, activation_bytes, stream);
            if (previous != nullptr) {
              runtime::StreamSynchronize(stream);
              arm.Deallocate(previous);
            }
            previous = activation;
          }
          void* kv = nullptr;
          if (Success(arm.Allocate(&kv, kv_step_bytes + layer * 512))) {
            kv_cache.push_back(kv);
          }
        }
        if (previous != nullptr) {
          runtime::StreamSynchronize(stream);
          arm.Deallocate(previous);
        }

        // Decode: a small activation per layer per step, plus a growing cache.
        for (std::size_t step = 0; step < decode_steps; ++step) {
          for (std::size_t layer = 0; layer < kLayers; ++layer) {
            void* activation = nullptr;
            if (Success(arm.Allocate(&activation, decode_bytes))) {
              runtime::MemsetAsync(activation, 0, decode_bytes, stream);
              runtime::StreamSynchronize(stream);
              arm.Deallocate(activation);
            }
            void* kv = nullptr;
            if (Success(arm.Allocate(&kv, kv_step_bytes + step * 256))) {
              kv_cache.push_back(kv);
            }
          }
        }

        // The sequence ends: its whole KV cache goes back at once. This is what
        // makes the benchmark more than a longer `ThreadScaling` -- it is the
        // moment a backing can drain, which is what arms the arena's shrink
        // scan and its cache reclamation, and those run inside the lock while
        // every other thread is still allocating.
        runtime::StreamSynchronize(stream);
        for (void* ptr : kv_cache) {
          arm.Deallocate(ptr);
        }
        kv_cache.clear();

        const auto end = std::chrono::steady_clock::now();
        samples[t].push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
      }
    });
  }

  const auto wall_start = std::chrono::steady_clock::now();
  go.store(true, std::memory_order_release);
  for (auto& worker : workers) {
    worker.join();
  }
  const auto wall_end = std::chrono::steady_clock::now();
  arm.Sync();

  ConcurrentInferenceResult result;
  result.upstream_allocs = arm.UpstreamAllocs() - before_upstream;
  result.bytes_reserved = arm.BytesReserved();
  result.tracks_bytes = arm.TracksBytes();

  const double wall_seconds =
      std::chrono::duration<double>(wall_end - wall_start).count();
  result.throughput = wall_seconds > 0.0
                          ? static_cast<double>(threads * sequences) /
                                wall_seconds
                          : 0.0;

  std::vector<double> all;
  all.reserve(threads * sequences);
  for (const auto& per_thread : samples) {
    all.insert(all.end(), per_thread.begin(), per_thread.end());
  }
  if (!all.empty()) {
    std::sort(all.begin(), all.end());
    result.p50 = all[all.size() / 2];
    const auto p99_index =
        std::min(all.size() - 1, static_cast<std::size_t>(
                                     static_cast<double>(all.size()) * 0.99));
    result.p99 = all[p99_index];
    result.max = all.back();
  }

  for (const auto& held : weights) {
    for (void* ptr : held) {
      arm.Deallocate(ptr);
    }
  }
  arm.Sync();
  for (auto& stream : handles) {
    runtime::StreamDestroy(stream);
  }

  const auto params =
      WithArm({perf::NumberParam("threads", threads),
               perf::NumberParam("layers", kLayers),
               perf::NumberParam("decode_steps", decode_steps)},
              Arm::kName);
  const std::size_t total = threads * sequences;
  perf::PrintResult("allocator_matrix.ConcurrentInferenceP50", params, total,
                    "ms", result.p50, result.p50);
  perf::PrintResult("allocator_matrix.ConcurrentInferenceP99", params, total,
                    "ms", result.p99, result.p99);
  perf::PrintResult("allocator_matrix.ConcurrentInferenceThroughput", params,
                    total, "seq_per_s", result.throughput, result.throughput);
  perf::PrintResult("allocator_matrix.ConcurrentInferenceUpstreamCalls", params,
                    total, "count",
                    static_cast<double>(result.upstream_allocs),
                    static_cast<double>(result.upstream_allocs));

  const std::string label = std::to_string(threads) + "T";
  RecordCell("ConcInfer/p50", label, Arm::kName, "ms", result.p50);
  RecordCell("ConcInfer/p99", label, Arm::kName, "ms", result.p99);
  RecordCell("ConcInfer/max", label, Arm::kName, "ms", result.max);
  RecordCell("ConcInfer/upstream", label, Arm::kName, "calls",
             static_cast<double>(result.upstream_allocs));
  RecordCell("ConcInfer/reserved", label, Arm::kName, "B",
             static_cast<double>(result.bytes_reserved), result.tracks_bytes);
  return result;
}

// --------------------------------------------------------------------------
// 11. Fragmentation under a long random-lifetime run
// --------------------------------------------------------------------------

// Everything else here allocates in a pattern. This one does not: random sizes
// spanning four orders of magnitude, random lifetimes, sustained long enough
// that the allocator's internal state is whatever the run made it rather than
// whatever it was designed for. Then it asks the question that matters at the
// end of such a run -- can you still get a large contiguous block?
//
// The two designs fail differently, which is why both the success rate and the
// retention are reported. `MemoryPool` never splits, so it cannot fragment
// internally at all: a large request either finds a matching size class or goes
// upstream, and it succeeds as long as the *device* has room. What it does
// instead is retain a block of every size class it ever saw, so its reserved
// bytes climb toward the sum of the whole size distribution. The arena splits
// and coalesces, so it reuses far more, but a large request needs a contiguous
// run inside one backing -- and if live blocks are scattered across every
// backing, that run may not exist even though the free bytes are there.
//
// The deterministic LCG is deliberate: two arms must see the identical sequence,
// or the comparison is between two workloads rather than two allocators.
template <typename Arm>
void BenchFragmentation(std::size_t operations) {
  // Held live at any moment; each slot is replaced when its lifetime expires.
  constexpr std::size_t kSlots = 256;
  // The probe: can a large contiguous block still be had at the end?
  const std::size_t probe_bytes = (32ull << 20) / kFootprintDivisor;
  constexpr std::size_t kProbes = 8;

  struct Slot {
    void* ptr = nullptr;
    std::size_t expires_at = 0;
  };

  Arm arm;
  std::vector<Slot> slots(kSlots);

  // Same constants as `std::minstd_rand`, inlined so the sequence cannot change
  // with the standard library.
  std::uint64_t state = 0x2545f4914f6cdd1dull;
  auto next = [&state] {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<std::uint32_t>(state >> 33);
  };

  // Sizes spanning 1 KiB to about 4 MiB, log-distributed so small allocations
  // dominate by count and large ones by bytes -- the shape a real mix has, and
  // the one that scatters small live blocks through the backings that a large
  // request needs whole.
  auto random_size = [&next] {
    const std::uint32_t decade = next() % 4;  // 1 KiB, 16 KiB, 256 KiB, 4 MiB
    const std::size_t base = 1024ull << (4 * decade);
    return base + (next() % base);
  };

  std::size_t failures = 0;
  for (std::size_t op = 0; op < operations; ++op) {
    Slot& slot = slots[next() % kSlots];
    if (slot.ptr != nullptr) {
      if (slot.expires_at > op) {
        continue;  // Not due yet: leave it live and let the hole persist.
      }
      arm.Deallocate(slot.ptr);
      slot.ptr = nullptr;
    }
    void* ptr = nullptr;
    if (Success(arm.Allocate(&ptr, random_size()))) {
      slot.ptr = ptr;
      // Lifetimes from a few operations to a few thousand, so short-lived
      // blocks churn through the holes long-lived ones leave behind.
      slot.expires_at = op + 1 + (next() % 4096);
    } else {
      ++failures;
    }
  }
  arm.Sync();

  const auto churn_reserved = arm.BytesReserved();
  const auto churn_upstream = arm.UpstreamAllocs();

  // Probe the fragmented state: several large blocks at once, so the question is
  // whether the pool can assemble contiguous runs and not merely find one.
  std::vector<void*> probes;
  probes.reserve(kProbes);
  for (std::size_t i = 0; i < kProbes; ++i) {
    void* ptr = nullptr;
    if (!Success(arm.Allocate(&ptr, probe_bytes)) || ptr == nullptr) {
      break;
    }
    probes.push_back(ptr);
  }
  arm.Sync();
  const double probe_rate =
      static_cast<double>(probes.size()) / static_cast<double>(kProbes);
  const auto probe_upstream = arm.UpstreamAllocs() - churn_upstream;

  for (void* ptr : probes) {
    arm.Deallocate(ptr);
  }
  for (Slot& slot : slots) {
    if (slot.ptr != nullptr) {
      arm.Deallocate(slot.ptr);
      slot.ptr = nullptr;
    }
  }
  arm.Sync();

  const std::string label = std::to_string(operations / 1000) + "k ops";
  RecordCell("Fragment/reserved", label, Arm::kName, "B",
             static_cast<double>(churn_reserved), arm.TracksBytes());
  RecordCell("Fragment/upstream", label, Arm::kName, "calls",
             static_cast<double>(churn_upstream));
  RecordCell("Fragment/fail", label, Arm::kName, "calls",
             static_cast<double>(failures));
  RecordCell("Fragment/probe ok", label, Arm::kName, "x", probe_rate);
  RecordCell("Fragment/probe up", label, Arm::kName, "calls",
             static_cast<double>(probe_upstream));

  const auto params =
      WithArm({perf::NumberParam("operations", operations),
               perf::NumberParam("probe_bytes", probe_bytes)},
              Arm::kName);
  perf::PrintResult("allocator_matrix.FragmentationReserved", params,
                    operations, "bytes", static_cast<double>(churn_reserved),
                    static_cast<double>(churn_reserved));
  perf::PrintResult("allocator_matrix.FragmentationProbeSuccess", params,
                    kProbes, "x", probe_rate, probe_rate);
  perf::PrintResult("allocator_matrix.FragmentationProbeUpstreamCalls", params,
                    kProbes, "count", static_cast<double>(probe_upstream),
                    static_cast<double>(probe_upstream));
}

// --------------------------------------------------------------------------
// 12. Allocation latency tail
// --------------------------------------------------------------------------

// Every timing above is a mean or a median over a loop, which is the right
// summary for throughput and the wrong one for a serving deployment: what a
// request feels is its own allocation, and the allocation that goes upstream
// costs three orders of magnitude more than the one that hits. A design with a
// better median and a worse tail is worse for serving, and no row here could
// currently tell you that.
//
// Timed per operation with a monotonic clock, which on a device backend is
// sound: the pools are synchronous, so a `Malloc` that misses blocks until the
// driver returns and the interval is the real cost. The clock's own overhead
// (tens of nanoseconds) is a visible fraction of a cache hit, so the p50 here
// reads slightly high compared with the loop-averaged rows -- consistently
// across arms, which is what keeps the comparison fair.
template <typename Arm>
void BenchLatencyTail(std::size_t size, std::size_t operations) {
  Arm arm;
  std::vector<double> samples;
  samples.reserve(operations);

  // Warm the arena's ramp and the pool's size class, so the measurement is of
  // the steady state rather than of first-touch growth. The cold path is what
  // `HighWaterGrowth` measures.
  for (std::size_t i = 0; i < 64; ++i) {
    void* ptr = nullptr;
    if (Success(arm.Allocate(&ptr, size))) {
      arm.Deallocate(ptr);
    }
  }
  arm.Sync();

  for (std::size_t op = 0; op < operations; ++op) {
    void* ptr = nullptr;
    const auto start = std::chrono::steady_clock::now();
    const auto status = arm.Allocate(&ptr, size);
    const auto end = std::chrono::steady_clock::now();
    if (Success(status)) {
      arm.Deallocate(ptr);
    }
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }
  arm.Sync();

  if (samples.empty()) {
    return;
  }
  std::sort(samples.begin(), samples.end());
  auto quantile = [&samples](double q) {
    const auto index = std::min(
        samples.size() - 1,
        static_cast<std::size_t>(static_cast<double>(samples.size()) * q));
    return samples[index];
  };

  const std::string label = DescribeSize(size);
  RecordCell("Latency/p50", label, Arm::kName, "us", quantile(0.50));
  RecordCell("Latency/p99", label, Arm::kName, "us", quantile(0.99));
  RecordCell("Latency/p999", label, Arm::kName, "us", quantile(0.999));
  RecordCell("Latency/max", label, Arm::kName, "us", samples.back());

  const auto params = WithArm(
      {perf::NumberParam("size_bytes", static_cast<std::uint64_t>(size))},
      Arm::kName);
  perf::PrintResult("allocator_matrix.LatencyP50", params, operations, "us",
                    quantile(0.50), quantile(0.50));
  perf::PrintResult("allocator_matrix.LatencyP99", params, operations, "us",
                    quantile(0.99), quantile(0.99));
  perf::PrintResult("allocator_matrix.LatencyP999", params, operations, "us",
                    quantile(0.999), quantile(0.999));
  perf::PrintResult("allocator_matrix.LatencyMax", params, operations, "us",
                    samples.back(), samples.back());
}

// --------------------------------------------------------------------------
// 13. Memory bandwidth through pooled memory
// --------------------------------------------------------------------------

// Every other row prices the allocator. This one asks whether using it costs
// anything *afterwards* -- whether a kernel reading a sliced block runs as fast
// as one reading a dedicated upstream allocation.
//
// There is a real mechanism to check for, not just due diligence. A block from
// `cudaMalloc` starts at a 256 B (in practice much coarser) boundary; a slice
// out of an arena backing is only guaranteed `kMinSliceAlignment` = 512 B, and
// after a split it can start at an arbitrary multiple of that. If that landed
// mid-page or misaligned against the memory transaction size, sustained
// bandwidth would drop. So the arena is measured on a *split* slice rather than
// on a fresh backing's first chunk, which is the case that could actually differ.
//
// Reported as GiB/s of `MemsetAsync` traffic, which is bandwidth-bound on a
// device and the closest thing to a STREAM kernel available through the
// dispatch API without a kernel-launch surface.
template <typename Arm>
void BenchBandwidth() {
  const std::size_t size = (64ull << 20) / kFootprintDivisor;
  constexpr std::size_t kIterations = 32;

  runtime::Stream stream{};
  if (!Success(runtime::StreamCreate(&stream))) {
    perf::SkipBenchmark("allocator_matrix.Bandwidth", "stream creation failed");
    return;
  }

  Arm arm;

  // Force the block under test to be a split remainder rather than a whole
  // backing: allocate a small block first so the large one starts at an offset,
  // which is the alignment case a fresh allocation would never exercise.
  void* leading = nullptr;
  arm.Allocate(&leading, 4096);

  void* buffer = nullptr;
  if (!Success(arm.Allocate(&buffer, size)) || buffer == nullptr) {
    if (leading != nullptr) {
      arm.Deallocate(leading);
    }
    runtime::StreamDestroy(stream);
    perf::SkipBenchmark("allocator_matrix.Bandwidth",
                        "device cannot hold the bandwidth buffer");
    return;
  }

  // Warm: first touch on a host backing faults pages in, and on a device the
  // first launch pays context setup. Neither is bandwidth.
  runtime::MemsetAsync(buffer, 0, size, stream);
  runtime::StreamSynchronize(stream);

  std::vector<double> samples;
  samples.reserve(kIterations);
  for (std::size_t i = 0; i < kIterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    runtime::MemsetAsync(buffer, static_cast<int>(i & 0xff), size, stream);
    runtime::StreamSynchronize(stream);
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    if (seconds > 0.0) {
      samples.push_back(static_cast<double>(size) / seconds /
                        (1024.0 * 1024.0 * 1024.0));
    }
  }

  arm.Deallocate(buffer);
  if (leading != nullptr) {
    arm.Deallocate(leading);
  }
  arm.Sync();
  runtime::StreamDestroy(stream);

  if (samples.empty()) {
    return;
  }
  const double median = perf::Median(samples);
  RecordCell("Bandwidth", DescribeSize(size), Arm::kName, "GiB/s", median);
  perf::PrintResult(
      "allocator_matrix.Bandwidth",
      WithArm({perf::NumberParam("size_bytes",
                                 static_cast<std::uint64_t>(size))},
              Arm::kName),
      kIterations, "GiB/s", perf::Mean(samples), median);
}

// --------------------------------------------------------------------------
// Reporting
// --------------------------------------------------------------------------

void PrintMatrix() {
  if (g_cells.empty()) {
    return;
  }

  // Rows in first-seen order, so the table follows the benchmark order rather
  // than an alphabetical one that would separate a timing from its cause.
  std::vector<std::pair<std::string, std::string>> rows;
  for (const Cell& cell : g_cells) {
    const auto key = std::make_pair(cell.workload, cell.params);
    if (std::find(rows.begin(), rows.end(), key) == rows.end()) {
      rows.push_back(key);
    }
  }

  std::cerr << "\n=== " << INFINI_RT_PERF_BACKEND_NAME
            << " allocator matrix (median; arena config: " << kConfigName
            << ") ===\n";
  std::cerr << std::left << std::setw(24) << "workload" << std::setw(14)
            << "params";
  for (const std::string& arm : g_arm_order) {
    std::cerr << std::right << std::setw(16) << arm;
  }
  std::cerr << std::right << std::setw(12) << "unit" << "\n";

  for (const auto& [workload, params] : rows) {
    std::cerr << std::left << std::setw(24) << workload << std::setw(14)
              << params;
    std::string unit;
    for (const std::string& arm : g_arm_order) {
      const auto found = std::find_if(
          g_cells.begin(), g_cells.end(), [&](const Cell& cell) {
            return cell.workload == workload && cell.params == params &&
                   cell.arm == arm;
          });
      if (found == g_cells.end()) {
        std::cerr << std::right << std::setw(16) << "-";
      } else if (!found->present) {
        // Absent by construction, not missing: `direct` reserves exactly what
        // is live, so retention is not an axis it has.
        std::cerr << std::right << std::setw(16) << "n/a";
        unit = found->unit;
      } else {
        std::cerr << std::right << std::setw(16) << std::fixed
                  << std::setprecision(found->unit == "calls" ? 0 : 2)
                  << found->value;
        unit = found->unit;
      }
    }
    std::cerr << std::right << std::setw(12) << unit << "\n";
  }

  std::cerr
      << "\nLower is better except `x` ratio rows. `calls` rows are exact\n"
         "counts, not timings, and are the cause behind the timing above them.\n"
         "`n/a` means the metric does not apply to that arm; `-` means the arm\n"
         "did not run that workload.\n"
         "cuda_async is stream-ordered: its `Deallocate` does not wait for\n"
         "pending device work, so it offers a weaker guarantee than the other\n"
         "three arms and its timings are not a drop-in speedup.\n";
  std::cerr << std::endl;
}

// --------------------------------------------------------------------------
// Driver
// --------------------------------------------------------------------------

bool PrepareRuntime() {
  int device_count = 0;
  if (!Success(runtime::GetDeviceCount(&device_count)) || device_count <= 0) {
    std::cerr << "perf_allocator_matrix skipped: no available device."
              << std::endl;
    return false;
  }
  if (!Success(runtime::SetDevice(0))) {
    std::cerr << "perf_allocator_matrix skipped: device 0 is not available."
              << std::endl;
    return false;
  }
  return true;
}

// Runs `body` for each arm in turn, skipping the stream-ordered one where the
// backend does not support it.
template <template <typename> class Body, typename... Args>
void ForEachArm(bool with_async, Args&&... args) {
  Body<DirectArm>{}(args...);
  Body<PoolArm>{}(args...);
  Body<ArenaArm>{}(args...);
  if (with_async) {
    Body<CudaAsyncArm>{}(args...);
  }
}

template <typename Arm>
struct RunCrossThreshold {
  void operator()(std::size_t iterations) const {
    BenchCrossThresholdRotation<Arm>(iterations);
  }
};

template <typename Arm>
struct RunLargeRecycle {
  void operator()(std::size_t iterations) const {
    BenchLargeBlockRecycle<Arm>(iterations);
  }
};

template <typename Arm>
struct RunHighWater {
  void operator()() const { BenchHighWaterGrowth<Arm>(); }
};

template <typename Arm>
struct RunTrimCost {
  void operator()(std::size_t cached) const { BenchTrimCost<Arm>(cached); }
};

template <typename Arm>
struct RunThreadScaling {
  void operator()(std::size_t threads, std::size_t ops) const {
    BenchThreadScaling<Arm>(threads, ops);
  }
};

template <typename Arm>
struct RunImplicitSync {
  void operator()() const { BenchImplicitSyncCost<Arm>(); }
};

template <typename Arm>
struct RunLedger {
  void operator()() const { MeasureLedgerAccuracy<Arm>(); }
};

template <typename Arm>
struct RunMultiStream {
  void operator()(std::size_t streams, std::size_t ops) const {
    BenchMultiStream<Arm>(streams, ops);
  }
};

template <typename Arm>
struct RunInference {
  void operator()(std::size_t steps) const {
    BenchLayerwiseInference<Arm>(steps);
  }
};

template <typename Arm>
struct RunConcurrentInference {
  void operator()(std::size_t threads, std::size_t sequences,
                  std::size_t steps) const {
    BenchConcurrentInference<Arm>(threads, sequences, steps);
  }
};

template <typename Arm>
struct RunFragmentation {
  void operator()(std::size_t operations) const {
    BenchFragmentation<Arm>(operations);
  }
};

template <typename Arm>
struct RunLatencyTail {
  void operator()(std::size_t size, std::size_t operations) const {
    BenchLatencyTail<Arm>(size, operations);
  }
};

template <typename Arm>
struct RunBandwidth {
  void operator()() const { BenchBandwidth<Arm>(); }
};

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--quick") {
      quick = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cerr << "usage: perf_allocator_matrix [--quick]\n"
                   "  --quick  fewer iterations and thread counts\n";
      return 0;
    }
  }

  if (!PrepareRuntime()) {
    return 0;
  }

  const bool with_async = CudaAsyncArm::Available();
  if (!with_async) {
    std::cerr << "cuda_async arm skipped: the backend does not provide a "
                 "stream-ordered allocator.\n";
  }

  const std::size_t rotations = quick ? 200u : 1000u;
  const std::size_t large_rotations = quick ? 60u : 200u;
  const std::size_t thread_ops = quick ? 5000u : 20000u;
  const std::size_t stream_ops = quick ? 200u : 1000u;
  const std::size_t decode_steps = quick ? 8u : 32u;

  ForEachArm<RunCrossThreshold>(with_async, rotations);
  ForEachArm<RunLargeRecycle>(with_async, large_rotations);
  ForEachArm<RunHighWater>(with_async);
  ForEachArm<RunTrimCost>(with_async, std::size_t{64});

  const std::vector<std::size_t> thread_counts =
      quick ? std::vector<std::size_t>{1, 8, 32}
            : std::vector<std::size_t>{1, 2, 4, 8, 16, 32, 64};
  for (const std::size_t threads : thread_counts) {
    ForEachArm<RunThreadScaling>(with_async, threads, thread_ops);
  }

  ForEachArm<RunImplicitSync>(with_async);
  ForEachArm<RunLedger>(with_async);

  const std::vector<std::size_t> stream_counts =
      quick ? std::vector<std::size_t>{2, 8}
            : std::vector<std::size_t>{2, 4, 8};
  for (const std::size_t streams : stream_counts) {
    ForEachArm<RunMultiStream>(with_async, streams, stream_ops);
  }

  ForEachArm<RunInference>(with_async, decode_steps);

  // Per-allocation latency distribution, at a size the arena's front cache
  // covers and one it does not -- 64 KiB is the last fast bin, 256 KiB is past
  // them, so the pair separates "the cache is working" from "the pool mutex and
  // the best-fit lookup are the cost".
  const std::size_t latency_ops = quick ? 20000u : 200000u;
  for (const std::size_t size : {64ull << 10, 256ull << 10}) {
    ForEachArm<RunLatencyTail>(with_async, static_cast<std::size_t>(size),
                               latency_ops);
  }

  ForEachArm<RunBandwidth>(with_async);

  // The one workload that is both multi-threaded and shaped like inference.
  const std::size_t sequences = quick ? 4u : 16u;
  const std::size_t conc_steps = quick ? 4u : 16u;
  const std::vector<std::size_t> inference_threads =
      quick ? std::vector<std::size_t>{1, 8}
            : std::vector<std::size_t>{1, 2, 4, 8, 16, 32};
  for (const std::size_t threads : inference_threads) {
    ForEachArm<RunConcurrentInference>(with_async, threads, sequences,
                                       conc_steps);
  }

  // Last, because it deliberately leaves the allocator in a fragmented state
  // and a fresh arm is constructed per benchmark anyway -- but running it ahead
  // of the others would still pay a device-wide `cudaFree` sync at teardown that
  // the next benchmark's first sample would absorb.
  ForEachArm<RunFragmentation>(with_async, quick ? 50000u : 500000u);

  PrintMatrix();
  return 0;
}
