// A/B benchmark: the same allocation workload run twice -- straight through the
// backend allocator, and through `ArenaMemoryPool` -- so the two arms are
// directly comparable.
//
// The arena reserves a large backing once and slices it, so an allocation goes
// upstream only when the arena has no room. The workloads below are chosen so
// that difference is visible rather than averaged away: `MissPath` forces a miss
// every iteration, and `MixedSizeClasses` rotates through sizes that a
// size-class design could not share but the arena can.
//
// Every paired benchmark emits one JSON row per arm, differing only in the
// `allocator` param (`direct` / `arena`) and sharing the same iteration count and
// unit, so a consumer can divide one by the other. A human-readable speedup table
// is written to stderr at the end; stdout stays pure JSON for
// `scripts/run_performance_tests.py`.
//
// `perf_allocator_matrix.cc` covers the shapes this file does not: gigabyte-scale
// growth, trim cost split into bookkeeping and upstream frees, device-only
// effects, and the vendor's own stream-ordered pool as a third arm. This file
// stays on the general shapes -- single block, working-set churn, mixed sizes,
// first-touch growth, thread scaling.
//
// The arena is instantiated over `DispatchUpstream`, an adapter over the
// `runtime::` dispatch API, so this one file measures whichever backend the
// library was built with (CPU, NVIDIA, ...) without per-backend variants.
#include <infini/rt.h>
#include <infini/rt/arena_memory_pool.h>

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
#include <vector>

#include "perf_common.h"

namespace {

namespace perf = infini::rt::perf;
namespace runtime = infini::rt::runtime;

// Satisfies the arena's upstream contract by forwarding to the dispatch API,
// which routes to whichever backend is active at runtime.
struct DispatchUpstream {
  using Error = runtime::Error;
  static constexpr Error kSuccess = runtime::kSuccess;

  static Error Malloc(void** ptr, std::size_t size) {
    return runtime::Malloc(ptr, size);
  }

  static Error Free(void* ptr) { return runtime::Free(ptr); }
};

// A megabyte-scale arena config. The production default reserves 64 MB per
// backing and ramps to 512 MB, which a shared or small device may not have to
// spare -- and reserving that much would make the arena's numbers a measure of
// the device's free memory rather than of the allocator. The ratios that matter
// are preserved: an 8x doubling headroom to the cap, and a small/large threshold
// well below it.
struct PerfArenaConfig {
  static constexpr std::size_t kInitialCapacity = 8ull << 20;   // 8 MB
  static constexpr std::size_t kMaxCapacity = 64ull << 20;      // 64 MB
  static constexpr std::size_t kSmallThreshold = 1ull << 20;    // 1 MB
  static constexpr std::size_t kMinSliceAlignment = 512;
  static constexpr std::size_t kMinSplitRemainder = 512;
  static constexpr std::size_t kShrinkThreshold = 16;
  static constexpr std::uint32_t kEmptyScansToDestroy = 2;
};

using Arena = infini::rt::ArenaMemoryPool<DispatchUpstream, PerfArenaConfig>;

bool Success(runtime::Error status) { return status == runtime::kSuccess; }

bool PrepareRuntime() {
  int device_count = 0;
  if (!Success(runtime::GetDeviceCount(&device_count)) || device_count <= 0) {
    std::cerr << "perf_memory_pool skipped: no available device." << std::endl;
    return false;
  }
  if (!Success(runtime::SetDevice(0))) {
    std::cerr << "perf_memory_pool skipped: device 0 is not available."
              << std::endl;
    return false;
  }
  return true;
}

std::vector<std::size_t> TestSizes() {
  std::vector<std::size_t> sizes{
      4 * 1024,
      64 * 1024,
      1024 * 1024,
      16 * 1024 * 1024,
  };

  if (std::getenv("INFINI_RT_PERF_ENABLE_LARGE") != nullptr) {
    sizes.push_back(256ULL * 1024ULL * 1024ULL);
  }

  return sizes;
}

// Device allocators cost microseconds per call, so large sizes get few
// iterations. Both arms of a pair always share this count.
std::size_t IterationsForSize(std::size_t size) {
  if (size <= 4 * 1024) {
    return 2000;
  }
  if (size <= 64 * 1024) {
    return 1000;
  }
  if (size <= 1024 * 1024) {
    return 500;
  }
  if (size <= 16 * 1024 * 1024) {
    return 100;
  }
  return 20;
}

std::vector<perf::Param> SizeParams(std::size_t size, const char* allocator) {
  return {perf::NumberParam("size_bytes", static_cast<std::uint64_t>(size)),
          perf::StringParam("allocator", allocator)};
}

// Renders a byte count for the stderr summary table.
std::string DescribeSize(std::size_t size) {
  if (size >= 1024 * 1024) {
    return std::to_string(size / (1024 * 1024)) + " MiB";
  }
  if (size >= 1024) {
    return std::to_string(size / 1024) + " KiB";
  }
  return std::to_string(size) + " B";
}

// One comparison row: the same workload measured without a pool and with the
// arena pool.
struct Comparison {
  std::string workload;
  std::string params;
  std::string unit;
  double direct_median = 0.0;
  double arena_median = 0.0;
  // Rows where the direct arm has no counterpart -- a trim has no meaning
  // without a cache -- print `n/a` in the `no pool` column rather than a zero
  // that would read as a measurement.
  bool has_direct = true;
  bool valid = false;
};

std::vector<Comparison> g_comparisons;

void Record(const std::string& workload, const std::string& params,
            const std::string& unit, const perf::Measurement& direct,
            const perf::Measurement& arena) {
  g_comparisons.push_back(Comparison{workload, params, unit, direct.median,
                                     arena.median, true, true});
}

// Probes whether `count` blocks of `size` can be held live at once. Large sizes
// on a small device would otherwise turn a benchmark into an OOM failure.
bool CanHoldLive(std::size_t size, std::size_t count) {
  std::vector<void*> blocks;
  blocks.reserve(count);
  bool ok = true;

  for (std::size_t i = 0; i < count; ++i) {
    void* ptr = nullptr;
    if (!Success(runtime::Malloc(&ptr, size)) || ptr == nullptr) {
      ok = false;
      break;
    }
    blocks.push_back(ptr);
  }

  for (void* ptr : blocks) {
    runtime::Free(ptr);
  }
  return ok;
}

// Workload A: allocate one block, free it, repeat. The simplest shape, and the
// one where the arena's caching should show its largest win: every iteration
// after the first is served from the backing, so the upstream allocator is
// never called.
void CompareSingleBlock(std::size_t size) {
  const auto iterations = IterationsForSize(size);

  if (!CanHoldLive(size, 1)) {
    perf::SkipBenchmark("perf_memory_pool.SingleBlock",
                        "device allocation failed during probe");
    return;
  }

  const auto direct = perf::RunBenchmarkMeasured(
      "perf_memory_pool.SingleBlock", SizeParams(size, "direct"), iterations,
      "us", [size] {
        void* ptr = nullptr;
        auto status = runtime::Malloc(&ptr, size);
        perf::DoNotOptimize(status);
        if (Success(status)) {
          status = runtime::Free(ptr);
          perf::DoNotOptimize(status);
        }
      });

  Arena arena;
  const auto arenaed = perf::RunBenchmarkMeasured(
      "perf_memory_pool.SingleBlock", SizeParams(size, "arena"), iterations,
      "us", [&arena, size] {
        void* ptr = nullptr;
        auto status = arena.Allocate(&ptr, size);
        perf::DoNotOptimize(status);
        if (Success(status)) {
          status = arena.Deallocate(ptr);
          perf::DoNotOptimize(status);
        }
      });

  Record("SingleBlock", DescribeSize(size), "us", direct, arenaed);
}

// Workload B: a rolling window of live blocks -- free the oldest, allocate a
// replacement -- the shape a layer-by-layer inference loop produces. Unlike
// workload A the allocator never sees an empty backing, so this is the more
// realistic steady state.
void CompareWorkingSetChurn(std::size_t size) {
  constexpr std::size_t kLiveBlocks = 8;
  const auto iterations = IterationsForSize(size);

  if (!CanHoldLive(size, kLiveBlocks + 1)) {
    perf::SkipBenchmark("perf_memory_pool.WorkingSetChurn",
                        "device cannot hold the working set");
    return;
  }

  auto make_params = [size](const char* allocator) {
    return std::vector<perf::Param>{
        perf::NumberParam("size_bytes", static_cast<std::uint64_t>(size)),
        perf::NumberParam("live_blocks", kLiveBlocks),
        perf::StringParam("allocator", allocator)};
  };

  std::vector<void*> direct_blocks(kLiveBlocks, nullptr);
  for (auto& block : direct_blocks) {
    if (!Success(runtime::Malloc(&block, size))) {
      for (void* held : direct_blocks) {
        if (held != nullptr) {
          runtime::Free(held);
        }
      }
      perf::SkipBenchmark("perf_memory_pool.WorkingSetChurn",
                          "prefill of the direct arm failed");
      return;
    }
  }

  std::size_t direct_cursor = 0;
  const auto direct = perf::RunBenchmarkMeasured(
      "perf_memory_pool.WorkingSetChurn", make_params("direct"), iterations,
      "us", [&direct_blocks, &direct_cursor, size] {
        void*& slot = direct_blocks[direct_cursor];
        direct_cursor = (direct_cursor + 1) % kLiveBlocks;
        auto status = runtime::Free(slot);
        slot = nullptr;
        if (Success(status)) {
          status = runtime::Malloc(&slot, size);
        }
        perf::DoNotOptimize(status);
      });

  for (void* block : direct_blocks) {
    if (block != nullptr) {
      runtime::Free(block);
    }
  }

  Arena arena;
  std::vector<void*> arena_blocks(kLiveBlocks, nullptr);
  for (auto& block : arena_blocks) {
    if (!Success(arena.Allocate(&block, size))) {
      perf::SkipBenchmark("perf_memory_pool.WorkingSetChurn",
                          "prefill of the arena arm failed");
      return;
    }
  }

  std::size_t arena_cursor = 0;
  const auto arenaed = perf::RunBenchmarkMeasured(
      "perf_memory_pool.WorkingSetChurn", make_params("arena"), iterations, "us",
      [&arena, &arena_blocks, &arena_cursor, size] {
        void*& slot = arena_blocks[arena_cursor];
        arena_cursor = (arena_cursor + 1) % kLiveBlocks;
        auto status = arena.Deallocate(slot);
        slot = nullptr;
        if (Success(status)) {
          status = arena.Allocate(&slot, size);
        }
        perf::DoNotOptimize(status);
      });

  for (void* block : arena_blocks) {
    if (block != nullptr) {
      arena.Deallocate(block);
    }
  }

  Record("WorkingSetChurn",
         DescribeSize(size) + " x" + std::to_string(kLiveBlocks) + " live",
         "us", direct, arenaed);
}

// Workload C: rotate through many distinct sizes. This is the arena's most
// favorable shape: one backing serves every size, and coalescing means a freed
// block of one size feeds a request of another. A direct allocation cannot reuse
// anything, so it pays the upstream cost on every call.
void CompareMixedSizeClasses() {
  constexpr std::size_t kClasses = 32;
  constexpr std::size_t kStride = 512;
  constexpr std::size_t kIterations = 2000;
  const std::string params = std::to_string(kClasses) + " classes";

  auto size_for = [](std::size_t index) { return (index + 1) * kStride; };

  if (!CanHoldLive(size_for(kClasses - 1), 1)) {
    perf::SkipBenchmark("perf_memory_pool.MixedSizeClasses",
                        "device allocation failed during probe");
    return;
  }

  std::vector<perf::Param> direct_params{
      perf::NumberParam("size_classes", kClasses),
      perf::NumberParam("stride_bytes", kStride),
      perf::StringParam("allocator", "direct")};
  std::vector<perf::Param> arena_params{
      perf::NumberParam("size_classes", kClasses),
      perf::NumberParam("stride_bytes", kStride),
      perf::StringParam("allocator", "arena")};

  std::size_t direct_index = 0;
  const auto direct = perf::RunBenchmarkMeasured(
      "perf_memory_pool.MixedSizeClasses", direct_params, kIterations, "us",
      [&direct_index, &size_for] {
        const auto size = size_for(direct_index);
        direct_index = (direct_index + 1) % kClasses;
        void* ptr = nullptr;
        auto status = runtime::Malloc(&ptr, size);
        perf::DoNotOptimize(status);
        if (Success(status)) {
          status = runtime::Free(ptr);
          perf::DoNotOptimize(status);
        }
      });

  Arena arena;
  std::size_t arena_index = 0;
  const auto arenaed = perf::RunBenchmarkMeasured(
      "perf_memory_pool.MixedSizeClasses", arena_params, kIterations, "us",
      [&arena, &arena_index, &size_for] {
        const auto size = size_for(arena_index);
        arena_index = (arena_index + 1) % kClasses;
        void* ptr = nullptr;
        auto status = arena.Allocate(&ptr, size);
        perf::DoNotOptimize(status);
        if (Success(status)) {
          status = arena.Deallocate(ptr);
          perf::DoNotOptimize(status);
        }
      });

  Record("MixedSizeClasses", params, "us", direct, arenaed);

  // The counters are the real story here, and unlike a timing they are exact:
  // the arena needs one upstream allocation per backing, however many sizes it
  // serves. Reported as its own row so a regression shows up without having to
  // read a latency delta.
  perf::PrintResult("perf_memory_pool.MixedSizeClassesUpstreamCalls",
                    arena_params, kIterations, "count",
                    static_cast<double>(arena.GetStats().upstream_alloc_count),
                    static_cast<double>(arena.GetStats().upstream_alloc_count));

  // Reserved bytes price the arena's retention: it holds whole backings, so it
  // trades memory for the upstream calls it avoids. Reported rather than
  // asserted, since which side of that trade is right depends on the device.
  perf::PrintResult("perf_memory_pool.MixedSizeClassesBytesReserved",
                    arena_params, kIterations, "bytes",
                    static_cast<double>(arena.GetStats().bytes_reserved),
                    static_cast<double>(arena.GetStats().bytes_reserved));

  // Also carried into the stderr table: a call count is the clearest single
  // number explaining the timing, and it belongs next to it rather than only in
  // the JSON. The direct arm has no counter of its own -- every direct
  // allocation is an upstream call by definition -- so the row is arena-only.
  g_comparisons.push_back(Comparison{
      "MixedSizeClasses/upstream", params, "calls", 0.0,
      static_cast<double>(arena.GetStats().upstream_alloc_count), false, true});
}

// Runs `op(thread_index, op_index)` concurrently on `threads` threads,
// `ops_per_thread` times each, and reports nanoseconds per operation. The
// indices let a workload give each thread a different size.
//
// `RunBenchmarkMeasured` cannot host this: its warmup would spawn the thread
// pool a thousand times, and its lambda measures a single operation rather than
// a whole concurrent batch.
// Target duration of one concurrent sample. A batch that finishes in a few
// microseconds measures thread startup and scheduler placement rather than the
// allocator, so the op count is calibrated to fill this window.
constexpr double kTargetSampleMs = 40.0;

// Times `op` single-threaded to estimate its cost, then returns the op count
// per thread needed to fill `kTargetSampleMs`. This keeps the measurement
// window comparable across backends: a CPU allocator at tens of nanoseconds
// gets a large count, a `cudaMalloc` at hundreds of microseconds a small one.
template <typename Op>
std::size_t CalibrateOps(std::size_t threads, Op&& op) {
  constexpr std::size_t kProbeOps = 64;
  constexpr std::size_t kMinOps = 50;
  constexpr std::size_t kMaxOps = 200000;

  for (std::size_t i = 0; i < 8; ++i) {  // warm the allocator's caches
    op(0, i);
  }

  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < kProbeOps; ++i) {
    op(0, i);
  }
  const auto elapsed = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - start)
                           .count();

  if (elapsed <= 0.0) {
    return kMaxOps;
  }

  const double per_op_ms = elapsed / static_cast<double>(kProbeOps);
  const double wanted =
      kTargetSampleMs / (per_op_ms * static_cast<double>(threads));
  if (wanted < static_cast<double>(kMinOps)) {
    return kMinOps;
  }
  if (wanted > static_cast<double>(kMaxOps)) {
    return kMaxOps;
  }
  return static_cast<std::size_t>(wanted);
}

template <typename Op>
perf::Measurement RunThreaded(const std::string& benchmark,
                              const std::vector<perf::Param>& params,
                              std::size_t threads, std::size_t ops_per_thread,
                              Op&& op) {
  constexpr std::size_t kSamples = 9;
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

    // Threads are parked on `go`, so thread creation stays out of the timed
    // region.
    const auto start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto& worker : workers) {
      worker.join();
    }
    const auto end = std::chrono::steady_clock::now();

    // The first pass is warmup.
    if (sample == 0) {
      continue;
    }
    const auto elapsed_ns =
        std::chrono::duration<double, std::nano>(end - start).count();
    samples.push_back(elapsed_ns / static_cast<double>(total_ops));
  }

  const perf::Measurement measurement{perf::Mean(samples),
                                      perf::Median(samples)};
  perf::PrintResult(benchmark, params, total_ops, "ns", measurement.mean,
                    measurement.median);
  return measurement;
}

// Workload D: concurrent allocation. The arena serializes every call on one
// mutex, so this is where it can lose to an allocator that scales.
void CompareThreadScaling(std::size_t threads) {
  constexpr std::size_t kSize = 4 * 1024;
  const std::string params = std::to_string(threads) + " threads";

  if (!CanHoldLive(kSize, threads)) {
    perf::SkipBenchmark("perf_memory_pool.ThreadScaling",
                        "device allocation failed during probe");
    return;
  }

  Arena arena;
  // Calibrate on the arena arm -- the faster of the two -- and give both arms
  // that count, since a pair is only comparable at equal op counts.
  const auto kOpsPerThread =
      CalibrateOps(threads, [&arena](std::size_t, std::size_t) {
        void* ptr = nullptr;
        if (Success(arena.Allocate(&ptr, kSize))) {
          arena.Deallocate(ptr);
        }
      });

  std::vector<perf::Param> direct_params{
      perf::NumberParam("threads", threads),
      perf::NumberParam("size_bytes", kSize),
      perf::NumberParam("ops_per_thread", kOpsPerThread),
      perf::StringParam("allocator", "direct")};
  std::vector<perf::Param> arena_params{
      perf::NumberParam("threads", threads),
      perf::NumberParam("size_bytes", kSize),
      perf::NumberParam("ops_per_thread", kOpsPerThread),
      perf::StringParam("allocator", "arena")};

  const auto direct =
      RunThreaded("perf_memory_pool.ThreadScaling", direct_params, threads,
                  kOpsPerThread, [](std::size_t, std::size_t) {
                    void* ptr = nullptr;
                    auto status = runtime::Malloc(&ptr, kSize);
                    if (Success(status)) {
                      status = runtime::Free(ptr);
                    }
                    perf::DoNotOptimize(status);
                  });

  const auto arenaed =
      RunThreaded("perf_memory_pool.ThreadScaling", arena_params, threads,
                  kOpsPerThread, [&arena](std::size_t, std::size_t) {
                    void* ptr = nullptr;
                    auto status = arena.Allocate(&ptr, kSize);
                    if (Success(status)) {
                      status = arena.Deallocate(ptr);
                    }
                    perf::DoNotOptimize(status);
                  });

  Record("ThreadScaling", params, "ns", direct, arenaed);
}

// Workload E: concurrent allocation spread across many sizes. Each thread starts
// at a different size and rotates, so at any instant the threads are mostly
// working on distinct extents -- the shape a multi-stream server produces.
// `ThreadScaling` hammers a single size instead.
void CompareConcurrentMixedSizes(std::size_t threads) {
  constexpr std::size_t kClasses = 16;
  constexpr std::size_t kStride = 512;
  const std::string params = std::to_string(threads) + "T x16cls";

  // Stride matches the arena's minimum slice alignment, so the sizes stay
  // distinct after rounding.
  auto size_for = [](std::size_t thread, std::size_t op) {
    return ((thread + op) % kClasses + 1) * kStride;
  };

  if (!CanHoldLive(kClasses * kStride, threads)) {
    perf::SkipBenchmark("perf_memory_pool.ConcurrentMixedSizes",
                        "device allocation failed during probe");
    return;
  }

  Arena arena;
  const auto kOpsPerThread =
      CalibrateOps(threads, [&arena, &size_for](std::size_t t, std::size_t op) {
        void* ptr = nullptr;
        if (Success(arena.Allocate(&ptr, size_for(t, op)))) {
          arena.Deallocate(ptr);
        }
      });

  std::vector<perf::Param> direct_params{
      perf::NumberParam("threads", threads),
      perf::NumberParam("size_classes", kClasses),
      perf::NumberParam("ops_per_thread", kOpsPerThread),
      perf::StringParam("allocator", "direct")};
  std::vector<perf::Param> arena_params{
      perf::NumberParam("threads", threads),
      perf::NumberParam("size_classes", kClasses),
      perf::NumberParam("ops_per_thread", kOpsPerThread),
      perf::StringParam("allocator", "arena")};

  const auto direct = RunThreaded(
      "perf_memory_pool.ConcurrentMixedSizes", direct_params, threads,
      kOpsPerThread, [&size_for](std::size_t thread, std::size_t op) {
        void* ptr = nullptr;
        auto status = runtime::Malloc(&ptr, size_for(thread, op));
        if (Success(status)) {
          status = runtime::Free(ptr);
        }
        perf::DoNotOptimize(status);
      });

  const auto arenaed = RunThreaded(
      "perf_memory_pool.ConcurrentMixedSizes", arena_params, threads,
      kOpsPerThread, [&arena, &size_for](std::size_t thread, std::size_t op) {
        void* ptr = nullptr;
        auto status = arena.Allocate(&ptr, size_for(thread, op));
        if (Success(status)) {
          status = arena.Deallocate(ptr);
        }
        perf::DoNotOptimize(status);
      });

  Record("ConcurrentMixedSizes", params, "ns", direct, arenaed);
}

// The arena's worst case: `ReleaseCached` every iteration empties the cache, so
// each allocation must call upstream. This isolates the bookkeeping the arena
// adds on top of a raw allocation. Arena-only -- the extra trim work does not
// exist for the direct arm -- so read it against this file's `SingleBlock`
// arena arm at the same size rather than as a speedup over `direct`.
void MeasureMissPath(std::size_t size) {
  const auto iterations = IterationsForSize(size);

  if (!CanHoldLive(size, 1)) {
    perf::SkipBenchmark("perf_memory_pool.MissPath",
                        "device allocation failed during probe");
    return;
  }

  Arena arena;
  perf::RunBenchmark("perf_memory_pool.MissPath",
                     SizeParams(size, "arena"), iterations, "us",
                     [&arena, size] {
                       void* ptr = nullptr;
                       auto status = arena.Allocate(&ptr, size);
                       if (Success(status)) {
                         status = arena.Deallocate(ptr);
                       }
                       perf::DoNotOptimize(status);
                       arena.ReleaseCached();
                     });
}

// The workload the arena was built for: a growing set of live blocks at mixed
// sizes, with nothing ever freed until the end. Nothing can be reused -- and
// that is the point. A direct allocation must call upstream once per block; the
// arena calls upstream once per backing and slices the rest. The upstream call
// count is reported alongside the timing because on a host `malloc` the timing
// understates the gap: an upstream call here costs tens of nanoseconds, where a
// `cudaMalloc` costs hundreds of microseconds.
void CompareFirstTouchGrowth() {
  constexpr std::size_t kBlocks = 2000;
  constexpr std::size_t kStride = 512;
  constexpr std::size_t kClasses = 64;
  const std::string params = std::to_string(kBlocks) + " blocks";

  auto size_for = [](std::size_t index) {
    return (index % kClasses + 1) * kStride;
  };

  // Probe the total footprint rather than a single block: this workload holds
  // every block live at once.
  if (!CanHoldLive(kClasses * kStride, 16)) {
    perf::SkipBenchmark("perf_memory_pool.FirstTouchGrowth",
                        "device allocation failed during probe");
    return;
  }

  // One sample is one full build-up-and-tear-down cycle, so the iteration count
  // is the block count and the reported unit is per-block.
  std::vector<void*> blocks;
  blocks.reserve(kBlocks);

  const auto direct = perf::RunBenchmarkMeasured(
      "perf_memory_pool.FirstTouchGrowth",
      {perf::NumberParam("blocks", kBlocks),
       perf::StringParam("allocator", "direct")},
      20, "us", [&blocks, &size_for] {
        for (std::size_t i = 0; i < kBlocks; ++i) {
          void* ptr = nullptr;
          if (Success(runtime::Malloc(&ptr, size_for(i)))) {
            blocks.push_back(ptr);
          }
        }
        for (void* ptr : blocks) {
          runtime::Free(ptr);
        }
        blocks.clear();
      });

  std::size_t arena_upstream = 0;
  const auto arenaed = perf::RunBenchmarkMeasured(
      "perf_memory_pool.FirstTouchGrowth",
      {perf::NumberParam("blocks", kBlocks),
       perf::StringParam("allocator", "arena")},
      20, "us", [&blocks, &size_for, &arena_upstream] {
        // A fresh arena each sample: a warm arena would serve the whole build-up
        // from its backing and measure the opposite of what this benchmark is
        // for.
        Arena arena;
        for (std::size_t i = 0; i < kBlocks; ++i) {
          void* ptr = nullptr;
          if (Success(arena.Allocate(&ptr, size_for(i)))) {
            blocks.push_back(ptr);
          }
        }
        arena_upstream = arena.GetStats().upstream_alloc_count;
        for (void* ptr : blocks) {
          arena.Deallocate(ptr);
        }
        blocks.clear();
      });

  Record("FirstTouchGrowth", params, "us", direct, arenaed);

  perf::PrintResult("perf_memory_pool.FirstTouchGrowthUpstreamCalls",
                    {perf::NumberParam("blocks", kBlocks),
                     perf::StringParam("allocator", "arena")},
                    kBlocks, "count", static_cast<double>(arena_upstream),
                    static_cast<double>(arena_upstream));

  // The direct arm's count is exact rather than measured: one upstream call per
  // block, by definition.
  g_comparisons.push_back(Comparison{"FirstTouchGrowth/upstream", params,
                                     "calls", static_cast<double>(kBlocks),
                                     static_cast<double>(arena_upstream), true,
                                     true});
}

// Concurrent miss path: `ReleaseCached` after every operation keeps the backing
// empty, so each `Allocate` calls upstream. This is the benchmark that prices
// holding the arena's lock across an upstream call -- if the lock is held, one
// thread's slow `cudaMalloc` blocks every other thread, and per-operation cost
// climbs with the thread count even though the work per thread is fixed.
//
// Arena-only: there is no meaningful `direct` arm, since the extra
// `ReleaseCached` work exists only for the arena. Read it as a scaling curve
// across thread counts, not as a ratio.
void MeasureConcurrentMissPath(std::size_t threads) {
  constexpr std::size_t kSize = 4 * 1024;

  if (!CanHoldLive(kSize, threads)) {
    perf::SkipBenchmark("perf_memory_pool.ConcurrentMissPath",
                        "device allocation failed during probe");
    return;
  }

  Arena arena;
  auto miss_op = [&arena](std::size_t, std::size_t) {
    void* ptr = nullptr;
    const auto status = arena.Allocate(&ptr, kSize);
    if (Success(status)) {
      arena.Deallocate(ptr);
    }
    perf::DoNotOptimize(status);
    arena.ReleaseCached();
  };
  const auto kOpsPerThread = CalibrateOps(threads, miss_op);

  RunThreaded("perf_memory_pool.ConcurrentMissPath",
              {perf::NumberParam("threads", threads),
               perf::NumberParam("size_bytes", kSize),
               perf::NumberParam("ops_per_thread", kOpsPerThread),
               perf::StringParam("allocator", "arena")},
              threads, kOpsPerThread, [&arena](std::size_t, std::size_t) {
                void* ptr = nullptr;
                auto status = arena.Allocate(&ptr, kSize);
                if (Success(status)) {
                  status = arena.Deallocate(ptr);
                }
                perf::DoNotOptimize(status);
                arena.ReleaseCached();
              });
}

// Cache hit with an alignment request. The arena splits the leading gap off as a
// reusable free chunk, so an aligned hit costs more bookkeeping than a plain
// one. Compare against the `SingleBlock` arena arm at the same size to price the
// alignment. Arena-only: the dispatch API has no aligned entry point to race it
// against.
void MeasureAlignedHit() {
  constexpr std::size_t kSize = 4 * 1024;
  constexpr std::size_t kAlignment = 256;
  constexpr std::size_t kIterations = 200000;

  Arena arena;
  void* probe = nullptr;
  if (!Success(arena.Allocate(&probe, kSize, kAlignment)) || probe == nullptr) {
    perf::SkipBenchmark("perf_memory_pool.AlignedHit",
                        "aligned allocation failed during probe");
    return;
  }
  arena.Deallocate(probe);

  perf::RunBenchmark("perf_memory_pool.AlignedHit",
                     {perf::NumberParam("size_bytes", kSize),
                      perf::NumberParam("alignment", kAlignment),
                      perf::StringParam("allocator", "arena")},
                     kIterations, "ns", [&arena] {
                       void* ptr = nullptr;
                       auto status = arena.Allocate(&ptr, kSize, kAlignment);
                       if (Success(status)) {
                         status = arena.Deallocate(ptr);
                       }
                       perf::DoNotOptimize(status);
                     });
}

// Cost of trimming the cache, measured per cached block. Each iteration refills
// the cache first, so the reported figure includes that refill.
//
// The upstream free count is reported next to the timing, because the timing
// alone attributes the cost to the wrong thing. A trim does two separable jobs:
// it walks bookkeeping (an ordered set plus a backing vector), and it issues
// upstream frees. On a host those cost about the same and the wall time is a
// fair summary. On a device they do not: every `cudaFree` implicitly
// synchronizes the whole device, so a design that issues one free per cached
// block stalls all pending work 64 times where one that frees whole backings
// stalls it once. Reporting both means the number that transfers to a device is
// visible even in a host run.
void MeasureReleaseCached() {
  constexpr std::size_t kSize = 4 * 1024;
  constexpr std::size_t kCachedBlocks = 64;
  constexpr std::size_t kIterations = 200;

  if (!CanHoldLive(kSize, kCachedBlocks)) {
    perf::SkipBenchmark("perf_memory_pool.ReleaseCached",
                        "device allocation failed during probe");
    return;
  }

  // The arena's trim is cheap by construction: 64 cached blocks live inside one
  // backing, so a trim is one upstream free rather than 64.
  Arena arena;
  std::vector<void*> blocks(kCachedBlocks, nullptr);
  perf::RunBenchmark("perf_memory_pool.ReleaseCached",
                     {perf::NumberParam("cached_blocks", kCachedBlocks),
                      perf::NumberParam("size_bytes", kSize),
                      perf::StringParam("allocator", "arena")},
                     kIterations, "us", [&arena, &blocks] {
                       for (void*& block : blocks) {
                         arena.Allocate(&block, kSize);
                       }
                       for (void*& block : blocks) {
                         arena.Deallocate(block);
                         block = nullptr;
                       }
                       arena.ReleaseCached();
                     });

  // Counted in a cycle of its own rather than divided out of the timed run
  // above: the runner's warmup and sample counts are its business, and dividing
  // by an assumed total would go quietly wrong the moment either changes.
  const auto before = arena.GetStats().upstream_free_count;
  for (void*& block : blocks) {
    arena.Allocate(&block, kSize);
  }
  for (void*& block : blocks) {
    arena.Deallocate(block);
    block = nullptr;
  }
  arena.ReleaseCached();
  const auto arena_frees =
      static_cast<double>(arena.GetStats().upstream_free_count - before);

  perf::PrintResult("perf_memory_pool.ReleaseCachedUpstreamFrees",
                    {perf::NumberParam("cached_blocks", kCachedBlocks),
                     perf::StringParam("allocator", "arena")},
                    1, "count", arena_frees, arena_frees);

  // The direct arm's count is exact rather than measured: freeing the working
  // set means one upstream free per block.
  g_comparisons.push_back(Comparison{
      "ReleaseCached/frees", std::to_string(kCachedBlocks) + " cached", "calls",
      static_cast<double>(kCachedBlocks), arena_frees, true, true});
}

// Accessors and no-op paths, all arena-only. `GetStats` reads the largest free
// extent out of the arena's ordered set, so it does real work and is measured
// rather than assumed free.
void MeasureBookkeeping() {
  constexpr std::size_t kIterations = 1000000;

  Arena arena;
  // Hold a spread of live blocks so `GetStats` walks a populated free set rather
  // than an empty one.
  std::vector<void*> live;
  for (std::size_t i = 0; i < 64; ++i) {
    void* ptr = nullptr;
    if (Success(arena.Allocate(&ptr, (i % 16 + 1) * 512))) {
      live.push_back(ptr);
    }
  }

  perf::RunBenchmark("perf_memory_pool.GetStats",
                     {perf::StringParam("allocator", "arena")}, kIterations,
                     "ns", [&arena] {
                       const auto stats = arena.GetStats();
                       perf::DoNotOptimize(stats.bytes_in_use);
                     });

  perf::RunBenchmark("perf_memory_pool.AllocateZeroBytes",
                     {perf::StringParam("allocator", "arena")}, kIterations,
                     "ns", [&arena] {
                       void* ptr = nullptr;
                       const auto status = arena.Allocate(&ptr, 0);
                       perf::DoNotOptimize(status);
                       perf::DoNotOptimize(ptr);
                     });

  perf::RunBenchmark("perf_memory_pool.DeallocateNullptr",
                     {perf::StringParam("allocator", "arena")}, kIterations,
                     "ns", [&arena] {
                       const auto status = arena.Deallocate(nullptr);
                       perf::DoNotOptimize(status);
                     });

  for (void* ptr : live) {
    arena.Deallocate(ptr);
  }
}

// Human-readable summary on stderr. stdout stays pure JSON.
void PrintComparisonTable() {
  if (g_comparisons.empty()) {
    return;
  }

  std::cerr << "\n=== " << INFINI_RT_PERF_BACKEND_NAME
            << ": no pool vs ArenaMemoryPool "
               "(median, lower is better) ===\n";
  std::cerr << std::left << std::setw(28) << "workload" << std::setw(16)
            << "params" << std::right << std::setw(14) << "no pool"
            << std::setw(14) << "arena" << std::setw(11) << "arena win"
            << "\n";

  for (const Comparison& row : g_comparisons) {
    if (!row.valid) {
      continue;
    }
    std::cerr << std::left << std::setw(28) << row.workload << std::setw(16)
              << row.params << std::right << std::fixed << std::setprecision(3);
    if (row.has_direct) {
      std::cerr << std::setw(11) << row.direct_median << " " << row.unit;
    } else {
      std::cerr << std::setw(14) << "n/a";
    }
    std::cerr << std::setw(11) << row.arena_median << " " << row.unit;

    if (row.has_direct && row.arena_median > 0.0 && row.direct_median > 0.0) {
      std::cerr << std::setw(10) << std::setprecision(2)
                << (row.direct_median / row.arena_median) << "x";
    } else {
      std::cerr << std::setw(11) << "n/a";
    }
    std::cerr << "\n";
  }

  std::cerr << "\n`arena win` > 1 means the arena is faster than going straight "
               "to the backend.\n`calls` rows are exact counts, not timings. "
               "`n/a` in the `no pool` column marks a\nrow with no direct "
               "counterpart -- a trim has no meaning without a cache.\n";
  std::cerr << std::endl;
}

}  // namespace

int main() {
  if (!PrepareRuntime()) {
    return 0;
  }

  for (const auto size : TestSizes()) {
    CompareSingleBlock(size);
    CompareWorkingSetChurn(size);
    MeasureMissPath(size);
  }

  CompareMixedSizeClasses();
  CompareFirstTouchGrowth();

  // The arena serializes on one mutex, so the curve past a handful of threads is
  // about how long it holds it. Stopping at 8 was well before the knee on a
  // many-core host: the interesting region is where the arena's critical section
  // (ordered-set lookup, split, coalesce) starts to outweigh its far smaller
  // number of upstream calls, and that only shows up once contention is real.
  // Counts above the machine's core count are dropped rather than
  // oversubscribed, which would measure the scheduler.
  const auto hardware_threads = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::thread::hardware_concurrency()));
  for (const std::size_t threads : {std::size_t{1}, std::size_t{2},
                                    std::size_t{4}, std::size_t{8},
                                    std::size_t{16}, std::size_t{32},
                                    std::size_t{64}}) {
    if (threads > hardware_threads) {
      break;
    }
    CompareThreadScaling(threads);
    CompareConcurrentMixedSizes(threads);
    MeasureConcurrentMissPath(threads);
  }

  MeasureAlignedHit();
  MeasureReleaseCached();
  MeasureBookkeeping();

  PrintComparisonTable();
  return 0;
}
