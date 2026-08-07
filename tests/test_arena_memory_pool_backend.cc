// Exercises `ArenaMemoryPool` over a *real* runtime backend (CPU, NVIDIA, ...).
//
// `test_arena_memory_pool.cc` already covers the pool's bookkeeping against a
// mock upstream at kilobyte scale. This test instead instantiates the pool over
// the backend's actual `runtime::Runtime` specialization, which is where the
// arena's central claim has to hold: a slice is an *interior offset* into one
// upstream allocation, so nothing but a real device round trip can prove that
// the pointer arithmetic lands inside genuine device memory and that two
// adjacent slices out of the same backing do not overwrite each other. Device
// pointers cannot be dereferenced from the host, so every check goes through
// `Memcpy`. The whole suite is skipped when no device is present.
#include <infini/rt.h>
#include <infini/rt/arena_memory_pool.h>
#include INFINI_RT_TEST_RUNTIME_HEADER

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "test_helper.h"

namespace {

using Runtime = infini::rt::runtime::Runtime<INFINI_RT_TEST_DEVICE_TYPE>;

// A megabyte-scale config. The production default reserves 64 MB per backing
// and ramps to 512 MB, which a shared CI device may not have to spare -- and at
// that scale a test would never reach the growth, oversize, or shrink paths
// within a reasonable number of allocations. The ratios that matter are
// preserved: an 8x doubling headroom to the cap, and a small/large threshold
// well below it.
struct BackendConfig {
  static constexpr std::size_t kInitialCapacity = 4ull << 20;  // 4 MB
  static constexpr std::size_t kMaxCapacity = 32ull << 20;     // 32 MB
  static constexpr std::size_t kSmallThreshold = 64ull << 10;  // 64 KB
  static constexpr std::size_t kMinSliceAlignment = 512;
  static constexpr std::size_t kMinSplitRemainder = 512;
  static constexpr std::size_t kShrinkThreshold = 8;
  static constexpr std::uint32_t kEmptyScansToDestroy = 2;
};

using Pool = infini::rt::ArenaMemoryPool<Runtime, BackendConfig>;

constexpr const char* kBackend = INFINI_RT_TEST_BACKEND_NAME;

bool SelectDevice() {
  int device_count = 0;
  if (Runtime::GetDeviceCount(&device_count) != Runtime::kSuccess ||
      device_count <= 0) {
    std::cout << kBackend << " arena memory pool skipped: no available device."
              << std::endl;
    return false;
  }
  if (Runtime::SetDevice(0) != Runtime::kSuccess) {
    std::cout << kBackend << " arena memory pool skipped: device 0 unavailable."
              << std::endl;
    return false;
  }
  return true;
}

// Writes `input` into device memory `ptr` and reads it back, asserting the
// bytes survive the round trip. This is the only host-safe way to confirm a
// device pointer is real and usable.
template <std::size_t N>
void ExpectUsable(infini::rt::test::TestContext* context, void* ptr,
                  const std::array<std::uint8_t, N>& input,
                  const char* message) {
  if (!context->Expect(ptr != nullptr, message)) {
    return;
  }
  std::array<std::uint8_t, N> output{};
  context->ExpectEqual(
      Runtime::Memcpy(ptr, input.data(), N, Runtime::kMemcpyHostToDevice),
      Runtime::kSuccess, "memcpy host-to-device should succeed");
  context->ExpectEqual(
      Runtime::Memcpy(output.data(), ptr, N, Runtime::kMemcpyDeviceToHost),
      Runtime::kSuccess, "memcpy device-to-host should succeed");
  context->ExpectEqual(output, input,
                       "pool-allocated memory should round-trip bytes");
}

// Fills device memory `[ptr, ptr + size)` with a pattern derived from `seed`.
bool FillDevice(void* ptr, std::size_t size, std::uint8_t seed) {
  std::vector<std::uint8_t> host(size);
  for (std::size_t i = 0; i < size; ++i) {
    host[i] = static_cast<std::uint8_t>(seed + (i & 0x3f));
  }
  return Runtime::Memcpy(ptr, host.data(), size,
                         Runtime::kMemcpyHostToDevice) == Runtime::kSuccess;
}

bool VerifyDevice(void* ptr, std::size_t size, std::uint8_t seed) {
  std::vector<std::uint8_t> host(size, 0);
  if (Runtime::Memcpy(host.data(), ptr, size, Runtime::kMemcpyDeviceToHost) !=
      Runtime::kSuccess) {
    return false;
  }
  for (std::size_t i = 0; i < size; ++i) {
    if (host[i] != static_cast<std::uint8_t>(seed + (i & 0x3f))) {
      return false;
    }
  }
  return true;
}

// A slice returned by the pool must be real, usable device memory -- not just a
// plausible-looking address computed off a backing base pointer.
void TestSliceIsUsableDeviceMemory(infini::rt::test::TestContext* context) {
  Pool pool;
  void* ptr = nullptr;
  context->ExpectEqual(pool.Allocate(&ptr, 256), Runtime::kSuccess,
                       "allocate should succeed on a real backend");
  const std::array<std::uint8_t, 8> input{0, 1, 2, 3, 4, 5, 6, 7};
  ExpectUsable(context, ptr, input, "allocation should produce a pointer");

  const Pool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.upstream_alloc_count, std::size_t{1},
                       "one device allocation backs the slice");
  context->Expect(stats.bytes_reserved >= BackendConfig::kInitialCapacity,
                  "the backing covers the configured initial capacity");
  context->ExpectEqual(pool.Deallocate(ptr), Runtime::kSuccess,
                       "deallocate should succeed");
}

// The arena's reason for existing: many slices come out of one device
// allocation, and every one of them addresses its own disjoint bytes. An
// off-by-one in the split arithmetic shows up here as one slice reading back a
// neighbor's pattern.
void TestSlicesAreDisjointOnDevice(infini::rt::test::TestContext* context) {
  Pool pool;
  constexpr std::size_t kSlices = 24;
  constexpr std::size_t kSize = 8192;

  std::vector<void*> blocks;
  for (std::size_t i = 0; i < kSlices; ++i) {
    void* ptr = nullptr;
    if (pool.Allocate(&ptr, kSize) != Runtime::kSuccess || ptr == nullptr) {
      context->Expect(false, "slicing allocation should succeed");
      break;
    }
    if (!FillDevice(ptr, kSize, static_cast<std::uint8_t>(i * 7 + 1))) {
      context->Expect(false, "writing a slice should succeed");
    }
    blocks.push_back(ptr);
  }

  bool disjoint = true;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    disjoint = disjoint &&
               VerifyDevice(blocks[i], kSize,
                            static_cast<std::uint8_t>(i * 7 + 1));
  }
  context->Expect(disjoint, "slices out of one backing must not overlap");

  const Pool::Stats stats = pool.GetStats();
  // 24 x 8 KB = 192 KB, comfortably inside one 4 MB backing.
  context->ExpectEqual(stats.upstream_alloc_count, std::size_t{1},
                       "24 slices need only one device allocation");
  context->ExpectEqual(stats.cache_hit_count, kSlices - 1,
                       "only the first allocation creates a backing");

  for (void* ptr : blocks) {
    pool.Deallocate(ptr);
  }
}

// Coalescing has to work on device memory too: after the slices are released
// the backing must serve one request spanning all of them, and that whole span
// must still round-trip bytes.
void TestCoalescedSpanIsUsable(infini::rt::test::TestContext* context) {
  Pool pool;
  std::vector<void*> blocks;
  for (std::size_t i = 0; i < 8; ++i) {
    void* ptr = nullptr;
    pool.Allocate(&ptr, 64 * 1024);
    blocks.push_back(ptr);
  }
  for (void* ptr : blocks) {
    pool.Deallocate(ptr);
  }

  // Round down to the slice granularity: a request is rounded *up* before it is
  // fitted, and the extent's own size need not be a multiple of it -- the
  // backing's head is trimmed by however much the device base pointer was
  // misaligned. Asking for the raw extent size would round past it and
  // legitimately need a second backing.
  const std::size_t span = pool.GetStats().largest_free_chunk /
                           BackendConfig::kMinSliceAlignment *
                           BackendConfig::kMinSliceAlignment;
  context->Expect(span >= 8 * 64 * 1024,
                  "the released slices coalesce into one extent");

  void* whole = nullptr;
  context->ExpectEqual(pool.Allocate(&whole, span), Runtime::kSuccess,
                       "the coalesced extent serves a full-span request");
  context->ExpectEqual(pool.GetStats().upstream_alloc_count, std::size_t{1},
                       "serving it needed no new device allocation");
  // Probe both ends of the span: the arithmetic that produced it has to be
  // right at the tail, not just at the base.
  const std::array<std::uint8_t, 8> input{2, 4, 6, 8, 10, 12, 14, 16};
  ExpectUsable(context, whole, input, "the span's head is usable");
  ExpectUsable(context, static_cast<char*>(whole) + span - 8, input,
               "the span's tail is usable");
  pool.Deallocate(whole);
}

// A requested power-of-two alignment must be honored by the slice, which must
// still be usable device memory.
void TestAlignment(infini::rt::test::TestContext* context) {
  Pool pool;
  constexpr std::size_t kAlignment = 4096;
  // Offset the arena first so the natural next slice is misaligned.
  void* filler = nullptr;
  pool.Allocate(&filler, 512);

  void* ptr = nullptr;
  context->ExpectEqual(pool.Allocate(&ptr, 1024, kAlignment), Runtime::kSuccess,
                       "aligned allocate should succeed");
  context->ExpectEqual(reinterpret_cast<std::uintptr_t>(ptr) % kAlignment,
                       std::uintptr_t{0},
                       "returned pointer should honor the alignment");
  const std::array<std::uint8_t, 8> input{1, 1, 2, 3, 5, 8, 13, 21};
  ExpectUsable(context, ptr, input, "aligned slice should be usable");
  pool.Deallocate(ptr);
  pool.Deallocate(filler);
}

// A request larger than the capacity cap gets its own exactly sized device
// allocation, and every byte of it is addressable.
void TestOversizeRequest(infini::rt::test::TestContext* context) {
  Pool pool;
  constexpr std::size_t kHuge = BackendConfig::kMaxCapacity + (4ull << 20);
  void* ptr = nullptr;
  if (pool.Allocate(&ptr, kHuge) != Runtime::kSuccess || ptr == nullptr) {
    std::cout << kBackend
              << " oversize case skipped: device cannot serve the request."
              << std::endl;
    return;
  }

  const Pool::Stats stats = pool.GetStats();
  context->Expect(stats.bytes_reserved >= kHuge,
                  "the oversize backing covers the request");
  // At least the request, possibly a little more: the tail left over in the
  // oversize backing is below `kMinSplitRemainder`, so it stays with the served
  // chunk as internal waste rather than becoming an unusable sliver.
  context->Expect(stats.bytes_in_use >= kHuge,
                  "the whole request is live");
  context->Expect(
      stats.bytes_in_use - kHuge < BackendConfig::kMinSplitRemainder +
                                       BackendConfig::kMinSliceAlignment,
      "an oversize backing wastes at most a sliver on the served chunk");
  const std::array<std::uint8_t, 8> input{9, 9, 8, 8, 7, 7, 6, 6};
  ExpectUsable(context, ptr, input, "the oversize slice's head is usable");
  ExpectUsable(context, static_cast<char*>(ptr) + kHuge - 8, input,
               "the oversize slice's tail is usable");
  pool.Deallocate(ptr);
}

// The headline behavior on a real device: a burst reserves extra backings, and
// once the workload returns to small requests the extras go back to the device
// while the resident backing stays warm.
void TestShrinkAfterBurst(infini::rt::test::TestContext* context) {
  Pool pool;
  void* resident = nullptr;
  pool.Allocate(&resident, 1024);

  std::vector<void*> burst;
  for (std::size_t i = 0; i < 24; ++i) {
    void* ptr = nullptr;
    if (pool.Allocate(&ptr, 1ull << 20) != Runtime::kSuccess) {
      break;
    }
    burst.push_back(ptr);
  }
  const std::size_t peak_backings = pool.GetStats().backing_count;
  if (peak_backings <= 1) {
    std::cout << kBackend
              << " shrink case skipped: the burst stayed in one backing."
              << std::endl;
    pool.Deallocate(resident);
    for (void* ptr : burst) {
      pool.Deallocate(ptr);
    }
    return;
  }

  for (void* ptr : burst) {
    pool.Deallocate(ptr);
  }
  context->ExpectEqual(pool.GetStats().backing_count, peak_backings,
                       "freeing alone does not release device memory");

  std::vector<void*> small;
  for (std::size_t i = 0;
       i < BackendConfig::kShrinkThreshold *
               (BackendConfig::kEmptyScansToDestroy + 2);
       ++i) {
    void* ptr = nullptr;
    pool.Allocate(&ptr, 1024);
    small.push_back(ptr);
  }

  const Pool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.backing_count, std::size_t{1},
                       "idle backings are returned to the device");
  context->Expect(stats.shrink_count > 0, "shrink was recorded");
  context->ExpectEqual(stats.upstream_free_count, stats.shrink_count,
                       "every shrink is one device free");

  // The resident backing survived, and the block living in it is intact.
  const std::array<std::uint8_t, 8> input{3, 1, 4, 1, 5, 9, 2, 6};
  ExpectUsable(context, resident, input,
               "the resident slice outlived the shrink");
  pool.Deallocate(resident);
  for (void* ptr : small) {
    pool.Deallocate(ptr);
  }
}

// `ReleaseCached` hands every drained backing back to the device, and the pool
// keeps working afterwards.
void TestReleaseCached(infini::rt::test::TestContext* context) {
  Pool pool;
  std::vector<void*> blocks;
  for (std::size_t i = 0; i < 8; ++i) {
    void* ptr = nullptr;
    pool.Allocate(&ptr, 128 * 1024);
    blocks.push_back(ptr);
  }
  for (void* ptr : blocks) {
    pool.Deallocate(ptr);
  }

  pool.ReleaseCached();
  Pool::Stats stats = pool.GetStats();
  context->ExpectEqual(stats.backing_count, std::size_t{0},
                       "an empty pool releases every backing");
  context->ExpectEqual(stats.bytes_reserved, std::size_t{0},
                       "reserved bytes drop to zero");
  context->ExpectEqual(stats.upstream_free_count, stats.upstream_alloc_count,
                       "every device allocation was freed exactly once");

  void* fresh = nullptr;
  context->ExpectEqual(pool.Allocate(&fresh, 4096), Runtime::kSuccess,
                       "the pool still works after a full release");
  const std::array<std::uint8_t, 4> input{7, 7, 7, 7};
  ExpectUsable(context, fresh, input, "a fresh slice is usable");
  pool.Deallocate(fresh);
}

// The production configuration has to instantiate and work over a real backend,
// not just the reduced one the rest of this file uses. One small allocation is
// enough to prove it: it reserves a default-sized backing and slices it.
void TestDefaultConfigInstantiates(infini::rt::test::TestContext* context) {
  infini::rt::ArenaMemoryPool<Runtime> pool;
  void* ptr = nullptr;
  if (pool.Allocate(&ptr, 4096) != Runtime::kSuccess || ptr == nullptr) {
    std::cout << kBackend
              << " default-config case skipped: device cannot reserve "
              << (infini::rt::DefaultArenaConfig::kInitialCapacity >> 20)
              << " MB." << std::endl;
    return;
  }
  const std::array<std::uint8_t, 8> input{1, 2, 3, 4, 5, 6, 7, 8};
  ExpectUsable(context, ptr, input, "a default-config slice is usable");
  context->ExpectEqual(pool.Deallocate(ptr), Runtime::kSuccess,
                       "deallocate should succeed");
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  if (!SelectDevice()) {
    return context.ExitCode();
  }

  TestSliceIsUsableDeviceMemory(&context);
  TestSlicesAreDisjointOnDevice(&context);
  TestCoalescedSpanIsUsable(&context);
  TestAlignment(&context);
  TestOversizeRequest(&context);
  TestShrinkAfterBurst(&context);
  TestReleaseCached(&context);
  TestDefaultConfigInstantiates(&context);

  return context.ExitCode();
}
