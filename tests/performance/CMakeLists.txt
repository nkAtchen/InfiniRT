set(INFINI_RT_PERF_BACKEND_NAME "cpu")
if(WITH_NVIDIA)
    set(INFINI_RT_PERF_BACKEND_NAME "nvidia")
elseif(WITH_ILUVATAR)
    set(INFINI_RT_PERF_BACKEND_NAME "iluvatar")
elseif(WITH_HYGON)
    set(INFINI_RT_PERF_BACKEND_NAME "hygon")
elseif(WITH_METAX)
    set(INFINI_RT_PERF_BACKEND_NAME "metax")
elseif(WITH_MARS)
    set(INFINI_RT_PERF_BACKEND_NAME "mars")
elseif(WITH_MOORE)
    set(INFINI_RT_PERF_BACKEND_NAME "moore")
elseif(WITH_CAMBRICON)
    set(INFINI_RT_PERF_BACKEND_NAME "cambricon")
elseif(WITH_ASCEND)
    set(INFINI_RT_PERF_BACKEND_NAME "ascend")
endif()

function(add_infini_rt_performance_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE infinirt)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(${target}
        PRIVATE "INFINI_RT_PERF_BACKEND_NAME=\"${INFINI_RT_PERF_BACKEND_NAME}\"")
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS performance)
endfunction()

add_infini_rt_performance_test(perf_runtime_dispatch perf_runtime_dispatch.cc)
add_infini_rt_performance_test(perf_memory perf_memory.cc)
add_infini_rt_performance_test(perf_memory_pool perf_memory_pool.cc)
add_infini_rt_performance_test(perf_tensor_view perf_tensor_view.cc)
add_infini_rt_performance_test(perf_tensor_view_footprint
                               perf_tensor_view_footprint.cc)

find_package(Threads REQUIRED)
target_link_libraries(perf_memory_pool PRIVATE Threads::Threads)

# Every allocation strategy the current backend offers, measured on one set of
# workloads. Unlike `arena_vs_pool` below, this one drives the real dispatch API,
# so the numbers are the backend's own -- which is the whole point on a device.
#
# `INFINI_RT_PERF_LARGE_ARENA_CONFIG` switches the arena to the production
# 64 MB -> 512 MB ramp and scales the footprints up to match. Only meaningful
# where memory is a device's: on a host build the same values would reserve
# gigabytes and measure the page allocator instead of the pool.
#
# Registered by hand rather than through the helper above, because the full sweep
# runs 64-thread and multi-stream arms across four allocators and the ctest
# target wants `--quick`: as a test the point is that every arm runs clean, not
# that the numbers are publication grade. Run the binary directly for those.
add_executable(perf_allocator_matrix perf_allocator_matrix.cc)
target_link_libraries(perf_allocator_matrix PRIVATE infinirt Threads::Threads)
target_include_directories(perf_allocator_matrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(perf_allocator_matrix
    PRIVATE "INFINI_RT_PERF_BACKEND_NAME=\"${INFINI_RT_PERF_BACKEND_NAME}\"")
if(NOT INFINI_RT_PERF_BACKEND_NAME STREQUAL "cpu")
    target_compile_definitions(perf_allocator_matrix
        PRIVATE INFINI_RT_PERF_LARGE_ARENA_CONFIG)
endif()
add_test(NAME perf_allocator_matrix COMMAND perf_allocator_matrix --quick)
set_tests_properties(perf_allocator_matrix PROPERTIES
    LABELS performance
    TIMEOUT 900)

# `MemoryPool` vs `ArenaMemoryPool` in one process. Unlike the targets above
# this one drives its own upstream stubs rather than the dispatch API -- one of
# them a calibrated busy-wait standing in for a synchronous `cudaMalloc`, which
# is the only condition under which the arena's amortization is visible on a
# machine without a device. It still links `infinirt` for the include paths and
# the backend-name definition the shared reporting header expects.
#
# Registered by hand rather than through the helper above, because the test
# needs `--quick`: the full run spends about half a minute in busy-waits, and as
# a ctest target the point is that the harness runs and its shared-contract
# checks pass, not that the numbers are publication grade. Run the binary
# directly for the longer arms.
add_executable(arena_vs_pool ab/arena_vs_pool.cc)
target_link_libraries(arena_vs_pool PRIVATE infinirt Threads::Threads)
target_include_directories(arena_vs_pool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(arena_vs_pool
    PRIVATE "INFINI_RT_PERF_BACKEND_NAME=\"${INFINI_RT_PERF_BACKEND_NAME}\"")
add_test(NAME arena_vs_pool COMMAND arena_vs_pool --quick)
set_tests_properties(arena_vs_pool PROPERTIES
    LABELS performance
    TIMEOUT 600)
