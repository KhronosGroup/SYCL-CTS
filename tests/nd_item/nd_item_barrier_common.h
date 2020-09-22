/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Provides common methods for nd_item barrier tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_ND_ITEM_ND_ITEM_BARRIER_COMMON_H
#define __SYCLCTS_TESTS_ND_ITEM_ND_ITEM_BARRIER_COMMON_H

#include "../common/common.h"
#include <numeric>
#include <vector>

namespace {

/**
 * @brief Test barrier works for local address space
 * @tparam kernelT Kernel to run onto
 * @tparam barrierCallT Type of barrier call. Deduced.
 * @param log Logger to use
 * @param queue Queue to use
 * @param barrier Lambda with the barrier call within
 */
template <class kernelT, typename barrierCallT>
bool test_barrier_local_space(sycl_cts::util::logger& log,
                              cl::sycl::queue &queue,
                              const barrierCallT& barrier)
{
  // Set workspace size
  const size_t globalSize = 64;
  const size_t localSize = 2;

  // Check work-group size limits; skip test in case it cannot be run
  if (!device_supports_wg_size(log, queue, localSize) ||
      !kernel_supports_wg_size<kernelT>(log, queue, localSize))
    return true;

  // Allocate and assign host data
  std::vector<int> data(globalSize);

  std::iota(data.begin(), data.end(), 0);

  // Init ranges
  cl::sycl::range<1> globalRange(globalSize);
  cl::sycl::range<1> localRange(localSize);
  cl::sycl::nd_range<1> NDRange(globalRange, localRange);

  // Run kernel to swap adjacent work item's id within data
  {
    cl::sycl::buffer<int, 1> buf(data.data(), globalRange);

    queue.submit([&](cl::sycl::handler &cgh) {
      auto ptr = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          tile(localRange, cgh);

      cgh.parallel_for<kernelT>(
          NDRange, [=](cl::sycl::nd_item<1> item) {
            const size_t idx = item.get_global_linear_id();
            const size_t pos = idx & 1;
            const size_t opp = pos ^ 1;

            tile[pos] = ptr[idx];
            barrier(item);
            ptr[idx] = tile[opp];
          });
    });
  }

  // Check correct results returned
  bool passed = true;
  for (int i = 0; i < globalSize; i += 2) {
    const int current = i;
    const int next = i + 1;
    if ((data[current] != next) || (data[next] != current)) {
      passed = false;
    }
  }

  return passed;
}

/**
 * @brief Test barrier works for global address space
 * @tparam kernelT Kernel to run onto
 * @tparam barrierCallT Type of barrier call. Deduced.
 * @param log Logger to use
 * @param queue Queue to use
 * @param barrier Lambda with the barrier call within
 */
template <class kernelT, typename barrierCallT>
bool test_barrier_global_space(sycl_cts::util::logger& log,
                               cl::sycl::queue &queue,
                               const barrierCallT& barrier)
{
  // Set workspace size
  const size_t globalSize = 64;
  const size_t localSize = 2;

  // Check work-group size limits; skip test in case it cannot be run
  if (!device_supports_wg_size(log, queue, localSize) ||
      !kernel_supports_wg_size<kernelT>(log, queue, localSize))
    return true;

  // Allocate and assign host data
  std::vector<int> data(globalSize);

  std::iota(data.begin(), data.end(), 0);

  // Init ranges
  cl::sycl::range<1> globalRange(globalSize);
  cl::sycl::range<1> localRange(localSize);
  cl::sycl::nd_range<1> NDRange(globalRange, localRange);

  // Run kernel to swap adjacent work item's id within data
  {
    cl::sycl::buffer<int, 1> buffer(data.data(), globalRange);

    queue.submit([&](cl::sycl::handler &cgh) {
      auto ptr = buffer.get_access<cl::sycl::access::mode::read_write>(cgh);

      cgh.parallel_for<kernelT>(
          NDRange, [=](cl::sycl::nd_item<1> item) {
            const size_t pos = item.get_global_linear_id();
            const size_t opp = pos ^ 1;

            const int val = ptr[pos];
            barrier(item);
            ptr[opp] = val;
          });
    });
  }

  // Check correct results returned
  bool passed = true;
  for (int i = 0; i < globalSize; i += 2) {
    int current = i;
    int next = i + 1;
    if ((data[current] != next) || (data[next] != current)) {
      passed = false;
    }
  }
  return passed;
}

} //namespace

#endif // __SYCLCTS_TESTS_ND_ITEM_ND_ITEM_BARRIER_COMMON_H