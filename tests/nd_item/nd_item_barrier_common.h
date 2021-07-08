/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
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

template<int dim>
bool check_result(std::vector<size_t>&, const sycl::range<dim>&, size_t) {
  return true;
}

template<>
bool check_result<1>(std::vector<size_t> &data,
                     const sycl::range<1> &globalRange,
                     size_t localSize) {
  size_t globalSize = globalRange[0];
  for (size_t id = 0; id < globalSize; id += localSize) {
    const size_t current = id;
    const size_t opp = id + 1;
    if ((data[current] != opp) || (data[opp] != current)) {
      return false;
    }
  }
  return true;
}

template<>
bool check_result<2>(std::vector<size_t> &data,
                     const sycl::range<2> &globalRange,
                     size_t localSize) {
  size_t globalSize0 = globalRange[0];
  size_t globalSize1 = globalRange[1];
  for (size_t id0 = 0; id0 < globalSize0; id0 += localSize) {
    for (size_t id1 = 0; id1 < globalSize1; id1 ++) {
      // id1 + (id0 · r2)
      const size_t current = id0 * globalSize0 + id1;
      const size_t opp = (id0 + 1) * globalSize0 + id1 ^ 1;
      if ((data[current] != opp) || (data[opp] != current)) {
        return false;
      }
    }
  }
  return true;
}

template<>
bool check_result<3>(std::vector<size_t>& data,
                     const sycl::range<3> &globalRange,
                     size_t localSize) {
  size_t globalSize0 = globalRange[0];
  size_t globalSize1 = globalRange[1];
  size_t globalSize2 = globalRange[2];
  for (size_t id0 = 0; id0 < globalSize0; id0 += localSize)
    for (size_t id1 = 0; id1 < globalSize1; id1 ++)
      for (size_t id2 = 0; id2 < globalSize2; id2 ++) {
        // id2 + (id1 · r2) + (id0 · r2 · r1)
        const size_t current = id0 * globalSize0 * globalSize0 +
                               id1 * globalSize1 + id2;
        const size_t opp = (id0 + 1) * globalSize0 * globalSize0 +
                           (id1 ^ 1) * globalSize1 + id2 ^ 1;
        if ((data[current] != opp) || (data[opp] != current)) {
          return false;
        }
  }
  return true;
}

/**
 * @brief Test barrier works for local address space
 * @tparam kernelT Kernel to run onto
 * @tparam barrierCallT Type of barrier call. Deduced.
 * @param log Logger to use
 * @param queue Queue to use
 * @param barrier Lambda with the barrier call within
 */
template <int dim, class kernelT, typename barrierCallT>
void test_barrier_local_space(sycl_cts::util::logger &log,
                              sycl::queue &queue,
                              const barrierCallT &barrier,
                              const std::string &errorMsg) {

  const size_t globalSizeD1 = (dim == 3) ? 4 : 8;
  const size_t globalSizeD2 = 4;
  const size_t totalGlobalSize = 64;
  const size_t localSize = 2;

  // Check work-group size limits; skip test in case it cannot be run
  if (!device_supports_wg_size(log, queue, localSize) ||
      !kernel_supports_wg_size<kernelT>(log, queue, localSize)) {
    log.note("skipping test because work-group size limits is exceeded.");
    return;
  }

  const auto globalRange =
      sycl_cts::util::get_cts_object::range<dim>::template
          get_fixed_size<totalGlobalSize>(globalSizeD1, globalSizeD2);
  const auto localRange =
      sycl_cts::util::get_cts_object::range<dim>::get(localSize, localSize,
                                                      localSize);
  sycl::nd_range<dim> NDRange(globalRange, localRange);

  // Allocate and assign host data
  std::vector<size_t> data(globalRange.size());

  std::iota(data.begin(), data.end(), 0);

  // Run kernel to swap adjacent work item's id within data
  {
    sycl::buffer<size_t, dim> buf(data.data(), globalRange);

    queue.submit([&](sycl::handler &cgh) {
      auto ptr = buf.template get_access<sycl::access_mode::read_write>(cgh);
      sycl::accessor<size_t, dim, sycl::access_mode::read_write,
                         sycl::target::local>
          tile(localRange, cgh);

      cgh.parallel_for<kernelT>(
          NDRange, [=](sycl::nd_item<dim> item) {
            sycl::id<dim> item_id = item.get_global_id();
            sycl::id<dim> pos_id;
            sycl::id<dim> opp_id;
            for (size_t i = 0; i < dim; i++){
              const size_t idx = item_id[i];
              const size_t pos = idx & 1;
              pos_id[i] = pos;
              const size_t opp = pos ^ 1;
              opp_id[i] = opp;
            }
            tile[pos_id] = ptr[item_id];
            barrier(item);
            ptr[item_id] = tile[opp_id];
          });
    });
  }

  // Check correct results returned
  if (!check_result<dim>(data, globalRange, localSize))
    FAIL(log, errorMsg + " for dim = " + std::to_string(dim));
}

/**
 * @brief Test barrier works for global address space
 * @tparam kernelT Kernel to run onto
 * @tparam barrierCallT Type of barrier call. Deduced.
 * @param log Logger to use
 * @param queue Queue to use
 * @param barrier Lambda with the barrier call within
 */
template <int dim, class kernelT, typename barrierCallT>
void test_barrier_global_space(sycl_cts::util::logger &log,
                               sycl::queue &queue,
                               const barrierCallT &barrier,
                               const std::string &errorMsg) {

  const size_t globalSizeD1 = (dim == 3) ? 4 : 8;
  const size_t globalSizeD2 = 4;
  const size_t totalGlobalSize = 64;
  const size_t localSize = 2;

  // Check work-group size limits; skip test in case it cannot be run
  if (!device_supports_wg_size(log, queue, localSize) ||
      !kernel_supports_wg_size<kernelT>(log, queue, localSize)) {
    log.note("skipping test because work-group size limits is exceeded.");
    return;
  }

  const auto globalRange =
      sycl_cts::util::get_cts_object::range<dim>::template
          get_fixed_size<totalGlobalSize>(globalSizeD1, globalSizeD2);
  const auto localRange =
      sycl_cts::util::get_cts_object::range<dim>::get(localSize, localSize,
                                                      localSize);
  sycl::nd_range<dim> NDRange(globalRange, localRange);

  // Allocate and assign host data
  std::vector<size_t> data(globalRange.size());

  std::iota(data.begin(), data.end(), 0);

  // Run kernel to swap adjacent work item's id within data
  {
    sycl::buffer<size_t, dim> buffer(data.data(), globalRange);

    queue.submit([&](sycl::handler &cgh) {
      auto ptr = buffer.template get_access<sycl::access_mode::read_write>(cgh);

      cgh.parallel_for<kernelT>(
          NDRange, [=](sycl::nd_item<dim> item) {
            sycl::id<dim> item_id = item.get_global_id();
            sycl::id<dim> opp_id;
            for (size_t i = 0; i < dim; i++){
              const size_t pos = item_id[i];
              const size_t opp = pos ^ 1;
              opp_id[i] = opp;
            }
            const size_t val = ptr[item_id];
            barrier(item);
            ptr[opp_id] = val;
          });
    });
  }

  // Check correct results returned
  if (!check_result<dim>(data, globalRange, localSize))
    FAIL(log, errorMsg + " for dim = " + std::to_string(dim));
}

} //namespace

#endif // __SYCLCTS_TESTS_ND_ITEM_ND_ITEM_BARRIER_COMMON_H
