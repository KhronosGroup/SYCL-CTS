/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common methods for nd_item mem_fence tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_ND_ITEM_ND_ITEM_MEM_FENCE_COMMON_H
#define __SYCLCTS_TESTS_ND_ITEM_ND_ITEM_MEM_FENCE_COMMON_H

#include "../common/common.h"
#include <functional>
#include <string>
#include <type_traits>

namespace {

/**
 * @brief Test memory fence works for global address space
 * @tparam kernelT Kernel to run onto
 * @tparam dim Nd-item dimension to use
 * @tparam readFenceCallT Type of read fence call. Deduced.
 * @tparam writeFenceCallT Type of write fence call. Deduced.
 * @param log Logger to use
 * @param queue Queue to use
 * @param readMemFence Lambda with the read (load) fence call within
 * @param writeMemFence Lambda with the write (store) fence call within
 */
template <class kernelT, int dim, typename readFenceCallT,
          typename writeFenceCallT>
bool test_rw_mem_fence_global_space(sycl_cts::util::logger& log,
                                    sycl::queue &queue,
                                    const readFenceCallT& readMemFence,
                                    const writeFenceCallT& writeMemFence)
{
  static_assert(dim == 1,
                "Multidimensional nd-items are not supported currently");
  // Set workspace size and cut-off read iterations limit
  const size_t globalSize = 64;
  const size_t localSize = 2;

  // Check work-group size limits; skip test in case it cannot be run
  if (!device_supports_wg_size(log, queue, localSize) ||
      !kernel_supports_wg_size<kernelT>(log, queue, localSize))
    return true;

  bool passed = true;

  // Init ranges
  sycl::range<1> globalRange(globalSize);
  sycl::range<1> localRange(localSize);
  sycl::nd_range<1> NDRange(globalRange, localRange);

  // Run kernel to verify memory ordering works for adjacent work-items
  {
    sycl::buffer<int, 1> data(globalRange);
    sycl::buffer<bool, 1> passedBuf(&passed, sycl::range<1>(1));

    // Initialize data state
    {
      auto ptr = data.get_access<sycl::access::mode::write>();
      for (size_t i = 0; i < ptr.get_count(); ++i)
        ptr[i] = -1;
    }
    queue.submit([&](sycl::handler &cgh) {
      auto ptr = data.get_access<sycl::access::mode::atomic>(cgh);
      auto pass = passedBuf.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for<kernelT>(
          NDRange, [=](sycl::nd_item<1> item) {
            // Use barrier to avoid dependency on thread start timing
            item.barrier(sycl::access::fence_space::global_space);

            const size_t current = item.get_global_linear_id();
            const size_t other = current ^ 1U;

            const int nWrites = 100;
            const int nReads = 100000;
            const int writesPerIteration =
                static_cast<int>(2U * (current + 1U));
            const int readsPerIteration =
                nReads / nWrites * writesPerIteration;

            int previousValue = ptr[other].load();
            int currentValue = 0;

            /** This sequence will work for any concurrent implementation,
             *  including the parallel ones. There is no strict forward
             *  guarantee requirement, so each work-item may execute
             *  independently. The test will fail only if memory ordering
             *  was broken, but still will succeed in case there was no update
             *  at all.
             */
            for (int i = 0; i < nWrites; i+= writesPerIteration) {
              // Run sequence of writes
              for (int j = 0; j < writesPerIteration; ++j) {
                ptr[current].store(i + j);
                writeMemFence(item);
              }
              // Run sequence of reads
              for (int k = 0; k < readsPerIteration; ++k) {
                // Memory fence on read; should be either read_write or read
                readMemFence(item);
                currentValue = ptr[other].load();

                // Verify memory order from other work-item
                if (currentValue < previousValue) {
                  pass[0] = false;
                }
                previousValue = currentValue;
              }
            }
          });
    });
  }

  return passed;
}

/**
 * @brief Test memory fence works for local address space
 * @tparam kernelT Kernel to run onto
 * @tparam dim Nd-item dimension to use
 * @tparam readFenceCallT Type of read fence call. Deduced.
 * @tparam writeFenceCallT Type of write fence call. Deduced.
 * @param log Logger to use
 * @param queue Queue to use
 * @param readMemFence Lambda with the read (load) fence call within
 * @param writeMemFence Lambda with the write (store) fence call within
 */
template <class kernelT, int dim, typename readFenceCallT,
          typename writeFenceCallT>
bool test_rw_mem_fence_local_space(sycl_cts::util::logger& log,
                                   sycl::queue &queue,
                                   const readFenceCallT& readMemFence,
                                   const writeFenceCallT& writeMemFence)
{
  static_assert(dim == 1,
                "Multidimensional nd-items are not supported currently");
  // Set workspace size and cut-off read iterations limit
  const size_t globalSize = 64;
  const size_t localSize = 2;

  // Check work-group size limits; skip test in case it cannot be run
  if (!device_supports_wg_size(log, queue, localSize) ||
      !kernel_supports_wg_size<kernelT>(log, queue, localSize))
    return true;

  bool passed = true;

  // Init ranges
  sycl::range<1> globalRange(globalSize);
  sycl::range<1> localRange(localSize);
  sycl::nd_range<1> NDRange(globalRange, localRange);

  // Run kernel to verify memory ordering works for adjacent work-items
  {
    sycl::buffer<bool, 1> passedBuf(&passed, sycl::range<1>(1));

    queue.submit([&](sycl::handler &cgh) {
      auto pass = passedBuf.get_access<sycl::access::mode::write>(cgh);
      sycl::accessor<int, 1, sycl::access::mode::atomic,
                         sycl::access::target::local>
          ptr(globalRange, cgh);

      cgh.parallel_for<kernelT>(
          NDRange, [=](sycl::nd_item<1> item) {
            const size_t current = item.get_global_linear_id();
            const size_t other = current ^ 1U;

            // Initialize data state
            ptr[current].store(-1);
            /** Use barrier to sync initialization and to avoid dependency on
             *  thread start timing
             */
            item.barrier(sycl::access::fence_space::global_and_local);

            const int nWrites = 100;
            const int nReads = 100000;
            const int writesPerIteration =
                static_cast<int>(2U * (current + 1U));
            const int readsPerIteration =
                nReads / nWrites * writesPerIteration;

            int previousValue = ptr[other].load();
            int currentValue = 0;

            for (int i = 0; i < nWrites; i+= writesPerIteration) {
              // Run sequence of writes
              for (int j = 0; j < writesPerIteration; ++j) {
                ptr[current].store(i + j);
                writeMemFence(item);
              }
              // Run sequence of reads
              for (int k = 0; k < readsPerIteration; ++k) {
                // Memory fence on read; should be either read_write or read
                readMemFence(item);
                currentValue = ptr[other].load();

                // Verify memory order from other work-item
                if (currentValue < previousValue) {
                  pass[0] = false;
                }
                previousValue = currentValue;
              }
            }
          });
    });
  }

  return passed;
}

/** Memory fence access group, by access_mode used for read and write access
 */
enum class access_group: int {
  useDefault = 0,
  useCombined,
  useSeparate
};

/**
 * @brief Test name factory, to use for logs
 * @param accessGroup Fence access group to use
 * @param dim Nd-item dimension
 */
template <access_group accessGroup, int dim>
class test_name
{
public:
  /**
   * @brief Retrieve test name for explicit fence space usage
   * @param fenceSpace Fence space value
   */
  static std::string get(sycl::access::fence_space fenceSpace)
  {
    switch (fenceSpace) {
    case sycl::access::fence_space::global_and_local:
      return "global_and_local space " + mem_fence_name();
    case sycl::access::fence_space::local_space:
      return "local space " + mem_fence_name();
    case sycl::access::fence_space::global_space:
      return "global space " + mem_fence_name();
    default:
      return "__unknown__";
    };
  }

  /**
   * @brief Retrieve test name for default fence space usage
   */
  static std::string get()
  {
    return "default space " + mem_fence_name();
  }
private:
  static std::string mem_fence_name() {
    const auto dimensions = std::to_string(dim);
    switch (accessGroup) {
      case access_group::useDefault:
        return "default memory fence(" + dimensions + ")";
      case access_group::useCombined:
        return "read_write memory fence(" + dimensions + ")";
      case access_group::useSeparate:
        return "read and write memory fences(" + dimensions + ")";
      default:
        return "__unknown__";
    };
  }
};

/**
 * @brief Provides access to stored lambdas using the access_group mode
 *        as selector.
 * @param defaultFenceCall Lambda for mem_fence call without access specified
 * @param readWriteFenceCall Lambda for mem_fence call with read_write access
 * @param readFenceCall Lambda for mem_fence call with read access
 * @param writeFenceCall Lambda for mem_fence call with write access
 */
template <typename defaultFenceCallT,
          typename readWriteFenceCallT,
          typename readFenceCallT,
          typename writeFenceCallT>
class fence_call_factory
{
  defaultFenceCallT defaultFenceCall;
  readWriteFenceCallT readWriteFenceCall;
  readFenceCallT readFenceCall;
  writeFenceCallT writeFenceCall;
public:
  fence_call_factory(defaultFenceCallT defaultCall,
                     readWriteFenceCallT readWriteCall,
                     readFenceCallT readCall,
                     writeFenceCallT writeCall):
    defaultFenceCall(defaultCall),
    readWriteFenceCall(readWriteCall),
    readFenceCall(readCall),
    writeFenceCall(writeCall) {
    }

  using defaultAccessT =
      std::integral_constant<access_group, access_group::useDefault>;
  using combinedAccessT =
      std::integral_constant<access_group, access_group::useCombined>;
  using separateAccessT =
      std::integral_constant<access_group, access_group::useSeparate>;

  /** Retrieve read fence call for 'default' access group
   */
  const defaultFenceCallT& get_read(defaultAccessT modeSelector) const {
    static_cast<void>(modeSelector);
    return defaultFenceCall;
  }
  /** Retrieve read fence call for 'combined' access group
   */
  const readWriteFenceCallT& get_read(combinedAccessT modeSelector) const {
    static_cast<void>(modeSelector);
    return readWriteFenceCall;
  }
  /** Retrieve read fence call for 'separate' access group
   */
  const readFenceCallT& get_read(separateAccessT modeSelector) const {
    static_cast<void>(modeSelector);
    return readFenceCall;
  }
  /** Retrieve write fence call for 'default' access group
   */
  const defaultFenceCallT& get_write(defaultAccessT modeSelector) const {
    static_cast<void>(modeSelector);
    return defaultFenceCall;
  }
  /** Retrieve write fence call for 'combined' access group
   */
  const readWriteFenceCallT& get_write(combinedAccessT modeSelector) const {
    static_cast<void>(modeSelector);
    return readWriteFenceCall;
  }
  /** Retrieve write fence call for 'separate' access group
   */
  const writeFenceCallT& get_write(separateAccessT modeSelector) const {
    static_cast<void>(modeSelector);
    return writeFenceCall;
  }
};

/**
 * @brief Object generator to store memory fence lambdas for different usage
 *        modes. Hides actual lambda type.
 * @param defaultFenceCall Lambda for mem_fence call without access specified
 * @param readWriteFenceCall Lambda for mem_fence call with read_write access
 * @param readFenceCall Lambda for mem_fence call with read access
 * @param writeFenceCall Lambda for mem_fence call with write access
 */
template <typename defaultFenceCallT,
          typename readWriteFenceCallT,
          typename readFenceCallT,
          typename writeFenceCallT>
fence_call_factory<defaultFenceCallT, readWriteFenceCallT, readFenceCallT,
                   writeFenceCallT>
  make_fence_call_factory(defaultFenceCallT defaultFenceCall,
                          readWriteFenceCallT readWriteFenceCall,
                          readFenceCallT readFenceCall,
                          writeFenceCallT writeFenceCall) {
  return fence_call_factory<defaultFenceCallT,
                            readWriteFenceCallT,
                            readFenceCallT,
                            writeFenceCallT>(
    defaultFenceCall, readWriteFenceCall, readFenceCall, writeFenceCall);
}

} //namespace

#endif // __SYCLCTS_TESTS_ND_ITEM_ND_ITEM_MEM_FENCE_COMMON_H
