/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common checks for async_work_group_copy and wait_for calls
//  for nd_item and group tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_ASYNC_WORK_GROUP_COPY_H
#define __SYCLCTS_TESTS_COMMON_ASYNC_WORK_GROUP_COPY_H

#include "../common/common.h"
#include "../common/once_per_unit.h"
#include "../common/type_coverage.h"
#include "../../util/array.h"
#include "../../util/variadic.h"

/**
 * @brief Initializes buffer given by pointer and size with series of 1 and 0
 * @retval Values used for initialization to use as the reference ones
 * @tparam bufferSize Size of buffer to initialize
 * @tparam T Deduced type of values to store
 * @tparam addressSpace Deduced address space
 * @param ptr Pointer to the start of buffer to initialize
 */
template <size_t bufferSize, typename T,
          sycl::access::address_space addressSpace>
sycl_cts::util::array<T, bufferSize> create_async_wg_copy_input(
    sycl::multi_ptr<T, addressSpace, sycl::access::decorated::yes> ptr) {
  sycl_cts::util::array<T, bufferSize> result;

  /** We use bool values because bool can be converted to the integral
   *  and to the floating point types
   */
  auto iteratorPtr = ptr;
  bool value = true;

  for (size_t i = 0; i < bufferSize; ++i, ++iteratorPtr, value = !value) {
    const T item{static_cast<T>(value)};
    /** In case of vec<> the static_cast calls an appropriate constructor, so
     *  all elements of the vector are initialized to the value given
     */
    result[i] = item;
    *iteratorPtr = item;
  }
  return result;
}

/**
 * @brief Initializes buffer given by pointer and size with value given
 * @retval Values used for initialization to use as the reference ones
 * @tparam bufferSize Size of buffer to initialize
 * @tparam value Value to use for initialization
 * @tparam T Deduced type of values to store
 * @tparam addressSpace Deduced address space
 * @param ptr Pointer to the start of buffer to initialize
 */
template <size_t bufferSize, bool value, typename T,
          sycl::access::address_space addressSpace>
sycl_cts::util::array<T, bufferSize> create_async_wg_copy_fixed(
    sycl::multi_ptr<T, addressSpace, sycl::access::decorated::yes> ptr) {
  sycl_cts::util::array<T, bufferSize> result;

  auto iteratorPtr = ptr;
  for (size_t i = 0; i < bufferSize; ++i, ++iteratorPtr) {
    const T item{static_cast<T>(value)};
    //In case of vec<> all elements of vector are initialized to the value given
    result[i] = item;
    *iteratorPtr = item;
  }
  return result;
}

/**
 * @brief Retrieve stride for async_work_group_copy call
 *        Stride is always 1 for the local_ptr
 */
template <size_t stride, typename T>
inline size_t get_async_wg_copy_stride(sycl::decorated_local_ptr<T>) {
  return 1;
}
/**
 * @brief Retrieve stride for async_work_group_copy call
 *        Stride is always the given one for the global_ptr
 */
template <size_t stride, typename T>
inline size_t get_async_wg_copy_stride(sycl::decorated_global_ptr<T>) {
  return stride;
}

/**
 * @brief Implementation for async_work_group_copy() call verification
 * @tparam bufferSize Size of local and global buffers accessible via pointers
 * @tparam stride Stride to use
 * @tparam instanceT Deduced type of object for async_work_group_copy() call
 * @tparam srcPtrT Deduced type of source pointer, local or global
 * @tparam dstPtrT Deduced type of destination pointer, local or global
 * @param instance Instance of object to call async_work_group_copy() method
 * @param srcPtr Source pointer to use for call
 * @param dstPtr Destination pointer to use for call
 */
template <size_t bufferSize, size_t stride, typename instanceT,
          typename srcPtrT, typename dstPtrT, typename ... callArgsT>
bool check_async_wg_copy_impl(instanceT&& instance,
                              srcPtrT srcPtr, dstPtrT dstPtr,
                              callArgsT ... callArgs) {
  const size_t numElements = bufferSize / stride;
  const size_t srcStride = get_async_wg_copy_stride<stride>(srcPtr);
  const size_t dstStride = get_async_wg_copy_stride<stride>(dstPtr);

  const auto referenceInput = create_async_wg_copy_input<bufferSize>(srcPtr);
  const auto referenceOutput =
      create_async_wg_copy_fixed<bufferSize, false>(dstPtr);

  // Force const-correctness
  typename std::decay<instanceT>::type const & constInstance = instance;

  // Use async_work_group_copy
  auto event = constInstance.async_work_group_copy(dstPtr, srcPtr,
                                                   numElements, callArgs...);
  constInstance.wait_for(event);

  bool succeed = true;
  auto input = srcPtr;
  auto output = dstPtr;

  // Verify source not changed
  for (size_t i = 0; i < bufferSize; ++i) {
    succeed &= check_equal_values(*input.get_raw(), referenceInput[i]);
    ++input;
  }
  // Verify elements copied to the destination at the right places
  input = srcPtr;
  for (size_t i = 0; i < numElements; ++i) {
    succeed &= check_equal_values(*output.get_raw(), *input.get_raw());
    ++output;
    for (size_t j = 1; j < dstStride; ++j) {
      succeed &= check_equal_values(*output.get_raw(),
                                    referenceOutput[i * dstStride + j]);
      ++output;
    }
    input += srcStride;
  }
  // Verify destination changed in scope only
  for (size_t i = numElements * dstStride; i < bufferSize; ++i) {
    succeed &= check_equal_values(*output.get_raw(), referenceOutput[i]);
    ++output;
  }
  return succeed;
}

/**
 * @brief Verify async_work_group_copy() call without stride parameter
 *        for the given instanceT object using pointers to local and
 *        global buffers with the given size
 * @tparam bufferSize Size of local and global buffers accessible via pointers
 * @tparam instanceT Deduced type of object for async_work_group_copy() call
 * @tparam srcPtrT Deduced type of source pointer, local or global
 * @tparam dstPtrT Deduced type of destination pointer, local or global
 * @param instance Instance of object to call async_work_group_copy() method
 * @param srcPtr Source pointer to use for call
 * @param dstPtr Destination pointer to use for call
 */
template <size_t bufferSize, typename instanceT,
          typename srcPtrT, typename dstPtrT>
bool check_async_wg_copy(instanceT&& instance,
                         srcPtrT&& srcPtr, dstPtrT&& dstPtr) {
  return check_async_wg_copy_impl<bufferSize, 1U>(instance, srcPtr, dstPtr);
}

/**
 * @brief Verify async_work_group_copy() call with stride parameter
 *        for the given instanceT object using pointers to local and
 *        global buffers with the given size
 * @tparam bufferSize Size of local and global buffers accessible via pointers
 * @tparam stride Stride to use
 * @tparam instanceT Deduced type of object for async_work_group_copy() call
 * @tparam srcPtrT Deduced type of source pointer, local or global
 * @tparam dstPtrT Deduced type of destination pointer, local or global
 * @param instance Instance of object to call async_work_group_copy() method
 * @param srcPtr Source pointer to use for call
 * @param dstPtr Destination pointer to use for call
 */
template <size_t bufferSize, size_t stride, typename instanceT,
          typename srcPtrT, typename dstPtrT>
bool check_async_wg_copy(instanceT&& instance,
                         srcPtrT&& srcPtr, dstPtrT&& dstPtr) {
  return check_async_wg_copy_impl<bufferSize, stride>(instance, srcPtr, dstPtr,
                                                      stride);
}

/**
 * @brief Functor with logic for wait_for tests
 * @tparam bufferSize Size of buffer to limit the pointers
 */
template <size_t bufferSize>
struct check_wait_for {
  using returnT = bool;

  /**
   * @brief Logic for wait_for tests. Runs some async_work_group_copy calls,
   *        stores events retrieved and waits for them.
   * @tparam instanceT Deduced type of object for wait_for() call
   * @tparam T Deduced type of underlying data for local and global pointers
   * @tparam eventsT Deduced type
   * @param instance Instance of object to call wait_for() method
   * @param srcPtr Pointer to the source buffer to use for call
   * @param dstPtr Pointer to the destination buffer to use for call
   * @param events Storage for device event instances
   */
  template <typename instanceT, typename T, typename... eventsT>
  returnT operator()(instanceT&& instance, sycl::decorated_local_ptr<T> srcPtr,
                     sycl::decorated_global_ptr<T> dstPtr,
                     eventsT... events) const {
    constexpr size_t nEvents = sizeof...(eventsT);
    constexpr size_t stride = nEvents;
    constexpr size_t numElements = bufferSize / stride;

    static_assert(numElements * stride == bufferSize,
      "Invalid number of events for the given buffer size");

    // Initialize input and output
    const auto referenceInput =
        create_async_wg_copy_fixed<bufferSize, true>(srcPtr);
    create_async_wg_copy_fixed<bufferSize, false>(dstPtr);

    // Force const-correctness
    typename std::decay<instanceT>::type const & constInstance = instance;

    // Create event objects
    auto source = srcPtr;
    auto destination = dstPtr;

    int packExpansion[nEvents] = {(
      events = constInstance.async_work_group_copy(destination, source,
                                                   numElements, stride),
      source+= numElements, // local_ptr has no stride
      ++destination,
      0 // Dummy initialization value
    )...};
    /** Every initializer clause is sequenced before any initializer clause
     *  that follows it in the braced-init-list. Every expression in comma
     *  operator is also strictly sequnced. So we can use increment safely.
     *  We still should discard dummy results, but this initialization
     *  should not be optimized out due side-effects
     */
    static_cast<void>(packExpansion);

    // Make wait_for call
    constInstance.wait_for(events...);

    // Verify all elements copied to the destination, stride doesn't matter
    bool succeed = true;
    auto output = dstPtr;

    for (size_t i = 0; i < numElements; ++i) {
      succeed &= check_equal_values(*output.get_raw(), referenceInput[i]);
      ++output;
    }
    return succeed;
  }
}; // check_wait_for

/**
 * @brief Common test logic for nd_item and group async_work_group_copy() calls
 * @tparam kernelInvokeT Kernel invocation functor
 * @tparam T Type of data to copy
 * @param queue SYCL queue to use for test
 * @param log Logger to use for test
 * @param instanceName The string naming the nd_item or group instance for logs
 * @param typeName The string naming the type of data for logs
 */
template<class kernelInvokeT, typename T>
void test_async_wg_copy(sycl::queue &queue, sycl_cts::util::logger &log,
                        const std::string& instanceName,
                        const std::string& typeName) {
  constexpr int dim = kernelInvokeT::dimensions;
  using instanceT = typename kernelInvokeT::parameterT;

  enum class check: size_t {
    local_to_global_no_stride = 0U,
    global_to_local_no_stride,
    local_to_global_with_stride,
    global_to_local_with_stride,
    nChecks //should be the latest one
  };
  std::array<bool, to_integral(check::nChecks)> result =
    {true, true, true, true};

  constexpr size_t RANGE_SIZE_TOTAL = 64;
  constexpr size_t RANGE_SIZE_1D = 2;
  constexpr size_t RANGE_SIZE_2D = 4;
  constexpr size_t BUFFER_SIZE = 10; //enough to verify stride

  const auto workItemRange =
      sycl_cts::util::get_cts_object::range<dim>::get(1, 1, 1);
  const auto workGroupRange =
      sycl_cts::util::get_cts_object::range<dim>::
          template get_fixed_size<RANGE_SIZE_TOTAL>(RANGE_SIZE_1D,
                                                    RANGE_SIZE_2D);
  {
    const size_t globalBufferSize = workGroupRange.size() * BUFFER_SIZE;
    auto buf = sycl::buffer<T, 1>(sycl::range<1>(globalBufferSize));
    auto resultBuffer =
        sycl::buffer<bool, 1>(result.data(),
                                  sycl::range<1>(result.size()));

    queue.submit([&](sycl::handler &cgh) {
    auto accResult =
        resultBuffer.template get_access<sycl::access_mode::write>(cgh);
    auto accGlobal =
        buf.template get_access<sycl::access_mode::read_write>(cgh);
    auto accLocal =
        sycl::local_accessor<T, 1>(sycl::range<1>(BUFFER_SIZE), cgh);

    kernelInvokeT{}(
        cgh, workGroupRange, workItemRange,
        [=](instanceT& instance, const size_t index) {
          // Each work-group uses its own part of global buffer,
          // single work-item per work-group
          using difference_type =
              typename sycl::decorated_global_ptr<T>::difference_type;
          const auto globalBufferOffset =
              static_cast<difference_type>(index * BUFFER_SIZE);

          auto ptrGlobal =
              accGlobal.template get_multi_ptr<sycl::access::decorated::yes>() +
              globalBufferOffset;
          auto ptrLocal =
              accLocal.template get_multi_ptr<sycl::access::decorated::yes>();

          if (!check_async_wg_copy<BUFFER_SIZE>(instance, ptrLocal, ptrGlobal)){
            const size_t resultIndex =
                to_integral(check::local_to_global_no_stride);
            accResult[resultIndex] = false;
          }
          if (!check_async_wg_copy<BUFFER_SIZE>(instance, ptrGlobal, ptrLocal)){
            const size_t resultIndex =
                to_integral(check::global_to_local_no_stride);
            accResult[resultIndex] = false;
          }
          constexpr size_t stride = 2;
          if (!check_async_wg_copy<BUFFER_SIZE, stride>(instance,
                                                        ptrLocal, ptrGlobal)){
            const size_t resultIndex =
                to_integral(check::local_to_global_with_stride);
            accResult[resultIndex] = false;
          }
          if (!check_async_wg_copy<BUFFER_SIZE, stride>(instance,
                                                        ptrGlobal, ptrLocal)){
            const size_t resultIndex =
                to_integral(check::global_to_local_with_stride);
            accResult[resultIndex] = false;
          }
        });
    });
  }
  if (!result[to_integral(check::local_to_global_no_stride)]) {
    FAIL(log, instanceName + "<" + std::to_string(dim) +
        ">: local_ptr to global_ptr without stride failed for " +
        type_name_string<T>::get(typeName));
  }
  if (!result[to_integral(check::global_to_local_no_stride)]) {
    FAIL(log, instanceName + "<" + std::to_string(dim) +
        ">: global_ptr to local_ptr without stride failed for " +
        type_name_string<T>::get(typeName));
  }
  if (!result[to_integral(check::local_to_global_with_stride)]) {
    FAIL(log, instanceName + "<" + std::to_string(dim) +
        ">: local_ptr to global_ptr with stride failed for " +
        type_name_string<T>::get(typeName));
  }
  if (!result[to_integral(check::global_to_local_with_stride)]) {
    FAIL(log, instanceName + "<" + std::to_string(dim) +
        ">: global_ptr to local_ptr with stride failed for " +
        type_name_string<T>::get(typeName));
  }
}

/**
 * @brief Common test logic for nd_item and group wait_for() calls
 * @tparam kernelInvokeT Kernel invocation functor
 * @tparam T Type of data to use for async_work_group_copy call
 * @param queue SYCL queue to use for test
 * @param log Logger to use for test
 * @param instanceName The string naming the nd_item or group instance for logs
 * @param typeName The string naming the type of data for logs
 */
template<class kernelInvokeT, typename T>
void test_wait_for(sycl::queue &queue, sycl_cts::util::logger &log,
                   const std::string& instanceName) {
  constexpr int dim = kernelInvokeT::dimensions;
  using instanceT = typename kernelInvokeT::parameterT;

  bool result = true;

  constexpr size_t RANGE_SIZE_TOTAL = 64;
  constexpr size_t RANGE_SIZE_1D = 2;
  constexpr size_t RANGE_SIZE_2D = 4;
  constexpr size_t BUFFER_SIZE = 128;
  constexpr size_t N_EVENTS_MAX = BUFFER_SIZE / 2;

  const auto workItemRange =
      sycl_cts::util::get_cts_object::range<dim>::get(1, 1, 1);
  const auto workGroupRange =
      sycl_cts::util::get_cts_object::range<dim>::
          template get_fixed_size<RANGE_SIZE_TOTAL>(RANGE_SIZE_1D,
                                                    RANGE_SIZE_2D);
  {
    const size_t globalBufferSize = workGroupRange.size() * BUFFER_SIZE;
    auto buf = sycl::buffer<T, 1>(sycl::range<1>(globalBufferSize));
    auto resultBuffer =
        sycl::buffer<bool, 1>(&result, sycl::range<1>(1));

    queue.submit([&](sycl::handler &cgh) {
    auto accResult =
        resultBuffer.template get_access<sycl::access_mode::write>(cgh);
    auto accGlobal =
        buf.template get_access<sycl::access_mode::read_write>(cgh);
    auto accLocal =
        sycl::local_accessor<T, 1>(sycl::range<1>(BUFFER_SIZE), cgh);

    auto events =
        sycl::accessor<sycl::device_event, 1,
                           sycl::access_mode::read_write,
                           sycl::target::local>(
            sycl::range<1>(N_EVENTS_MAX), cgh);

    kernelInvokeT{}(
        cgh, workGroupRange, workItemRange,
        [=](instanceT& instance, const size_t index) {

          // Each work-group uses its own part of global buffer,
          // single work-item per work-group
          using difference_type =
              typename sycl::decorated_global_ptr<T>::difference_type;
          const auto globalBufferOffset =
              static_cast<difference_type>(index * BUFFER_SIZE);

          auto ptrGlobal =
              accGlobal.template get_multi_ptr<sycl::access::decorated::yes>() +
              globalBufferOffset;
          auto ptrLocal =
              accLocal.template get_multi_ptr<sycl::access::decorated::yes>();

          if (!run_variadic<check_wait_for<BUFFER_SIZE>>::with<1>(events,
                  instance, ptrLocal, ptrGlobal)){
            accResult[0] = false;
          }
          if (!run_variadic<check_wait_for<BUFFER_SIZE>>::with<2>(events,
                  instance, ptrLocal, ptrGlobal)){
            accResult[0] = false;
          }
          if (!run_variadic<check_wait_for<BUFFER_SIZE>>::with<N_EVENTS_MAX>(
                  events, instance, ptrLocal, ptrGlobal)){
            accResult[0] = false;
          }
        });
    });
  }
  if (!result) {
    FAIL(log, instanceName + "<" + std::to_string(dim) + ">: wait_for failed");
  }
}

/**
 * @brief Common logic to check all dimensions for nd_item and group
 *        with queue constructed already
 * @tparam action Functor template for action to run
 * @tparam actionArgsT Parameter pack to instantiate functor template with
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param queue SYCL queue instance to use
 * @param log Logger instance to use
 * @param args Arguments to forward into the call
 */
template<template<int, typename...> class action,
         typename ... actionArgsT, typename ... argsT>
void check_all_dims(sycl::queue &queue, sycl_cts::util::logger &log,
                    argsT&& ... args) {
    action<1, actionArgsT...>{}(queue, log, std::forward<argsT>(args)...);
    action<2, actionArgsT...>{}(queue, log, std::forward<argsT>(args)...);
    action<3, actionArgsT...>{}(queue, log, std::forward<argsT>(args)...);
    queue.wait_and_throw();
}

/**
 * @brief Common logic to check all dimensions for nd_item and group
 *        with queue constructed within
 * @tparam action Functor template for action to run
 * @tparam actionArgsT Parameter pack to instantiate functor template with
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param log Logger instance to use
 * @param args Arguments to forward into the call
 */
template<template<int, typename...> class action,
         typename ... actionArgsT, typename ... argsT>
void check_all_dims(sycl_cts::util::logger &log, argsT&& ... args) {
  {
    auto queue = once_per_unit::get_queue();

    check_all_dims<action, actionArgsT...>(queue, log,
                                           std::forward<argsT>(args)...);
  }
}

#endif  // __SYCLCTS_TESTS_COMMON_ASYNC_WORK_GROUP_COPY_H
