/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common checks for sycl::usm_allocator api
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_USM_ALLOCATOR_API_H
#define __SYCLCTS_TESTS_USM_ALLOCATOR_API_H

#include "../common/common.h"
#include "usm.h"
#include "usm_allocations_helper.h"

namespace usm_allocator_api {
using namespace sycl_cts;

using allocate_aligned = std::true_type;
using allocate_general = std::false_type;

template <typename c_id, usm_alloc_help::memb_func_index index>
struct usm_allctr_krnl_name;

/** @brief Represents the way the allocator was created
 */
enum class creation_way : int {
  by_context_device = 0,
  by_context_device_prop_list,
  by_queue,
  by_queue_prop_list,
  copy_ctor,
  move_ctor,
  copy_ctor_diff_type,
  move_assign
};

template <creation_way way>
std::string_view creation_way_hint() {
  if constexpr (way == creation_way::by_context_device) {
    return "constructed by context and device";
  } else if constexpr (way == creation_way::by_context_device_prop_list) {
    return "constructed by context, device and property_list";
  } else if constexpr (way == creation_way::by_queue) {
    return "constructed by queue";
  } else if constexpr (way == creation_way::by_queue_prop_list) {
    return "constructed by queue and property_list";
  } else if constexpr (way == creation_way::copy_ctor) {
    return "copy constructed";
  } else if constexpr (way == creation_way::move_ctor) {
    return "move constructed";
  } else if constexpr (way == creation_way::copy_ctor_diff_type) {
    return "copy constructed from allocator with different type";
  } else if constexpr (way == creation_way::move_assign) {
    return "move assigned";
  } else {
    static_assert(way != way, "Unknown creation way");
  }
}

/** @brief Defines a case and used to make unique kernel name
 *  @tparam T Testing type
 *  @tparam alKind sycl::usm::alloc
 *  @tparam crWay Creation way of usm_allocator
 */
template <typename T, sycl::usm::alloc alKind, creation_way crWay,
          typename Aligned>
struct case_id {
  using Type = T;
  static constexpr sycl::usm::alloc kind = alKind;
  static constexpr creation_way way = crWay;
  using aligned = Aligned;
};

template <typename c_id>
std::string log_message(const char *check) {
  return "case for 'allocate' (usm_allocator is " +
         std::string(creation_way_hint<c_id::way>()) + ") - " +
         std::string(check) + " for kind: " +
         std::string(usm::get_allocation_description<c_id::kind>());
}

/** @brief Perform check for allocated memory
 */
template <typename c_id, std::size_t count, std::size_t align, typename UPtr>
static void fill_and_check(util::logger &log, const UPtr &u_ptr,
                           sycl::queue &q) {
  using namespace usm_alloc_help;
  using T = typename c_id::Type;
  using Aligned = typename c_id::aligned;
  // Prepare unique kernel names
  using fill_kernel = usm_allctr_krnl_name<c_id, memb_func_index::fill>;
  using simple_kernel = usm_allctr_krnl_name<c_id, memb_func_index::simple>;
  using host_kernel = usm_allctr_krnl_name<c_id, memb_func_index::host>;

  T *ptr = u_ptr.get();

  // Check alignment
  if constexpr (Aligned::value) {
    if (!check_alignment(ptr, align)) {
      FAIL(log,
           "Wrong aligned pointer in " + log_message<c_id>("alignment check"));
    }
  }

  fill<T, count, fill_kernel>(ptr, q);

  // Regular check
  if (!check_simple<T, count, simple_kernel>(log, ptr, q)) {
    FAIL(log, log_message<c_id>("default check"));
  }

  // Memory allocated as 'shared' should be accessible to memory allocated
  // on 'host'
  if constexpr (c_id::kind == sycl::usm::alloc::shared) {
    if (allocation_supported<sycl::usm::alloc::host>(log, q)) {
      if (!check_by_index<T, memb_func_index::host, count, host_kernel>(
              log, ptr, q)) {
        FAIL(log, log_message<c_id>("check on host"));
      }
    } else {
      log.note(log_message<c_id>("(skipped check on host)"));
    }
  }
}

/** @brief Run tests for given usm::alloc
 */
template <typename T, sycl::usm::alloc kind, typename Aligned>
static void run_api_test_for_kind(util::logger &log) {
  using namespace usm_alloc_help;
  auto queue = util::get_cts_object::queue();

  if (!usm_alloc_help::allocation_supported<kind>(log, queue)) {
    log.note(" (test skipped) for kind: " +
             std::string(usm::get_allocation_description<kind>()));
    return;
  }
  auto ctx = queue.get_context();
  auto dev = queue.get_device();
  sycl::property_list pl{};
  constexpr std::size_t count = 10;
  /** Possible universal values for alignment are powers of two (8, 16, etc)
   *  Behaviour is backend-specific if incorrect alignment passed
   */
  constexpr std::size_t align = 8;

  using AllocatorT =
      typename std::conditional<Aligned::value,
                                sycl::usm_allocator<T, kind, align>,
                                sycl::usm_allocator<T, kind>>::type;
  // Custom deleter
  using DeleterT = usm_custom_deleter<T, AllocatorT, count>;
  using PtrType = std::unique_ptr<T, DeleterT>;

  /** Check (context, device) constructor
   */
  {
    using c_id = case_id<T, kind, creation_way::by_context_device, Aligned>;
    AllocatorT allctr(ctx, dev);
    PtrType ptr(allctr.allocate(count), DeleterT{allctr});
    fill_and_check<c_id, count, align>(log, ptr, queue);
  }

  // TODO: Case for (context, device, property_list) constructor will be added
  // here

  /** Check (queue) constructor
   */
  {
    using c_id = case_id<T, kind, creation_way::by_queue, Aligned>;
    AllocatorT allctr(queue);
    PtrType ptr(allctr.allocate(count), DeleterT{allctr});
    fill_and_check<c_id, count, align>(log, ptr, queue);
  }

  // TODO: Case for (queue, property_list) constructor will be added here

  /** Check copy constructor
   */
  {
    using c_id = case_id<T, kind, creation_way::copy_ctor, Aligned>;
    AllocatorT other(ctx, dev);
    AllocatorT allctr(other);
    PtrType ptr(allctr.allocate(count), DeleterT{allctr});
    fill_and_check<c_id, count, align>(log, ptr, queue);
  }

  /** Check move constructor
   */
  {
    using c_id = case_id<T, kind, creation_way::move_ctor, Aligned>;
    AllocatorT other(ctx, dev);
    AllocatorT allctr(std::move(other));
    PtrType ptr(allctr.allocate(count), DeleterT{allctr});
    fill_and_check<c_id, count, align>(log, ptr, queue);
  }

  /** Check copy constructor for allocator with different T type
   */
  {
    using c_id = case_id<T, kind, creation_way::copy_ctor_diff_type, Aligned>;
    using U = custom_type;
    using AllocDiffTType =
        typename std::conditional<Aligned::value,
                                  sycl::usm_allocator<U, kind, align>,
                                  sycl::usm_allocator<U, kind>>::type;
    AllocDiffTType other(ctx, dev);
    AllocatorT allctr(other);
    PtrType ptr(allctr.allocate(count), DeleterT{allctr});
    fill_and_check<c_id, count, align>(log, ptr, queue);
  }

  /** Check move assignment
   */
  {
    using c_id = case_id<T, kind, creation_way::move_assign, Aligned>;
    AllocatorT other(queue);
    AllocatorT allctr(ctx, dev);
    allctr = std::move(other);
    PtrType ptr(allctr.allocate(count), DeleterT{allctr});
    fill_and_check<c_id, count, align>(log, ptr, queue);
  }
}

/** @brief Main test-suite
 *  @tparam T Testing type
 */
template <typename T, typename Aligned>
class check_usm_allocator_api {
 public:
  void operator()(util::logger &log) {
    run_api_test_for_kind<T, sycl::usm::alloc::host, Aligned>(log);
    run_api_test_for_kind<T, sycl::usm::alloc::shared, Aligned>(log);
  }
};

}  // namespace usm_allocator_api

#endif  // __SYCLCTS_TESTS_USM_ALLOCATOR_API_H
