/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common checks for sycl::usm_allocator constructors
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_USM_ALLOCATOR_CONSTRUCTORS_H
#define __SYCLCTS_TESTS_USM_ALLOCATOR_CONSTRUCTORS_H

#include "../common/common.h"
#include "usm.h"
#include "usm_allocations_helper.h"
#include <memory>

namespace usm_allocator_constructors {
using namespace sycl_cts;

/** @brief Run tests for given usm::alloc
 */
template <typename T, sycl::usm::alloc kind>
static void run_ctors_test_for_kind(util::logger &log) {
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

  using AllocatorT = sycl::usm_allocator<T, kind, align>;
  // Custom deleter
  using DeleterT = usm_custom_deleter<T, AllocatorT, count>;
  using PtrType = std::unique_ptr<T, DeleterT>;

  /** Check (context, device) constructor
   */
  {
    AllocatorT allctr(ctx, dev);
    PtrType ptr(allctr.allocate(count), DeleterT{allctr});
    if (!check_ptr_kind(ptr, ctx, kind)) {
      FAIL(log, "Constructor (context, device) for kind: " +
                std::string(usm::get_allocation_description<kind>()));
    }
  }

  // TODO: Case for (context, device, property_list) ctor will be added here

  /** Check (queue) constructor
   */
  {
    AllocatorT allctr(queue);
    PtrType ptr(allctr.allocate(count), DeleterT{allctr});
    if (!check_ptr_kind(ptr, ctx, kind)) {
      FAIL(log, "Constructor (queue) for kind: " +
                std::string(usm::get_allocation_description<kind>()));
    }
  }

  // TODO: Case for (queue, property_list) ctor will be added here

  /** Check copy constructor
   */
  {
    AllocatorT other(ctx, dev);
    PtrType ptr_other(other.allocate(count), DeleterT{other});

    AllocatorT allctr(other);
    PtrType ptr(allctr.allocate(count), DeleterT{allctr});

    if (!compare_ptrs_kind(ptr, ptr_other, ctx)) {
      FAIL(log, "Copy constructor for kind: " +
                std::string(usm::get_allocation_description<kind>()));
    }
  }

  /** Check move constructor
   */
  {
    AllocatorT other(ctx, dev);
    T* other_ptr = other.allocate(count);

    AllocatorT allctr(std::move(other));
    // Both pointers should be deleted by 'allctr' because 'other' was moved
    PtrType ptr_other(other_ptr, DeleterT{allctr});
    PtrType ptr(allctr.allocate(count), DeleterT{allctr});

    if (!compare_ptrs_kind(ptr, ptr_other, ctx)) {
      FAIL(log, "Move constructor for kind: " +
                std::string(usm::get_allocation_description<kind>()));
    }
  }

  /** Check copy constructor for allocator with different T type
   */
  {
    using U = custom_type;
    using AllocDiffTType = sycl::usm_allocator<U, kind, align>;
    using DeleterDiffTType = usm_custom_deleter<U, AllocDiffTType, count>;
    using OtherPtrType = std::unique_ptr<U, DeleterDiffTType>;

    AllocDiffTType other(ctx, dev);
    OtherPtrType ptr_other(other.allocate(count), DeleterDiffTType{other});

    AllocatorT allctr(other);
    PtrType ptr(allctr.allocate(count), DeleterT{allctr});

    if (!compare_ptrs_kind(ptr, ptr_other, ctx)) {
      FAIL(log, "Copy constructor (usm_allocator<U,...>) for kind: " +
                std::string(usm::get_allocation_description<kind>()));
    }
  }

  /** Check move assignment
   */
  {
    AllocatorT other(queue);
    T* other_ptr = other.allocate(count);

    AllocatorT allctr(ctx, dev);
    allctr = std::move(other);
    // Both pointers should be deleted by 'allctr' because 'other' was moved
    PtrType ptr_other(other_ptr, DeleterT{allctr});
    PtrType ptr(allctr.allocate(count), DeleterT{allctr});

    if (!compare_ptrs_kind(ptr, ptr_other, ctx)) {
      FAIL(log, "Move assignment for kind: " +
                std::string(usm::get_allocation_description<kind>()));
    }
  }

  /** Check equality operator
   */
  {
    using AllocatorDiffT = sycl::usm_allocator<T, other_kind<kind>(), align>;
    AllocatorT allctrA(ctx, dev);
    AllocatorT allctrB(allctrA);
    AllocatorT allctrC(queue);
    allctrC = allctrA;
    AllocatorDiffT allctrD(ctx, dev);
    std::string kind_hint{usm::get_allocation_description<kind>()};

    if (!(allctrA == allctrB)) {
      FAIL(log,
           "usm_allocator equality operator does not work correctly"
           "(copy constructed) for kind: " +
               kind_hint);
    }
    if (!(allctrA == allctrC)) {
      FAIL(log,
           "usm_allocator equality operator does not work correctly"
           "(copy assigned) for kind: " +
               kind_hint);
    }
    if (allctrA != allctrB) {
      FAIL(log,
           "usm_allocator non-equality operator does not work correctly"
           "(copy constructed) for kind: " +
               kind_hint);
    }
    if (allctrA != allctrC) {
      FAIL(log,
           "usm_allocator non-equality operator does not work correctly"
           "(copy constructed) for kind: " +
               kind_hint);
    }
    if (allctrC == allctrD) {
      FAIL(log,
           "usm_allocator equality operator does not work correctly"
           "(comparing with different) for kind: " +
               kind_hint);
    }
    if (!(allctrC != allctrD)) {
      FAIL(log,
           "usm_allocator non-equality operator does not work correctly"
           "(comparing with different) for kind: " +
               kind_hint);
    }
  }
}

/** @brief Main test-suite
 *  @tparam T Testing type
 */
template <typename T>
class check_usm_allocator_constructors {
 public:
  void operator()(util::logger &log) {
    run_ctors_test_for_kind<T, sycl::usm::alloc::host>(log);
    run_ctors_test_for_kind<T, sycl::usm::alloc::shared>(log);
  }
};

}  // namespace usm_allocator_constructors

#endif  // __SYCLCTS_TESTS_USM_ALLOCATOR_CONSTRUCTORS_H
