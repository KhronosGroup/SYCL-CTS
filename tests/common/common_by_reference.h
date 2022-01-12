/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
// Provides verification for common by-reference semantics
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_COMMON_BY_REFERENCE_H
#define __SYCLCTS_TESTS_COMMON_COMMON_BY_REFERENCE_H

#include "common.h"
#include "common_semantics.h"
#include "../../util/exceptions.h"

#include <string>
#include <utility>

namespace common_by_reference {
using namespace sycl_cts;

/** @brief Can be used as condition in custom mutator
 */
enum class mutation : int {
  const_correctness,
  mutate_b,
  mutate_a,
  SIZE  // This should be last
};

/** @brief Predefined functor without mutation of passed object
 */
struct no_mutation {
  template <typename T>
  void operator()(T&, mutation) const {} // No mutation expected
};

/** @brief Check that object follows common reference semantics.
 *  @attention Test for this check must be independent or last because passed
 *             object will be moved during this check. So we can't use object
 *             anymore.
 */
template <typename T, typename StorageT, typename MutatorT>
void check_on_host(sycl_cts::util::logger &log, T& a, T& other,
                   const std::string &testName, MutatorT mutator) {
  common_semantics::check_on_host(log, a, other, testName);
  // Create storage for following checks
  StorageT storage;

  auto mkInfo = [&](const std::string &check_name){
    return testName + " Common reference semantics. " + check_name;
  };

  // Check that hashes are equal for a and copy, but different for a and other
  std::hash<T> hasher;
  T copy_to_hash = a;
  if (hasher(a) != hasher(copy_to_hash)) {
    FAIL(log, testName + "(Hashes of 'a' and 'b' are not equal)");
  }
  if (hasher(a) == hasher(other)) {
    FAIL(log, "(Hashes should not be equal for 'a' and 'other')");
  }

  // Check that object is copy constructable
  storage.store(a);
  T copy_constructed_obj(a);
  if (a != copy_constructed_obj) {
    FAIL(log, testName + " is not copy constructible");
  }
  util::run_check(log, mkInfo("Copy construction."), [&]{
    if (!storage.check(copy_constructed_obj)) {
      FAIL(log, testName + " is not copy constructible (storage::check())");
    }
  });

  // Check that obect is copy assignable
  storage.store(a);
  T copy_assigned_obj = a;
  if (a != copy_assigned_obj) {
    FAIL(log, testName + " is not copy assignable");
  }
  util::run_check(log, mkInfo("Copy assignment."), [&]{
    if (!storage.check(copy_assigned_obj)) {
      FAIL(log, testName + " is not copy assignable (storage::check())");
    }
  });

  // Check const-correctness
  const auto &const_obj = a;
  storage.store(const_obj);
  mutator(a, mutation::const_correctness);
  util::run_check(log, mkInfo("Const-correcntess."), [&]{
    if (!storage.check(const_obj)) {
      FAIL(log, testName + " ignores const correctness (storage::check())");
    }
  });

  // Check that mutation applied to both object and copy of object
  T mutable_obj = a;
  mutator(mutable_obj, mutation::mutate_b);
  if (a != mutable_obj) {
    FAIL(log, testName + " does not follow common reference semantics. (a != b"
              "after mutation of 'b')");
  }

  mutator(a, mutation::mutate_a);
  if (a != mutable_obj) {
    FAIL(log, testName + " does not follow common reference semantics. (a != b"
              "after mutation of 'a')");
  }

  // Move construction check
  // Store object data before it is moved
  storage.store(a);
  T move_constructed_obj(std::move(a));
  util::run_check(log, mkInfo("Move construction"), [&]{
    if (!storage.check(move_constructed_obj)) {
      FAIL(log, testName + " is not move constructible (storage::check())");
    }
  });

  // Move assignment check
  // Store object data before it is moved
  storage.store(move_constructed_obj);
  T move_assigned_obj = std::move(move_constructed_obj);
  util::run_check(log, mkInfo("Move assignment."), [&]{
    if (!storage.check(move_assigned_obj)) {
      FAIL(log, testName + " is not move assignable (storage::check())");
    }
  });
}

}  // common_by_reference

#endif  // __SYCLCTS_TESTS_COMMON_COMMON_BY_REFERENCE_H
