/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide verification that accessor's iterator is conforming to named
//  requirement LegacyRandomAccessIterator
//
*******************************************************************************/
#include "../../util/named_requirement_verification/legacy_random_access_iterator.h"
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::accessor is implemented
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)
#include "accessor_common.h"
#endif

namespace accessor_iterator_requirement {

using string_view = std::basic_string_view<char>;

/**
 * @brief Function helps to fail catch2 test and print errors from array through
 * a INFO invocations
 *
 * @tparam N Size of array with error messages
 * @param errors Array with error messages
 */
template <size_t N>
inline void print_errors(const std::array<string_view, N>& errors) {
  const bool false_to_fail_test = false;
  for (size_t i = 0; i < N; ++i) {
    if (!errors[i].empty()) {
      INFO(errors[i]);
      CHECK(false_to_fail_test);
    }
  }
}

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("LegacyRandomAccessIterator requirement verification for sycl::accessor "
 "iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto q = util::get_cts_object::queue();

  constexpr size_t size_of_res_array =
      legacy_random_access_iterator_requirement::count_of_possible_errors;
  std::basic_string_view<char> errors[size_of_res_array];

  constexpr size_t size_of_dummy = 5;
  int dummy[size_of_dummy] = {1, 2, 3, 4, 5};
  {
    sycl::buffer<std::basic_string_view<char>, 1> res_buf(
        errors, sycl::range(size_of_res_array));
    sycl::buffer<int, 1> dummy_buf(dummy, sycl::range(size_of_dummy));

    q.submit([&](sycl::handler& cgh) {
      auto res_acc = res_buf.get_access<sycl::access_mode::write>(cgh);
      auto dummy_acc = res_buf.get_access<sycl::access_mode::read_write>(cgh);
      cgh.single_task([=] {
        auto dummy_acc_it = dummy_acc.begin();
        auto verification_result =
            legacy_random_access_iterator_requirement{}.is_satisfied_for(
                dummy_acc_it, size_of_dummy);
        if (!verification_result.first) {
          for (int i = 0; i < size_of_res_array; ++i) {
            if (!verification_result.second[i].empty()) {
              // Copy errors to the host side
              res_acc[i] = verification_result.second[i];
            }
          }
        }
      });
    });
    q.wait_and_throw();
  }
  print_errors(errors);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("LegacyRandomAccessIterator requirement verification for sycl::local_accessor "
 "iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto q = util::get_cts_object::queue();

  constexpr size_t size_of_res_array =
      legacy_random_access_iterator_requirement::count_of_possible_errors;
  std::basic_string_view<char> errors[size_of_res_array];

  constexpr size_t alloc_size = 5;
  {
    sycl::buffer<std::basic_string_view<char>, 1> res_buf(
        errors, sycl::range(size_of_res_array));

    q.submit([&](sycl::handler& cgh) {
      auto res_acc = res_buf.get_access<sycl::access_mode::write>(cgh);
      sycl::local_accessor<int, 1> dummy_acc(sycl::range(alloc_size), cgh);

      cgh.single_task([=] {
        for (int i = 0; i < alloc_size; ++i) {
          dummy_acc[i] = i + 1;
        }

        auto dummy_acc_it = dummy_acc.begin();
        auto verification_result =
            legacy_random_access_iterator_requirement{}.is_satisfied_for(
                dummy_acc_it, size_of_dummy);
        if (!verification_result.first) {
          for (int i = 0; i < size_of_res_array; ++i) {
            if (!verification_result.second[i].empty()) {
              // Copy errors to the host side
              res_acc[i] = verification_result.second[i];
            }
          }
        }
      });
    });
    q.wait_and_throw();
  }
  print_errors(errors);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("LegacyRandomAccessIterator requirement verification for sycl::host_accessor "
 "iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto q = util::get_cts_object::queue();

  constexpr size_t size_of_dummy = 5;
  int dummy[size_of_dummy] = {1, 2, 3, 4, 5};
  {
    sycl::buffer<int, 1> dummy_buf(dummy, sycl::range(size_of_dummy));

    sycl::host_accessor<sycl::access_mode::read_write> dummy_acc(dummy_buf);
    auto dummy_acc_it = dummy_acc.begin();
    auto verification_result =
        legacy_random_access_iterator_requirement{}.is_satisfied_for(
            dummy_acc_it, size_of_dummy);
    if (!verification_result.first) {
      print_errors(verification_result.second);
    }
  }
});

}  // namespace accessor_iterator_requirement
