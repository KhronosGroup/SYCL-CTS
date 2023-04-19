/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

//  Provide verification that accessor's iterators are conforming to named
//  requirement LegacyRandomAccessIterator

#include <tuple>

#include "../../util/named_requirement_verification/legacy_random_access_iterator.h"
#include "../common/disabled_for_test_case.h"
#include "../common/get_cts_object.h"
#include "catch2/catch_test_macros.hpp"

namespace accessor_iterator_requirement {

/**
 * @brief Function helps to fail catch2 test and print errors from array through
 * a FAIL_CHECK invocations
 *
 * @tparam N Size of array with error messages
 * @param errors Array with error messages
 */
template <size_t N>
inline void print_errors(const std::array<int, N>& errors) {
  named_requirement_verification::error_messages error_messages;
  const bool false_to_fail_test = false;
  for (size_t i = 0; i < N; ++i) {
    if (errors[i] != 0) {
      FAIL_CHECK(error_messages.error_message(errors[i]));
    }
  }
}

template <typename AccT, typename IteratorT>
class kernel_fill_errors;

template <typename ConstructAcc, typename GetIterator>
inline auto fill_errors(std::tuple<ConstructAcc, GetIterator> ftuple) {
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  using AccT = sycl::accessor<int, 1, sycl::access::mode::read_write>;
  using LocAccT = sycl::local_accessor<int, 1>;

  auto q = once_per_unit::get_queue();

  constexpr size_t size_of_res_array =
      legacy_random_access_iterator_requirement::count_of_possible_errors;
  std::array<int, size_of_res_array> errors;
  errors.fill(0);
  constexpr size_t size_of_dummy = 1;
  int dummy[size_of_dummy] = {1};
  {
    sycl::buffer<int, 1> res_buf(errors.data(), sycl::range(size_of_res_array));
    sycl::buffer<int, 1> dummy_buf(dummy, sycl::range(size_of_dummy));

    auto action = [](auto& dummy_acc, auto& res_acc, auto& ftuple) {
      auto dummy_acc_it = std::get<1>(ftuple)(dummy_acc);
      auto verification_result =
          legacy_random_access_iterator_requirement{}.is_satisfied_for(
              dummy_acc_it);
      if (!verification_result.first) {
        for (int i = 0; i < size_of_res_array; ++i) {
          if (verification_result.second[i] != 0) {
            // Copy errors to the host side
            res_acc[i] = verification_result.second[i];
          }
        }
      }
    };

    q.submit([&](sycl::handler& cgh) {
      auto res_acc = res_buf.get_access<sycl::access_mode::write>(cgh);
      auto dummy_acc = std::get<0>(ftuple)(cgh, dummy_buf);
      // obtain a unique kernel name using the accessor and iterator type
      using kname = kernel_fill_errors<
          AccT, std::invoke_result_t<GetIterator, decltype(dummy_acc)&>>;
      if constexpr (std::is_same_v<decltype(dummy_acc), AccT>) {
        cgh.single_task<kname>([=] { action(dummy_acc, res_acc, ftuple); });
      } else if constexpr (std::is_same_v<decltype(dummy_acc), LocAccT>) {
        sycl::range<1> r(1);
        cgh.parallel_for<kname>(
            sycl::nd_range(r, r),
            [=](sycl::nd_item<1> item) { action(dummy_acc, res_acc, ftuple); });
      } else {
        auto dummy_acc_it = std::get<1>(ftuple)(dummy_acc);
        auto verification_result =
            legacy_random_access_iterator_requirement{}.is_satisfied_for(
                dummy_acc_it);
        if (!verification_result.first) {
          errors = verification_result.second;
        }
      }
    });
    q.wait_and_throw();
  }
  return errors;
}

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("LegacyRandomAccessIterator requirement verification for sycl::accessor "
 "iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto t = std::make_tuple(
      [](sycl::handler& cgh, auto& buf) {
        return buf.template get_access<sycl::access_mode::read_write>(cgh);
      },
      [](auto& dummy_acc) { return dummy_acc.begin(); });
  auto errors = fill_errors(t);
  print_errors(errors);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("LegacyRandomAccessIterator requirement verification for sycl::accessor "
 "const iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto t = std::make_tuple(
      [](sycl::handler& cgh, auto& buf) {
        return buf.template get_access<sycl::access_mode::read_write>(cgh);
      },
      [](auto& dummy_acc) { return dummy_acc.cbegin(); });
  auto errors = fill_errors(t);
  print_errors(errors);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("LegacyRandomAccessIterator requirement verification for sycl::accessor "
 "reverse iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto t = std::make_tuple(
      [](sycl::handler& cgh, auto& buf) {
        return buf.template get_access<sycl::access_mode::read_write>(cgh);
      },
      [](auto& dummy_acc) { return dummy_acc.rbegin(); });
  auto errors = fill_errors(t);
  print_errors(errors);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("LegacyRandomAccessIterator requirement verification for sycl::accessor "
 "const reverse iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto t = std::make_tuple(
      [](sycl::handler& cgh, auto& buf) {
        return buf.template get_access<sycl::access_mode::read_write>(cgh);
      },
      [](auto& dummy_acc) { return dummy_acc.crbegin(); });
  auto errors = fill_errors(t);
  print_errors(errors);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("LegacyRandomAccessIterator requirement verification for sycl::local_accessor "
 "iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto t = std::make_tuple(
      [](sycl::handler& cgh, [[maybe_unused]] auto& buf) {
        sycl::local_accessor<int, 1> dummy_acc(sycl::range(1), cgh);
        return dummy_acc;
      },
      [](auto& dummy_acc) { return dummy_acc.begin(); });
  auto errors = fill_errors(t);
  print_errors(errors);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("LegacyRandomAccessIterator requirement verification for sycl::local_accessor "
 "const iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto t = std::make_tuple(
      [](sycl::handler& cgh, [[maybe_unused]] auto& buf) {
        sycl::local_accessor<int, 1> dummy_acc(sycl::range(1), cgh);
        return dummy_acc;
      },
      [](auto& dummy_acc) { return dummy_acc.cbegin(); });
  auto errors = fill_errors(t);
  print_errors(errors);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("LegacyRandomAccessIterator requirement verification for sycl::local_accessor "
 "reverse iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto t = std::make_tuple(
      [](sycl::handler& cgh, [[maybe_unused]] auto& buf) {
        sycl::local_accessor<int, 1> dummy_acc(sycl::range(1), cgh);
        return dummy_acc;
      },
      [](auto& dummy_acc) { return dummy_acc.rbegin(); });
  auto errors = fill_errors(t);
  print_errors(errors);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("LegacyRandomAccessIterator requirement verification for sycl::local_accessor "
 "const reverse iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto t = std::make_tuple(
      [](sycl::handler& cgh, [[maybe_unused]] auto& buf) {
        sycl::local_accessor<int, 1> dummy_acc(sycl::range(1), cgh);
        return dummy_acc;
      },
      [](auto& dummy_acc) { return dummy_acc.crbegin(); });
  auto errors = fill_errors(t);
  print_errors(errors);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("LegacyRandomAccessIterator requirement verification for sycl::host_accessor "
 "iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto t = std::make_tuple(
      []([[maybe_unused]] sycl::handler& cgh, auto& buf) {
        sycl::host_accessor<int, 1> dummy_acc(buf);
        return dummy_acc;
      },
      [](auto& dummy_acc) { return dummy_acc.begin(); });
  auto errors = fill_errors(t);
  print_errors(errors);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("LegacyRandomAccessIterator requirement verification for sycl::host_accessor "
 "const iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto t = std::make_tuple(
      []([[maybe_unused]] sycl::handler& cgh, auto& buf) {
        sycl::host_accessor<int, 1> dummy_acc(buf);
        return dummy_acc;
      },
      [](auto& dummy_acc) { return dummy_acc.cbegin(); });
  auto errors = fill_errors(t);
  print_errors(errors);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("LegacyRandomAccessIterator requirement verification for sycl::host_accessor "
 "reverse iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto t = std::make_tuple(
      []([[maybe_unused]] sycl::handler& cgh, auto& buf) {
        sycl::host_accessor<int, 1> dummy_acc(buf);
        return dummy_acc;
      },
      [](auto& dummy_acc) { return dummy_acc.rbegin(); });
  auto errors = fill_errors(t);
  print_errors(errors);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("LegacyRandomAccessIterator requirement verification for sycl::host_accessor "
 "const reverse iterator",
 "[accessor]")({
  using namespace sycl_cts;
  using namespace named_requirement_verification;

  auto t = std::make_tuple(
      []([[maybe_unused]] sycl::handler& cgh, auto& buf) {
        sycl::host_accessor<int, 1> dummy_acc(buf);
        return dummy_acc;
      },
      [](auto& dummy_acc) { return dummy_acc.crbegin(); });
  auto errors = fill_errors(t);
  print_errors(errors);
});

}  // namespace accessor_iterator_requirement
