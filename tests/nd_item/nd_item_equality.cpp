/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2018-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2022-2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/invoke.h"
#include "../common/semantics_by_value.h"

#define TEST_NAME nd_item_equality

namespace TEST_NAMESPACE {
using namespace sycl_cts;

template <int numDims>
struct nd_item_equality_kernel;

/** test sycl::device initialization
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const final {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <int numDims>
  void test_equality(util::logger& log) {
    using item_t = sycl::nd_item<numDims>;
    using kernel_t = nd_item_equality_kernel<numDims>;

    // Store comparison results from kernel into a success array
    std::array<bool,
               to_integral(common_by_value_semantics::current_check::size)>
        success;
    std::fill(std::begin(success), std::end(success), true);

    {
      sycl::buffer<bool> successBuf(success.data(),
                                    sycl::range<1>(success.size()));

      const auto oneElemRange =
          util::get_cts_object::range<numDims>::get(1, 1, 1);

      auto queue = util::get_cts_object::queue();
      queue
          .submit([&](sycl::handler& cgh) {
            auto successAcc =
                successBuf.get_access<sycl::access_mode::write>(cgh);

            cgh.parallel_for<kernel_t>(
                sycl::nd_range<numDims>(oneElemRange, oneElemRange),
                [=](item_t item) {
                  common_by_value_semantics::check_equality(item, successAcc);
                });
          })
          .wait_and_throw();
    }

    for (int i = 0; i < success.size(); ++i) {
      INFO(std::string(TOSTRING(TEST_NAME)) + " is " +
           common_by_value_semantics::get_error_string(i));
      CHECK(success[i]);
    }
  }

  /** execute the test
   */
  void run(util::logger& log) final {
    test_equality<1>(log);
    test_equality<2>(log);
    test_equality<3>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
