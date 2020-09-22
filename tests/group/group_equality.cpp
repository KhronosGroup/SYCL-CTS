/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/common_by_value.h"
#include "../group/group_common.h"

#include <array>

#define TEST_NAME group_equality

namespace TEST_NAMESPACE {
using namespace sycl_cts;

template <int numDims>
struct group_setup_kernel;

template <int numDims>
struct group_equality_kernel;

/** test cl::sycl::device initialization
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
    try {
      using item_t = cl::sycl::group<numDims>;

      // group is not default constructible, store two objects into the array
      static constexpr size_t numItems = 2;
      using item_array_t = std::array<item_t, numItems>;
      char rawItems[sizeof(item_array_t)];
      auto& items = *reinterpret_cast<item_array_t*>(rawItems);

      store_group_instances<group_setup_kernel<numDims>>(items);

      // Check nd_item equality operator on the device side
      equality_comparable_on_device<item_t>::
          template check_on_kernel<group_equality_kernel<numDims>>(
              log, items, "group " + std::to_string(numDims) + " (device)");

      // Check group equality operator on the host side
      check_equality_comparable_generic(log, items[0],
                                        "group " + std::to_string(numDims) +
                                        " (host)");
    } catch (const cl::sycl::exception& e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
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

}  // namespace TEST_NAME
