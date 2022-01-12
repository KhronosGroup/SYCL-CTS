/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/common_semantics.h"

#include <array>

#define TEST_NAME nd_range_equality

namespace {

template <int numDims>
struct nd_range_equality_kernel;

/**
 * @brief Provides a safe index for checking an operation
 */
enum class current_check {
  equal_self,
  not_equal_self,
  equal_other_same_global,
  not_equal_other_same_global,
  equal_other_same_local,
  not_equal_other_same_local,
  equal_other_different,
  not_equal_other_different,
  SIZE  // This should be last
};

struct success_array {
 public:
  success_array() {
    std::fill(std::begin(m_rawArray), std::end(m_rawArray), true);
  }
  bool& operator[](current_check index) {
    return m_rawArray[static_cast<size_t>(index)];
  }

 private:
  std::array<bool, static_cast<size_t>(current_check::SIZE)> m_rawArray;
};

}  // namespace

namespace TEST_NAME {
using namespace sycl_cts;

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
    {
      // Prepare ranges
      const auto range2 = util::get_cts_object::range<numDims>::get(2, 1, 1);
      const auto range4 = util::get_cts_object::range<numDims>::get(4, 1, 1);
      const auto range8 = util::get_cts_object::range<numDims>::get(8, 1, 1);

      // Prepare an array of nd_range
      // The nd_range objects are designed so that:
      //    0 and 1 have the same global range
      //    1 and 2 have the same local range
      //    0 and 2 are completely different
      using nd_range_t = sycl::nd_range<numDims>;
      std::array<nd_range_t, 3> objects = {nd_range_t(range8, range4),
                                           nd_range_t(range8, range2),
                                           nd_range_t(range4, range2)};

      // Store comparison results into a success array
      success_array success;

      // Perform comparisons on the stored nd_range objects
      const auto& object0 = objects[0];
      const auto& object1 = objects[1];
      const auto& object2 = objects[2];
      {
        auto& currentSuccess = success[current_check::equal_self];
        currentSuccess = (object0 == object0);
      }
      {
        auto& currentSuccess = success[current_check::not_equal_self];
        currentSuccess = (object0 != object0);
      }
      {
        auto& currentSuccess = success[current_check::equal_other_same_global];
        currentSuccess = (object0 == object1);
      }
      {
        auto& currentSuccess =
            success[current_check::not_equal_other_same_global];
        currentSuccess = (object0 != object1);
      }
      {
        auto& currentSuccess = success[current_check::equal_other_same_local];
        currentSuccess = (object1 == object2);
      }
      {
        auto& currentSuccess =
            success[current_check::not_equal_other_same_local];
        currentSuccess = (object1 != object2);
      }
      {
        auto& currentSuccess = success[current_check::equal_other_different];
        currentSuccess = (object0 == object2);
      }
      {
        auto& currentSuccess =
            success[current_check::not_equal_other_different];
        currentSuccess = (object0 != object2);
      }

      // Check nd_range equality operator
      common_semantics::check_on_host(log, objects[0],
                                      "nd_range " + std::to_string(numDims));
      CHECK_VALUE(log, success[current_check::equal_self], true, numDims);
      CHECK_VALUE(log, success[current_check::not_equal_self], false, numDims);
      CHECK_VALUE(log, success[current_check::equal_other_same_global], false,
                  numDims);
      CHECK_VALUE(log, success[current_check::not_equal_other_same_global],
                  true, numDims);
      CHECK_VALUE(log, success[current_check::equal_other_same_local], false,
                  numDims);
      CHECK_VALUE(log, success[current_check::not_equal_other_same_local], true,
                  numDims);
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
