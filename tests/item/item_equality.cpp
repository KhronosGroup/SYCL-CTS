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

#define TEST_NAME item_equality

namespace {

template <int numDims>
struct item_setup_kernel;
template <int numDims>
struct item_equality_kernel;

/**
 * @brief Provides a safe index for checking an operation
 */
enum class current_check {
  equal_self,
  not_equal_self,
  equal_other,
  not_equal_other,
  SIZE  // This should be last
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
      using item_t = sycl::item<numDims>;

      // item is not default constructible, store two objects
      static constexpr size_t numItems = 2;
      using item_array_t = std::array<item_t, numItems>;
      char rawItems[sizeof(item_array_t)];
      auto& items = *reinterpret_cast<item_array_t*>(rawItems);

      // Store comparison results from kernel into a success array
      using success_array_t =
          std::array<bool, static_cast<size_t>(current_check::SIZE)>;
      success_array_t success;
      std::fill(std::begin(success), std::end(success), true);

      // First store two instances of item into the array
      {
        auto testQueue = util::get_cts_object::queue();

        const auto oneElemRange =
            util::get_cts_object::range<numDims>::get(1, 1, 1);
        const auto itemRange =
            util::get_cts_object::range<numDims>::get(numItems, 1, 1);

        // Retrieve two item objects and store them
        sycl::buffer<item_t> itemBuf(items.data(),
                                     sycl::range<1>(items.size()));
        testQueue.submit([&](sycl::handler& cgh) {
          auto itemAcc =
              itemBuf.template get_access<sycl::access_mode::write>(cgh);

          cgh.parallel_for<item_setup_kernel<numDims>>(
              itemRange,
              [=](item_t item) { itemAcc[item.get_linear_id()] = item; });
        });

        // Perform comparisons on the stored item objects
        sycl::buffer<bool> successBuf(success.data(),
                                      sycl::range<1>(success.size()));
        testQueue.submit([&](sycl::handler& cgh) {
          auto itemAcc =
              itemBuf.template get_access<sycl::access_mode::read>(cgh);
          auto successAcc =
              successBuf.get_access<sycl::access_mode::write>(cgh);

          cgh.single_task<item_equality_kernel<numDims>>([=]() {
            const auto& item0 = itemAcc[0];
            const auto& item1 = itemAcc[1];

            {
              auto& currentSuccess =
                  successAcc[static_cast<size_t>(current_check::equal_self)];
              currentSuccess = (item0 == item0);
            }
            {
              auto& currentSuccess = successAcc[static_cast<size_t>(
                  current_check::not_equal_self)];
              currentSuccess = (item0 != item0);
            }
            {
              auto& currentSuccess =
                  successAcc[static_cast<size_t>(current_check::equal_other)];
              currentSuccess = (item0 == item1);
            }
            {
              auto& currentSuccess = successAcc[static_cast<size_t>(
                  current_check::not_equal_other)];
              currentSuccess = (item0 != item1);
            }
          });
        });
      }

      // Check item equality operator
      common_semantics::check_on_host(log, items[0],
                                      "item " + std::to_string(numDims));
      CHECK_VALUE(log, success[static_cast<size_t>(current_check::equal_self)],
                  true, numDims);
      CHECK_VALUE(log,
                  success[static_cast<size_t>(current_check::not_equal_self)],
                  false, numDims);
      CHECK_VALUE(log, success[static_cast<size_t>(current_check::equal_other)],
                  false, numDims);
      CHECK_VALUE(log,
                  success[static_cast<size_t>(current_check::not_equal_other)],
                  true, numDims);
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
