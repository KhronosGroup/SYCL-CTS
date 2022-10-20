/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for device_global subscript operator
//
//  Tests executes in the kernel. These types of tests have result arrays,
//  that kernel uses through a buffer accessor. For result arrays defined enum
//  classes that have to be used for addressing elements in the array. Arrays
//  with strings used to print error corresponding to the index in result array.
//  This test is provided only for array types.
//
*******************************************************************************/

#include "../../../util/array.h"
#include "../../common/common.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_api_subscript_operator

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace device_global_common_functions;

#if defined(SYCL_EXT_ONEAPI_PROPERTIES) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
using namespace sycl::ext::oneapi::experimental;

template <typename T, size_t sizeOfArray, test_names name>
struct device_global_kernel_name;

namespace subscript_operator {
// Creating instance with default constructor
template <typename T, size_t sizeOfArray>
device_global<T[sizeOfArray]> dev_global;
template <typename T, size_t sizeOfArray>
const device_global<T[sizeOfArray]> const_dev_global;

// Used to address elements in the result array
enum class indx : size_t {
  correct_def_val_const,
  correct_def_val_non_const,
  same_type_const,
  same_type_non_const,
  correct_changed_val,
  size  // must be last
};
constexpr size_t integral(const indx& i) { return to_integral(i); }

/** @brief The function tests that device_global correctly operating with
 * subscript operator and returns underlying value
 *  @tparam T Type of underlying device_global data
 *  @tparam sizeOfArray size of underlying data array
 */
template <typename T, size_t sizeOfArray>
void run_test(util::logger& log, const std::string& type_name) {
  using kernel =
      device_global_kernel_name<T, sizeOfArray, test_names::subscript_operator>;

  std::string error_strings[integral(indx::size)]{
      "Wrong default value returned by subscript operator of const instance",
      "Wrong default value returned by subscript operator of non-const "
      "instance",
      "Wrong type returned by subscript operator of const instance",
      "Wrong type returned by subscript operator of non-const instance",
      "Wrong value after change instance through subscript operator",
  };

  util::array<bool, integral(indx::size)> result;
  {
    sycl::buffer<bool, 1> result_buf(result.values,
                                     sycl::range<1>(integral(indx::size)));

    auto queue = util::get_cts_object::queue();
    queue.submit([&](sycl::handler& cgh) {
      auto result_acc =
          result_buf.template get_access<sycl::access_mode::read_write>(cgh);

      T value_ref_zero_init{};
      std::memset(&value_ref_zero_init, 0, sizeof(value_ref_zero_init));
      cgh.single_task<kernel>([=] {
        for (size_t i = 0; i < sizeOfArray; ++i) {
          // Check that array element contains default value
          result_acc[integral(indx::correct_def_val_const)] =
              const_dev_global<T, sizeOfArray>[i] == value_ref_zero_init;
          result_acc[integral(indx::correct_def_val_non_const)] =
              dev_global<T, sizeOfArray>[i] == value_ref_zero_init;

          // Check that array element contains correct type
          result_acc[integral(indx::same_type_const)] =
              std::is_same<decltype(const_dev_global<T, sizeOfArray>[i]),
                           typename device_global<T>::element_type&>::value;
          result_acc[integral(indx::same_type_non_const)] =
              std::is_same<decltype(dev_global<T, sizeOfArray>[i]),
                           typename device_global<T>::element_type&>::value;

          const T new_value{1};
          dev_global<T, sizeOfArray>[i] = new_value;
          // Check that array element contains new value
          result_acc[integral(indx::correct_changed_val)] =
              dev_global<T, sizeOfArray>[i] == new_value;

          // If any check fail, then sum will not be equal to size, then
          // we can break the loop, because test is failed
          const int sum_of_checks =
              result_acc[integral(indx::correct_def_val_const)] +
              result_acc[integral(indx::correct_def_val_non_const)] +
              result_acc[integral(indx::same_type_const)] +
              result_acc[integral(indx::same_type_non_const)] +
              result_acc[integral(indx::correct_changed_val)];
          if (sum_of_checks != integral(indx::size)) break;
        }
      });
    });
  }
  for (size_t i = integral(indx::correct_def_val_const);
       i < integral(indx::size); ++i) {
    if (!result[i]) {
      std::string fail_msg = get_case_description("Device global: operator[]()",
                                                  error_strings[i], type_name);
      FAIL(log, fail_msg);
    }
  }
}
}  // namespace subscript_operator

template <typename T>
class check_device_global_api_subscript_operator_for_type {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    // Run tests for subscript operator with different size of array
    subscript_operator::run_test<T, 1>(log, type_name);
    subscript_operator::run_test<T, 5>(log, type_name);
  }
};
#endif

/** test device_global subscript operator
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger& log) override {
#if !defined(SYCL_EXT_ONEAPI_PROPERTIES)
    WARN("SYCL_EXT_ONEAPI_PROPERTIES is not defined, test is skipped");
#elif !defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    WARN("SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined, test is skipped");
#else
    auto types = device_global_types::get_types();
    for_all_types<check_device_global_api_subscript_operator_for_type>(types,
                                                                       log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
