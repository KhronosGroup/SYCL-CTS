/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides functional tests for device_global
//
//  Run several kernels on one device. The test modify device_global value in
//  the first_kernel and then expect to read the same value from the second
//  kernel. The test have a result array, that kernel uses through a buffer
//  accessor. For the result array defined enum class is defined that have to be
//  used for addressing elements in the array.
//
*******************************************************************************/

#include "../../../util/array.h"
#include "../../common/common.h"
#include "../../common/type_coverage.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_functional_several_kernel_in_one_device

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace device_global_common_functions;
#if defined(SYCL_EXT_ONEAPI_PROPERTIES) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
namespace oneapi = sycl::ext::oneapi;

namespace several_kernel_in_one_device {
template <typename T>
oneapi::experimental::device_global<T> dev_global;
template <typename T>
const oneapi::experimental::device_global<T> const_dev_global;

template <typename T>
struct first_kernel;
template <typename T>
struct second_kernel;

// Used to address elements in the result array
enum class indx : size_t {
  const_expected,
  non_const_expected,
  size  // must be last
};
constexpr auto integral(const indx& i) { return to_integral<indx>(i); }

/**
 * @brief The function tests that device_global value changes correctly being
 * modified and read from the different kernels on one device
 * @tparam T Type of underlying device_global value
 */
template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  // Using remove_extent for get type of element when T is array
  // If T is not array then element_type == T
  // As we fill the array with the same numbers with type of the array element,
  // we can store only one number to compare whole array with this number
  using element_type = std::remove_extent_t<T>;

  // Creating variables for store result of comparing
  constexpr size_t checks_size = integral(indx::size);

  element_type new_value;
  value_operations::assign(new_value, 42);

  // Changing value in first kernel
  auto queue = util::get_cts_object::queue();
  {
    queue.submit([&](sycl::handler& cgh) {
      cgh.single_task<first_kernel<T>>(
          [=]() { value_operations::assign(dev_global<T>, new_value); });
    });
    queue.wait_and_throw();
  }

  util::array<element_type, checks_size> expected_value;

  // For non-const expecting changed value
  value_operations::assign(expected_value[integral(indx::non_const_expected)],
                           new_value);

  // For const value expecting that value is zero-initialized
  std::memset(&expected_value[integral(indx::const_expected)], 0,
              sizeof(element_type));

  util::array<bool, checks_size> changed_correct;
  {
    sycl::buffer<bool> changed_corr_buf(changed_correct.values,
                                        sycl::range<1>(checks_size));

    queue.submit([&](sycl::handler& cgh) {
      auto changed_corr =
          changed_corr_buf.template get_access<sycl::access_mode::write>(cgh);
      // Comparing current device_global val with expected in second kernel
      cgh.single_task<second_kernel<T>>([=](sycl::kernel_handler h) {
        changed_corr[integral(indx::const_expected)] =
            value_operations::are_equal(
                const_dev_global<T>,
                expected_value[integral(indx::const_expected)]);
        changed_corr[integral(indx::non_const_expected)] =
            value_operations::are_equal(
                dev_global<T>,
                expected_value[integral(indx::non_const_expected)]);
      });
    });
    queue.wait_and_throw();
  }

  for (size_t i = 0; i < checks_size; i++) {
    if (changed_correct[i] == false) {
      FAIL(log, get_case_description(
                    "device_global: Running two kernels on one device",
                    "Value changed incorrectly after modify in one kernel and "
                    "read from another",
                    type_name));
    }
  }
}
}  // namespace several_kernel_in_one_device

template <typename T>
class check_device_global_serveal_kernels_one_device_for_type {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    several_kernel_in_one_device::run_test<T>(log, type_name);
    several_kernel_in_one_device::run_test<T[5]>(log, type_name);
  }
};
#endif

/** test device_global functional
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
    for_all_types<check_device_global_serveal_kernels_one_device_for_type>(
        types, log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
