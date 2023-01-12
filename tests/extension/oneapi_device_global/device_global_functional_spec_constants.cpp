/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides functional test for device_global
//
//  Checks interaction of device_global instance with specialization
//  constants. At the first kernel run check that device_global contains default
//  value and then change the device_global to value from specialization
//  constant. At the second run change specialization constant value and check
//  that the device_global value contains new value from first run.
//
*******************************************************************************/

#include "../../common/common.h"
#include "../../common/type_coverage.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_functional_spec_constants

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace device_global_common_functions;

#if defined(SYCL_EXT_ONEAPI_PROPERTIES) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
namespace oneapi = sycl::ext::oneapi;

namespace spec_constants_interaction {
template <typename T>
oneapi::experimental::device_global<T> dev_global;

constexpr sycl::specialization_id<int> spec_const_id;

template <typename T>
struct kernel_read_then_write;

/**
 * @brief The class provide static functions to execute read and write
 * operations in the kernel
 */
template <typename T>
class read_and_write_in_kernel {
 public:
  /**
   * @brief The function reads value from device_global instance and then writes
   * new value from specialization constants in instance. Test will fail if
   * value from device_global instance not equal to default value
   */
  static inline void expect_def_val(sycl::queue& queue, util::logger& log,
                                    const std::string& type_name) {
    run(queue, true, "Expect to read default value", log, type_name);
  }

  /**
   * @brief The function reads value from device_global instance and then writes
   * new value from specialization constants in instance. Test will be failed if
   * value from device_global instance not equal to T{1}
   */
  static inline void expect_new_val(sycl::queue& queue, util::logger& log,
                                    const std::string& type_name) {
    run(queue, false, "Expect to read modified value", log, type_name);
  }

 private:
  /**
   * @brief The function reads value from device_global instance and then
   * writes new value from specialization constants in instance.
   * @param is_def_val_expected The flag shows if default value expected
   * @param error_info String to display, when test fails
   */
  static inline void run(sycl::queue& queue, const bool is_def_val_expected,
                         const std::string& error_info, util::logger& log,
                         const std::string& type_name) {
    constexpr int initial_sc_val = 1;
    constexpr int changed_sc_val = 2;
    const int sc_val = is_def_val_expected ? initial_sc_val : changed_sc_val;

    // Default value of type T in case if we expect to read default value
    T def_val{};
    // Changed value of type T in case if we expect to read modified value
    T new_val{};
    value_operations::assign(new_val, initial_sc_val);

    // is_read_correct will be set to true if device_global value is equal to
    // the expected_val inside kernel
    bool is_read_correct{false};

    {
      // Creating result buffer
      sycl::buffer<bool, 1> is_read_corr_buf(&is_read_correct,
                                             sycl::range<1>(1));
      queue.submit([&](sycl::handler& cgh) {
        using kernel = kernel_read_then_write<T>;

        cgh.template set_specialization_constant<spec_const_id>(sc_val);

        auto is_read_correct_acc =
            is_read_corr_buf.template get_access<sycl::access_mode::write>(cgh);

        // Kernel that will compare device_global variable with expected and
        // then write new value from specialization constants
        cgh.single_task<kernel>([=](sycl::kernel_handler h) {
          if (is_def_val_expected) {
            is_read_correct_acc[0] =
                value_operations::are_equal(dev_global<T>, def_val);
          } else {
            is_read_correct_acc[0] =
                value_operations::are_equal(dev_global<T>, new_val);
          }

          // Get specialization constant value for change device_global instance
          int sc_val = h.template get_specialization_constant<spec_const_id>();
          value_operations::assign(dev_global<T>, sc_val);
        });
      });
      queue.wait_and_throw();
    }
    if (is_read_correct == false) {
      std::string fail_msg = get_case_description(
          "device_global: Interaction with specialization constants",
          error_info, type_name);
      FAIL(log, fail_msg);
    }
  }
};

/**
 * @brief The function tests that the device_global value is correctly read and
 * changed while interacting with specialization constants
 * @tparam T Type of underlying device_global value
 */
template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  using VerifierT = read_and_write_in_kernel<T>;
  auto queue = util::get_cts_object::queue();

  // At the first run expect default value
  VerifierT::expect_def_val(queue, log, type_name);

  // At the second run expect changed value
  VerifierT::expect_new_val(queue, log, type_name);
}
}  // namespace spec_constants_interaction

template <typename T>
class check_device_global_spec_constants {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    spec_constants_interaction::run_test<T>(log, type_name);
    spec_constants_interaction::run_test<T[5]>(log, type_name);
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
    SKIP("SYCL_EXT_ONEAPI_PROPERTIES is not defined");
#elif !defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    SKIP("SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined");
#else
    auto types = device_global_types::get_types();
    for_all_types<check_device_global_spec_constants>(types, log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
