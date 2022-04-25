/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides functional test for device_global
//
//  Tests interaction with kernel bundle. In test defines kernel that reads and
//  writes the device_global value, then create the kernel bundle from the
//  defined kernel and invokes it. After that, the test builds and invokes the
//  kernel the second time and checks that device_global value correctly changed
//  from the first invocation.
//
*******************************************************************************/

#include "../../common/common.h"
#include "../../common/type_coverage.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_functional_kernel_bundle

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace device_global_common_functions;

#if defined(SYCL_EXT_ONEAPI_PROPERTY_LIST) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
namespace oneapi = sycl::ext::oneapi;

namespace kernel_bundle_interaction {
template <typename T>
oneapi::device_global<T> dev_global;

template <typename T>
struct kernel_read_then_write;

/**
 * @brief The class provide static functions to execute read and write
 * operations in the kernel. Method get_kernel_bundle() will be called to force
 * an online compilation of kernel before invoking
 */
template <typename T>
class read_and_write_in_kernel {
 public:
  /**
   * @brief The function reads value from device_global instance and then writes
   * new value in instance. Test will be failed if value from device_global
   * instance not equal to default value
   */
  static inline void expect_def_val(util::logger& log,
                                    const std::string& type_name) {
    run(true,
        "Value read incorrectly from kernel on first invocation. Default "
        "value expected",
        log, type_name);
  }

  /**
   * @brief The function reads value from device_global instance and then writes
   * new value in instance. Test will be failed if value from device_global
   * instance not equal to T{1}
   */
  static inline void expect_new_val(util::logger& log,
                                    const std::string& type_name) {
    run(false,
        "Value read incorrectly from kernel on second invocation. "
        "Changed value expected",
        log, type_name);
  }

 private:
  /**
   * @brief The function reads value from device_global instance and then
   * writes new value in instance
   * @param is_def_val_expected The flag shows if default value expected
   * @param error_info String to display, when test fails
   */
  static inline void run(const bool is_def_val_expected,
                         const std::string& error_info, util::logger& log,
                         const std::string& type_name) {
    // is_read_correct will be set to true if device_global value is equal to
    // the expected_value inside kernel
    bool is_read_correct{true};

    // Default value of type T in case if we expect to read default value
    T def_val{};
    // Changed value of type T in case if we expect to read modified value
    T new_val{};
    // The function change_val have default second parameter, so expect that all
    // values will change the same
    value_operations::change_val<T>(new_val, 42);

    {
      // Creating result buffer
      sycl::buffer<bool, 1> is_read_corr_buf(&is_read_correct,
                                             sycl::range<1>(1));
      // Setting kernel bundle
      auto queue = util::get_cts_object::queue();
      auto ctxt = queue.get_context();
      // Force online compilation
      auto bundle =
          sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctxt);

      queue.submit([&](sycl::handler& cgh) {
        using kernel = kernel_read_then_write<T>;
        auto is_read_correct_acc =
            is_read_corr_buf.template get_access<sycl::access_mode::write>(cgh);

        // Using pre-compiled kernel from bundle
        cgh.use_kernel_bundle(bundle);

        // Kernel that will compare device_global variable with expected and
        // then write new value
        cgh.single_task<kernel>([=] {
          if (is_def_val_expected) {
            is_read_correct_acc[0] =
                value_operations::are_equal<T>(dev_global<T>, def_val);
          } else {
            is_read_correct_acc[0] =
                value_operations::are_equal<T>(dev_global<T>, new_val);
          }
          value_operations::change_val<T>(dev_global<T>, 42);
        });
      });
      queue.wait_and_throw();
    }
    if (is_read_correct == false) {
      FAIL(log, get_case_description(
                    "device_global: Interaction with kernel bundles",
                    error_info, type_name));
    }
  }
};

/**
 * @brief The function tests that the device_global value is correctly read and
 * changed while interacting with kernel bundles
 * @tparam T Type of underlying device_global value
 */
template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  using VerifierT = read_and_write_in_kernel<T>;
  // At the first run expect default value
  VerifierT::expect_def_val(log, type_name);

  // At the second run expect changed value
  VerifierT::expect_new_val(log, type_name);
}
}  // namespace kernel_bundle_interaction

template <typename T>
class check_device_global_kernel_bundle {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    kernel_bundle_interaction::run_test<T>(log, type_name);
    kernel_bundle_interaction::run_test<T[5]>(log, type_name);
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
#if !defined(SYCL_EXT_ONEAPI_PROPERTY_LIST)
    WARN("SYCL_EXT_ONEAPI_PROPERTY_LIST is not defined, test is skipped");
#elif !defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    WARN("SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined, test is skipped");
#else
    auto types = device_global_types::get_types();
    for_all_types<check_device_global_kernel_bundle>(types, log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
