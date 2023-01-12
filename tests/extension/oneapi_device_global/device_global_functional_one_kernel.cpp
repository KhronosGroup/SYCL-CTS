/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides functional tests for device_global
//  Run one kernel multiple times on one device with the different
//  properies. The test reads a value from device_global instance and write the
//  new value. At the next run it expects to read the same value, that we wrote
//  before.
//
*******************************************************************************/

#include "../../common/common.h"
#include "../../common/type_coverage.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_functional_one_kernel

namespace TEST_NAMESPACE {
using namespace sycl_cts;
#if defined(SYCL_EXT_ONEAPI_PROPERTIES) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
namespace oneapi = sycl::ext::oneapi;
using namespace device_global_common_functions;

namespace one_kernel_multiple_times {
template <typename T, typename properties_t>
oneapi::experimental::device_global<T, properties_t> dev_global;

template <typename T, property_tag tag>
struct kernel_read_then_write;

/**
 * @brief The class provide static functions to execute read and write
 * operations in the kernel
 */
template <typename T, typename properties_t, property_tag tag>
class read_and_write_in_kernel {
 public:
  /**
   * @brief The function reads value from device_global instance and then writes
   * new value in instance. Test will be failed if value from the device_global
   * instance not equal to default value
   */
  static inline void expect_def_val(sycl::queue& queue, util::logger& log,
                                    const std::string& type_name) {
    run(queue, true,
        "Value read incorrectly from kernel on first invocation. Default "
        "value expected",
        log, type_name);
  }

  /**
   * @brief The function reads value from device_global instance and then writes
   * new value in instance. Test will be failed if value from device_global
   * instance not equal to T{1}
   */
  static inline void expect_new_val(sycl::queue& queue, util::logger& log,
                                    const std::string& type_name) {
    run(queue, false,
        "Value read incorrectly from kernel on second invocation. "
        "Changed value expected",
        log, type_name);
  }

 private:
  /**
   * @brief The function reads value from device_global instance and then
   * writes new value in instance
   * @param expect_def_value Flag that shows if default value expected after
   * read in kernel or modified - T{1} value expected
   * @param error_info String with additional info to display, when test fails
   * @param type_name Name of testing type for display if test fails
   */
  static inline void run(sycl::queue& queue, const bool expect_def_value,
                         const std::string& error_info, util::logger& log,
                         const std::string& type_name) {
    // Default value of type T in case if we expect to read default value
    T def_val{};

    // Changed value of type T in case if we expect to read modified value
    T new_val{};
    value_operations::assign(new_val, 42);

    // is_read_correct will be set to true if device_global value is equal to
    // the expected value inside kernel
    bool is_read_correct{false};
    {
      // Creating result buffer
      sycl::buffer<bool, 1> is_read_corr_buf(&is_read_correct,
                                             sycl::range<1>(1));
      queue.submit([&](sycl::handler& cgh) {
        using kernel = kernel_read_then_write<T, tag>;
        auto is_read_correct_acc =
            is_read_corr_buf.template get_access<sycl::access_mode::write>(cgh);

        // Kernel that will compare device_global variable with expected and
        // then write new value
        cgh.single_task<kernel>([=] {
          if (expect_def_value) {
            is_read_correct_acc[0] = value_operations::are_equal(
                dev_global<T, properties_t>, def_val);
          } else {
            is_read_correct_acc[0] = value_operations::are_equal(
                dev_global<T, properties_t>, new_val);
          }
          value_operations::assign(dev_global<T, properties_t>, 42);
        });
      });
      queue.wait_and_throw();
    }
    if (is_read_correct == false) {
      std::string fail_msg = get_case_description(
          "device_global: Running one kernel multiple times", error_info,
          type_name);
      FAIL(log, fail_msg);
    }
  }
};

/**
 * @brief The function tests that the device_global value is correctly read and
 * changed from a single kernel executed multiple times
 * @tparam T Type of underlying device_global value
 * @tparam prop_value_t Type of property_value that included in properties
 * @tparam tag For kernel naming
 */
template <typename T, typename prop_value_t, property_tag tag>
void run_test(util::logger& log, const std::string& type_name) {
  using VerifierT = read_and_write_in_kernel<T, prop_value_t, tag>;
  auto queue = util::get_cts_object::queue();

  // At first run expecting default values
  VerifierT::expect_def_val(queue, log, type_name);

  // At the second run expect changed value
  VerifierT::expect_new_val(queue, log, type_name);
}
/**
 * @brief The function runs a test with properties that can include in
 * device_global properties
 * @tparam T Type of underlying value in device_global
 */
template <typename T>
void run_tests_with_properties(sycl_cts::util::logger& log,
                               const std::string& type_name) {
  // Using a property_tag for kernel name

  // Run without any properies
  run_test<T, decltype(oneapi::experimental::properties{}), property_tag::none>(
      log, type_name);

  {
    using oneapi::experimental::device_image_scope;
    // Run with device_image_scope property
    run_test<T, decltype(oneapi::experimental::properties{device_image_scope}),
             property_tag::dev_image_scope>(log, type_name);
  }

  {
    using oneapi::experimental::host_access;
    using oneapi::experimental::host_access_enum;
    // Run with different host_access properies
    run_test<T,
             decltype(oneapi::experimental::properties{
                 host_access<host_access_enum::read>}),
             property_tag::host_access_r>(log, type_name);
    run_test<T,
             decltype(oneapi::experimental::properties{
                 host_access<host_access_enum::write>}),
             property_tag::host_access_w>(log, type_name);
    run_test<T,
             decltype(oneapi::experimental::properties{
                 host_access<host_access_enum::read_write>}),
             property_tag::host_access_r_w>(log, type_name);
    run_test<T,
             decltype(oneapi::experimental::properties{
                 host_access<host_access_enum::none>}),
             property_tag::host_access_none>(log, type_name);
  }

  {
    using oneapi::experimental::init_mode;
    using oneapi::experimental::init_mode_enum;
    // Run with different init_mode properies
    run_test<T,
             decltype(oneapi::experimental::properties{
                 init_mode<init_mode_enum::reprogram>}),
             property_tag::init_mode_trig_reprogram>(log, type_name);
    run_test<T,
             decltype(oneapi::experimental::properties{
                 init_mode<init_mode_enum::reset>}),
             property_tag::init_mode_trig_reset>(log, type_name);
  }

  {
    using oneapi::experimental::implement_in_csr;
    // Run with different implement_in_csr properies
    run_test<T,
             decltype(oneapi::experimental::properties{implement_in_csr<true>}),
             property_tag::impl_in_csr_true>(log, type_name);
    run_test<
        T, decltype(oneapi::experimental::properties{implement_in_csr<false>}),
        property_tag::impl_in_csr_false>(log, type_name);
  }
}

}  // namespace one_kernel_multiple_times

template <typename T>
class check_device_global_one_kernel_for_type {
 public:
  inline void operator()(sycl_cts::util::logger& log,
                         const std::string& type_name) const {
    using namespace one_kernel_multiple_times;
    // Run tests with T and array T
    run_tests_with_properties<T>(log, type_name);
    run_tests_with_properties<T[5]>(log, type_name);
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
    for_all_types<check_device_global_one_kernel_for_type>(types, log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
