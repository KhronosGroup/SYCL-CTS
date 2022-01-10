/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::errc_for
//
*******************************************************************************/

#include "exceptions.h"

#define TEST_NAME exceptions_errc_for

namespace TEST_NAMESPACE {

template <template <sycl::backend> class arg>
struct check_template_exists {};

bool check_opencl_supporting(const sycl::queue &q,
                             sycl_cts::util::logger &log) {
  bool opencl_supported{false};
#ifdef SYCL_BACKEND_OPENCL
  opencl_supported = q.get_backend() == sycl::backend::opencl;
#endif  // SYCL_BACKEND_OPENCL
  if (!opencl_supported) {
    log.note("OpenCL backend is not supported on this device");
  }
  return opencl_supported;
}

using namespace sycl_cts;

/** Test instance
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    if (!check_opencl_supporting(util::get_cts_object::queue(), log)) {
      return;
    }
    // check that sycl::errc_for exist
    check_template_exists<sycl::errc_for>();

#ifdef SYCL_BACKEND_OPENCL
    using sycl_errc_enum_t = sycl::errc_for<sycl::backend::opencl>;
    // check that sycl::errc_for is enum and scoped enum
    if (!std::is_enum_v<sycl_errc_enum_t>) {
      FAIL(log, "sycl::errc_for is not enum");
    } else if (!std::is_convertible_v<sycl_errc_enum_t,
                                      std::underlying_type<sycl_errc_enum_t>>) {
      FAIL(log,
           "sycl::errc_for is not a scoped enum cause he can't be implicitly "
           "converted to int");
    }
    const auto errc_value{static_cast<sycl_errc_enum_t>(0)};
    std::error_code err_code(errc_value,
                             sycl::error_category_for<sycl::backend::opencl>());
    if (err_code.default_error_condition() !=
        std::error_condition(
            errc_value, sycl::error_category_for<sycl::backend::opencl>())) {
      FAIL(log,
           "error_code::default_error_condition() is not equal to "
           "std::error_condition");
    }
    if (!std::is_error_code_enum_v<sycl_errc_enum_t>) {
      FAIL(log, "sycl::errc_for is not a error code enumeration");
    }
    if (std::is_error_condition_enum_v<sycl_errc_enum_t>) {
      FAIL(log, "sycl::errc_for is a error condition enumeration");
    }
    if (sycl::error_category_for<sycl::backend::opencl>().name() != "opencl") {
      FAIL(log,
           "sycl::error_category_for<sycl::backend::opencl> name is not "
           "equal to \"opencl\"");
    }
#endif  // SYCL_BACKEND_OPENCL
  }
};

util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
