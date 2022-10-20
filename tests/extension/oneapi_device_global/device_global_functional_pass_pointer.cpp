/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides functional test for device_global
//
//  Tests passing pointer to the underlying data to another kernel
//
//  In this test the first kernel called write_kernel change the value of
//  device_global instance and store address of returned value from .get()
//  method to buffer accessor, that contains T* ptr.
//
//  In the second kernel called read_kernel attempt
//  to read value by dereferencing the pointer from first step. Test will pass
//  if dereferenced value from the second kernel will be equal to the
//  expected value
//
*******************************************************************************/

#include "../../common/common.h"
#include "../../common/type_coverage.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_functional_pass_pointer

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace device_global_common_functions;

#if defined(SYCL_EXT_ONEAPI_PROPERTIES) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
namespace oneapi = sycl::ext::oneapi;

namespace pass_pointer_to_another_kernel {
template <typename T>
oneapi::experimental::device_global<T> dev_global;

template <typename T>
struct write_kernel;
template <typename T>
struct read_kernel;

/**
 * @brief The function tests that pointer to device_global value correctly
 * passes from one kernel to another
 * @tparam T Type of the underlying device_global data
 */
template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  // Pointer to T for store address of underlying value in device_global
  T* ptr;

  auto queue = util::get_cts_object::queue();
  {
    sycl::buffer<T*, 1> ptr_buf(&ptr, sycl::range<1>(1));
    queue.submit([&](sycl::handler& cgh) {
      using kernel = write_kernel<T>;
      auto ptr_acc = ptr_buf.template get_access<sycl::access_mode::write>(cgh);

      cgh.single_task<kernel>([=] {
        // Change the device_global instance and store address to the ptr
        // through accessor
        value_operations::assign(dev_global<T>, 42);
        ptr_acc[0] = &(dev_global<T>.get());
      });
    });
    queue.wait_and_throw();
  }

  // Expecting to read the new_value that was set to the device_global instance
  // in previous kernel
  T new_val{};
  // The function assign have default second parameter, so expect that all
  // values will change the same
  value_operations::assign(new_val, 42);

  bool is_read_correct{true};
  {
    // Creating result buffer
    sycl::buffer<bool, 1> is_read_corr_buf(&is_read_correct, sycl::range<1>(1));

    queue.submit([&](sycl::handler& cgh) {
      using kernel = read_kernel<T>;

      auto is_read_correct_acc =
          is_read_corr_buf.template get_access<sycl::access_mode::write>(cgh);

      cgh.single_task<kernel>([=] {
        is_read_correct_acc[0] =
            (value_operations::are_equal(*(ptr), new_val));
      });
    });
    queue.wait_and_throw();
  }
  if (is_read_correct == false) {
    std::string fail_msg = get_case_description(
        "device_global: Passing a pointer to the "
        "underlying value to another kernel",
        "Wrong value after dereferencing pointer in another kernel", type_name);
    FAIL(log, fail_msg);
  }
}
}  // namespace pass_pointer_to_another_kernel

template <typename T>
class check_device_global_pass_pointer {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    pass_pointer_to_another_kernel::run_test<T>(log, type_name);
    pass_pointer_to_another_kernel::run_test<T[5]>(log, type_name);
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
    for_all_types<check_device_global_pass_pointer>(types, log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
