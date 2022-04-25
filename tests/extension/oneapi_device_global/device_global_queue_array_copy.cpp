/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides test for array for overloads for queue::copy for device_global
//
*******************************************************************************/

#include "../../common/common.h"
#include "../../common/type_coverage.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_queue_array_copy

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace device_global_common_functions;

#if defined(SYCL_EXT_ONEAPI_PROPERTY_LIST) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
namespace oneapi = sycl::ext::oneapi;

template <typename T>
struct queue_array_copy_to_dg_kernel;

template <typename T>
struct queue_array_change_dg_kernel;

template <typename T>
oneapi::device_global<T> dev_global_out;

/** @brief The function tests that queue copy overloads correctly copy
 *  a single element to device_global
 *  @tparam T Type of underlying struct
 */
template <typename T, size_t N>
void run_test_copy_to_device_global_array(util::logger& log,
                                          const std::string& type_name) {
  T data[N];
  value_operations<T[N]>::change_val(data, 1);
  const auto src_data = &data[0];
  size_t num_element = N / 2;

  auto queue = util::get_cts_object::queue();

  auto event = queue.copy(src_data, dev_global_out<T[N]>, 1, num_element);
  event.wait();

  bool is_copied_correctly = false;
  {
    sycl::buffer<bool, 1> is_cop_corr_buf(&is_copied_correctly,
                                          sycl::range<1>(1));
    queue.submit([&](sycl::handler& cgh) {
      auto is_cop_corr_acc =
          is_cop_corr_buf.template get_access<sycl::access_mode::write>(cgh);
      cgh.single_task<queue_array_copy_to_dg_kernel<T>>([=] {
        is_cop_corr_acc[0] =
            dev_global_out<T[N]>[num_element] == data[num_element];
      });
    });
    queue.wait_and_throw();
  }
  if (!is_copied_correctly) {
    FAIL(log,
         get_case_description(
             "Overload of sycl::queue::copy for device_global",
             "Didn't copy correct element to device_global array", type_name));
  }
}

// Creating instance with default constructor
template <typename T>
oneapi::device_global<T> dev_global_in;

/** @brief The function tests that queue copy overloads correctly copy
 * single element from device_global
 *  @tparam T Type of underlying struct
 */
template <typename T, size_t N>
void run_test_copy_from_device_global_array(util::logger& log,
                                            const std::string& type_name) {
  T new_val[N];
  value_operations<T[N]>::change_val(new_val, 5);
  T data[N];
  auto dst_data = &data[0];
  size_t num_element = N / 2;

  auto queue = util::get_cts_object::queue();

  queue.submit([&](sycl::handler& cgh) {
    cgh.single_task<queue_array_change_dg_kernel<T>>(
        [=] { value_operations<T[N]>::change_val(dev_global_in<T[N]>, new_val); });
  });
  queue.wait_and_throw();

  auto event = queue.copy(dev_global_in<T[N]>, dst_data, 1, num_element);
  event.wait();

  if (data[num_element] != new_val[num_element]) {
    FAIL(log, get_case_description(
                  "Overload of sycl::queue::copy for device_global",
                  "Didn't copy correct element from device_global array",
                  type_name));
  }
}

template <typename T>
class check_queue_overloads_for_device_global_array_for_type {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    // Run test for queue overloads
    run_test_copy_to_device_global_array<T, 5>(log, type_name);
    run_test_copy_from_device_global_array<T, 5>(log, type_name);
  }
};
#endif

/** test overloads for queue::copy for device_global
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
    for_all_types<check_queue_overloads_for_device_global_array_for_type>(types,
                                                                          log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
