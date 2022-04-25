/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides functional test for overloads for queue::copy for device_global
//
*******************************************************************************/

#include "../../common/common.h"
#include "../../common/type_coverage.h"
#include "../../usm/usm_api.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_queue_copy

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace device_global_common_functions;

#if defined(SYCL_EXT_ONEAPI_PROPERTY_LIST) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
namespace oneapi = sycl::ext::oneapi;

template <typename T>
struct check_copy_to_dg_kernel;

template <typename T>
struct change_dg_kernel;

// Creating instance with default constructor
template <typename T>
oneapi::device_global<T> dev_global1;

template <typename T>
oneapi::device_global<T> dev_global2;

template <typename T>
oneapi::device_global<T> dev_global3;

/** @brief The function tests that queue copy overloads correctly copy to
 * device_global
 *  @tparam T Type of underlying struct
 */
template <typename T>
void run_test_copy_to_device_global(util::logger& log,
                                    const std::string& type_name) {
  using element_type = std::remove_all_extents_t<T>;
  T data{};
  value_operations<T>::change_val(data, 1);
  const auto src_data = pointer_helper(data);

  // to generate events with generator from usm_api.h
  // gens will generate events that will fill array arr_src
  // in single_task that will take a significant amount of time
  constexpr size_t numEvents = 5;
  constexpr size_t gen_buf_size = 1000;
  std::array<usm_api::event_generator<element_type, gen_buf_size>, numEvents>
      gens{};
  element_type init_value{1};

  auto queue = util::get_cts_object::queue();

  auto event1 = queue.copy(src_data, dev_global1<T>);
  event1.wait();

  // List of events that tested usm operation should wait
  std::vector<sycl::event> depEvents(numEvents);
  // Run time-consuming tasks to generate events
  for (size_t i = 0; i < depEvents.size(); ++i) {
    depEvents[i] = gens[i].init(queue, init_value);
  }

  // Call overloads with events and check if they wait for events to pass.
  // Use fast copy arrays for future check. If overloads do not
  // wait for the passed events to complete, then an incompletely
  // initialized 'arr_src' will be copied to 'arr_dst' and the test
  // will fail with generator function check().

  auto event2 =
      queue.copy(src_data, dev_global2<T>,
                      sizeof(T) / sizeof(element_type), 0, depEvents[0]);
  event2.wait();
  gens[0].copy_arrays(queue);

  auto event3 = queue.copy(src_data, dev_global3<T>,
                                sizeof(T) / sizeof(element_type), 0, depEvents);
  event3.wait();
  // Reverse order is used to increase probability of data race.
  // Array in gens[0] is already copied after event2,
  // so it's not copied here to not to rewrite result for previous case.
  for (size_t i = numEvents - 1; i > 1; --i) {
    gens[i].copy_arrays(queue);
  }

  bool events_result = true;
  for (size_t i = 0; i < numEvents; ++i)
    events_result &= gens[i].check(init_value);

  if (!events_result)
    FAIL(log,
         "One or more generators completed work before the verifier."
         "Copy overloads to device_global didn't wait for depEvents to "
         "complete");

  bool is_copied_correctly = false;
  {
    sycl::buffer<bool, 1> is_cop_corr_buf(&is_copied_correctly,
                                          sycl::range<1>(1));
    queue.submit([&](sycl::handler& cgh) {
      auto is_cop_corr_acc =
          is_cop_corr_buf.template get_access<sycl::access_mode::write>(cgh);
      cgh.single_task<check_copy_to_dg_kernel<T>>([=] {
        is_cop_corr_acc[0] = value_operations::are_equal<T>(dev_global1<T>, data);
        is_cop_corr_acc[0] &=
            value_operations::are_equal<T>(dev_global2<T>, data);
        is_cop_corr_acc[0] &=
            value_operations::are_equal<T>(dev_global3<T>, data);
      });
    });
    queue.wait_and_throw();
  }
  if (!is_copied_correctly) {
    FAIL(log, get_case_description(
                  "Overloads of sycl::queue::copy for device_global",
                  "Didn't copy correct data to device_global", type_name));
  }
}

template <typename T>
oneapi::device_global<T> dev_global;

/** @brief The function tests that queue copy overloads correctly copy from
 * device_global
 *  @tparam T Type of underlying struct
 */
template <typename T>
void run_test_copy_from_device_global(util::logger& log,
                                      const std::string& type_name) {
  using element_type = std::remove_all_extents_t<T>;
  T new_val{};
  value_operations<T>::change_val(new_val, 5);
  T data1{}, data2{}, data3{};
  auto dst_data1 = pointer_helper(data1);
  auto dst_data2 = pointer_helper(data2);
  auto dst_data3 = pointer_helper(data3);

  sycl::queue queue;
  auto queue = util::get_cts_object::queue();

  queue.submit([&](sycl::handler& cgh) {
    cgh.single_task<change_dg_kernel<T>>(
        [=] { value_operations<T>::change_val(dev_global<T>, new_val); });
  });
  queue.wait_and_throw();

  auto event1 = queue.copy(dev_global<T>, dst_data1);
  event1.wait();

  // to generate events with generator from usm_api.h
  // gens will generate events that will fill array arr_src
  // in single_task that will take a significant amount of time
  constexpr size_t numEvents = 5;
  constexpr size_t gen_buf_size = 1000;
  std::array<usm_api::event_generator<element_type, gen_buf_size>, numEvents>
      gens{};
  element_type init_value{1};

  // List of events that tested usm operation should wait
  std::vector<sycl::event> depEvents(numEvents);
  // Run time-consuming tasks to generate events
  for (size_t i = 0; i < depEvents.size(); ++i) {
    depEvents[i] = gens[i].init(queue, init_value);
  }

  // Call overloads with events and check if they wait for events to pass.
  // Use fast copy arrays for future check. If overloads do not
  // wait for the passed events to complete, then an incompletely
  // initialized 'arr_src' will be copied to 'arr_dst' and the test
  // will fail with generator function check().

  auto event2 =
      queue.copy(dev_global<T>, dst_data2,
                      sizeof(T) / sizeof(element_type), 0, depEvents[0]);
  event2.wait();
  gens[0].copy_arrays(queue);

  auto event3 = queue.copy(dev_global<T>, dst_data3,
                                sizeof(T) / sizeof(element_type), 0, depEvents);
  event3.wait();
  // Reverse order is used to increase probability of data race.
  // Array in gens[0] is already copied after event2,
  // so it's not copied here to not to rewrite result for previous case.
  for (size_t i = numEvents - 1; i > 1; --i) {
    gens[i].copy_arrays(queue);
  }

  bool events_result = true;
  for (size_t i = 0; i < numEvents; ++i)
    events_result &= gens[i].check(init_value);

  if (!events_result)
    FAIL(log,
         "One or more generators completed work before the verifier."
         "Copy overloads from device_global didn't wait for depEvents to "
         "complete");

  if (!value_operations::are_equal<T>(data1, new_val) ||
      !value_operations::are_equal<T>(data2, new_val) ||
      !value_operations::are_equal<T>(data3, new_val)) {
    FAIL(log, get_case_description(
                  "Overloads of sycl::queue::copy for device_global",
                  "Didn't copy correct data from device_global", type_name));
  }
}

template <typename T>
class check_queue_overloads_for_device_global_for_type {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    // Run test for queue overloads
    run_test_copy_to_device_global<T>(log, type_name);
    run_test_copy_from_device_global<T>(log, type_name);

    run_test_copy_to_device_global<T[5]>(log, type_name);
    run_test_copy_from_device_global<T[5]>(log, type_name);
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
    for_all_types<check_queue_overloads_for_device_global_for_type>(types, log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
