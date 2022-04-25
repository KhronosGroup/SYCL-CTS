/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides functional test for device_global
//
//  Runs several tests on multiple devices. The test checks that the value of
//  the device_global instance will be unique for every device. There have to be
//  at least 2 devices to run the test.
//
*******************************************************************************/

#include "../../common/common.h"
#include "../../common/type_coverage.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_functional_unique_for_device

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace device_global_common_functions;
#if defined(SYCL_EXT_ONEAPI_PROPERTY_LIST) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
namespace oneapi = sycl::ext::oneapi;

/**
 * @brief The function tries to partition the device that was given in the
 * parameter into at least 2 subdevices
 * @param dev The sycl::device instance that needs to be partitioned
 * @return std::vector<sycl::device> The resulting vector contains subdevices if
 * the device can be partitioned. Otherwise, the vector will be empty.
 */
std::vector<sycl::device> try_to_get_sub_devices(const sycl::device& dev) {
  using namespace sycl::info;
  // Get partition properties that the device supports
  auto props = dev.template get_info<device::partition_properties>();
  // Try to partition by each property
  for (const auto& prop : props) {
    switch (prop) {
      case partition_property::partition_equally:
        return dev.create_sub_devices<partition_property::partition_equally>(1);
      case partition_property::partition_by_counts:
        return dev.create_sub_devices<partition_property::partition_by_counts>(
            {1, 1});
      case partition_property::partition_by_affinity_domain: {
        auto supported_domains =
            dev.get_info<device::partition_affinity_domains>();
        for (const auto& domain : supported_domains) {
          auto sub_devices = dev.create_sub_devices<
              partition_property::partition_by_affinity_domain>(domain);
          // Have to partition in at least 2 subdevices
          if (sub_devices.size() > 1) {
            return sub_devices;
          }
        }
      }
      // Return an empty vector if device doesn't support partition
      case partition_property::no_partition:
        return {};
    }
  }
  // Return an empty vector if fails to partition in at least 2 devices
  return {};
}

namespace unique_for_every_device {
template <typename T>
oneapi::device_global<T> dev_global;

template <typename T>
struct write_kernel;
template <typename T>
struct read_kernel;

/**
 * @brief The function invokes the kernel, that changes device_global instance
 * @tparam T Type of underlying value
 * @param q The sycl::queue object of a device
 */
template <typename T>
void call_write_kernel(sycl::queue& q) {
  using kernel = write_kernel<T>;
  q.submit([&](sycl::handler& cgh) {
    cgh.single_task<kernel>(
        [=] { value_operations::change_val<T>(dev_global<T>, 42); });
  });
}

/**
 * @brief The function invokes the kernel, that reads value from device_global
 * instance and compares it with default
 * @tparam T Type of underlying value
 * @param q The sycl::queue object of a device
 */
template <typename T>
void call_read_kernel(sycl::queue& q, util::logger& log,
                      const std::string& type_name) {
  using kernel = write_kernel<T>;
  bool is_default_val{false};
  {
    sycl::buffer is_default_val_buf(&is_default_val, sycl::range<1>(1));
    q.submit([&](sycl::handler& cgh) {
      auto is_default_val_acc =
          is_default_val_buf.template get_access<sycl::access_mode::write>(cgh);
      cgh.single_task<kernel>([=] {
        T def_value{};
        is_default_val_acc[0] =
            value_operations::are_equal<T>(dev_global<T>, def_value);
      });
    });
  }
  // Test fails if non-default value read from the device_global instance
  if (!is_default_val)
    FAIL(log, get_case_description("device_global: Unique for every device",
                                   "Value changed on another device. "
                                   "Expect change only on one device",
                                   type_name));
}
/**
 * @brief The function tests that the device_global instance is unique for every
 * device
 * @tparam T Type of the underlying value
 */
template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  std::vector<sycl::queue> queues;

  auto platforms = sycl::platform::get_platforms();
  // Collect at least two devices to the vector with queues
  for (auto& platform : platforms) {
    auto cur_platform_devices = platform.get_devices();
    for (const auto& device : cur_platform_devices) {
      // Try to partition device into subdevices
      auto subdevices = try_to_get_sub_devices(device);
      if (subdevices.size() > 1) {
        // Create sycl::queue instances for the subdevices and put it to the end
        // of queues vector
        queues.insert(queues.end(), subdevices.begin(), subdevices.end());
      } else {
        // If device doesn't support partition then create sycl::queue
        // instance for the root device and put it to the end of queues vector
        queues.emplace_back(device);
      }
    }
    // If already have more than one device, then break the loop
    if (queues.size() > 1) break;
  }

  const size_t q_size = queues.size();
  if (q_size < 2) {
    WARN(
        "device_global: Unique for every device. Not enough devices for the "
        "test. At least 2 required.");
    return;
  }

  // Check that device_global is unique for every device
  call_write_kernel<T>(queues[0]);
  call_read_kernel<T>(queues[1], log, type_name);
}
}  // namespace unique_for_every_device

template <typename T>
class check_device_global_serveal_kernels_one_device_for_type {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    unique_for_every_device::run_test<T>(log, type_name);
    unique_for_every_device::run_test<T[5]>(log, type_name);
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
    for_all_types<check_device_global_serveal_kernels_one_device_for_type>(
        types, log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
