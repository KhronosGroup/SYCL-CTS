/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_api

namespace device_api__ {
using namespace sycl_cts;

/** tests the api for cl::sycl::device
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  virtual void run(util::logger &log) override {
    try {
      cts_selector selector;
      cl::sycl::device device(selector);

      /** check has_extensions() method
      */
      auto extensionSupported =
          device.has_extension(cl::sycl::string_class("cl_khr_fp64"));
      if (typeid(extensionSupported) != typeid(bool))
        FAIL(log, "has_extension() does not return bool");

      /** check get_info() method
      */
      auto platformName = device.has_extension(cl::sycl::info::platform::name);
      if (typeid(platformName) != typeid(cl::sycl::string_class)) {
        FAIL(log, "has_extension() does not string_class");
      }

      /** check is_host() method
      */
      auto isHost = device.is_host();
      if (typeid(isHost) != typeid(bool)) {
        FAIL(log, "is_host() does not return bool");
      }

      /** check is_cpu() method
      */
      auto isCPU = device.is_cpu();
      if (typeid(isCPU) != typeid(bool)) {
        FAIL(log, "is_cpu() does not return bool");
      }

      /** check is_host() method
      */
      auto isGPU = device.is_gpu();
      if (typeid(isGPU) != typeid(bool)) {
        FAIL(log, "is_gpu() does not return bool");
      }

      /** check get_platform() method
      */
      auto parentPlatform = device.get_platform();
      if (typeid(parentPlatform) != typeid(cl::sycl::platform)) {
        FAIL(log, "get_platform() does not return platform");
      }

      /** check create_sub_devices_equally(size_t) method
      */
      {
        if (device.has_extension("cl_ext_device_fission")) {
          auto subDevices = device.create_sub_devices_equally(2);
          if (typeid(subDevices) !=
              typeid(cl::sycl::vector_class<cl::sycl::device>)) {
            FAIL(log,
                 "create_sub_devices_equally() does not return "
                 "vector_class<device>");
          }
        }
      }

      /** check create_sub_devices_by_count(vector_size<size_t>) method
      */
      {
        if (device.has_extension("cl_ext_device_fission")) {
          cl::sycl::vector_class<size_t> devicePartitionCounts;
          devicePartitionCounts.push_back(3);
          devicePartitionCounts.push_back(1);
          auto subDevices =
              device.create_sub_devices_by_count(devicePartitionCounts);
          if (typeid(subDevices) !=
              typeid(cl::sycl::vector_class<cl::sycl::device>)) {
            FAIL(log,
                 "create_sub_devices_by_count() does not return "
                 "vector_class<device>");
          }
        }
      }

      /** check create_sub_devices_by_affinity(vector_size<affinity_domain>)
       * method
      */
      {
        if (device.has_extension("cl_ext_device_fission")) {
          auto subDevices = device.create_sub_devices_by_count(
              cl::sycl::info::cl_device_affinity_domain_next_partitionable);
          if (typeid(subDevices) !=
              typeid(cl::sycl::vector_class<cl::sycl::device>)) {
            FAIL(log,
                 "create_sub_devices_by_affinity() does not return "
                 "vector_class<device>");
          }
        }
      }

      /** check get_devices() static method
      */
      auto devices = cl::sycl::device::get_devices();
      if (typeid(devices) != typeid(cl::sycl::vector_class<cl::sycl::device>)) {
        FAIL(
            log,
            "get_devices() does not return cl::sycl::vector<cl::sycl::device>");
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "a sycl exception was caught");
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_api__ */
