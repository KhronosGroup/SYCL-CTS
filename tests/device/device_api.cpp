/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_api

namespace TEST_NAMESPACE {
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

  /** returns true if the device supports a particular partition property
   */
  static bool supports_partition_property(
      const cl::sycl::device &dev,
      cl::sycl::info::partition_property partitionProp) {
    auto supported =
        dev.get_info<cl::sycl::info::device::partition_properties>();
    for (cl::sycl::info::partition_property prop : supported) {
      if (prop == partitionProp) {
        return true;
      }
    }
    return false;
  }

  /** returns true if the device supports a particular affinity domain
   */
  static bool supports_affinity_domain(
      const cl::sycl::device &dev,
      cl::sycl::info::partition_property partitionProp,
      cl::sycl::info::partition_affinity_domain domain) {
    if (partitionProp !=
        cl::sycl::info::partition_property::partition_by_affinity_domain) {
      return true;
    }
    auto supported =
        dev.get_info<cl::sycl::info::device::partition_affinity_domains>();
    for (cl::sycl::info::partition_affinity_domain dom : supported) {
      if (dom == domain) {
        return true;
      }
    }
    return false;
  }

  /** execute this test
   */
  virtual void run(util::logger &log) override {
    try {
      /** check get_platform() member function
       */
      {
        cts_selector selector;
        auto dev = util::get_cts_object::device(selector);
        auto parentPlatform = dev.get_platform();
        check_return_type<cl::sycl::platform>(log, parentPlatform,
                                              "device::get_platform()");
      }

      /** check get() member function
       */
      {
        cts_selector selector;
        auto dev = util::get_cts_object::device(selector);
        if (!selector.is_host()) {
          auto clDeviceId = dev.get();
          check_return_type<cl_device_id>(log, clDeviceId, "device::get()");
        }
      }

      /** check is_host() member function
       */
      {
        cts_selector selector;
        auto dev = util::get_cts_object::device(selector);
        auto isHost = dev.is_host();
        check_return_type<bool>(log, isHost, "device::is_host()");
      }

      /** check is_cpu() member function
       */
      {
        cts_selector selector;
        auto dev = util::get_cts_object::device(selector);
        auto isCPU = dev.is_cpu();
        check_return_type<bool>(log, isCPU, "device::is_cpu()");
      }

      /** check is_gpu() member function
       */
      {
        cts_selector selector;
        auto dev = util::get_cts_object::device(selector);
        auto isGPU = dev.is_gpu();
        check_return_type<bool>(log, isGPU, "device::is_gpu()");
      }

      /** check is_accelerator() member function
       */
      {
        cts_selector selector;
        auto dev = util::get_cts_object::device(selector);
        auto isAcc = dev.is_accelerator();
        check_return_type<bool>(log, isAcc, "device::is_accelerator()");
      }

      /** check get_info() member function
       */
      {
        cts_selector selector;
        auto dev = util::get_cts_object::device(selector);
        auto platformName = dev.get_info<cl::sycl::info::device::name>();
        check_return_type<cl::sycl::string_class>(log, platformName,
                                                  "device::get_info()");
      }

      /** check has_extensions() member function
      */
      {
        cts_selector selector;
        auto dev = util::get_cts_object::device(selector);
        auto extensionSupported =
            dev.has_extension(cl::sycl::string_class("cl_khr_fp64"));
        check_return_type<bool>(log, extensionSupported,
                                "device::has_extension(string_class)");
      }

      /** check
       * create_sub_devices<info::partition_property::partition_equally>(size_t)
       * member function
       */
      {
        cts_selector selector;
        auto dev = util::get_cts_object::device(selector);
        if (supports_partition_property(
                dev, cl::sycl::info::partition_property::partition_equally)) {
          auto subDevices = dev.create_sub_devices<
              cl::sycl::info::partition_property::partition_equally>(2);
          check_return_type<cl::sycl::vector_class<cl::sycl::device>>(
              log, subDevices, "device::create_sub_devices(size_t)");
        }
      }

      /** check
       * create_sub_devices<info::partition_property::partition_by_counts>(vector_class<size_t>)
       * member function
      */
      {
        cts_selector selector;
        auto dev = util::get_cts_object::device(selector);
        if (supports_partition_property(
                dev, cl::sycl::info::partition_property::partition_by_counts)) {
          cl::sycl::vector_class<size_t> devicePartitionCounts;
          devicePartitionCounts.push_back(3);
          devicePartitionCounts.push_back(1);
          auto subDevices = dev.create_sub_devices<
              cl::sycl::info::partition_property::partition_by_counts>(
              devicePartitionCounts);
          check_return_type<cl::sycl::vector_class<cl::sycl::device>>(
              log, subDevices,
              "device::create_sub_devices(vector_class<size_t>)");
        }
      }

      /** check
       * create_sub_devices<info::partition_property::partition_by_affinity_domain>(affinity_domain)
       * member function
      */
      {
        cts_selector selector;
        auto dev = util::get_cts_object::device(selector);
        cl::sycl::info::partition_property partitionProperty =
            cl::sycl::info::partition_property::partition_by_affinity_domain;
        cl::sycl::info::partition_affinity_domain affinityDomain =
            cl::sycl::info::partition_affinity_domain::next_partitionable;
        if (supports_partition_property(dev, partitionProperty)) {
          if (supports_affinity_domain(dev, partitionProperty,
                                       affinityDomain)) {
            auto subDevices =
                dev.create_sub_devices<cl::sycl::info::partition_property::
                                           partition_by_affinity_domain>(
                    affinityDomain);
            check_return_type<cl::sycl::vector_class<cl::sycl::device>>(
                log, subDevices,
                "device::create_sub_device(info::partition_affinity_domain)");
          }
        }
      }

      /** check get_devices() static member function
      */
      {
        auto devs = cl::sycl::device::get_devices();
        check_return_type<cl::sycl::vector_class<cl::sycl::device>>(
            log, devs, "device::get_devices()");
      }

      /** check get_devices(info::device_type::all) static member function
      */
      {
        auto devs =
            cl::sycl::device::get_devices(cl::sycl::info::device_type::all);
        check_return_type<cl::sycl::vector_class<cl::sycl::device>>(
            log, devs, "device::get_devices(info::device_type::all)");
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_api__ */
