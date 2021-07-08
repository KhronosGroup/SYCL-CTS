/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_CTS_SELECTOR_H
#define __SYCLCTS_TESTS_COMMON_CTS_SELECTOR_H

#include "sycl.h"

#include "../../util/selector.h"

/** test suite specific device selector
 */
class cts_selector : public sycl::device_selector {
 public:
  /** Returns true if the default platform of the selector is host.
   * @return boolean specifying whether the default platform is host. */
  bool is_host() const {
    using namespace sycl_cts::util;

    return (get<selector>().get_default_platform() == selector::ctsplat::host);
  }

  /**
   * Scores devices according to the type requested by the CTS
   * @param type, the type of device currently being scored
   * @param desired, the device type requested by the selector class
   * @result = an integer score
   */
  int score(sycl::info::device_type type,
            sycl_cts::util::selector::ctsdevice desired) const {
    using namespace sycl_cts::util;

    int result = -1;

    // type == all device matches all, return early
    if (type == sycl::info::device_type::all) {
      result = 1000;
    }

    switch (desired) {
      case selector::ctsdevice::host:
        if (type == sycl::info::device_type::host) {
          result = 1000;
        }
        break;
      case selector::ctsdevice::opencl_cpu:
        if (type == sycl::info::device_type::cpu) {
          result = 1000;
        }
        break;
      case selector::ctsdevice::opencl_gpu:
        if (type == sycl::info::device_type::gpu) {
          result = 1000;
        }
        break;
      case selector::ctsdevice::opencl_accelerator:
        if (type == sycl::info::device_type::accelerator) {
          result = 1000;
        }
        break;
      case selector::ctsdevice::custom:
        if (type == sycl::info::device_type::custom) {
          result = 1000;
        }
        break;
      // Looking for a device type that doesn't exist, therefore
      // select no devices
      default:
        result = -1;
    }
    // Device does not match the requested type
    return result;
  }

  /** device selection operator
   *  return <  0  : device will never be selected
   *  return >= 0  : positive device rating
   */
  virtual int operator()(const sycl::device &dev) const {
    using namespace sycl_cts;
    using namespace sycl_cts::util;

    selector::ctsdevice ctsDevType = get<selector>().get_default_device();
    selector::ctsplat ctsPlatform = get<selector>().get_default_platform();

    // Early exit for host device
    if (dev.is_host()) {
      if (ctsDevType == selector::ctsdevice::host) {
        return 1000;
      } else {
        return -1;
      }
    }

    sycl::string_class vendor =
        dev.get_platform().get_info<sycl::info::platform::vendor>();
    sycl::info::device_type type =
        dev.get_info<sycl::info::device::device_type>();

    int result = -1;

    switch (ctsPlatform) {
      case selector::ctsplat::amd:
        if (vendor.find("AMD") != std::string::npos) {
          result = score(type, ctsDevType);
        }
        break;
      case selector::ctsplat::arm:
        if (vendor.find("ARM") != std::string::npos) {
          result = score(type, ctsDevType);
        }
        break;
      case selector::ctsplat::intel:
        if (vendor.find("Intel") != std::string::npos) {
          result = score(type, ctsDevType);
        }
        break;
      case selector::ctsplat::nvidia:
        if (vendor.find("NVIDIA") != std::string::npos) {
          result = score(type, ctsDevType);
        }
        break;
      default:
        result = -1;
    }

    return result;
  }
};

#endif  // __SYCLCTS_TESTS_COMMON_CTS_SELECTOR_H
