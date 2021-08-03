/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
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
class cts_selector : public cl::sycl::device_selector {
 public:
  /** Checks whether the default device is a host device
   * @return True if the default device is host
   */
  bool is_host() const {
    using namespace sycl_cts::util;

    return (get<selector>().get_default_device() == selector::ctsdevice::host);
  }

  /**
   * Scores devices according to the type requested by the CTS
   * @param type, the type of device currently being scored
   * @param desired, the device type requested by the selector class
   * @result = an integer score
   */
  int score(cl::sycl::info::device_type type,
            sycl_cts::util::selector::ctsdevice desired) const {
    using namespace sycl_cts::util;

    int result = -1;

    // type == all device matches all, return early
    if (type == cl::sycl::info::device_type::all) {
      result = 1000;
    }

    switch (desired) {
      case selector::ctsdevice::host:
        if (type == cl::sycl::info::device_type::host) {
          result = 1000;
        }
        break;
      case selector::ctsdevice::opencl_cpu:
        if (type == cl::sycl::info::device_type::cpu) {
          result = 1000;
        }
        break;
      case selector::ctsdevice::opencl_gpu:
        if (type == cl::sycl::info::device_type::gpu) {
          result = 1000;
        }
        break;
      case selector::ctsdevice::opencl_accelerator:
        if (type == cl::sycl::info::device_type::accelerator) {
          result = 1000;
        }
        break;
      case selector::ctsdevice::custom:
        if (type == cl::sycl::info::device_type::custom) {
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
  virtual int operator()(const cl::sycl::device &dev) const {
    using namespace sycl_cts;
    using namespace sycl_cts::util;

    selector::ctsdevice ctsDevType = get<selector>().get_default_device();

    cl::sycl::info::device_type type =
        dev.get_info<cl::sycl::info::device::device_type>();

    int result = score(type, ctsDevType);

    return result;
  }
};

#endif  // __SYCLCTS_TESTS_COMMON_CTS_SELECTOR_H
