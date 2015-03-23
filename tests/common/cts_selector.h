/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include "sycl.h"
#include "../../util/selector.h"

/** test suite specific device selector
 */
class cts_selector : public cl::sycl::device_selector {
 public:
  /** device selection operator
   *  return <  0  : device will never be selected
   *  return >= 0  : positive device rating
   */
  virtual int operator()(const cl::sycl::device &dev) const {
    sycl_cts::util::selector::ctsdevice ctsDevType =
        sycl_cts::util::get<sycl_cts::util::selector>().get_default();

    if (dev.is_host()) {
      if (ctsDevType == sycl_cts::util::selector::ctsdevice::host)
        return 1000;
      else
        return 0;
    }

    cl_device_id devid = dev.get();
    cl_device_type devtype;

    cl_int error = clGetDeviceInfo(devid, CL_DEVICE_TYPE,
                                   sizeof(cl_device_type), &devtype, nullptr);

    if (error != CL_SUCCESS) throw "clGetDeviceInfo failed";

    if ((ctsDevType == sycl_cts::util::selector::ctsdevice::opencl_cpu) &&
        (devtype & CL_DEVICE_TYPE_CPU))
      return 1000;

    if ((ctsDevType == sycl_cts::util::selector::ctsdevice::opencl_gpu) &&
        (devtype & CL_DEVICE_TYPE_GPU))
      return 1000;

    return -1;
  }
};
