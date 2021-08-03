/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_SELECTOR_H
#define __SYCLCTS_UTIL_SELECTOR_H

#include "stl.h"
#include "singleton.h"

namespace sycl_cts {
namespace util {

/** the selector class keeps track of what device type
 *  we have asked the test suite to run with
 */
class selector : public singleton<selector> {
 public:
  /** SYCL device types
   */
  enum class ctsdevice {
    unknown = 0,
    host,
    opencl_cpu,
    opencl_gpu,
    opencl_accelerator,
    custom,
  };

  /** constructor
   */
  selector();

  /** set the default device to use for the SYCL CTS
   *  @param name, the name of the device to use.
   *  valid options are:
   *      'host'
   *      'opencl_cpu'
   *      'opencl_gpu'
   *      'opencl_accelerator'
   */
  void set_default_device(const std::string &name);

  /** set the default device type via enum
   */
  void set_default_device(ctsdevice deviceType);

  /** return a enum of cts_device type specifying the
   *  requested default device type for this run of the cts
   */
  ctsdevice get_default_device() const;

 protected:
  // default SYCL device type to use
  ctsdevice m_device;
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_SELECTOR_H
