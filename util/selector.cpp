/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "selector.h"
#include "cmdarg.h"

namespace sycl_cts {
namespace util {

/** constructor
 */
selector::selector()
    : m_platform(ctsplat::unknown), m_device(ctsdevice::unknown) {}

void selector::set_default_platform(const std::string &name) {
  if (name == "host")
    m_platform = ctsplat::host;
  else if (name == "amd")
    m_platform = ctsplat::amd;
  else if (name == "arm")
    m_platform = ctsplat::arm;
  else if (name == "intel")
    m_platform = ctsplat::intel;
  else if (name == "nvidia")
    m_platform = ctsplat::nvidia;
}

/** set the default device to use for the SYCL CTS
 *  @param name, the name of the device to use.
 *  valid options are:
 *      'host'
 *      'opencl_cpu'
 *      'opencl_gpu'
 */
void selector::set_default_device(const std::string &name) {
  if (name == "host") {
    m_device = ctsdevice::host;
  }
  if (name == "opencl_cpu") {
    m_device = ctsdevice::opencl_cpu;
  }
  if (name == "opencl_gpu") {
    m_device = ctsdevice::opencl_gpu;
  }
}

/** set the default platform via enum
 */
void selector::set_default_platform(ctsplat platform) { m_platform = platform; }

/** set the default device type via enum
 */
void selector::set_default_device(ctsdevice deviceType) {
  m_device = deviceType;
}

/** return the default platform of choice for this cts run
 */
selector::ctsplat selector::get_default_platform() const {
  // return the cached device type
  return m_platform;
}

/** return the default device of choice for this cts run
 */
selector::ctsdevice selector::get_default_device() const {
  // return the cached device type
  return m_device;
}

}  // namespace util
}  // namespace sycl_cts
