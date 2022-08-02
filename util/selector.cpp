/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#include "selector.h"
#include "cmdarg.h"

namespace sycl_cts {
namespace util {

/** constructor
 */
selector::selector() : m_device(ctsdevice::unknown) {}

/** set the default device to use for the SYCL CTS
 *  @param name, the name of the device to use.
 *  valid options are:
 *      'host'
 *      'opencl_cpu'
 *      'opencl_gpu'
 *      'opencl_accelerator'
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
  if (name == "opencl_accelerator") {
    m_device = ctsdevice::opencl_accelerator;
  }
}

/** set the default device type via enum
 */
void selector::set_default_device(ctsdevice deviceType) {
  m_device = deviceType;
}

/** return the default device of choice for this cts run
 */
selector::ctsdevice selector::get_default_device() const {
  // return the cached device type
  return m_device;
}

}  // namespace util
}  // namespace sycl_cts
