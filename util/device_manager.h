/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2020-2022 The Khronos Group Inc.
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

#ifndef __SYCLCTS_UTIL_TEST_MANAGER_H
#define __SYCLCTS_UTIL_TEST_MANAGER_H

#include "singleton.h"

#include <optional>
#include <regex>

namespace sycl_cts {
namespace util {

class device_manager : public singleton<device_manager> {
 public:
  void set_device_regex(std::regex re) { device_regex = std::move(re); }

  /**
   * @return The regex set by the `--device` CLI parameter, used for selecting
   * the CTS device.
   */
  const std::optional<std::regex>& get_device_regex() const {
    return device_regex;
  }

  /**
   * Lists all available devices, indicating the currently selected one.
   */
  void list_devices() const;

  /**
   * Dumps information about the device used for this CTS run to a
   * file, to be used by the conformance report generation script.
   */
  void dump_info(const std::string& infoDumpFile);

 private:
  std::optional<std::regex> device_regex;
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_TEST_MANAGER_H
