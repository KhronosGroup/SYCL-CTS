/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2021-2022 The Khronos Group Inc.
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

#ifndef __SYCLCTS_TESTS_COMMON_CTS_SELECTOR_H
#define __SYCLCTS_TESTS_COMMON_CTS_SELECTOR_H

#include <sycl/sycl.hpp>

#include "../../util/device_manager.h"

#include <regex>

namespace {

/** device selection operator
 *  return <  0  : device will never be selected
 *  return >= 0  : positive device rating
 */
inline int cts_selector(const sycl::device& dev) {
  using namespace sycl_cts;
  using namespace sycl_cts::util;

  auto& device_regex = get<device_manager>().get_device_regex();

  if (!device_regex.has_value()) {
    return sycl::default_selector_v(dev);
  }

  const auto platform_name =
      dev.get_platform().get_info<sycl::info::platform::name>();
  const auto device_name = dev.get_info<sycl::info::device::name>();

  if (std::regex_search(platform_name + " / " + device_name, *device_regex)) {
    return 1000;
  }

  return -1;
}

}  // namespace
#endif  // __SYCLCTS_TESTS_COMMON_CTS_SELECTOR_H
