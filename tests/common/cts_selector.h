/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2017-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2021-2022 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
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
