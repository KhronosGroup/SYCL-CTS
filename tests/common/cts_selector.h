/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_CTS_SELECTOR_H
#define __SYCLCTS_TESTS_COMMON_CTS_SELECTOR_H

#include <sycl/sycl.hpp>

#include "../../util/test_manager.h"

#include <regex>

/** test suite specific device selector
 */
class cts_selector : public sycl::device_selector {
 public:
  /** Returns true if the default platform of the selector is host.
   * @return boolean specifying whether the default platform is host. */
  bool is_host() const {
    sycl::device device(*this);
    return device.get_info<sycl::info::device::device_type>() ==
           sycl::info::device_type::host;
  }

  /** device selection operator
   *  return <  0  : device will never be selected
   *  return >= 0  : positive device rating
   */
  virtual int operator()(const sycl::device& dev) const {
    using namespace sycl_cts;
    using namespace sycl_cts::util;

    auto& device_regex = get<test_manager>().get_device_regex();

    if (!device_regex.has_value()) {
      return sycl::default_selector{}(dev);
    }

    const auto platform_name =
        dev.get_platform().get_info<sycl::info::platform::name>();
    const auto device_name = dev.get_info<sycl::info::device::name>();

    if (std::regex_search(platform_name + " / " + device_name, *device_regex)) {
      return 1000;
    }

    return -1;
  }
};

#endif  // __SYCLCTS_TESTS_COMMON_CTS_SELECTOR_H
