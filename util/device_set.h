/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide generic device sets support for tests that require multiple
//  devices to run
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_DEVICE_SET_H
#define __SYCLCTS_UTIL_DEVICE_SET_H

#include "../tests/common/common.h"
#include "kernel_restrictions.h"

#include <string>
#include <unordered_set>
#include <vector>

namespace sycl_cts::util {

/** @brief Provides toolkit for working with a set of devices
 *  @details For example, to have a greedy tests using all devices with and
 *           without specific aspects we can:
 *               const auto platforms = sycl::platform::get_platforms();
 *               for (const auto& platform: platforms) {
 *                 sycl::context ctx(platform.get_devices());
 *
 *                 auto devs_all = device_set(ctx);
 *                 auto devs_fp16 = device_set::filtered(devs_all,
 *                                                       sycl::aspect::fp16);
 *                 auto devs_fp64 = device_set::filtered(devs_all,
 *                                                       sycl::aspect::fp64);
 *                 auto devs_core = devs_all;
 *                 devs_core.removeDevsWith(sycl::aspect::fp16);
 *                 devs_core.substract(devs_fp64);
 *               }
 */
class device_set {
  using StorageType = std::unordered_set<sycl::device>;
  sycl::context context;
  StorageType devices;

 public:
  device_set() = delete;
  device_set(const sycl::context& ctx, util::logger& log);

  void join(device_set other);
  void substract(const device_set& other);
  void intersect(const device_set& other);

  /** @brief Remove all devices with aspect given
   */
  void removeDevsWith(sycl::aspect aspect);

  /** @brief Remove all devices with aspects given
   */
  void removeDevsWith(std::initializer_list<sycl::aspect> aspects);

  /** @brief Remove all devices without aspect given
   */
  void removeDevsWithout(sycl::aspect aspect);

  /** @brief Keep compatible with kernel_restrictions devices only
   */
  void removeDevsWithout(const kernel_restrictions& restriction);

  /** @brief Make and return filtered version of device_set given
   *  @details Returned device_set contains devices with aspect given
   */
  static device_set filtered(const device_set& other, sycl::aspect aspect);

  /** @brief Make and return filtered version of device_set given
   *  @details Returned device set contains compatible with kernel_restrictions
   *           devices only
   */
  static device_set filtered(const device_set& other,
                             const kernel_restrictions& restrictions);

  sycl::context get_context() const;
  std::vector<sycl::device> get_devices() const;

  auto cbegin() const { return devices.cbegin(); }
  auto cend() const { return devices.cend(); }
};

}  // namespace sycl_cts::util

#endif  // __SYCLCTS_UTIL_DEVICE_SET_H
