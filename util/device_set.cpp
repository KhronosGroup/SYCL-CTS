/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide generic device sets support for tests that require multiple
//  devices to run
//
*******************************************************************************/

#include "device_set.h"
#include "cpp_compat.h"

#include <stdexcept>

namespace sycl_cts::util {

device_set::device_set(const sycl::context& ctx, util::logger& log)
    : context(ctx) {
  auto vector = context.get_devices();

  devices.reserve(vector.size());
  for (const auto& device : vector) {
    devices.insert(device);
  }
}

void device_set::join(device_set other) {
  if (context != other.context)
    throw std::logic_error("Different contexts on device_set join");

  devices.merge(other.devices);
}

void device_set::substract(const device_set& other) {
  if (context != other.context)
    throw std::logic_error("Different contexts on device_set substract");

  for (const auto& device : other.devices) {
    devices.erase(device);
  }
}

void device_set::intersect(const device_set& other) {
  if (context != other.context)
    throw std::logic_error("Different contexts on device_set intersect");

  auto condition = [&](const StorageType::iterator& it) {
    const auto& device = *it;
    return other.devices.find(device) == other.devices.end();
  };
  erase_if(devices, condition);
}

void device_set::removeDevsWith(sycl::aspect aspect) {
  auto condition = [&](const StorageType::iterator& it) {
    const auto& device = *it;
    return device.has(aspect);
  };
  erase_if(devices, condition);
}

void device_set::removeDevsWith(std::initializer_list<sycl::aspect> aspects) {
  for (auto aspect : aspects) {
    removeDevsWith(aspect);
  }
}

void device_set::removeDevsWithout(sycl::aspect aspect) {
  auto condition = [&](const StorageType::iterator& it) {
    const auto& device = *it;
    return !device.has(aspect);
  };
  erase_if(devices, condition);
}

void device_set::removeDevsWithout(const kernel_restrictions& restriction) {
  auto condition = [&](const StorageType::iterator& it) {
    const auto& device = *it;
    return !restriction.is_compatible(device);
  };
  erase_if(devices, condition);
}

device_set device_set::filtered(const device_set& other, sycl::aspect aspect) {
  device_set result(other);
  result.removeDevsWithout(aspect);
  return result;
}

device_set device_set::filtered(const device_set& other,
                                const kernel_restrictions& restrictions) {
  device_set result(other);
  result.removeDevsWithout(restrictions);
  return result;
}

sycl::context device_set::get_context() const { return context; }

std::vector<sycl::device> device_set::get_devices() const {
  std::vector<sycl::device> result;

  result.reserve(devices.size());
  for (const auto& device : devices) {
    result.push_back(device);
  }
  return result;
}

}  // namespace sycl_cts::util
