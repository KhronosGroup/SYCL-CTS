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

#include <algorithm>
#include <stdexcept>

namespace sycl_cts::util {

device_set::device_set(const sycl::context& ctx, util::logger&)
    : m_context(ctx) {
  auto devices = ctx.get_devices();

  m_devices.reserve(devices.size());
  std::copy(devices.begin(), devices.end(),
            std::inserter(m_devices, m_devices.end()));
}

void device_set::join(device_set other) {
  if (m_context != other.m_context)
    throw std::logic_error("Different contexts on device_set join");

  m_devices.merge(other.m_devices);
}

void device_set::substract(const device_set& other) {
  if (m_context != other.m_context)
    throw std::logic_error("Different contexts on device_set substract");

  for (const auto& device : other.m_devices) {
    m_devices.erase(device);
  }
}

void device_set::intersect(const device_set& other) {
  if (m_context != other.m_context)
    throw std::logic_error("Different contexts on device_set intersect");

  auto condition = [&](const StorageType::iterator& it) {
    const auto& device = *it;
    return other.m_devices.find(device) == other.m_devices.end();
  };
  erase_if(m_devices, condition);
}

void device_set::removeDevsWith(sycl::aspect aspect) {
  auto condition = [&](const StorageType::iterator& it) {
    const auto& device = *it;
    return device.has(aspect);
  };
  erase_if(m_devices, condition);
}

void device_set::removeDevsWith(std::initializer_list<sycl::aspect> aspects) {
  for (const auto& aspect : aspects) {
    removeDevsWith(aspect);
  }
}

void device_set::removeDevsWithout(sycl::aspect aspect) {
  auto condition = [&](const StorageType::iterator& it) {
    const auto& device = *it;
    return !device.has(aspect);
  };
  erase_if(m_devices, condition);
}

void device_set::removeDevsWithout(const kernel_restrictions& restriction) {
  auto condition = [&](const StorageType::iterator& it) {
    const auto& device = *it;
    return !restriction.is_compatible(device);
  };
  erase_if(m_devices, condition);
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

sycl::context device_set::get_context() const { return m_context; }

std::vector<sycl::device> device_set::get_devices() const {
  std::vector<sycl::device> result;

  result.reserve(m_devices.size());
  std::copy(m_devices.begin(), m_devices.end(),
            std::inserter(result, result.end()));
  return result;
}

}  // namespace sycl_cts::util
