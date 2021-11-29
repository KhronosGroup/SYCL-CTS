/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides descriptor for specific kernel requirements
//
*******************************************************************************/

#include "kernel_restrictions.h"

#include <algorithm>

namespace sycl_cts {
namespace util {

kernel_restrictions::kernel_restrictions() { reset(); }

void kernel_restrictions::set_aspects(const aspect::aspect_set& value) {
  m_aspects = value;
}

void kernel_restrictions::set_sub_group_size(size_t value) {
  sub_group_size.second = value;
  sub_group_size.first = true;
}

void kernel_restrictions::reset() {
  m_aspects = aspect::aspect_set{};
  sub_group_size.first = false;
  work_group_size_dims = 0;
}

void kernel_restrictions::add_aspect(const sycl::aspect& asp) {
  m_aspects.insert(asp);
}

void kernel_restrictions::add_aspects(const aspect::aspect_set& asp) {
  m_aspects.insert(asp.begin(), asp.end());
}

bool kernel_restrictions::is_compatible(const sycl::device& device,
                                        std::string& info) const {
  bool compatible = true;
  // Verify optional aspects support if required
  if (!m_aspects.empty()) {
    aspect::aspect_set incompat_aspects;
    for (const auto aspect : m_aspects) {
      if (!device.has(aspect)) {
        compatible = false;
        incompat_aspects.insert(aspect);
      }
    }
    if (!incompat_aspects.empty()) {
      info += "incompatible with aspects: (" +
              aspect::to_string(incompat_aspects) + "); ";
    }
  }

  // Verify sub_group_size restriction if any
  if (sub_group_size.first) {
    const size_t requested = sub_group_size.second;
    const auto available =
        device.get_info<sycl::info::device::sub_group_sizes>();
    const auto begin = available.begin();
    const auto end = available.end();

    const bool has_sg_size = std::find(begin, end, requested) != end;
    compatible &= has_sg_size;
    if (!has_sg_size) {
      info += "incompatible with sub_group_size: (" +
              std::to_string(requested) + "); ";
    }
  }

  // Verify work_group_size restriction if any
  if (work_group_size_dims > 0) {
    size_t requested = 1;
    for (int dim = 0; dim < work_group_size_dims; ++dim) {
      requested *= work_group_size[dim];
    }

    const auto available =
        device.get_info<sycl::info::device::max_work_group_size>();
    const bool has_wg_size = requested <= available;
    compatible &= has_wg_size;
    if (!has_wg_size) {
      info += "incompatible with work_group_size (" +
              std::to_string(requested) + ");";
    }
  }

  if (compatible) {
    info = "compatible with kernel_restrictions";
  }

  return compatible;
}

bool kernel_restrictions::is_compatible(const sycl::device& device) const {
  std::string dummy_string;
  return is_compatible(device, dummy_string);
}

aspect::aspect_set kernel_restrictions::get_aspects() const {
  return m_aspects;
}

bool kernel_restrictions::has_sub_group_size() const {
  return sub_group_size.first;
}

size_t kernel_restrictions::get_sub_group_size() const {
  return sub_group_size.second;
}

std::string kernel_restrictions::to_string() const {
  std::string result;
  if (!m_aspects.empty()) {
    result += "aspects (" + aspect::to_string(m_aspects) + ") ";
  }

  if (sub_group_size.first) {
    result += "sub_group_size (" + std::to_string(sub_group_size.second) + ") ";
  }

  if (work_group_size_dims > 0) {
    static const std::string delimiter = ",";
    result += "work_group_size (";
    size_t requested = 1;
    for (int dim = 0; dim < work_group_size_dims; ++dim) {
      requested *= work_group_size[dim];
      result += std::to_string(work_group_size[dim]) + delimiter;
    }
    result.resize(result.size() - delimiter.size());
    result += ") = total (" + std::to_string(requested) + ")";
  }

  return result;
}

}  // namespace util
}  // namespace sycl_cts
