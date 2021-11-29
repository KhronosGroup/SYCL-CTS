/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides descriptor for specific kernel requirements
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_KERNEL_RESTRICTIONS_H
#define __SYCLCTS_UTIL_KERNEL_RESTRICTIONS_H

#include "../tests/common/common.h"

#include "aspect_set.h"
#include <string>
#include <utility>

namespace sycl_cts {
namespace util {

/** @brief Descriptor for specific kernel requirements
 *  @details See SYCL2020 rev.3 par. 5.7
 */
class kernel_restrictions {
  /** @brief Stores set of aspects specific kernel requires
   */
  aspect::aspect_set m_aspects;

  std::pair<bool, size_t> sub_group_size;

  size_t work_group_size[3];
  int work_group_size_dims;
 public:
  kernel_restrictions();

  void set_aspects(const aspect::aspect_set& aspects);
  void set_sub_group_size(size_t value);

  template <int dims>
  void set_work_group_size(sycl::id<dims> value) {
    work_group_size_dims = dims;
    for (int i = 0; i < dims; ++i) {
      work_group_size[i] = value[i];
    }
  }

  void add_aspect(const sycl::aspect& asp);

  void add_aspects(const aspect::aspect_set& asp);
  
  void reset();

  bool is_compatible(const sycl::device& device, std::string& info) const;
  bool is_compatible(const sycl::device& device) const;

  aspect::aspect_set get_aspects() const;
  bool has_sub_group_size() const;
  size_t get_sub_group_size() const;

  std::string to_string() const;
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_KERNEL_RESTRICTIONS_H
