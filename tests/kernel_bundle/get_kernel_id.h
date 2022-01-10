/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common functions for sycl::get_kernel_id or sycl::get_kernel_id
//  tests.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_GET_KERNEL_ID_H
#define __SYCLCTS_TESTS_GET_KERNEL_ID_H

#include "../../util/exceptions.h"
#include "../common/common.h"

namespace sycl_cts {
namespace tests {
namespace get_kernel_id {

constexpr bool by_handler{true};
constexpr bool by_queue{false};

/** @brief Srorage class for saving kernel_id properties for future checks.
 */
class kernel_id_storage {
  std::string k_id_name;

 public:
  void store(const sycl::kernel_id &k_id) {
    if (k_id.get_name() == nullptr)
      throw util::fail_check(
          "kernel_id::get_name() returned nullptr (in "
          "kernel_id_storage::store()). ");
    k_id_name = k_id.get_name();
  }

  bool check(const sycl::kernel_id &k_id) const {
    if (k_id.get_name() == nullptr)
      throw util::fail_check(
          "kernel_id::get_name() returned nullptr (in "
          "kernel_id_storage::check()). ");
    return k_id.get_name() == k_id_name;
  }
};

/** @brief Run test by sycl::queue or sycl::handler
 */
template <bool ByHandler, typename TestFunctorT>
void run_verification(sycl_cts::util::logger &log, TestFunctorT action) {
  auto queue = sycl_cts::util::get_cts_object::queue();
  if constexpr (ByHandler) {
    queue.submit(action);
  } else {
    action(queue);
  }
}

}  // namespace get_kernel_id
}  // namespace tests
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_GET_KERNEL_ID_H
