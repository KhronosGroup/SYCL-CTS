/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for sycl::link tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SYCL_LINK_H
#define __SYCLCTS_TESTS_SYCL_LINK_H

#include "../common/common.h"
#include "kernel_bundle.h"
#include "kernels.h"
#include <vector>

namespace sycl_cts {
namespace tests {
namespace sycl_link {

using vector_with_object_bundles =
    std::vector<sycl::kernel_bundle<sycl::bundle_state::object>>;

template <sycl::bundle_state BundleState>
class TestCaseDescription
    : public sycl_cts::tests::kernel_bundle::TestCaseDescriptionBase<
          BundleState> {
 public:
  constexpr TestCaseDescription(std::string_view functionOverload)
      : sycl_cts::tests::kernel_bundle::TestCaseDescriptionBase<BundleState>(
            "sycl::link", functionOverload){};
};

}  // namespace sycl_link
}  // namespace tests
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_SYCL_LINK_H
