/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for interaction reductions with sycl::half variable type
//  without identity param.
//
*******************************************************************************/

#include "reduction_without_identity_param_common.h"

namespace reduction_without_identity_param_fp16 {
using namespace sycl_cts;
using namespace reduction_without_identity;
using namespace reduction_common;
  
TEST_CASE("reduction_without_identity_param_fp16", "[reduction]") {
    auto queue = util::get_cts_object::queue();
    
    if (!queue.get_device().has(sycl::aspect::fp16)) {
        SKIP("Device does not support half precision floating point operations");
    }
    
    run_tests_for_all_functors<sycl::half, run_test_without_property>()(
        range, queue, "sycl::half");
    run_tests_for_all_functors<sycl::half, run_test_with_property>()(
        nd_range, queue, "sycl::half");
}
} // reduction_without_identity_param_fp16
