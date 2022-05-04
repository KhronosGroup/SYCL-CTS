/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests the API for sycl::handler::copy
//
*******************************************************************************/

#include "handler_copy_common.h"

#include "catch2/catch_test_macros.hpp"

namespace handler_copy_core {
using namespace handler_copy_common;

TEST_CASE("Tests the API for sycl::handler::copy", "[handler]") {
  auto queue = util::get_cts_object::queue();

  log_helper lh;

  test_all_variants<int>(lh, queue);

#if defined(SYCL_CTS_ENABLE_FULL_CONFORMANCE)
  test_all_variants<char>(lh, queue);
  test_all_variants<short>(lh, queue);
  test_all_variants<long>(lh, queue);
  test_all_variants<float>(lh, queue);

  test_all_variants<sycl::char2>(lh, queue);
  test_all_variants<sycl::short3>(lh, queue);
  test_all_variants<sycl::int4>(lh, queue);
  test_all_variants<sycl::long8>(lh, queue);
  test_all_variants<sycl::float8>(lh, queue);
#endif
}

}  // namespace handler_copy_core
