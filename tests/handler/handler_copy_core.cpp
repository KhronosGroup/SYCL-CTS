/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests the API for sycl::handler::copy
//
*******************************************************************************/

#include "handler_copy_common.h"

#include "catch2/catch_test_macros.hpp"

#include "../common/type_coverage.h"

#include "../common/string_makers.h"

#include "../common/disabled_for_test_case.h"

namespace handler_copy_core {
using namespace handler_copy_common;

TEST_CASE("Tests the API for sycl::handler::copy", "[handler]") {
  auto queue = util::get_cts_object::queue();

  log_helper lh;

  test_all_variants<int>(lh, queue);

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
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

// FIXME: re-enable when sycl::errc is implemented in computecpp
DISABLED_FOR_TEST_CASE(ComputeCpp)
("Check exception on copy(accessor, accessor) in case of invalid "
 "destination accessor size",
 "[handler]")({
  auto queue = util::get_cts_object::queue();

  const auto types =
      named_type_pack<int
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
                      ,
                      char, short, long, float, sycl::char2, sycl::short3,
                      sycl::int4, sycl::long8, sycl::float8
#endif
                      >::generate("int"
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
                                  ,
                                  "char", "short", "long", "float",
                                  "sycl::char2", "sycl::short3", "sycl::int4",
                                  "sycl::long8", "sycl::float8"
#endif
      );

  const auto dims = value_pack<int, 1, 2, 3>::generate_named(
      "one dim range", "two dim range", "three dim range");

  const auto src_modes =
      value_pack<sycl::access_mode, sycl::access_mode::read,
                 sycl::access_mode::read_write>::generate_named();

  const auto dst_modes =
      value_pack<sycl::access_mode, sycl::access_mode::write,
                 sycl::access_mode::read_write,
                 sycl::access_mode::discard_write,
                 sycl::access_mode::discard_read_write>::generate_named();

  for_all_combinations<CheckCopyAccToAccException>(types, dims, dims, src_modes,
                                                   dst_modes, queue);
});

}  // namespace handler_copy_core
