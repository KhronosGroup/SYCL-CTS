/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Compile-time tests for the stream constructor added by the
//  sycl_khr_properties extension. The runtime effects are tested in
//  runtime.cpp.
//
//  The extension does not define any property that can be used with the
//  stream class and applications cannot define their own properties, so the
//  only PropertyOrList argument that satisfies the constraints is the empty
//  properties list.
//
*******************************************************************************/

#include "khr_properties_common.h"

#if SYCL_CTS_KHR_PROPERTIES_SPLIT_HEADERS
#include <sycl/khr/split_headers/handler.hpp>
#include <sycl/khr/split_headers/stream.hpp>
#endif

#include <cstddef>

namespace khr_properties::tests {

template <typename P>
inline constexpr bool can_construct_stream_v =
    std::is_constructible_v<sycl::stream, std::size_t, std::size_t,
                            sycl::handler&, P>;

TEST_CASE("the new stream constructor is constrained on PropertyOrList",
          "[khr_properties][stream]") {
  STATIC_CHECK(can_construct_stream_v<khr::empty_properties_t>);
  STATIC_CHECK_FALSE(can_construct_stream_v<prop::in_order>);
  STATIC_CHECK_FALSE(can_construct_stream_v<prop::enable_profiling>);
  STATIC_CHECK_FALSE(can_construct_stream_v<queue_props_t>);
  STATIC_CHECK_FALSE(can_construct_stream_v<not_a_property>);
}

TEST_CASE("only the empty properties list applies to the stream class",
          "[khr_properties][stream][traits]") {
  check_class_without_properties<sycl::stream>();
}

}  // namespace khr_properties::tests
