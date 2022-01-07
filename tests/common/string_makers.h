/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_STRING_MAKERS_H
#define __SYCLCTS_TESTS_COMMON_STRING_MAKERS_H

#include <sstream>

#include <catch2/catch_tostring.hpp>
#include <sycl/sycl.hpp>

namespace Catch {
template <int Dimensions>
struct StringMaker<sycl::id<Dimensions>> {
  static std::string convert(const sycl::id<Dimensions>& id) {
    std::stringstream ss;
    ss << "{";
    for (int d = 0; d < Dimensions; ++d) {
      ss << id[d];
      if (d != Dimensions - 1) {
        ss << ", ";
      }
    }
    ss << "}";
    return ss.str();
  }
};
}  // namespace Catch

#endif  // __SYCLCTS_TESTS_COMMON_STRING_MAKERS_H