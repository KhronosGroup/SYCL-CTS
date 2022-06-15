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
template <>
struct StringMaker<sycl::access_mode> {
  using type = sycl::access_mode;
  static std::string convert(type value) {
    switch (value) {
      case type::read:
        return "access_mode::read";
      case type::write:
        return "access_mode::write";
      case type::read_write:
        return "access_mode::read_write";
      case type::discard_write:
        return "access_mode::discard_write (deprecated)";
      case type::discard_read_write:
        return "access_mode::discard_read_write (deprecated)";
      case type::atomic:
        return "access_mode::atomic (deprecated)";
      default:
        return "unknown access mode";
    }
  }
};

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

template <>
struct StringMaker<sycl::target> {
  using type = sycl::target;
  static std::string convert(type value) {
    switch (value) {
      case type::device:
        return "target::device";

#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)
      case type::host_task:
        return "target::host_task";
#endif

      case type::constant_buffer:
        return "target::constant_buffer (deprecated)";
      case type::local:
        return "target::local (deprecated)";
      case type::host_buffer:
        return "target::host_buffer (deprecated)";
      default:
        return "unknown target";
    }
  }
};

template <>
struct StringMaker<sycl::access::address_space> {
  using type = sycl::access::address_space;
  static std::string convert(type value) {
    switch (value) {
      case type::global_space:
        return "access::address_space::global_space";
      case type::local_space:
        return "access::address_space::local_space";
      case type::private_space:
        return "access::address_space::private_space";
      case type::generic_space:
        return "access::address_space::generic_space";
      default:
        // no stringification for deprecated ones
        return "unknown or deprecated address_space";
    }
  }
};

template <>
struct StringMaker<sycl::access::decorated> {
  using type = sycl::access::decorated;
  static std::string convert(type value) {
    switch (value) {
      case type::yes:
        return "access::decorated::yes";
      case type::no:
        return "access::decorated::no";
      case type::legacy:
        return "access::decorated::legacy";
      default:
        return "unknown";
    }
  }
};
}  // namespace Catch

#endif  // __SYCLCTS_TESTS_COMMON_STRING_MAKERS_H
