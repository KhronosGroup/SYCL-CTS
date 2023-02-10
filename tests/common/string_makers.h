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

#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP && \
    !SYCL_CTS_COMPILING_WITH_DPCPP
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
struct StringMaker<sycl::aspect> {
  using type = sycl::aspect;
  static std::string convert(type value) {
    switch (value) {
      case type::cpu:
        return "aspect::cpu";
      case type::gpu:
        return "aspect::gpu";
      case type::accelerator:
        return "aspect::accelerator";
      case type::custom:
        return "aspect::custom";
#ifndef SYCL_CTS_COMPILING_WITH_DPCPP
      case type::emulated:
        return "aspect::emulated";
#endif  // SYCL_CTS_COMPILING_WITH_DPCPP
      case type::host_debuggable:
        return "aspect::host_debuggable";
      case type::fp16:
        return "aspect::fp16";
      case type::fp64:
        return "aspect::fp64";
      case type::atomic64:
        return "aspect::atomic64";
      case type::image:
        return "aspect::image";
      case type::online_compiler:
        return "aspect::online_compiler";
      case type::online_linker:
        return "aspect::online_linker";
      case type::queue_profiling:
        return "aspect::queue_profiling";
      case type::usm_device_allocations:
        return "aspect::usm_device_allocations";
      case type::usm_host_allocations:
        return "aspect::usm_host_allocations";
      case type::usm_atomic_host_allocations:
        return "aspect::usm_atomic_host_allocations";
      case type::usm_shared_allocations:
        return "aspect::usm_shared_allocations";
      case type::usm_atomic_shared_allocations:
        return "aspect::usm_atomic_shared_allocations";
      case type::usm_system_allocations:
        return "aspect::usm_system_allocations";
      default:
        return "unknown aspect";
    }
  }
};

template <>
struct StringMaker<sycl::memory_order> {
  using type = sycl::memory_order;
  static std::string convert(const type& order) {
    switch (order) {
      case type::relaxed:
        return "memory_order::relaxed";
      case type::acq_rel:
        return "memory_order::acq_rel";
      case type::seq_cst:
        return "memory_order::seq_cst";
      case type::acquire:
        return "memory_order::acquire";
      case type::release:
        return "memory_order::release";
      default:
        return "unknown memory_order";
    }
  }
};

template <>
struct StringMaker<sycl::memory_scope> {
  using type = sycl::memory_scope;
  static std::string convert(const type& scope) {
    switch (scope) {
      case type::work_item:
        return "memory_scope::work_item";
      case type::sub_group:
        return "memory_scope::sub_group";
      case type::work_group:
        return "memory_scope::work_group";
      case type::device:
        return "memory_scope::device";
      case type::system:
        return "memory_scope::system";
      default:
        return "unknown memory_scope";
    }
  }
};

template <>
struct StringMaker<sycl::access::address_space> {
  using type = sycl::access::address_space;
  static std::string convert(const type& addr_space) {
    switch (addr_space) {
      case type::global_space:
        return "address_space::global_space";
      case type::local_space:
        return "address_space::local_space";
      case type::generic_space:
        return "address_space::generic_space";
      case type::private_space:
        return "address_space::private_space";
      case type::constant_space:
        return "address_space::constant_space (deprecated)";
      default:
        return "unknown address_space";
    }
  }
};

}  // namespace Catch

#endif  // __SYCLCTS_TESTS_COMMON_STRING_MAKERS_H
