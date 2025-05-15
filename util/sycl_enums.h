/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Enum stringification for the SYCL CTS
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_SYCL_ENUMS_H
#define __SYCLCTS_UTIL_SYCL_ENUMS_H

#include <catch2/catch_tostring.hpp>
#include <sycl/sycl.hpp>

CATCH_REGISTER_ENUM(sycl::errc, sycl::errc::success, sycl::errc::runtime,
                    sycl::errc::kernel, sycl::errc::accessor,
                    sycl::errc::nd_range, sycl::errc::event,
                    sycl::errc::kernel_argument, sycl::errc::build,
                    sycl::errc::invalid, sycl::errc::memory_allocation,
                    sycl::errc::platform, sycl::errc::profiling,
                    sycl::errc::feature_not_supported,
                    sycl::errc::kernel_not_supported,
                    sycl::errc::backend_mismatch);

CATCH_REGISTER_ENUM(sycl::access::decorated, sycl::access::decorated::yes,
                    sycl::access::decorated::no,
                    sycl::access::decorated::legacy)

CATCH_REGISTER_ENUM(sycl::access::address_space,
                    sycl::access::address_space::global_space,
                    sycl::access::address_space::local_space,
                    sycl::access::address_space::generic_space,
                    sycl::access::address_space::private_space)

CATCH_REGISTER_ENUM(sycl::memory_scope, sycl::memory_scope::work_item,
                    sycl::memory_scope::sub_group,
                    sycl::memory_scope::work_group, sycl::memory_scope::device,
                    sycl::memory_scope::system)

CATCH_REGISTER_ENUM(sycl::memory_order, sycl::memory_order::relaxed,
                    sycl::memory_order::acq_rel, sycl::memory_order::seq_cst,
                    sycl::memory_order::acquire, sycl::memory_order::release)

CATCH_REGISTER_ENUM(sycl::target, sycl::target::device,
                    sycl::target::constant_buffer, sycl::target::local,
                    sycl::target::host_buffer, sycl::target::host_task)

CATCH_REGISTER_ENUM(sycl::access_mode, sycl::access_mode::read,
                    sycl::access_mode::write, sycl::access_mode::read_write,
                    sycl::access_mode::discard_write,
                    sycl::access_mode::discard_read_write,
                    sycl::access_mode::atomic)

CATCH_REGISTER_ENUM(sycl::aspect, sycl::aspect::cpu, sycl::aspect::gpu,
                    sycl::aspect::accelerator, sycl::aspect::custom,
                    sycl::aspect::emulated, sycl::aspect::host_debuggable,
                    sycl::aspect::fp16, sycl::aspect::fp64,
                    sycl::aspect::atomic64, sycl::aspect::image,
                    sycl::aspect::online_compiler, sycl::aspect::online_linker,
                    sycl::aspect::queue_profiling,
                    sycl::aspect::usm_device_allocations,
                    sycl::aspect::usm_host_allocations,
                    sycl::aspect::usm_atomic_host_allocations,
                    sycl::aspect::usm_shared_allocations,
                    sycl::aspect::usm_atomic_shared_allocations,
                    sycl::aspect::usm_system_allocations)

#endif  // __SYCLCTS_UTIL_SYCL_ENUMS_H
