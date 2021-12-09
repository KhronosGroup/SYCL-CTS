/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide generic compile-time aspect sets support
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_ASPECT_SET_H
#define __SYCLCTS_UTIL_ASPECT_SET_H

#include "../tests/common/common.h"

#include <set>
#include <string>

namespace sycl_cts::util::aspect {

/** @brief Provides set of aspects to work with
 *  @details Example of usage:
 *             util::aspect::aspect_set set1{
 *               sycl::aspect::cpu,
 *               sycl::aspect::gpu,
 *               sycl::aspect::fp16,
 *               sycl::aspect::fp64
 *             };
 *             util::aspect::aspect_set set2{
 *               sycl::aspect::custom,
 *               sycl::aspect::queue_profiling,
 *               sycl::aspect::fp16,
 *               sycl::aspect::fp64
 *             } ;
 *             util::aspect::aspect_set intersection;
 *             std::set_intersection(set1.begin(), set1.end(),
 *                                   set2.begin(), set2.end(),
 *                                   std::back_inserter(intersection));
 *             log.note(util::aspect::to_string(intersections));
 */
using aspect_set = std::set<sycl::aspect>;

/** @brief Provides string representation of sycl::aspect
 */
std::string to_string(sycl::aspect asp);

/** @brief Provides string representation of util::aspect::aspect_set
 */
std::string to_string(const aspect_set &asp_set);

}  // namespace sycl_cts::util::aspect

#endif  // __SYCLCTS_UTIL_ASPECT_SET_H
