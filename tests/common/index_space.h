/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2025 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_INDEX_SPACE_H
#define __SYCLCTS_TESTS_COMMON_INDEX_SPACE_H

#include <sycl/sycl.hpp>

/// Linearizes a multi-dimensional index according to the specification.
template <unsigned int dimension>
size_t linearize(sycl::range<dimension> range, sycl::id<dimension> id);

inline size_t linearize(sycl::range<1> range, sycl::id<1> id) {
  static_cast<void>(range);
  return id[0];
}

inline size_t linearize(sycl::range<2> range, sycl::id<2> id) {
  return id[1] + id[0] * range[1];
}

inline size_t linearize(sycl::range<3> range, sycl::id<3> id) {
  return id[2] + id[1] * range[2] + id[0] * range[1] * range[2];
}

/**
Computes a multi-dimensional index such that id = unlinearize(range,
linearize(range, id)) if id is a valid index in range. */
template <unsigned int dimension>
sycl::id<dimension> unlinearize(sycl::range<dimension> range, size_t id);

inline sycl::id<1> unlinearize(sycl::range<1>, size_t id) { return {id}; }

inline sycl::id<2> unlinearize(sycl::range<2> range, size_t id) {
  size_t id0 = id / range[1];
  size_t id1 = id % range[1];
  return {id0, id1};
}

inline sycl::id<3> unlinearize(sycl::range<3> range, size_t id) {
  size_t id0 = id / (range[1] * range[2]);
  size_t rem = id % (range[1] * range[2]);
  size_t id1 = rem / range[2];
  size_t id2 = rem % range[2];
  return {id0, id1, id2};
}

#endif  // __SYCLCTS_TESTS_COMMON_INDEX_SPACE_H
