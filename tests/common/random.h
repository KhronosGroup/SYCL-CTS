/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
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

#ifndef __SYCLCTS_TESTS_COMMON_CTS_RANDOM_H
#define __SYCLCTS_TESTS_COMMON_CTS_RANDOM_H

/**
 * Provides compile-time random number generation.
 *
 * Usage:
 * using rng = minstd_rand<seed>;
 * unsigned int rand0 = rng::value;
 * using rng_next = rng::next;
 * unsigned int rand1 = rng_next::value;
 */

/** Equivalent to \p offset invocations of \p engine::next. */
template <typename engine, unsigned int offset>
struct discard {
  using type = typename discard<typename engine::next, offset - 1>::type;
};

template <typename engine>
struct discard<engine, 0> {
  using type = engine;
};

/**
 * Linear congruential engine with a multiplier term \p a,
 * increment term \p c, modulus term \p m, and seed/state \p seed */
template <unsigned int a, unsigned int c, unsigned int m, unsigned int seed = 1>
struct linear_congruential_engine {
  static constexpr unsigned int value = (a * seed + c) % m;
  using next = linear_congruential_engine<a, c, m, value>;
};

/** Compile-time variant of \p std::minstd_rand. */
template <unsigned int seed = 1>
using minstd_rand = linear_congruential_engine<48271, 0, 2147483647, seed>;

#endif  // __SYCLCTS_TESTS_COMMON_CTS_RANDOM_H
