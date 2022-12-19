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

#ifndef SYCL_CTS_TEST_MARRAY_MARRAY_COMMON_H
#define SYCL_CTS_TEST_MARRAY_MARRAY_COMMON_H

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "../common/type_list.h"
#include "marray_custom_type.h"

namespace marray_common {

/** @brief Execute the tests with the same values for NumElements. */
inline auto get_num_elements() {
  return value_pack<int, 1, 2, 3, 4, 5, 8, 16, 32, 64>::generate_unnamed();
}

/** @brief Execute the tests with the same values for DataT. */
inline auto get_types() {
#ifdef SYCL_CTS_COMPILING_WITH_COMPUTECPP
  WARN(
      "ComputeCPP does not support custom types other than sycl::half."
      "Skipping the test case for custom types.");
#endif
  return named_type_pack<
      char, int, float, std::int8_t, std::int32_t
  // does not work with any custom type other than sycl::half
#ifndef SYCL_CTS_COMPILING_WITH_COMPUTECPP
      ,
      custom_int
#endif
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
      ,
      unsigned char, short, unsigned short, unsigned int, long, unsigned long,
      long long, unsigned long long, bool, std::uint8_t, std::int16_t,
      std::uint16_t, std::uint32_t, std::int64_t, std::uint64_t
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
      >::generate("char", "int", "float", "std::int8_t", "std::int32_t"
#ifndef SYCL_CTS_COMPILING_WITH_COMPUTECPP
                  ,
                  "custom_type"
#endif
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
                  ,
                  "unsigned char", "short", "unsigned short", "unsigned int",
                  "long", "unsigned long", "long long", "unsigned long long",
                  "bool", "std::uint8_t", "std::int16_t", "std::uint16_t",
                  "std::uint32_t", "std::int64_t", "std::uint64_t"
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  );
}

/** @brief Helper for constexpr constructor. */
template <typename DataT, std::size_t NumElements, int InitialValue = 0>
struct ctor {
  using marray_t = sycl::marray<DataT, NumElements>;

  template <int N, int... Rest>
  struct ctor_impl {
    static constexpr auto& value = ctor_impl<N - 1, N, Rest...>::value;
  };

  template <int... Rest>
  struct ctor_impl<InitialValue, Rest...> {
    static constexpr marray_t value{InitialValue, Rest...};
  };

  template <std::size_t num_elements>
  struct ctor_ {
    static_assert(num_elements != 0);
    static constexpr marray_t value =
        ctor_impl<InitialValue + num_elements - 1>::value;
  };

  static constexpr marray_t value = ctor_<NumElements>::value;
};

}  // namespace marray_common

#endif  // SYCL_CTS_TEST_MARRAY_MARRAY_COMMON_H
