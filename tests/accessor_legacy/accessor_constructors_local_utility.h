/*************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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
//  This file is a common utility for the implementation of
//  accessor_constructors.cpp and accessor_api.cpp.
//
**************************************************************************/
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_LOCAL_UTILITY_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_LOCAL_UTILITY_H

#include "../common/common.h"
#include "accessor_constructors_utility.h"

#ifndef TEST_NAME
#error Invalid test namespace
#endif

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** @brief Creates a local accessor and checks all its members for correctness
 */
template <typename accTag, typename ... propertyListT>
class check_accessor_constructor_local {
public:
  /** @brief Overload to verify all constructors w/o range
   */
  static void check(sycl_cts::util::logger &log,
                    const std::string& constructorName,
                    const std::string& typeName,
                    sycl::handler& handler,
                    const propertyListT& ... properties) {
    // construct the accessor
    typename accTag::type accessor(handler,
                                   properties...);

    // check the accessor
    check_accessor_members<accTag>::check(
        log, accessor, constructorName, typeName,
        accessor_members::size{sizeof(typename accTag::dataT)},
        accessor_members::count{1});
  }
  /** @brief Overload to verify all constructors with range
   */
  static void check(sycl::range<accTag::dataDims> range,
                    sycl_cts::util::logger &log,
                    const std::string& constructorName,
                    const std::string& typeName,
                    sycl::handler& handler,
                    const propertyListT& ... properties) {
    // construct the accessor
    typename accTag::type accessor(range, handler,
                                   properties...);

    // check the accessor
    check_accessor_members<accTag>::check(
        log, accessor, constructorName, typeName,
        accessor_members::size{range.size() * sizeof(typename accTag::dataT)},
        accessor_members::count{range.size()},
        accessor_members::range<accTag::dataDims>{range});
  }
};

/** @brief Checks all constructors available
 */
template <typename T, size_t dims, sycl::target target>
class check_all_accessor_constructors_local {
public:
  template <sycl::access_mode mode, typename ... rangeArgsT>
  static void check(sycl::handler &handler,
                    sycl_cts::util::logger &log,
                    const std::string& typeName,
                    rangeArgsT&& ... range) {
    using accTag = accessor_type_info<T, dims, mode, target>;

    constexpr bool usesRange = sizeof...(rangeArgsT) != 0;
    {
      using verifier = check_accessor_constructor_local<accTag>;

      const auto constructorName = usesRange ?
          "constructor(range, handler)" :
          "constructor(handler)";
      verifier::check(range...,
                      log, constructorName, typeName, handler);
    }
    {
      using property_list = sycl::property_list;
      using verifier = check_accessor_constructor_local<accTag, property_list>;

      property_list properties {};
      // no specific properties for local accessor in spec

      const auto constructorName = usesRange ?
          "constructor(range, handler, property_list)" :
          "constructor(handler, property_list)";
      verifier::check(range...,
                      log, constructorName, typeName, handler, properties);
    }
  }
};

/** @brief Check common-by-reference semantics
 */
template <typename T, size_t dims, sycl::target target>
class check_accessor_common_by_reference_local {
public:
  template <sycl::access_mode mode, typename ... rangeArgsT>
  static void check(sycl::handler &handler,
                    sycl_cts::util::logger &log,
                    const std::string& typeName,
                    rangeArgsT&& ... range) {
    using accTag = accessor_type_info<T, dims, mode, target>;
    {
      using verifier = check_accessor_copy_constructable<accTag>;

      typename accTag::type srcAccessor(range..., handler);

      verifier::check(srcAccessor, log, typeName);
    }
    {
      using verifier = check_accessor_copy_assignable<accTag>;

      typename accTag::type srcAccessor(range..., handler);
      typename accTag::type dstAccessor(range..., handler);

      verifier::check(srcAccessor, dstAccessor, log, typeName);
    }
    {
      using verifier = check_accessor_move_constructable<accTag>;

      typename accTag::type srcAccessor(range..., handler);

      verifier::check(srcAccessor, log, typeName);
    }
    {
      using verifier = check_accessor_move_assignable<accTag>;

      typename accTag::type srcAccessor(range..., handler);
      typename accTag::type dstAccessor(range..., handler);

      verifier::check(srcAccessor, dstAccessor, log, typeName);
    }
  }
};

/** @brief Used to test the local accessor combinations for n > 0 dimensions
 */
template <typename T, typename kernelName, size_t dims>
class local_accessor_dims {
public:
  static void check(util::logger &log, sycl::queue &queue,
                    const std::string& typeName) {
    int size = 32;
    auto range =
        sycl_cts::util::get_cts_object::range<dims>::get(size, size, size);

    /** check buffer accessor constructors for local
     */
    {
      constexpr auto target = sycl::target::local;
      using verifier = check_all_accessor_constructors_local<T, dims, target>;
      using semantics_verifier =
          check_accessor_common_by_reference_local<T, dims, target>;

      queue.submit([&](sycl::handler &h) {
        /** check local accessor constructor for different modes
         */
        {
          constexpr auto mode = sycl::access_mode::read;
          verifier::template check<mode>(h, log, typeName, range);
        }
        {
          constexpr auto mode = sycl::access_mode::atomic;
          verifier::template check<mode>(h, log, typeName, range);
        }
        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = sycl::access_mode::read_write;
          semantics_verifier::template check<mode>(h, log, typeName, range);
        }

        /** dummy kernel as no kernel is required for these checks
         */
        using dummy =
            dummy_functor<kernelName, sycl::target::local>;
        h.single_task<dummy>(dummy{});
      });
      queue.wait_and_throw();
    }
  }
};

/** @brief Used to test the 0 dimensional local accessor combinations
*/
template <typename T, typename kernelName>
class local_accessor_dims<T, kernelName, 0> {
 public:
  static void check(util::logger &log, sycl::queue &queue,
                    const std::string& typeName) {
    /** check buffer accessor constructors for local
     */
    {
      constexpr auto target = sycl::target::local;
      constexpr size_t dims = 0;
      using verifier = check_all_accessor_constructors_local<T, dims, target>;
      using semantics_verifier =
          check_accessor_common_by_reference_local<T, dims, target>;

      queue.submit([&](sycl::handler &h) {
        /** check local accessor constructor for different modes
         */
        {
          constexpr auto mode = sycl::access_mode::read;
          verifier::template check<mode>(h, log, typeName);
        }
        {
          constexpr auto mode = sycl::access_mode::atomic;
          verifier::template check<mode>(h, log, typeName);
        }
        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = sycl::access_mode::read_write;
          semantics_verifier::template check<mode>(h, log, typeName);
        }

        /** dummy kernel as no kernel is required for these checks
         */
        using dummy =
            dummy_functor<kernelName, sycl::target::local>;
        h.single_task<dummy>(dummy{});
      });
      queue.wait_and_throw();
    }
  }
};

/** @brief Run tests for all local accessor dimensions
 */
template <typename T, typename /*extensionTag*/, typename kernelName>
class local_accessor_all_dims {
public:
  template <typename ... argsT>
  void operator()(argsT&& ... args) {
    local_accessor_dims<T, kernelName, 0>::check(std::forward<argsT>(args)...);
    local_accessor_dims<T, kernelName, 1>::check(std::forward<argsT>(args)...);
    local_accessor_dims<T, kernelName, 2>::check(std::forward<argsT>(args)...);
    local_accessor_dims<T, kernelName, 3>::check(std::forward<argsT>(args)...);
  }
};
}  // namespace accessor_utility__

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_LOCAL_UTILITY_H
