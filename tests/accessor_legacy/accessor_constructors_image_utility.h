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
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_IMAGE_UTILITY_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_IMAGE_UTILITY_H

#include "../common/common.h"
#include "accessor_constructors_utility.h"

#ifndef TEST_NAME
#error Invalid test namespace
#endif

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** @brief Creates an image accessor and checks all its members for correctness
 */
template <typename accTag, typename ... propertyListT>
class check_accessor_constructor_image {
public:
  template <typename allocatorT, typename ... handlerArgsT>
  static void check(sycl::image<accTag::dataDims, allocatorT> &image,
                    sycl::range<accTag::dataDims> range,
                    sycl_cts::util::logger &log,
                    const std::string& constructorName,
                    const std::string& typeName,
                    const propertyListT& ... properties,
                    handlerArgsT&& ... handler) {
    // construct the accessor
    typename accTag::type accessor(image,
                                   std::forward<handlerArgsT>(handler)...,
                                   properties...);

    // check the accessor
    check_accessor_members<accTag>::check(
        log, accessor, constructorName, typeName,
        accessor_members::count{range.size()},
        accessor_members::range<accTag::dataDims>{range});
  }
};

/** @brief Checks all image accessor constructors available
 */
template <typename T, size_t dims, sycl::target target>
class check_all_accessor_constructors_image {
public:
  template <sycl::access_mode mode,
            int imageDims, typename allocatorT, typename ... handlerArgsT>
  static void check(sycl::image<imageDims, allocatorT> &image,
                    sycl::range<imageDims> range,
                    sycl_cts::util::logger &log,
                    const std::string& typeName,
                    handlerArgsT&& ... handler) {
    using accTag = accessor_type_info<T, dims, mode, target>;

    constexpr bool usesHander = sizeof...(handlerArgsT) != 0;
    {
      using verifier = check_accessor_constructor_image<accTag>;

      const auto constructorName = usesHander ?
          "constructor(image, handler)" :
          "constructor(image)";
      verifier::check(image, range,
                      log, constructorName, typeName, handler...);
    }
    {
      using property_list = sycl::property_list;
      using verifier = check_accessor_constructor_image<accTag, property_list>;

      auto context = util::get_cts_object::context();
      property_list properties {
          sycl::property::buffer::context_bound(context)};

      const auto constructorName = usesHander ?
          "constructor(image, handler, property_list)" :
          "constructor(image, property_list)";
      verifier::check(image, range,
                      log, constructorName, typeName, properties, handler...);
    }
  }
};

/** @brief Check common-by-reference semantics
 */
template <typename T, size_t dims, sycl::target target>
class check_accessor_common_by_reference_image {
public:
  template <sycl::access_mode mode,
            int imageDims, typename allocatorT, typename ... handlerArgsT>
  static void check(sycl::image<imageDims, allocatorT> &image,
                    sycl::image<imageDims, allocatorT> &image2,
                    sycl_cts::util::logger &log,
                    const std::string& typeName,
                    handlerArgsT&& ... handler) {
    using accTag = accessor_type_info<T, dims, mode, target>;
    {
      using verifier = check_accessor_copy_constructable<accTag>;

      typename accTag::type srcAccessor(image, handler...);

      verifier::check(srcAccessor, log, typeName);
    }
    {
      using verifier = check_accessor_copy_assignable<accTag>;

      typename accTag::type srcAccessor(image, handler...);
      typename accTag::type dstAccessor(image2, handler...);

      verifier::check(srcAccessor, dstAccessor, log, typeName);
    }
    {
      using verifier = check_accessor_move_constructable<accTag>;

      typename accTag::type srcAccessor(image, handler...);

      verifier::check(srcAccessor, log, typeName);
    }
    {
      using verifier = check_accessor_move_assignable<accTag>;

      typename accTag::type srcAccessor(image, handler...);
      typename accTag::type dstAccessor(image2, handler...);

      verifier::check(srcAccessor, dstAccessor, log, typeName);
    }
  }
};

/** @brief Used to test the image accessor combinations for image and host_image
 */
template <typename T, size_t dims, typename ... allocatorT>
class image_accessor_dims {
  using image_t = sycl::image<dims, allocatorT...>;
 public:
  static void check(util::logger &log, sycl::queue &queue,
                    const std::string& typeName) {
    int size = 32;
    auto range =
        sycl_cts::util::get_cts_object::range<dims>::get(size, size,size);
    std::vector<cl_float> data(range.size() * 4, 0.0f);
    image_t image(data.data(),
                  sycl::image_channel_order::rgba,
                  sycl::image_channel_type::fp32, range);
    std::vector<cl_float> data2(range.size() * 4, 0.0f);
    image_t image2(data2.data(),
                   sycl::image_channel_order::rgba,
                   sycl::image_channel_type::fp32, range);

    /** check image accessor constructors for image
     */
    {
      constexpr auto target = sycl::target::image;
      using verifier = check_all_accessor_constructors_image<T, dims, target>;
      using semantics_verifier =
          check_accessor_common_by_reference_image<T, dims, target>;

      queue.submit([&](sycl::handler &h) {
        /** check image constructors for different modes
         */
        {
          constexpr auto mode = sycl::access_mode::read;
          verifier::template check<mode>(image, range, log, typeName, h);
        }
        {
          constexpr auto mode = sycl::access_mode::write;
          verifier::template check<mode>(image, range, log, typeName, h);
        }
        {
          constexpr auto mode = sycl::access_mode::discard_write;
          verifier::template check<mode>(image, range, log, typeName, h);
        }
        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = sycl::access_mode::read;
          semantics_verifier::template check<mode>(image, image2,
                                                   log, typeName, h);
        }

        /** dummy kernel as no kernel is required for these checks
         */
        using dummy = dummy_functor<T, sycl::target::image>;
        h.single_task<dummy>(dummy{});
      });
      queue.wait_and_throw();
    }

    /** check host_image accessor constructors for host_image
     */
    {
      constexpr auto target = sycl::target::host_image;
      using verifier =
          check_all_accessor_constructors_image<T, dims, target>;
      using semantics_verifier =
          check_accessor_common_by_reference_image<T, dims, target>;

      /** check host_image constructor for different modes
       */
      {
        constexpr auto mode = sycl::access_mode::read;
        verifier::template check<mode>(image, range, log, typeName);
      }
      {
        constexpr auto mode = sycl::access_mode::write;
        verifier::template check<mode>(image, range, log, typeName);
      }
      {
        constexpr auto mode = sycl::access_mode::discard_write;
        verifier::template check<mode>(image, range, log, typeName);
      }
      /** check common-by-reference semantics
       */
      {
          constexpr auto mode = sycl::access_mode::read;
          semantics_verifier::template check<mode>(image, image2,
                                                   log, typeName);
      }
    }
  }
};

/** @brief Used to test the image array accessor combinations
 */
template <typename T, size_t dims, typename ... allocatorT>
class image_array_accessor_dims {
  static constexpr auto target = sycl::target::image_array;
  static constexpr size_t dataDims = acc_data_dims<target, dims>::get();
  using image_t = sycl::image<dataDims, allocatorT...>;
public:
  static void check(util::logger &log, sycl::queue &queue,
                    const std::string& typeName) {
    int size = 32;
    auto range =
        sycl_cts::util::get_cts_object::range<dataDims>::get(size, size, size);
    std::vector<cl_float> data(range.size() * 4, 0.0f);
    image_t image(data.data(),
                  sycl::image_channel_order::rgba,
                  sycl::image_channel_type::fp32, range);
    std::vector<cl_float> data2(range.size() * 4, 0.0f);
    image_t image2(data2.data(),
                   sycl::image_channel_order::rgba,
                   sycl::image_channel_type::fp32, range);

    /** check image array accessor constructors for image
     */
    {
      using verifier = check_all_accessor_constructors_image<T, dims, target>;
      using semantics_verifier =
          check_accessor_common_by_reference_image<T, dims, target>;

      queue.submit([&](sycl::handler &h) {
        /** check image array constructor for different modes
         */
        {
          constexpr auto mode = sycl::access_mode::read;
          verifier::template check<mode>(image, range, log, typeName, h);
        }
        {
          constexpr auto mode = sycl::access_mode::write;
          verifier::template check<mode>(image, range, log, typeName, h);
        }
        {
          constexpr auto mode = sycl::access_mode::discard_write;
          verifier::template check<mode>(image, range, log, typeName, h);
        }
        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = sycl::access_mode::read;
          semantics_verifier::template check<mode>(image, image2,
                                                   log, typeName, h);
        }

        /** dummy kernel as no kernel is required for these checks
         */
        using dummy = dummy_functor<T, sycl::target::image_array>;
        h.single_task<dummy>(dummy{});
      });
      queue.wait_and_throw();
    }
  }
};
/** @brief Run tests for all image accessor dimensions and targets
 */
template <typename T, typename ... allocatorT, typename ... argsT>
void image_accessor_allocator(argsT&& ... args) {
  image_accessor_dims<T, 1, allocatorT...>::check(
      std::forward<argsT>(args)...);
  image_accessor_dims<T, 2, allocatorT...>::check(
      std::forward<argsT>(args)...);
  image_accessor_dims<T, 3, allocatorT...>::check(
      std::forward<argsT>(args)...);
  image_array_accessor_dims<T, 1, allocatorT...>::check(
      std::forward<argsT>(args)...);
  image_array_accessor_dims<T, 2, allocatorT...>::check(
      std::forward<argsT>(args)...);
}

template <typename T, typename /*extensionTag*/>
class image_accessor_type {
public:
  template <typename ... argsT>
  void operator()(argsT&& ... args) {
    using user_allocator = std::allocator<T>;

    image_accessor_allocator<T>(std::forward<argsT>(args)...);
    image_accessor_allocator<T, user_allocator>(std::forward<argsT>(args)...);
  }
};
}  // namespace accessor_utility__

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_IMAGE_UTILITY_H
