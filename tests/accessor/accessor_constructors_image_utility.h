/*************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
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
template <typename accTag>
class check_accessor_constructor_image {
public:
  template <typename ... handlerArgsT>
  static void check(cl::sycl::image<accTag::dataDims> &image,
                    cl::sycl::range<accTag::dataDims> range,
                    sycl_cts::util::logger &log,
                    const std::string& constructorName,
                    const std::string& typeName,
                    handlerArgsT&& ... handler) {
    // construct the accessor
    typename accTag::type accessor(image,
                                   std::forward<handlerArgsT>(handler)...);

    // check the accessor
    check_accessor_members<accTag>::check(
        log, accessor, constructorName, typeName,
        accessor_members::count{range.size()},
        accessor_members::range<accTag::dataDims>{range});
  }
};

/** @brief Checks all image accessor constructors available
 */
template <typename T, size_t dims, cl::sycl::access::target target>
class check_all_accessor_constructors_image {
public:
  template <cl::sycl::access::mode mode,
            int imageDims, typename ... handlerArgsT>
  static void check(cl::sycl::image<imageDims> &image,
                    cl::sycl::range<imageDims> range,
                    sycl_cts::util::logger &log,
                    const std::string& typeName,
                    handlerArgsT&& ... handler) {
    using accTag = accessor_type_info<T, dims, mode, target>;
    using verifier = check_accessor_constructor_image<accTag>;

    constexpr bool usesHander = sizeof...(handlerArgsT) != 0;
    {
      const auto constructorName = usesHander ?
          "constructor(image, handler)" :
          "constructor(image)";
      verifier::check(image, range,
                      log, constructorName, typeName, handler...);
    }
  }
};

/** @brief Check common-by-reference semantics
 */
template <typename T, size_t dims, cl::sycl::access::target target>
class check_accessor_common_by_reference_image {
public:
  template <cl::sycl::access::mode mode,
            int imageDims, typename ... handlerArgsT>
  static void check(cl::sycl::image<imageDims> &image,
                    cl::sycl::image<imageDims> &image2,
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
template <typename T, size_t dims>
class image_accessor_dims {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue,
                    const std::string& typeName) {
    int size = 32;
    auto range =
        sycl_cts::util::get_cts_object::range<dims>::get(size, size,size);
    std::vector<cl_float> data(range.size() * 4, 0.0f);
    cl::sycl::image<dims> image(data.data(),
                                cl::sycl::image_channel_order::rgba,
                                cl::sycl::image_channel_type::fp32, range);
    std::vector<cl_float> data2(range.size() * 4, 0.0f);
    cl::sycl::image<dims> image2(data2.data(),
                                 cl::sycl::image_channel_order::rgba,
                                 cl::sycl::image_channel_type::fp32, range);

    /** check image accessor constructors for image
     */
    {
      constexpr auto target = cl::sycl::access::target::image;
      using verifier =
          check_all_accessor_constructors_image<T, dims, target>;
      using semantics_verifier =
          check_accessor_common_by_reference_image<T, dims, target>;

      queue.submit([&](cl::sycl::handler &h) {
        /** check image constructors for different modes
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          verifier::template check<mode>(image, range, log, typeName, h);
        }
        {
          constexpr auto mode = cl::sycl::access::mode::write;
          verifier::template check<mode>(image, range, log, typeName, h);
        }
        {
          constexpr auto mode = cl::sycl::access::mode::discard_write;
          verifier::template check<mode>(image, range, log, typeName, h);
        }
        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          semantics_verifier::template check<mode>(image, image2,
                                                   log, typeName, h);
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor<T, cl::sycl::access::target::image>{});
      });
      queue.wait_and_throw();
    }

    /** check host_image accessor constructors for host_image
     */
    {
      constexpr auto target = cl::sycl::access::target::host_image;
      using verifier =
          check_all_accessor_constructors_image<T, dims, target>;
      using semantics_verifier =
          check_accessor_common_by_reference_image<T, dims, target>;

      /** check host_image constructor for different modes
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        verifier::template check<mode>(image, range, log, typeName);
      }
      {
        constexpr auto mode = cl::sycl::access::mode::write;
        verifier::template check<mode>(image, range, log, typeName);
      }
      {
        constexpr auto mode = cl::sycl::access::mode::discard_write;
        verifier::template check<mode>(image, range, log, typeName);
      }
      /** check common-by-reference semantics
       */
      {
          constexpr auto mode = cl::sycl::access::mode::read;
          semantics_verifier::template check<mode>(image, image2,
                                                   log, typeName);
      }
    }
  }
};

/** @brief Used to test the image array accessor combinations
 */
template <typename T, size_t dims>
class image_array_accessor_dims {
  static constexpr auto target = cl::sycl::access::target::image_array;
  static constexpr size_t dataDims = acc_data_dims<target, dims>::get();
public:
  static void check(util::logger &log, cl::sycl::queue &queue,
                    const std::string& typeName) {
    int size = 32;
    auto range =
        sycl_cts::util::get_cts_object::range<dataDims>::get(size, size, size);
    std::vector<cl_float> data(range.size() * 4, 0.0f);
    cl::sycl::image<dataDims> image(data.data(),
                                    cl::sycl::image_channel_order::rgba,
                                    cl::sycl::image_channel_type::fp32, range);
    std::vector<cl_float> data2(range.size() * 4, 0.0f);
    cl::sycl::image<dataDims> image2(data2.data(),
                                     cl::sycl::image_channel_order::rgba,
                                     cl::sycl::image_channel_type::fp32, range);

    /** check image array accessor constructors for image
     */
    {
      using verifier =
          check_all_accessor_constructors_image<T, dims, target>;
      using semantics_verifier =
          check_accessor_common_by_reference_image<T, dims, target>;

      queue.submit([&](cl::sycl::handler &h) {
        /** check image array constructor for different modes
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          verifier::template check<mode>(image, range, log, typeName, h);
        }
        {
          constexpr auto mode = cl::sycl::access::mode::write;
          verifier::template check<mode>(image, range, log, typeName, h);
        }
        {
          constexpr auto mode = cl::sycl::access::mode::discard_write;
          verifier::template check<mode>(image, range, log, typeName, h);
        }
        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          semantics_verifier::template check<mode>(image, image2,
                                                   log, typeName, h);
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(
            dummy_functor<T, cl::sycl::access::target::image_array>{});
      });
      queue.wait_and_throw();
    }
  }
};

}  // namespace accessor_utility__

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_IMAGE_UTILITY_H
