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

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** Creates an image accessor and checks all its members for correctness.
 */
template <typename accTag>
class check_accessor_constructor_image {
  static constexpr size_t imageDims =
      accTag::dims + (accTag::target == cl::sycl::access::target::image_array);
public:
  template <typename ... handlerArgsT>
  static void check(cl::sycl::image<imageDims> &image,
                    cl::sycl::range<imageDims> range,
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
        accessor_members::count{getElementsCount<imageDims>(range)},
        accessor_members::range<imageDims>{range});
  }
};

/** check accessor is Copy Constructible
  */
template <typename accTag>
class check_image_accessor_copy_constructable {
  static constexpr size_t imageDims =
      accTag::dims + (accTag::target == cl::sycl::access::target::image_array);
public:
  static void check(const typename accTag::type& a,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    auto b{a};

    check_accessor_members<accTag>::check(
        log, b, "copy construction", typeName,
        accessor_members::count{a.get_count()},
        accessor_members::range<imageDims>{a.get_range()});
  }
};

/** check accessor is Copy Assignable
 */
template <typename accTag>
class check_image_accessor_copy_assignable {
  static constexpr size_t imageDims =
      accTag::dims + (accTag::target == cl::sycl::access::target::image_array);
public:
  static void check(const typename accTag::type& a,
                    typename accTag::type& b,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    b = a;

    check_accessor_members<accTag>::check(
        log, b, "copy assignment", typeName,
        accessor_members::count{a.get_count()},
        accessor_members::range<imageDims>{a.get_range()});
  }
};

/** check accessor is Move Constructible
 */
template <typename accTag>
class check_image_accessor_move_constructable {
  static constexpr size_t imageDims =
      accTag::dims + (accTag::target == cl::sycl::access::target::image_array);
public:
  static void check(const typename accTag::type& a,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    auto b{std::move(a)};

    check_accessor_members<accTag>::check(
        log, b, "move construction", typeName,
        accessor_members::count{a.get_count()},
        accessor_members::range<imageDims>{a.get_range()});
  }
};

/** check accessor is Move Assignable
 */
template <typename accTag>
class check_image_accessor_move_assignable {
  static constexpr size_t imageDims =
      accTag::dims + (accTag::target == cl::sycl::access::target::image_array);
public:
  static void check(const typename accTag::type& a,
                    typename accTag::type& b,
                    sycl_cts::util::logger &log,
                    const std::string& typeName) {
    b = std::move(a);

    check_accessor_members<accTag>::check(
        log, b, "move assignment", typeName,
        accessor_members::count{a.get_count()},
        accessor_members::range<imageDims>{a.get_range()});
  }
};

/** Used to test the image accessor combinations for image and host_image
 */
template <typename T, size_t dims>
class image_accessor_dims {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue,
                    const std::string& typeName) {
    int size = 32;
    cl::sycl::range<dims> range = getRange<dims>(size);
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

      queue.submit([&](cl::sycl::handler &h) {
        /** check constructor for reading image
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target>;
          using verifier = check_accessor_constructor_image<accTag>;

          verifier::check(image, range,
                          log, "constructor(image, handler)",
                          typeName,
                          h);
        }
        /** check constructor for writing image
         */
        {
          constexpr auto mode = cl::sycl::access::mode::write;
          using accTag = accessor_type_info<T, dims, mode, target>;
          using verifier = check_accessor_constructor_image<accTag>;

          verifier::check(image, range,
                          log, "constructor(image, handler)",
                          typeName,
                          h);
        }
        /** check constructor for discard_write image
         */
        {
          constexpr auto mode = cl::sycl::access::mode::discard_write;
          using accTag = accessor_type_info<T, dims, mode, target>;
          using verifier = check_accessor_constructor_image<accTag>;

          verifier::check(image, range,
                          log, "constructor(image, handler)",
                          typeName,
                          h);
        }

         /** check common-by-reference semantics
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target>;
          {
            using verifier = check_image_accessor_copy_constructable<accTag>;

            typename accTag::type srcAccessor(image, h);
            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_image_accessor_copy_assignable<accTag>;

            typename accTag::type srcAccessor(image, h);
            typename accTag::type dstAccessor(image, h);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
          {
            using verifier = check_image_accessor_move_constructable<accTag>;

            typename accTag::type srcAccessor(image, h);

            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_image_accessor_move_assignable<accTag>;

            typename accTag::type srcAccessor(image, h);
            typename accTag::type dstAccessor(image, h);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
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

      /** check constructor for reading image
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target>;
        using verifier = check_accessor_constructor_image<accTag>;

        verifier::check(image, range,
                        log, "constructor(image)",
                        typeName);
      }
      /** check constructor for writing image
       */
      {
        constexpr auto mode = cl::sycl::access::mode::write;
        using accTag = accessor_type_info<T, dims, mode, target>;
        using verifier = check_accessor_constructor_image<accTag>;

        verifier::check(image, range,
                        log, "constructor(image)",
                        typeName);
      }
      /** check constructor for discard_write image
       */
      {
        constexpr auto mode = cl::sycl::access::mode::discard_write;
        using accTag = accessor_type_info<T, dims, mode, target>;
        using verifier = check_accessor_constructor_image<accTag>;

        verifier::check(image, range,
                        log, "constructor(image)",
                        typeName);
      }

      /** check common-by-reference semantics
       */
      {
        constexpr auto mode = cl::sycl::access::mode::read;
        using accTag = accessor_type_info<T, dims, mode, target>;
        {
          using verifier = check_image_accessor_copy_constructable<accTag>;

          typename accTag::type srcAccessor(image);
          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_image_accessor_copy_assignable<accTag>;

          typename accTag::type srcAccessor(image);
          typename accTag::type dstAccessor(image);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
        {
          using verifier = check_image_accessor_move_constructable<accTag>;

          typename accTag::type srcAccessor(image);

          verifier::check(srcAccessor, log, typeName);
        }
        {
          using verifier = check_image_accessor_move_assignable<accTag>;

          typename accTag::type srcAccessor(image);
          typename accTag::type dstAccessor(image);

          verifier::check(srcAccessor, dstAccessor, log, typeName);
        }
      }
    }
  }
};

/** Used to test the imagearray accessor combinations
 */
template <typename T, size_t dims>
class image_array_accessor_dims {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue,
                    const std::string& typeName) {
    int size = 32;
    cl::sycl::range<dims + 1> range = getRange<dims + 1>(size);
    std::vector<cl_float> data(range.size() * 4, 0.0f);
    cl::sycl::image<dims + 1> image(data.data(),
                                    cl::sycl::image_channel_order::rgba,
                                    cl::sycl::image_channel_type::fp32, range);

    /** check image array accessor constructors for image
     */
    {
      constexpr auto target = cl::sycl::access::target::image_array;

      queue.submit([&](cl::sycl::handler &h) {
        /** check constructor for reading image
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target>;
          using verifier = check_accessor_constructor_image<accTag>;

          verifier::check(image, range,
                          log, "constructor(image, handler)",
                          typeName,
                          h);
        }
        /** check constructor for writing image
         */
        {
          constexpr auto mode = cl::sycl::access::mode::write;
          using accTag = accessor_type_info<T, dims, mode, target>;
          using verifier = check_accessor_constructor_image<accTag>;

          verifier::check(image, range,
                          log, "constructor(image, handler)",
                          typeName,
                          h);
        }
        /** check constructor for discard_write image
         */
        {
          constexpr auto mode = cl::sycl::access::mode::discard_write;
          using accTag = accessor_type_info<T, dims, mode, target>;
          using verifier = check_accessor_constructor_image<accTag>;

          verifier::check(image, range,
                          log, "constructor(image, handler)",
                          typeName,
                          h);
        }

         /** check common-by-reference semantics
         */
        {
          constexpr auto mode = cl::sycl::access::mode::read;
          using accTag = accessor_type_info<T, dims, mode, target>;
          {
            using verifier = check_image_accessor_copy_constructable<accTag>;

            typename accTag::type srcAccessor(image, h);
            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_image_accessor_copy_assignable<accTag>;

            typename accTag::type srcAccessor(image, h);
            typename accTag::type dstAccessor(image, h);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
          {
            using verifier = check_image_accessor_move_constructable<accTag>;

            typename accTag::type srcAccessor(image, h);

            verifier::check(srcAccessor, log, typeName);
          }
          {
            using verifier = check_image_accessor_move_assignable<accTag>;

            typename accTag::type srcAccessor(image, h);
            typename accTag::type dstAccessor(image, h);

            verifier::check(srcAccessor, dstAccessor, log, typeName);
          }
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
