/*************************************************************************
//
//  SYCL Conformance Test Suite
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

template <typename T, size_t dims, cl::sycl::access::mode kMode,
          cl::sycl::access::target kTarget>
class check_accessor_constructor_image;

template <typename T, size_t dims, cl::sycl::access::mode kMode>
class check_accessor_constructor_image<T, dims, kMode,
                                       cl::sycl::access::target::image> {
 public:
  static void check(cl::sycl::image<dims> &image, cl::sycl::handler &h,
                    cl::sycl::range<dims> range, util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<T, dims, kMode, cl::sycl::access::target::image,
                       cl::sycl::access::placeholder::false_t>
        a(image, h);

    int elementSize = sizeof(cl_float) * 4;  // each element contains 4 channels

    // check the accessor
    check_accessor_members<T, dims, kMode, cl::sycl::access::target::image,
                           cl::sycl::access::placeholder::false_t>::
        check(a, getElementsCount<dims>(range) * elementSize,
              getElementsCount<dims>(range), "constructor(image, handler)",
              log);
  }
};

template <typename T, size_t dims, cl::sycl::access::mode kMode>
class check_accessor_constructor_image<T, dims, kMode,
                                       cl::sycl::access::target::host_image> {
 public:
  static void check(cl::sycl::image<dims> &image, cl::sycl::range<dims> range,
                    util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<T, dims, kMode, cl::sycl::access::target::host_image,
                       cl::sycl::access::placeholder::false_t>
        a(image);

    int elementSize = sizeof(cl_float) * 4;  // each element contains 4 channels

    // check the accessor
    check_accessor_members<T, dims, kMode, cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>::
        check(a, getElementsCount<dims>(range) * elementSize,
              getElementsCount<dims>(range), "constructor(image)", log);
  }
};

template <typename T, size_t dims, cl::sycl::access::mode kMode>
class check_accessor_constructor_image<T, dims, kMode,
                                       cl::sycl::access::target::image_array> {
 public:
  static void check(cl::sycl::image<dims + 1> &image, cl::sycl::handler &h,
                    cl::sycl::range<dims + 1> range, util::logger &log) {
    // construct the accessor
    cl::sycl::accessor<T, dims, kMode, cl::sycl::access::target::image_array,
                       cl::sycl::access::placeholder::false_t>
        a(image, h);

    int elementSize = sizeof(cl_float) * 4;  // each element contains 4 channels

    // check the accessor
    check_accessor_members<T, dims, kMode,
                           cl::sycl::access::target::image_array,
                           cl::sycl::access::placeholder::false_t>::
        check(a, getElementsCount<dims + 1>(range) * elementSize,
              getElementsCount<dims + 1>(range), "constructor(image, handler)",
              log);
  }
};

template <typename T, size_t dims>
class image_accessor_dims {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    int size = 32;
    cl::sycl::range<dims> range = getRange<dims>(size);
    std::vector<cl_float> data(range.size() * 4, 0.0f);
    cl::sycl::image<dims> image(data.data(),
                                cl::sycl::image_channel_order::rgba,
                                cl::sycl::image_channel_type::fp32, range);

    int elementSize = sizeof(cl_float) * 4;  // each element contains 4 channels

    /** check image accessor constructors for image
    */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (image, handler) constructor for reading image
        */
        check_accessor_constructor_image<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::image>::check(image, h, range, log);

        /** check (image, handler) constructor for writing image
        */
        check_accessor_constructor_image<
            T, dims, cl::sycl::access::mode::write,
            cl::sycl::access::target::image>::check(image, h, range, log);

        /** check (image, handler) constructor for discard_write image
        */
        check_accessor_constructor_image<
            T, dims, cl::sycl::access::mode::discard_write,
            cl::sycl::access::target::image>::check(image, h, range, log);

        /** check accessor is Copy Constructible
        */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          auto b{a};

          check_accessor_members<T, dims, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::image,
                                 cl::sycl::access::placeholder::false_t>::
              check(b, a.get_size(), a.get_count(), "copy construction", log);
        }

        /** check accessor is Copy Assignable
        */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              b(image, h);
          b = a;

          check_accessor_members<
              T, dims, cl::sycl::access::mode::read,
              cl::sycl::access::target::image,
              cl::sycl::access::placeholder::false_t>::check(b, a.get_size(),
                                                             a.get_count(),
                                                             "copy assignment",
                                                             log);
        }

        /** check accessor is Move Constructible
        */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          auto b{std::move(a)};

          check_accessor_members<T, dims, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::image,
                                 cl::sycl::access::placeholder::false_t>::
              check(b, image.get_size(), image.get_count(), "move construction",
                    log);
        }

        /** check accessor is Move Assignable
        */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              b(image, h);
          b = std::move(a);

          check_accessor_members<
              T, dims, cl::sycl::access::mode::read,
              cl::sycl::access::target::image,
              cl::sycl::access::placeholder::false_t>::check(b,
                                                             image.get_size(),
                                                             image.get_count(),
                                                             "move assignment",
                                                             log);
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
      /** check (image) constructor for reading host_image
      */
      check_accessor_constructor_image<
          T, dims, cl::sycl::access::mode::read,
          cl::sycl::access::target::host_image>::check(image, range, log);

      /** check (image) constructor for writing host_image
      */
      check_accessor_constructor_image<
          T, dims, cl::sycl::access::mode::write,
          cl::sycl::access::target::host_image>::check(image, range, log);

      /** check (image) constructor for discard_write host_image
      */
      check_accessor_constructor_image<
          T, dims, cl::sycl::access::mode::discard_write,
          cl::sycl::access::target::host_image>::check(image, range, log);

      /** check accessor is Copy Constructible
      */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);
        auto b{a};

        check_accessor_members<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::host_image,
            cl::sycl::access::placeholder::false_t>::check(b, a.get_size(),
                                                           a.get_count(),
                                                           "copy construction",
                                                           log);
      }

      /** check accessor is Copy Assignable
      */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            b(image);
        b = a;

        check_accessor_members<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::host_image,
            cl::sycl::access::placeholder::false_t>::check(b, a.get_size(),
                                                           a.get_count(),
                                                           "copy assignment",
                                                           log);
      }

      /** check accessor is Move Constructible
      */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);
        auto b{std::move(a)};

        check_accessor_members<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::host_image,
            cl::sycl::access::placeholder::false_t>::check(b, image.get_size(),
                                                           image.get_count(),
                                                           "move construction",
                                                           log);
      }

      /** check accessor is Move Assignable
      */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            b(image);
        b = std::move(a);

        check_accessor_members<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::host_image,
            cl::sycl::access::placeholder::false_t>::check(b, image.get_size(),
                                                           image.get_count(),
                                                           "move assignment",
                                                           log);
      }
    }
  }
};

template <typename T, size_t dims>
class image_array_accessor_dims {
 public:
  static void check(util::logger &log, cl::sycl::queue &queue) {
    int size = 32;
    cl::sycl::range<dims + 1> range = getRange<dims + 1>(size);
    std::vector<cl_float> data(range.size() * 4, 0.0f);
    cl::sycl::image<dims + 1> image(data.data(),
                                    cl::sycl::image_channel_order::rgba,
                                    cl::sycl::image_channel_type::fp32, range);
    int elementSize = sizeof(cl_float) * 4;  // each element contains 4 channels

    /** check image array accessor constructors for image
    */
    {
      queue.submit([&](cl::sycl::handler &h) {
        /** check (image, handler) constructor for reading image
        */
        check_accessor_constructor_image<
            T, dims, cl::sycl::access::mode::read,
            cl::sycl::access::target::image_array>::check(image, h, range, log);

        /** check (image, handler) constructor for writing image
        */
        check_accessor_constructor_image<
            T, dims, cl::sycl::access::mode::write,
            cl::sycl::access::target::image_array>::check(image, h, range, log);

        /** check (image, handler) constructor for discard_write image
        */
        check_accessor_constructor_image<
            T, dims, cl::sycl::access::mode::discard_write,
            cl::sycl::access::target::image_array>::check(image, h, range, log);

        /** check accessor is Copy Constructible
        */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          auto b{a};

          check_accessor_members<T, dims, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::image_array,
                                 cl::sycl::access::placeholder::false_t>::
              check(a, b.get_size(), b.get_count(), "copy construction", log);
        }

        /** check accessor is Copy Assignable
        */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              b(image, h);
          b = a;

          check_accessor_members<
              T, dims, cl::sycl::access::mode::read,
              cl::sycl::access::target::image_array,
              cl::sycl::access::placeholder::false_t>::check(b, a.get_size(),
                                                             a.get_count(),
                                                             "copy assignment",
                                                             log);
        }

        /** check accessor is Move Constructible
        */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          auto b{std::move(a)};

          check_accessor_members<T, dims, cl::sycl::access::mode::read,
                                 cl::sycl::access::target::image_array,
                                 cl::sycl::access::placeholder::false_t>::
              check(b, image.get_size(), image.get_count(), "move construction",
                    log);
        }

        /** check accessor is Move Assignable
        */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              b(image, h);
          b = std::move(a);

          check_accessor_members<
              T, dims, cl::sycl::access::mode::read,
              cl::sycl::access::target::image_array,
              cl::sycl::access::placeholder::false_t>::check(b,
                                                             image.get_size(),
                                                             image.get_count(),
                                                             "move assignmnet",
                                                             log);
        }

        /** dummy kernel as no kernel is required for these checks
        */
        h.single_task(dummy_functor<T, cl::sycl::access::target::image_array>{});
      });
      queue.wait_and_throw();
    }
  }
};

}  // namespace accessor_utility__

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_H
