/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define TEST_NAME accessor_constructors_image

#include "../common/common.h"
#include "../accessor/accessor_constructors_utility.h"

namespace TEST_NAMESPACE {
/** unique dummy_functor per file
 *  this is a hack until the CMake script is fixed; kill both the alias and the
 *  dummy class once it is fixed
 */
class dummy_accessor_constructors_image {};
using dummy_functor = ::dummy_functor<dummy_accessor_constructors_image>;

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
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);

          if (a.get_size() < getElementsCount<dims>(range) * elementSize) {
            FAIL(log,
                 "image accessor for read is not constructed correctly "
                 "(get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "image accessor for read is not constructed correctly "
                 "(get_count)");
          }
        }

        /** check (image, handler) constructor for writing image
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::write,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);

          if (a.get_size() < getElementsCount<dims>(range) * elementSize) {
            FAIL(log,
                 "image accessor for write is not constructed correctly "
                 "(get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "image accessor for write is not constructed correctly "
                 "(get_count)");
          }
        }

        /** check (image, handler) constructor for discard_write image
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_write,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);

          if (a.get_size() < getElementsCount<dims>(range) * elementSize) {
            FAIL(log,
                 "image accessor for discard_write is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims>(range)) {
            FAIL(log,
                 "image accessor for discard_write is not constructed "
                 "correctly (get_count)");
          }
        }

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          auto b{a};

          if (a.get_size() < b.get_size()) {
            FAIL(log, "image accessor is not copy constructible (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log, "image accessor is not copy constructible (get_count)");
          }
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

          if (a.get_size() < b.get_size()) {
            FAIL(log, "image accessor is not copy assignable (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log, "image accessor is not copy assignable (get_count)");
          }
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          auto b{std::move(a)};

          if (b.get_size() < getElementsCount<dims>(range) * elementSize) {
            FAIL(log, "image accessor is not move constructible (get_size)");
          }

          if (b.get_count() != getElementsCount<dims>(range)) {
            FAIL(log, "image accessor is not move constructible (get_count)");
          }
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

          if (b.get_size() < getElementsCount<dims>(range) * elementSize) {
            FAIL(log, "image accessor is not move assignable (get_size)");
          }

          if (b.get_count() != getElementsCount<dims>(range)) {
            FAIL(log, "image accessor is not move assignable (get_count)");
          }
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor{});
      });
      queue.wait_and_throw();
    }

    /** check host_image accessor constructors for host_image
     */
    {
      /** check (image, handler) constructor for reading host_image
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);

        if (a.get_size() < getElementsCount<dims>(range) * elementSize) {
          FAIL(log,
               "host image accessor for read is not constructed correctly "
               "(get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host image accessor for read is not constructed correctly "
               "(get_count)");
        }
      }

      /** check (image, handler) constructor for writing host_image
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::write,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);

        if (a.get_size() < getElementsCount<dims>(range) * elementSize) {
          FAIL(log,
               "host image accessor for write is not constructed correctly "
               "(get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host image accessor for write is not constructed correctly "
               "(get_count)");
        }
      }

      /** check (image, handler) constructor for discard_write host_image
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_write,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);

        if (a.get_size() < getElementsCount<dims>(range) * elementSize) {
          FAIL(log,
               "host image accessor for discard_write is not constructed "
               "correctly (get_size)");
        }

        if (a.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host image accessor for discard_write is not constructed "
               "correctly (get_count)");
        }
      }

      /** check accessor is Copy Constructible
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);
        auto b{a};

        if (a.get_size() < b.get_size()) {
          FAIL(log, "host image accessor is not copy constructible (get_size)");
        }

        if (a.get_count() != b.get_count()) {
          FAIL(log,
               "host image accessor is not copy constructible (get_count)");
        }
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

        if (a.get_size() < b.get_size()) {
          FAIL(log, "host image accessor is not copy assignable (get_size)");
        }

        if (a.get_count() != b.get_count()) {
          FAIL(log, "host image accessor is not copy assignable (get_count)");
        }
      }

      /** check accessor is Move Constructible
       */
      {
        cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                           cl::sycl::access::target::host_image,
                           cl::sycl::access::placeholder::false_t>
            a(image);
        auto b{std::move(a)};

        if (b.get_size() < getElementsCount<dims>(range) * elementSize) {
          FAIL(log, "host image accessor is not move constructible (get_size)");
        }

        if (b.get_count() != getElementsCount<dims>(range)) {
          FAIL(log,
               "host image accessor is not move constructible (get_count)");
        }
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

        if (b.get_size() < getElementsCount<dims>(range) * elementSize) {
          FAIL(log, "host image accessor is not move assignable (get_size)");
        }

        if (b.get_count() != getElementsCount<dims>(range)) {
          FAIL(log, "host image accessor is not move assignable (get_count)");
        }
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
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);

          if (a.get_size() < getElementsCount<dims + 1>(range) * elementSize) {
            FAIL(log,
                 "image array accessor for read is not constructed correctly "
                 "(get_size)");
          }

          if (a.get_count() != getElementsCount<dims + 1>(range)) {
            FAIL(log,
                 "image array accessor for read is not constructed correctly "
                 "(get_count)");
          }
        }

        /** check (image, handler) constructor for writing image
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::write,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);

          if (a.get_size() < getElementsCount<dims + 1>(range) * elementSize) {
            FAIL(log,
                 "image array accessor for write is not constructed correctly "
                 "(get_size)");
          }

          if (a.get_count() != getElementsCount<dims + 1>(range)) {
            FAIL(log,
                 "image array accessor for write is not constructed correctly "
                 "(get_count)");
          }
        }

        /** check (image, handler) constructor for discard_write image
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::discard_write,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);

          if (a.get_size() < getElementsCount<dims + 1>(range) * elementSize) {
            FAIL(log,
                 "image array accessor for discard_write is not constructed "
                 "correctly (get_size)");
          }

          if (a.get_count() != getElementsCount<dims + 1>(range)) {
            FAIL(log,
                 "image array accessor for discard_write is not constructed "
                 "correctly (get_count)");
          }
        }

        /** check accessor is Copy Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          auto b{a};

          if (a.get_size() < b.get_size()) {
            FAIL(log,
                 "image array accessor is not copy constructible (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "image array accessor is not copy constructible (get_count)");
          }
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

          if (a.get_size() < b.get_size()) {
            FAIL(log, "image array accessor is not copy assignable (get_size)");
          }

          if (a.get_count() != b.get_count()) {
            FAIL(log,
                 "image array accessor is not copy assignable (get_count)");
          }
        }

        /** check accessor is Move Constructible
         */
        {
          cl::sycl::accessor<T, dims, cl::sycl::access::mode::read,
                             cl::sycl::access::target::image_array,
                             cl::sycl::access::placeholder::false_t>
              a(image, h);
          auto b{std::move(a)};

          if (b.get_size() < getElementsCount<dims + 1>(range) * elementSize) {
            FAIL(log,
                 "image array accessor is not move constructible (get_size)");
          }

          if (b.get_count() != getElementsCount<dims + 1>(range)) {
            FAIL(log,
                 "image array accessor is not move constructible (get_count)");
          }
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

          if (b.get_size() < getElementsCount<dims + 1>(range) * elementSize) {
            FAIL(log, "image array accessor is not move assignable (get_size)");
          }

          if (b.get_count() != getElementsCount<dims + 1>(range)) {
            FAIL(log,
                 "image array accessor is not move assignable (get_count)");
          }
        }

        /** dummy kernel as no kernel is required for these checks
         */
        h.single_task(dummy_functor{});
      });
      queue.wait_and_throw();
    }
  }
};

/** tests the constructors for cl::sycl::accessor
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename T>
  void check_all_dims(util::logger &log, cl::sycl::queue &queue) {
    image_accessor_dims<T, 1>::check(log, queue);
    image_accessor_dims<T, 2>::check(log, queue);
    image_accessor_dims<T, 3>::check(log, queue);
    image_array_accessor_dims<T, 1>::check(log, queue);
    image_array_accessor_dims<T, 2>::check(log, queue);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

      /** check accessor constructors for cl_int4
       */
      check_all_dims<cl::sycl::cl_int4>(log, queue);

      /** check accessor constructors for cl_uint4
       */
      check_all_dims<cl::sycl::cl_uint4>(log, queue);

      /** check accessor constructors for cl_float4
       */
      check_all_dims<cl::sycl::cl_float4>(log, queue);

      queue.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
