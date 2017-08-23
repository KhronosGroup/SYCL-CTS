/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "image_common.h"
#include "../common/common.h"

#define TEST_NAME image_api

template <int dims>
struct image_kernel_read;
template <int dims>
struct image_kernel_write;

namespace image_api__ {
using namespace sycl_cts;
using namespace cl::sycl;

template <int dims>
class image_ctors {
 public:
  void operator()(util::logger &log, range<dims> &r,
                  range<dims - 1> *p = nullptr) const {
    log.note(
        "Testing image combination: dims[%d], range[%d, %d, %d], pitch[%d]",
        dims, r[0], r[1], r[2], (p != nullptr));

    // This combination is required to be supported by the OpenCL spec
    const auto channelOrder = image_channel_order::rgba;
    const auto channelType = image_channel_type::fp32;

    // Prepare variables
    const auto channelCount = get_channel_order_count(channelOrder);
    const auto channelTypeSize = get_channel_type_size(channelType);
    const auto elementSize = channelTypeSize * channelCount;
    const auto numElems = static_cast<int>(r[0] * r[1] * r[2]);

    // Create image host data
    auto imageHost = get_image_host<dims>(channelTypeSize, channelCount);

    // Creates an image, either with a pitch or without
    // The pitch is not used for a 1D image or when null
    image<dims> img = image_generic<dims>::create_with_pitch(
        static_cast<void *>(imageHost.get()), channelOrder, channelType, r, p);

    // Check get_range()
    if ((!CHECK_VALUE_SCALAR(log, img.get_range()[0], r.get(0))) ||
        (!CHECK_VALUE_SCALAR(log, img.get_range()[1], r.get(1))) ||
        (!CHECK_VALUE_SCALAR(log, img.get_range()[2], r.get(2)))) {
      FAIL(log, "Ranges are not the same.");
    }

    // Check get_pitch()
    if (!image_generic<dims>::compare_pitch(log, img, p)) {
      FAIL(log, "Pitches are not the same.");
    }

    // Check get_size()
    if (img.get_size() < (numElems * elementSize)) {
      string_class message =
          string_class("Sizes are not the same: expected at least ") +
          std::to_string(numElems * elementSize) + ", got " +
          std::to_string(img.get_size());
      FAIL(log, message);
    }

    // Check get_count()
    if (!CHECK_VALUE_SCALAR(log, img.get_count(), numElems)) {
      FAIL(log, "Counts are not the same.");
    }

    // Check get_allocator()
    {
      using AllocatorT = std::allocator<unsigned char>;

      /* create another image with a custom allocator */
      auto imgAlloc =
          image_generic<dims>::template create_with_pitch_and_allocator<
              AllocatorT>(static_cast<void *>(imageHost.get()), channelOrder,
                          channelType, r, p);

      auto allocator = imgAlloc.get_allocator();

      check_return_type<AllocatorT>(log, allocator, "get_allocator()");

      auto ptr = allocator.allocate(1);
      if (ptr == nullptr) {
        FAIL(log, "get_allocator() returned an invalid allocator");
      }
      allocator.deallocate(ptr, 1);
    }

    // Check set_write_back()
    img.set_write_back(false);
    img.set_write_back(true);

    auto queue = util::get_cts_object::queue();

    queue.submit([&](handler &cgh) {
      auto img_acc =
          img.template get_access<float4, cl::sycl::access::mode::read>(cgh);
      auto myKernel = [=](item<1> item) {
        // Read image data using integer coordinates
        float4 dataFromInt = img_acc.read(image_access<dims>::get_int(item));
        // Read image data using floating point coordinates
        float4 dataFromFloat =
            img_acc.read(image_access<dims>::get_float(item));
      };
      cgh.parallel_for<image_kernel_read<dims>>(r, myKernel);
    });

    queue.submit([&](handler &cgh) {
      auto img_acc =
          img.template get_access<float4, cl::sycl::access::mode::write>(cgh);
      auto myKernel = [=](item<1> item) {
        // Write image data
        img_acc.write(image_access<dims>::get_int(item), float4(0.5f));
      };
      cgh.parallel_for<image_kernel_write<dims>>(r, myKernel);
    });

    queue.wait_and_throw();

    // Check image values
    for (int i = 0; i < numElems; i++) {
      if (!CHECK_VALUE(log, imageHost.get()[i], 0.5f, i)) {
        FAIL(log, "Image contains wrong values.");
      }
    }
  }
};

/**
 * test cl::sycl::image initialization
 */
class TEST_NAME : public util::test_base_opencl {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      // Ensure the image always has 64 elements
      const int elemsPerDim1 = 64;
      const int elemsPerDim2 = 8;
      const int elemsPerDim3 = 4;

      range<1> range_1d(elemsPerDim1);
      range<2> range_2d(elemsPerDim2, elemsPerDim2);
      range<3> range_3d(elemsPerDim3, elemsPerDim3, elemsPerDim3);

      // Test without pitch
      {
        image_ctors<1> img_1d;
        image_ctors<2> img_2d;
        image_ctors<3> img_3d;
        img_1d(log, range_1d);
        img_2d(log, range_2d);
        img_3d(log, range_3d);
      }

      // Test with pitch
      {
        range<1> pitch_1d(elemsPerDim2);
        range<2> pitch_2d(elemsPerDim3, elemsPerDim3 * elemsPerDim3);

        image_ctors<2> img_2d;
        image_ctors<3> img_3d;
        img_2d(log, range_2d, &pitch_1d);
        img_3d(log, range_3d, &pitch_2d);
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace image_api__ */
