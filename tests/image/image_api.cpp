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
template <int dims>
struct image_kernel_get_access;

namespace image_api__ {
using namespace sycl_cts;

template <int dims>
class image_ctors {
 public:
  void operator()(util::logger &log, cl::sycl::range<dims> &r,
                  cl::sycl::range<dims - 1> *p = nullptr) const {
    auto numElems = 1;
    switch (dims) {
      case 1:
        log.note("Testing image combination: dims[%d], range[%d]", dims, r[0]);
        numElems = r[0];
        break;
      case 2:
        log.note(
            "Testing image combination: dims[%d], range[%d, %d], pitch[%d]",
            dims, r[0], r[1], (p != nullptr));
        numElems = r[0] * r[1];
        break;
      case 3:
        log.note(
            "Testing image combination: dims[%d], range[%d, %d, %d], pitch[%d]",
            dims, r[0], r[1], r[2], (p != nullptr));
        numElems = r[0] * r[1] * r[2];
        break;
      default:
        break;
    }

    // This combination is required to be supported by the OpenCL spec
    const auto channelOrder = cl::sycl::image_channel_order::rgba;
    const auto channelType = cl::sycl::image_channel_type::fp32;

    // Prepare variables
    const auto channelCount = get_channel_order_count(channelOrder);
    const auto channelTypeSize = get_channel_type_size(channelType);
    const auto elementSize = channelTypeSize * channelCount;

    // Pitch has to be at least as large as the multiple of element size
    image_generic<dims>::multiply_pitch(p, elementSize);

    // Create image host data
    auto imageHost =
        get_image_host<dims>(numElems, channelTypeSize, channelCount);

    // Creates an image, either with a pitch or without
    // The pitch is not used for a 1D image or when null
    cl::sycl::image<dims> img = image_generic<dims>::create_with_pitch(
        static_cast<void *>(imageHost.data()), channelOrder, channelType, r, p);

    // Check get_range()
    if (dims == 3) {
      if ((!CHECK_VALUE_SCALAR(log, img.get_range()[0], r.get(0))) ||
          (!CHECK_VALUE_SCALAR(log, img.get_range()[1], r.get(1))) ||
          (!CHECK_VALUE_SCALAR(log, img.get_range()[2], r.get(2)))) {
        FAIL(log, "Ranges are not the same.");
      }
    }
    if (dims == 2) {
      if ((!CHECK_VALUE_SCALAR(log, img.get_range()[0], r.get(0))) ||
          (!CHECK_VALUE_SCALAR(log, img.get_range()[1], r.get(1)))) {
        FAIL(log, "Ranges are not the same.");
      }
    }
    if (dims == 1) {
      if ((!CHECK_VALUE_SCALAR(log, img.get_range()[0], r.get(0)))) {
        FAIL(log, "Ranges are not the same.");
      }
    }

    // Check get_pitch()
    if (!image_generic<dims>::compare_pitch(log, img, p)) {
      FAIL(log, "Pitches are not the same.");
    }

    // Check get_size()
    if (img.get_size() < (numElems * elementSize)) {
      cl::sycl::string_class message =
          cl::sycl::string_class("Sizes are not the same: expected at least ") +
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
      using AllocatorT = std::allocator<cl::sycl::byte>;

      /* create another image with a custom allocator */
      auto imgAlloc =
          image_generic<dims>::template create_with_pitch_and_allocator<
              AllocatorT>(static_cast<void *>(imageHost.data()), channelOrder,
                          channelType, r, p);

      auto allocator = imgAlloc.get_allocator();

      check_return_type<AllocatorT>(log, allocator, "get_allocator()");

      auto ptr = allocator.allocate(1);
      if (ptr == nullptr) {
        FAIL(log, "get_allocator() returned an invalid allocator");
      }
      allocator.deallocate(ptr, 1);
    }

    /* Check image properties */
    {
      cl::sycl::mutex_class mutex;
      auto context = util::get_cts_object::context();
      const cl::sycl::property_list propList{
          cl::sycl::property::image::use_host_ptr(),
          cl::sycl::property::image::use_mutex(mutex),
          cl::sycl::property::image::context_bound(context)};

      // Create another image
      auto img2 = cl::sycl::image<dims>(static_cast<void *>(imageHost.data()),
                                        channelOrder, channelType, r, propList);

      /* check has_property() */

      auto hasHostPtrProperty =
          img2.template has_property<cl::sycl::property::image::use_host_ptr>();
      check_return_type<bool>(log, hasHostPtrProperty,
                              "has_property<use_host_ptr>()");

      auto hasUseMutexProperty =
          img2.template has_property<cl::sycl::property::image::use_mutex>();
      check_return_type<bool>(log, hasUseMutexProperty,
                              "has_property<use_mutex>()");

      auto hasContentBoundProperty = img2.template has_property<
          cl::sycl::property::image::context_bound>();
      check_return_type<bool>(log, hasContentBoundProperty,
                              "has_property<context_bound>()");

      /* check get_property() */

      auto hostPtrProperty =
          img2.template get_property<cl::sycl::property::image::use_host_ptr>();
      check_return_type<cl::sycl::property::image::use_host_ptr>(
          log, hostPtrProperty, "get_property<use_host_ptr>()");

      auto useMutexProperty =
          img2.template get_property<cl::sycl::property::image::use_mutex>();
      check_return_type<cl::sycl::property::image::use_mutex>(
          log, useMutexProperty, "get_property<use_mutex>()");
      check_return_type<cl::sycl::mutex_class *>(
          log, useMutexProperty.get_mutex_ptr(),
          "image::use_mutex::get_mutex_ptr()");

      auto contextBoundProperty = img2.template get_property<
          cl::sycl::property::image::context_bound>();
      check_return_type<cl::sycl::property::image::context_bound>(
          log, contextBoundProperty, "get_property<context_bound>()");
      check_return_type<cl::sycl::context>(
          log, contextBoundProperty.get_context(),
          "image::context_bound::get_context()");
    }

    // Check image enum classes
    {
      // Check image channel orders
      check_enum_class_value(cl::sycl::image_channel_order::a);
      check_enum_class_value(cl::sycl::image_channel_order::r);
      check_enum_class_value(cl::sycl::image_channel_order::rx);
      check_enum_class_value(cl::sycl::image_channel_order::rg);
      check_enum_class_value(cl::sycl::image_channel_order::rgx);
      check_enum_class_value(cl::sycl::image_channel_order::ra);
      check_enum_class_value(cl::sycl::image_channel_order::rgb);
      check_enum_class_value(cl::sycl::image_channel_order::rgbx);
      check_enum_class_value(cl::sycl::image_channel_order::rgba);
      check_enum_class_value(cl::sycl::image_channel_order::argb);
      check_enum_class_value(cl::sycl::image_channel_order::bgra);
      check_enum_class_value(cl::sycl::image_channel_order::intensity);
      check_enum_class_value(cl::sycl::image_channel_order::luminance);

      // Check image channel types
      check_enum_class_value(cl::sycl::image_channel_type::snorm_int8);
      check_enum_class_value(cl::sycl::image_channel_type::snorm_int16);
      check_enum_class_value(cl::sycl::image_channel_type::unorm_int8);
      check_enum_class_value(cl::sycl::image_channel_type::unorm_int16);
      check_enum_class_value(cl::sycl::image_channel_type::unorm_short_565);
      check_enum_class_value(cl::sycl::image_channel_type::unorm_short_555);
      check_enum_class_value(cl::sycl::image_channel_type::unorm_int_101010);
      check_enum_class_value(cl::sycl::image_channel_type::signed_int8);
      check_enum_class_value(cl::sycl::image_channel_type::signed_int16);
      check_enum_class_value(cl::sycl::image_channel_type::signed_int32);
      check_enum_class_value(cl::sycl::image_channel_type::unsigned_int8);
      check_enum_class_value(cl::sycl::image_channel_type::unsigned_int16);
      check_enum_class_value(cl::sycl::image_channel_type::unsigned_int32);
      check_enum_class_value(cl::sycl::image_channel_type::fp16);
      check_enum_class_value(cl::sycl::image_channel_type::fp32);
    }

    // Check set_write_back()
    {
      img.set_write_back();
      img.set_write_back(false);
      img.set_write_back(true);
    }

    // Check set_final_data()
    {
      // Create another image
      auto imageHost2 =
          get_image_host<dims>(numElems, channelTypeSize, channelCount);
      auto img2 = cl::sycl::image<dims>(static_cast<void *>(imageHost2.data()),
                                        channelOrder, channelType, r);

      auto rawPtr = imageHost2.data();
      auto rawPtrVoid = static_cast<void *>(rawPtr);
      auto rawPtrFloat = static_cast<float *>(rawPtrVoid);
      auto sharedPtrVoid =
          cl::sycl::shared_ptr_class<void>(rawPtrVoid, [](void *) {});
      auto sharedPtrFloat =
          cl::sycl::shared_ptr_class<float>(rawPtrFloat, [](float *) {});
      auto weakPtrVoid = cl::sycl::weak_ptr_class<void>(sharedPtrVoid);
      auto weakPtrFloat = cl::sycl::weak_ptr_class<float>(sharedPtrFloat);
      auto iterator = imageHost2.begin();

      img.set_final_data();
      img.set_final_data(nullptr);
      img.set_final_data(rawPtr);
      img.set_final_data(rawPtrVoid);
      img.set_final_data(rawPtrFloat);
      img.set_final_data(sharedPtrVoid);
      img.set_final_data(sharedPtrFloat);
      img.set_final_data(weakPtrVoid);
      img.set_final_data(weakPtrFloat);
      img.set_final_data(iterator);
    }

    // Check get_access()
    {
      // Create another image
      auto img2 = cl::sycl::image<dims>(static_cast<void *>(imageHost.data()),
                                        channelOrder, channelType, r);
      {
        // target::host_image
        img2.template get_access<cl::sycl::float4,
                                 cl::sycl::access::mode::read_write>();
      }

      auto queue = util::get_cts_object::queue();
      try {
        queue.submit([&](cl::sycl::handler &cgh) {
          // target::image
          auto imgAcc =
              img2.template get_access<cl::sycl::float4,
                                       cl::sycl::access::mode::write>(cgh);
          cgh.single_task<image_kernel_get_access<dims>>([]() {});
        });

        queue.wait_and_throw();
      } catch (const cl::sycl::feature_not_supported &fnse) {
        if (!queue.get_device()
                 .template get_info<cl::sycl::info::device::image_support>()) {
          log.note("device does not support images -- skipping check");
        } else {
          throw;
        }
      }
    }

    const auto expected = 0.5f;

    // Check read/write APIs.
    {
      auto queue = util::get_cts_object::queue();

      try {
        {
          cl::sycl::image<dims> img2 = image_generic<dims>::create_with_pitch(
              static_cast<void *>(imageHost.data()), channelOrder, channelType,
              r, p);
          queue.submit([&](cl::sycl::handler &cgh) {
            auto img_acc =
                img2.template get_access<cl::sycl::float4,
                                         cl::sycl::access::mode::read>(cgh);

            auto sampler = cl::sycl::sampler(
                cl::sycl::coordinate_normalization_mode::unnormalized,
                cl::sycl::addressing_mode::clamp,
                cl::sycl::filtering_mode::nearest);

            auto myKernel = [img_acc, sampler](cl::sycl::item<dims> item) {
              // Read image data using integer coordinates
              cl::sycl::float4 dataFromInt =
                  img_acc.read(image_access<dims>::get_int(item));
              (void)dataFromInt;  // silent warning
              // Read image data using integer coordinates and a sampler
              cl::sycl::float4 dataFromIntWithSampler =
                  img_acc.read(image_access<dims>::get_int(item), sampler);
              (void)dataFromIntWithSampler;  // silent warning
              // Read image data using floating point coordinates
              // Only works with a sampler
              cl::sycl::float4 dataFromFloat =
                  img_acc.read(image_access<dims>::get_float(item), sampler);
              (void)dataFromFloat;  // silent warning
            };
            cgh.parallel_for<image_kernel_read<dims>>(r, myKernel);
          });

          queue.submit([&](cl::sycl::handler &cgh) {
            auto img_acc =
                img2.template get_access<cl::sycl::float4,
                                         cl::sycl::access::mode::write>(cgh);
            auto myKernel = [expected, img_acc](cl::sycl::item<dims> item) {
              // Write image data
              img_acc.write(image_access<dims>::get_int(item),
                            cl::sycl::float4(expected));
            };
            cgh.parallel_for<image_kernel_write<dims>>(r, myKernel);
          });

          queue.wait_and_throw();
        }  // End of scope for img2 object.

        // Check image values
        const auto floatData = reinterpret_cast<float *>(imageHost.data());
        for (int i = 0; i < numElems; ++i) {
          if (!CHECK_VALUE(log, floatData[i], 0.5f, i)) {
            FAIL(log, "Image contains wrong values.");
          }
        }

      } catch (const cl::sycl::feature_not_supported &fnse) {
        if (!queue.get_device()
                 .template get_info<cl::sycl::info::device::image_support>()) {
          log.note("device does not support images -- skipping check");
        } else {
          throw;
        }
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
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      // Ensure the image always has 64 elements
      const int elemsPerDim1 = 64;
      const int elemsPerDim2 = 8;
      const int elemsPerDim3 = 4;

      cl::sycl::range<1> range_1d(elemsPerDim1);
      cl::sycl::range<2> range_2d(elemsPerDim2, elemsPerDim2);
      cl::sycl::range<3> range_3d(elemsPerDim3, elemsPerDim3, elemsPerDim3);

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
        cl::sycl::range<1> pitch_1d(elemsPerDim2);
        cl::sycl::range<2> pitch_2d(elemsPerDim3, elemsPerDim3 * elemsPerDim3);

        image_ctors<2> img_2d;
        image_ctors<3> img_3d;
        img_2d(log, range_2d, &pitch_1d);
        img_3d(log, range_3d, &pitch_2d);
      }
    } catch (const cl::sycl::exception &e) {
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
