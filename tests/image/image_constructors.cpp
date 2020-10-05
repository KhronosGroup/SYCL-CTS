/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "image_common.h"
#include "../common/common.h"

#define TEST_NAME image_constructors

namespace image_constructors__ {
using namespace sycl_cts;

void no_delete(void *) {}

template <int dims, typename AllocatorT>
inline void check_constructed_correctly(util::logger &log,
                                        cl::sycl::image<dims, AllocatorT> &img,
                                        int numElems, unsigned int elementSize,
                                        bool &combinationSuccess) {
  // Check get_size()
  if (img.get_size() < (numElems * elementSize)) {
    cl::sycl::string_class message =
        cl::sycl::string_class("Sizes are not the same: expected at least ") +
        std::to_string(numElems * elementSize) + ", got " +
        std::to_string(img.get_size());
    combinationSuccess = false;
    FAIL(log, message);
  }
}

template <typename AllocatorT, int dims>
void test_constructors_no_pitch(util::logger &log, void *imageHostPtr,
                                cl::sycl::range<dims> &r, int numElems,
                                unsigned int elementSize,
                                cl::sycl::image_channel_order channelOrder,
                                cl::sycl::image_channel_type channelType,
                                bool &combinationSuccess,
                                const cl::sycl::property_list &propList) {
  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<dims>&,
   *              const property_list& = {})
   */
  {
    cl::sycl::image<dims, AllocatorT> img =
        cl::sycl::image<dims, AllocatorT>(imageHostPtr, channelOrder, channelType, r);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<dims>&, const property_list&)
   */
  {
    cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
        imageHostPtr, channelOrder, channelType, r, propList);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<dims>&, allocator,
   *              const property_list& = {})
   */
  {
    AllocatorT imgAlloc;
    cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
        imageHostPtr, channelOrder, channelType, r, imgAlloc);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<dims>&, allocator,
   *              const property_list&)
   */
  {
    AllocatorT imgAlloc;
    cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
        imageHostPtr, channelOrder, channelType, r, imgAlloc, propList);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (const void*, image_channel_order,
   *              image_channel_type, const range<dims>&,
   *              const property_list& = {})
   */
  {
    const auto constHostPtr = imageHostPtr;
    cl::sycl::image<dims, AllocatorT> img =
        cl::sycl::image<dims, AllocatorT>(constHostPtr, channelOrder, channelType, r);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (const void*, image_channel_order,
   *              image_channel_type, const range<dims>&, const property_list&)
   */
  {
    const auto constHostPtr = imageHostPtr;
    cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
        constHostPtr, channelOrder, channelType, r, propList);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (const void*, image_channel_order,
   *              image_channel_type, const range<dims>&, allocator,
   *              const property_list& = {})
   */
  {
    const auto constHostPtr = imageHostPtr;
    AllocatorT imgAlloc;
    cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
        constHostPtr, channelOrder, channelType, r, imgAlloc);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (const void*, image_channel_order,
   *              image_channel_type, const range<dims>&, allocator,
   *              const property_list&)
   */
  {
    const auto constHostPtr = imageHostPtr;
    AllocatorT imgAlloc;
    cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
        constHostPtr, channelOrder, channelType, r, imgAlloc, propList);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<dims>&,
   *              const property_list& = {})
   */
  {
    auto hostPointer =
        cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
    cl::sycl::image<dims, AllocatorT> img =
        cl::sycl::image<dims, AllocatorT>(hostPointer, channelOrder, channelType, r);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<dims>&, const property_list&)
   */
  {
    auto hostPointer =
        cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
    cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(hostPointer, channelOrder,
                                                      channelType, r, propList);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<dims>&, allocator,
   *              const property_list& = {})
   */
  {
    AllocatorT imgAlloc;
    auto hostPointer =
        cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
    cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(hostPointer, channelOrder,
                                                      channelType, r, imgAlloc);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<dims>&, allocator,
   *              const property_list&)
   */
  {
    AllocatorT imgAlloc;
    auto hostPointer =
        cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
    cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
        hostPointer, channelOrder, channelType, r, imgAlloc, propList);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<dims>&, const property_list& = {})
   */
  {
    cl::sycl::image<dims, AllocatorT> img =
        cl::sycl::image<dims, AllocatorT>(channelOrder, channelType, r);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<dims>&, const property_list&)
   */
  {
    cl::sycl::image<dims, AllocatorT> img =
        cl::sycl::image<dims, AllocatorT>(channelOrder, channelType, r, propList);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<dims>&, allocator, const property_list& = {})
   */
  {
    AllocatorT imgAlloc;
    cl::sycl::image<dims, AllocatorT> img =
        cl::sycl::image<dims, AllocatorT>(channelOrder, channelType, r, imgAlloc);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<dims>&, allocator, const property_list&)
   */
  {
    AllocatorT imgAlloc;
    cl::sycl::image<dims, AllocatorT> img =
        cl::sycl::image<dims, AllocatorT>(channelOrder, channelType, r, imgAlloc, propList);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }
}

template <typename AllocatorT, int dims>
struct test_constructors_with_pitch {
  test_constructors_with_pitch(util::logger &log, void *imageHostPtr,
                               cl::sycl::range<dims> &r, int numElems,
                               unsigned int elementSize,
                               cl::sycl::image_channel_order channelOrder,
                               cl::sycl::image_channel_type channelType,
                               cl::sycl::range<dims - 1> *pitch,
                               bool &combinationSuccess,
                               const cl::sycl::property_list &propList) {
    /* Constructor (void *, image_channel_order,
     *              image_channel_type, const range<dims>&,
     *              const range<dims - 1>&, const property_list& = {})
     */
    {
      cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
          imageHostPtr, channelOrder, channelType, r, *pitch);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (void *, image_channel_order,
     *              image_channel_type, const range<dims>&,
     *              const range<dims - 1>&, const property_list&)
     */
    {
      cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
          imageHostPtr, channelOrder, channelType, r, *pitch, propList);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (void *, image_channel_order,
     *              image_channel_type, const range<dims>&,
     *              const range<dims - 1>&, allocator,
     *              const property_list& = {})
     */
    {
      AllocatorT imgAlloc;
      cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
          imageHostPtr, channelOrder, channelType, r, *pitch, imgAlloc);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (void *, image_channel_order,
     *              image_channel_type, const range<dims>&,
     *              const range<dims - 1>&, allocator, const property_list&)
     */
    {
      AllocatorT imgAlloc;
      cl::sycl::image<dims, AllocatorT> img =
          cl::sycl::image<dims, AllocatorT>(imageHostPtr, channelOrder, channelType, r,
                                *pitch, imgAlloc, propList);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (shared_ptr_class<void>&, image_channel_order,
     *              image_channel_type, const range<dims>&,
     *              const range<dims - 1>&, const property_list& = {})
     */
    {
      auto hostPointer =
          cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
      cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
          hostPointer, channelOrder, channelType, r, *pitch);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (shared_ptr_class<void>&, image_channel_order,
     *              image_channel_type, const range<dims>&,
     *              const range<dims - 1>&, const property_list&)
     */
    {
      auto hostPointer =
          cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
      cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
          hostPointer, channelOrder, channelType, r, *pitch, propList);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (shared_ptr_class<void>&, image_channel_order,
     *              image_channel_type, const range<dims>&,
     *              const range<dims - 1>&, allocator,
     *              const property_list& = {})
     */
    {
      AllocatorT imgAlloc;
      auto hostPointer =
          cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
      cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
          hostPointer, channelOrder, channelType, r, *pitch, imgAlloc);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (shared_ptr_class<void>&, image_channel_order,
     *              image_channel_type, const range<dims>&,
     *              const range<dims - 1>&, allocator, const property_list&)
     */
    {
      AllocatorT imgAlloc;
      auto hostPointer =
          cl::sycl::shared_ptr_class<void>(imageHostPtr, &no_delete);
      cl::sycl::image<dims, AllocatorT> img =
          cl::sycl::image<dims, AllocatorT>(hostPointer, channelOrder, channelType, r,
                                *pitch, imgAlloc, propList);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (image_channel_order, image_channel_type,
     *              const range<dims>&, const range<dims - 1>&,
     *              const property_list& = {})
     */
    {
      cl::sycl::image<dims, AllocatorT> img =
          cl::sycl::image<dims, AllocatorT>(channelOrder, channelType, r, *pitch);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (image_channel_order, image_channel_type,
     *              const range<dims>&, const range<dims - 1>&,
     *              const property_list&)
     */
    {
      cl::sycl::image<dims, AllocatorT> img =
          cl::sycl::image<dims, AllocatorT>(channelOrder, channelType, r, *pitch, propList);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (image_channel_order, image_channel_type,
     *              const range<dims>&, const range<dims - 1>&, allocator,
     *              const property_list& = {})
     */
    {
      AllocatorT imgAlloc;
      cl::sycl::image<dims, AllocatorT> img =
          cl::sycl::image<dims, AllocatorT>(channelOrder, channelType, r, *pitch, imgAlloc);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (image_channel_order, image_channel_type,
     *              const range<dims>&, const range<dims - 1>&, allocator,
     *              const property_list&)
     */
    {
      AllocatorT imgAlloc;
      cl::sycl::image<dims, AllocatorT> img = cl::sycl::image<dims, AllocatorT>(
          channelOrder, channelType, r, *pitch, imgAlloc, propList);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }
  }
};

template <typename AllocatorT>
struct test_constructors_with_pitch<AllocatorT, 1> {
  test_constructors_with_pitch(util::logger &log, void *imageHostPtr,
                               cl::sycl::range<1> &r, int numElems,
                               unsigned int elementSize,
                               cl::sycl::image_channel_order channelOrder,
                               cl::sycl::image_channel_type channelType,
                               void *pitch, bool &combinationSuccess,
                               const cl::sycl::property_list &propList) {
    // 1D images don't take pitch, ignore test
  }
};

template <int dims>
class image_ctors {
 public:
  void operator()(util::logger &log, cl::sycl::range<dims> &r,
                  const cl::sycl::property_list &propList,
                  cl::sycl::range<dims - 1> *pitch = nullptr) {
    switch (dims) {
      case 1:
        log.note("Testing image combination: dims[%d], range[%d]", dims, r[0]);
        break;
      case 2:
        log.note("Testing image combination: dims[%d], range[%d, %d]", dims,
                 r[0], r[1]);
        break;
      case 3:
        log.note("Testing image combination: dims[%d], range[%d, %d, %d]", dims,
                 r[0], r[1], r[2]);
        break;
      default:
        break;
    }

    size_t itOrder = 0;
    size_t itType = 0;

    // Pitch has to be at least as large as the multiple of element size,
    // just multiply by largest possible supported element size
    // (RGBA float as an example)
    const auto pitchElementSize =
        (get_channel_order_count(cl::sycl::image_channel_order::rgba) *
         get_channel_type_size(cl::sycl::image_channel_type::fp32));
    image_generic<dims>::multiply_pitch(pitch, pitchElementSize);

    // For each channel order
    for (itOrder = 0; itOrder < MINIMUM_CHANNEL_ORDERS; ++itOrder) {
      // Set up the test set for each channel separately
      auto testSet = get_test_set_full(g_channelOrderCount[itOrder].order);

      // Get number of channels
      const auto channelOrder = testSet.order;
      const auto channelCount = get_channel_order_count(channelOrder);
      const auto numChannelTypes = static_cast<size_t>(testSet.numChannelTypes);

      // For each channel type
      for (itType = 0; itType < numChannelTypes; ++itType) {
        bool combinationSuccess = true;

        // Prepare variables
        const auto channelType = testSet.typeArray[itType];
        const auto channelTypeSize = get_channel_type_size(channelType);
        const auto elementSize = channelTypeSize * channelCount;

        // Create image host data
        constexpr auto numElems = 1;
        auto imageHost =
            get_image_host<dims>(numElems, channelTypeSize, channelCount);
        void *imageHostPtr = static_cast<void *>(imageHost.data());
        auto imageHost2 =
            get_image_host<dims>(numElems, channelTypeSize, channelCount);
        void *imageHost2Ptr = static_cast<void *>(imageHost2.data());

        // Test all constructors that don't take a pitch
        test_constructors_no_pitch<cl::sycl::image_allocator>(log, imageHostPtr, r, numElems, elementSize,
                                   channelOrder, channelType,
                                   combinationSuccess, propList);

        test_constructors_no_pitch<std::allocator<cl::sycl::byte>>(log, imageHostPtr, r, numElems, elementSize,
                                   channelOrder, channelType,
                                   combinationSuccess, propList);

        // Test all constructors that take a pitch
        if (pitch != nullptr) {
          test_constructors_with_pitch<cl::sycl::image_allocator, dims>(
              log, imageHostPtr, r, numElems, elementSize, channelOrder,
              channelType, pitch, combinationSuccess, propList);

          test_constructors_with_pitch<std::allocator<cl::sycl::byte>, dims>(
              log, imageHostPtr, r, numElems, elementSize, channelOrder,
              channelType, pitch, combinationSuccess, propList);
        }

        // Check copy constructor
        {
          cl::sycl::image<dims> imgA(imageHostPtr, channelOrder, channelType, r,
                                     propList);
          cl::sycl::image<dims> imgB(imgA);

          if (imgA.get_range() != imgB.get_range()) {
            FAIL(log, "image was not copy assigned correctly. (get_range)");
            combinationSuccess = false;
          }
          if (!image_generic<dims>::compare_pitch(log, imgA, imgB)) {
            FAIL(log, "image was not copy assigned correctly. (get_pitch)");
            combinationSuccess = false;
          }
          if (imgA.get_size() != imgB.get_size()) {
            FAIL(log, "image was not copy assigned correctly. (get_size)");
            combinationSuccess = false;
          }
          if (imgA.get_count() != imgB.get_count()) {
            FAIL(log, "image was not copy assigned correctly. (get_count)");
            combinationSuccess = false;
          }
        }

        /* Check copy assignment */
        {
          cl::sycl::image<dims> imgA(imageHostPtr, channelOrder, channelType, r,
                                     propList);
          cl::sycl::image<dims> imgB(imageHost2Ptr, channelOrder, channelType,
                                     r);
          imgB = imgA;

          if (imgA.get_range() != imgB.get_range()) {
            FAIL(log, "image was not copy assigned correctly. (get_range)");
            combinationSuccess = false;
          }
          if (!image_generic<dims>::compare_pitch(log, imgA, imgB)) {
            FAIL(log, "image was not copy assigned correctly. (get_pitch)");
            combinationSuccess = false;
          }
          if (imgA.get_size() != imgB.get_size()) {
            FAIL(log, "image was not copy assigned correctly. (get_size)");
            combinationSuccess = false;
          }
          if (imgA.get_count() != imgB.get_count()) {
            FAIL(log, "image was not copy assigned correctly. (get_count)");
            combinationSuccess = false;
          }
        }

        /* check move constructor */
        {
          const cl::sycl::property_list propertyList{
              cl::sycl::property::image::use_host_ptr()};

          cl::sycl::image<dims> imgA(imageHostPtr, channelOrder, channelType, r,
                                     propertyList);
          cl::sycl::image<dims> imgB(std::move(imgA));

          bool hasHostPtrProperty = imgB.template has_property<
              cl::sycl::property::image::use_host_ptr>();

          if (!hasHostPtrProperty) {
            FAIL(log,
                 "image was not copy assigned properly. "
                 "(has_property<use_host_ptr>)");
          }
        }

        /* check move assignment */
        {
          const cl::sycl::property_list propertyList{
              cl::sycl::property::image::use_host_ptr()};

          cl::sycl::image<dims> imgA(imageHostPtr, channelOrder, channelType, r,
                                     propertyList);
          cl::sycl::image<dims> imgB(imageHost2Ptr, channelOrder, channelType,
                                     r);

          imgB = std::move(imgA);

          bool hasHostPtrProperty = imgB.template has_property<
              cl::sycl::property::image::use_host_ptr>();

          if (!hasHostPtrProperty) {
            FAIL(log,
                 "image was not move assigned properly. "
                 "(has_property<use_host_ptr>)");
          }
        }

        /* check equality operator */
        {
          const auto r2 = r * 2;

          cl::sycl::image<dims> imgA(imageHostPtr, channelOrder, channelType,
                                     r);
          cl::sycl::image<dims> imgB(imgA);
          cl::sycl::image<dims> imgC(imageHostPtr, channelOrder, channelType,
                                     r2);
          imgC = imgA;
          cl::sycl::image<dims> imgD(imageHostPtr, channelOrder, channelType,
                                     r2);

          // check equality
          bool equality = (imgA == imgB);
          if (!(imgA == imgB)) {
            FAIL(log, "image equality comparison failed. (copy constructed)");
            combinationSuccess = false;
          }
          if (!(imgA == imgC)) {
            FAIL(log, "image equality comparison failed. (copy assigned)");
            combinationSuccess = false;
          }
          if (imgA != imgB) {
            FAIL(log,
                 "image non-equality does not work correctly"
                 "(copy constructed)");
          }
          if (imgA != imgC) {
            FAIL(log,
                 "image non-equality does not work correctly"
                 "(copy assigned)");
          }
          if (imgC == imgD) {
            FAIL(log,
                 "image equality does not work correctly"
                 "(comparing same)");
          }
          if (!(imgC != imgD)) {
            FAIL(log,
                 "image non-equality does not work correctly"
                 "(comparing same)");
          }
        }

        /* check hash */
        {
          cl::sycl::image<dims> imgA(imageHostPtr, channelOrder, channelType,
                                     r);
          cl::sycl::image<dims> imgB = imgA;

          cl::sycl::hash_class<cl::sycl::image<dims>> hasher;

          if (hasher(imgA) != hasher(imgB)) {
            FAIL(log, "image hashing failed. (hashing of equals)");
            combinationSuccess = false;
          }
        }

        if (!combinationSuccess) {
          log.note("Failed with combination {dims[%d], pitch[%d], %s, %s}",
                   dims, (pitch != nullptr),
                   get_channel_order_string(testSet.order),
                   get_channel_type_string(testSet.typeArray[itType]));
        }
      }
    }
  }
};

/**
 * test cl::sycl::image initialization
 */
class TEST_NAME : public util::test_base {
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

      cl::sycl::range<1> pitch_1d(elemsPerDim2);
      cl::sycl::range<2> pitch_2d(elemsPerDim3, elemsPerDim3 * elemsPerDim3);

      image_ctors<1> img_1d;
      image_ctors<2> img_2d;
      image_ctors<3> img_3d;

      image_ctors<1> img_1d_with_properties;
      image_ctors<2> img_2d_with_properties;
      image_ctors<3> img_3d_with_properties;

      /* create property lists */
      const cl::sycl::property_list emptyPropList{};
      cl::sycl::mutex_class mutex;
      auto context = util::get_cts_object::context();
      const cl::sycl::property_list propList{
          cl::sycl::property::image::use_mutex(mutex),
          cl::sycl::property::image::context_bound(context)};

      img_1d(log, range_1d, emptyPropList);
      img_2d(log, range_2d, emptyPropList, &pitch_1d);
      img_3d(log, range_3d, emptyPropList, &pitch_2d);

      img_1d_with_properties(log, range_1d, propList);
      img_2d_with_properties(log, range_2d, propList, &pitch_1d);
      img_3d_with_properties(log, range_3d, propList, &pitch_2d);
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

} /* namespace image_constructors__ */
