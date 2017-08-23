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
using namespace cl::sycl;

void no_delete(void *) {}

template <int dims>
inline void check_constructed_correctly(util::logger &log, image<dims> &img,
                                        int numElems, unsigned int elementSize,
                                        bool &combinationSuccess) {
  // Check get_size()
  if (img.get_size() < (numElems * elementSize)) {
    string_class message =
        string_class("Sizes are not the same: expected at least ") +
        std::to_string(numElems * elementSize) + ", got " +
        std::to_string(img.get_size());
    combinationSuccess = false;
    FAIL(log, message);
  }
}

template <int dims>
void test_constructors_no_pitch(util::logger &log, void *imageHostPtr,
                                range<dims> &r, int numElems,
                                unsigned int elementSize,
                                image_channel_order channelOrder,
                                image_channel_type channelType,
                                bool &combinationSuccess) {
  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<dims>&)
   */
  {
    image<dims> img = image<dims>(imageHostPtr, channelOrder, channelType, r);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<dims>&)
   */
  {
    auto hostPointer = shared_ptr_class<void>(imageHostPtr, &no_delete);
    image<dims> img = image<dims>(hostPointer, channelOrder, channelType, r);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (shared_ptr_class<void>&, image_channel_order,
   *              image_channel_type, const range<dims>&,
   *              mutex_class&)
   */
  {
    auto hostPointer = shared_ptr_class<void>(imageHostPtr, &no_delete);
    mutex_class mutex;
    image<dims> img =
        image<dims>(hostPointer, channelOrder, channelType, r, mutex);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<dims>&)
   */
  {
    image<dims> img = image<dims>(channelOrder, channelType, r);
    check_constructed_correctly(log, img, numElems, elementSize,
                                combinationSuccess);
  }
}

template <int dims>
struct test_constructors_with_pitch {
  test_constructors_with_pitch(util::logger &log, void *imageHostPtr,
                               range<dims> &r, int numElems,
                               unsigned int elementSize,
                               image_channel_order channelOrder,
                               image_channel_type channelType,
                               range<dims - 1> *pitch,
                               bool &combinationSuccess) {
    /* Constructor (void *, image_channel_order,
     *              image_channel_type, const range<dims>&,
     *              const range<dims - 1>&)
     */
    {
      image<dims> img =
          image<dims>(imageHostPtr, channelOrder, channelType, r, *pitch);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (shared_ptr_class<void>&, image_channel_order,
     *              image_channel_type, const range<dims>&,
     *              const range<dims - 1>&)
     */
    {
      auto hostPointer = shared_ptr_class<void>(imageHostPtr, &no_delete);
      image<dims> img =
          image<dims>(hostPointer, channelOrder, channelType, r, *pitch);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (shared_ptr_class<void>&, image_channel_order,
     *              image_channel_type, const range<dims>&,
     *              const range<dims - 1>&, mutex_class&)
     */
    {
      auto hostPointer = shared_ptr_class<void>(imageHostPtr, &no_delete);
      mutex_class mutex;
      image<dims> img =
          image<dims>(hostPointer, channelOrder, channelType, r, *pitch, mutex);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }

    /* Constructor (image_channel_order, image_channel_type,
     *              const range<dims>&, const range<dims - 1>&)
     */
    {
      image<dims> img = image<dims>(channelOrder, channelType, r, *pitch);
      check_constructed_correctly(log, img, numElems, elementSize,
                                  combinationSuccess);
    }
  }
};

template <>
struct test_constructors_with_pitch<1> {
  test_constructors_with_pitch(util::logger &log, void *imageHostPtr,
                               range<1> &r, int numElems,
                               unsigned int elementSize,
                               image_channel_order channelOrder,
                               image_channel_type channelType, void *pitch,
                               bool &combinationSuccess) {
    // 1D images don't take pitch, ignore test
  }
};

template <int dims>
class image_ctors {
 public:
  void operator()(util::logger &log, range<dims> &r,
                  range<dims - 1> *pitch = nullptr) {
    log.note("Testing image combination: dims[%d], range[%d, %d, %d]", dims,
             r[0], r[1], r[2]);

    size_t itOrder = 0;
    size_t itType = 0;
    const auto numElems = static_cast<int>(r[0] * r[1] * r[2]);

    // For each channel order
    for (itOrder = 0; itOrder < MINIMUM_CHANNEL_ORDERS; ++itOrder) {
      // Set up the test set for each channel separately
      auto testSet = get_test_set_minimum(g_channelOrderCount[itOrder].order);

      // Get number of channels
      const auto channelOrder = testSet.order;
      const auto channelCount = get_channel_order_count(channelOrder);

      // For each channel type
      for (itType = 0; itType < testSet.numChannelTypes; ++itType) {
        bool combinationSuccess = true;

        // Prepare variables
        const auto channelType = testSet.typeArray[itType];
        const auto channelTypeSize = get_channel_type_size(channelType);
        const auto elementSize = channelTypeSize * channelCount;

        // Create image host data
        auto imageHost = get_image_host<dims>(channelTypeSize, channelCount);
        void *imageHostPtr = static_cast<void *>(imageHost.get());

        // Test all constructors that don't take a pitch
        test_constructors_no_pitch(log, imageHostPtr, r, numElems, elementSize,
                                   channelOrder, channelType,
                                   combinationSuccess);

        // Test all constructors that take a pitch
        if (pitch != nullptr) {
          test_constructors_with_pitch<dims>(
              log, imageHostPtr, r, numElems, elementSize, channelOrder,
              channelType, pitch, combinationSuccess);
        }

        // Check copy constructor
        {
          image<dims> imgA(imageHostPtr, channelOrder, channelType, r);
          image<dims> imgB(imgA);

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
          image<dims> imgA(imageHostPtr, channelOrder, channelType, r);
          image<dims> imgB = imgA;

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
          image<dims> imgA(imageHostPtr, channelOrder, channelType, r);
          image<dims> imgB(std::move(imgA));
        }

        /* check move assignment */
        {
          image<dims> imgA(imageHostPtr, channelOrder, channelType, r);
          image<dims> imgB = std::move(imgA);
        }

        /* check equality operator */
        {
          image<dims> imgA(imageHostPtr, channelOrder, channelType, r);
          image<dims> imgB(imgA);
          image<dims> imgC = imgA;

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
        }

        /* check hash */
        {
          image<dims> imgA(imageHostPtr, channelOrder, channelType, r);
          image<dims> imgB = imgA;

          cl::sycl::hash_class<image<dims>> hasher;

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

      range<1> pitch_1d(elemsPerDim2);
      range<2> pitch_2d(elemsPerDim3, elemsPerDim3 * elemsPerDim3);

      image_ctors<1> img_1d;
      image_ctors<2> img_2d;
      image_ctors<3> img_3d;

      img_1d(log, range_1d);
      img_2d(log, range_2d, &pitch_1d);
      img_3d(log, range_3d, &pitch_2d);
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

} /* namespace image_constructors__ */
