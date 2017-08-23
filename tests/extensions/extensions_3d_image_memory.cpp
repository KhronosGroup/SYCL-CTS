/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME extension_3d_image_memory

namespace extension_3d_image_memory__ {
using namespace sycl_cts;
using namespace cl::sycl;

template <int dims, int size>
class image_ctors {
 public:
  void operator()(util::logger &log, range<dims> &r) {
    /* determine sizes */
    int numElements = size;
    for (int i = 1; i < dims; i++) {
      numElements *= size;
    }
    int sizeInBytes = numElements * sizeof(float);

    /* allocate host side buffer, too large for stack */
    cl::sycl::unique_ptr_class<float> image_host(new float[sizeInBytes]);

    /* white block */
    ::memset(image_host.get(), 0.0f, sizeInBytes);

    const float val = 0.2f;
    {
      image<dims> img((void *)image_host.get(), image_channel_order::rgba,
                      image_channel_type::fp32, r);

      auto queue = util::get_cts_object::queue();
      queue.submit([&](handler &cgh) {
        auto img_acc =
            img.template get_access<float4, cl::sycl::access::mode::write>(cgh);
        auto my_range = nd_range<3>(r, r);
        auto my_kernel = ([=](nd_item<3> item) {
          img_acc[item.get_global()] = float4(val, val, val, val);
        });

        /* execute the kernel */
        cgh.parallel_for<class TEST_NAME>(my_range, my_kernel);
      });

      queue.wait_and_throw();
    }
    // check if the value has been written back
    for (int i = 0; i < numElements; i++) {
      if (image_host.get()[i] != val) {
        FAIL(log, "Image host is not as expected.");
      }
    }
  }
};

/**
 * test cl::sycl::buffer initialization
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
      const int size = 4;
      range<3> range_3d(size, size, size);
      image_ctors<3, size> img_3d;
      img_3d(log, range_3d);
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
