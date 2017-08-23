/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME extension_3d_image_memory

namespace extension_3d_image_memory__ {
using namespace sycl_cts;
using namespace cl::sycl;

cl_channel_order g_order[] = {CL_RGBA, 0};

cl_channel_type g_type[] = {CL_FLOAT, 0};

template <int dims, int size>
class image_ctors {
 public:
  void operator()(util::logger &log, range<dims> &r) {
    /* allocate host side buffer, too large for stack */
    util::UNIQUE_PTR<float> image_host(new float[4 * size]);

    float l_float = 1.0f;

    // white block
    memset(image_host.get(), (*(uint32_t *)&l_float), sizeof(float) * 4 * size);

    size_t l_type_itter, l_order_itter;
    // for each type
    for (l_type_itter = 0; g_type[l_type_itter] != 0; l_type_itter++) {
      // for each order
      for (l_order_itter = 0; g_order[l_order_itter] != 0; l_order_itter++) {
        const float val = 0.2f;
        // cl_channel_type
        {
          image<dims> img((void *)image_host.get(), g_order[l_order_itter],
                          g_type[l_type_itter], r);

          default_selector sel;
          queue queue(sel);
          queue.submit([&](handler &cgh) {
            auto img_acc =
                img.template get_access<float4, cl::sycl::access::mode::write>(
                    cgh);
            auto my_range = nd_range<3>(range<1>(4 * size), range<1>(4 * size));
            auto my_kernel = ([=](item<3> item) {
              img_acc[item.get_global(0)][item.get_global(1)]
                     [item.get_global(2)] = val;
            });

            /* execute the kernel */
            cgh.parallel_for<class TEST_NAME>(my_range, my_kernel);
          });

          queue.wait_and_throw();
        }
        // check if the value has been written back
        for (int i = 0; i < 4 * size; i++) {
          if (image_host.get()[i] != val) {
            FAIL(log, "Image host is not as expected.");
          }
        }
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
      const int size = 32;
      range<3> range_3d(size, size, size);
      image_ctors<3, size * size * size> img_3d;
      img_3d(log, range_3d);
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace image_constructors__ */
