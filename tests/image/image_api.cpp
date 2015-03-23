/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME image_api

namespace image_api__ {
using namespace sycl_cts;
using namespace cl::sycl;

struct channel_type_size {
  cl_channel_type idx;
  unsigned int size;
};

channel_type_size g_channel_size[] = {{CL_SNORM_INT8, 1},
                                      {CL_SNORM_INT16, 2},
                                      {CL_UNORM_INT8, 1},
                                      {CL_UNORM_INT16, 2},
                                      {CL_UNORM_SHORT_565, 2},
                                      {CL_UNORM_SHORT_555, 2},
                                      {CL_UNORM_INT_101010, 4},
                                      {CL_SIGNED_INT8, 1},
                                      {CL_SIGNED_INT16, 2},
                                      {CL_SIGNED_INT32, 4},
                                      {CL_UNSIGNED_INT8, 1},
                                      {CL_UNSIGNED_INT16, 2},
                                      {CL_UNSIGNED_INT32, 4},
                                      {CL_FLOAT, sizeof(float)},
                                      {CL_HALF_FLOAT, sizeof(cl::sycl::half)},
                                      {0, 0}};

unsigned int get_channel_type_size(unsigned int idx) {
  unsigned int itter;
  for (itter = 0; g_channel_size[itter].idx != 0; itter++) {
    if (g_channel_size[itter].idx == idx) return g_channel_size[itter].size;
  }

  // that should be never reached
  return 0;
}

struct channels {
  cl_channel_order idx;
  unsigned int size;
};

channels g_channels[] = {{CL_R, 1},
                         {CL_A, 1},
                         {CL_RG, 2},
                         {CL_RA, 2},
                         {CL_RGB, 3},
                         {CL_RGBA, 4},
                         {CL_BGRA, 4},
                         {CL_ARGB, 4},
                         {CL_INTENSITY, 1},
                         {CL_LUMINANCE, 1},
                         {CL_Rx, 4},
                         {CL_RGx, 4},
                         {CL_RGBx, 4},
                         {0, 0}};

unsigned int get_channel_count(unsigned int idx) {
  unsigned int itter;
  for (itter = 0; g_channels[itter].idx != 0; itter++) {
    if (g_channels[itter].idx == idx) return g_channels[itter].size;
  }

  // that should be never reached
  return 0;
}

struct collection {
  cl_channel_order m_order;
  cl_channel_type m_type[17];
};

collection g_test_set[] = {
    {CL_R,
     {CL_SNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16,
      CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010,
      CL_SIGNED_INT8, CL_SIGNED_INT16, CL_SIGNED_INT32, CL_UNSIGNED_INT8,
      CL_UNSIGNED_INT16, CL_UNSIGNED_INT32, CL_HALF_FLOAT, CL_FLOAT, 0}},
    {CL_A,
     {CL_SNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16,
      CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010,
      CL_SIGNED_INT8, CL_SIGNED_INT16, CL_SIGNED_INT32, CL_UNSIGNED_INT8,
      CL_UNSIGNED_INT16, CL_UNSIGNED_INT32, CL_HALF_FLOAT, CL_FLOAT, 0}},
    {CL_RG,
     {CL_SNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16,
      CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010,
      CL_SIGNED_INT8, CL_SIGNED_INT16, CL_SIGNED_INT32, CL_UNSIGNED_INT8,
      CL_UNSIGNED_INT16, CL_UNSIGNED_INT32, CL_HALF_FLOAT, CL_FLOAT, 0}},
    {CL_RA,
     {CL_SNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16,
      CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010,
      CL_SIGNED_INT8, CL_SIGNED_INT16, CL_SIGNED_INT32, CL_UNSIGNED_INT8,
      CL_UNSIGNED_INT16, CL_UNSIGNED_INT32, CL_HALF_FLOAT, CL_FLOAT, 0}},
    {CL_RGBA,
     {CL_SNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16,
      CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010,
      CL_SIGNED_INT8, CL_SIGNED_INT16, CL_SIGNED_INT32, CL_UNSIGNED_INT8,
      CL_UNSIGNED_INT16, CL_UNSIGNED_INT32, CL_HALF_FLOAT, CL_FLOAT, 0}},
    {CL_Rx,
     {CL_SNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16,
      CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010,
      CL_SIGNED_INT8, CL_SIGNED_INT16, CL_SIGNED_INT32, CL_UNSIGNED_INT8,
      CL_UNSIGNED_INT16, CL_UNSIGNED_INT32, CL_HALF_FLOAT, CL_FLOAT, 0}},
    {CL_RGx,
     {CL_SNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16,
      CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010,
      CL_SIGNED_INT8, CL_SIGNED_INT16, CL_SIGNED_INT32, CL_UNSIGNED_INT8,
      CL_UNSIGNED_INT16, CL_UNSIGNED_INT32, CL_HALF_FLOAT, CL_FLOAT, 0}},
    {CL_ARGB,
     {CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8, CL_UNSIGNED_INT8, 0}},
    {CL_BGRA,
     {CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8, CL_UNSIGNED_INT8, 0}},
    {CL_INTENSITY,
     {CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16,
      CL_HALF_FLOAT, CL_FLOAT, 0}},
    {CL_LUMINANCE,
     {CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16,
      CL_HALF_FLOAT, CL_FLOAT, 0}},
    {CL_RGB, {CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010, 0}},
    {CL_RGBx, {CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010, 0}},
    {CL_RG,
     {CL_SNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16,
      CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010,
      CL_SIGNED_INT8, CL_SIGNED_INT16, CL_SIGNED_INT32, CL_UNSIGNED_INT8,
      CL_UNSIGNED_INT16, CL_UNSIGNED_INT32, CL_HALF_FLOAT, CL_FLOAT, 0}},
    {0, {0}}};

template <int dims, int size>
class image_ctors {
 public:
  void operator()(util::logger& log, range<dims>& r,
                  range<dims - 1>* p = nullptr) {
    size_t l_order_itter, l_type_itter;

    // for each chanell order
    for (l_order_itter = 0; g_test_set[l_order_itter].m_order != 0;
         l_order_itter++) {
      // get number of the chanells
      unsigned int l_channels_count =
          get_channel_count(g_test_set[l_order_itter].m_order);

      // for each chanell type
      for (l_type_itter = 0; g_test_set[l_order_itter].m_type[l_type_itter];
           l_type_itter++) {
        unsigned int l_channels_type_size = get_channel_type_size(
            g_test_set[l_order_itter].m_type[l_type_itter]);

        util::UNIQUE_PTR<char[]> image_host(
            new char[size * l_channels_type_size * l_channels_count]);

        for (int ii = 0; ii < size * l_channels_count; ii++) {
          for (int ij = 0; ij < l_channels_type_size; ij++) {
            image_host[ii * l_channels_type_size + ij] = ij + 1;
          }
        }

        image<dims> img((void*)image_host.get(),
                        g_test_set[l_order_itter].m_order,
                        g_test_set[l_order_itter].m_type[l_type_itter], r);

        if (r.get(0) != img.get_range()[0] || r.get(1) != img.get_range()[1] ||
            r.get(2) != img.get_range()[2]) {
          FAIL(log, "Renges are not the same.");
        }

        if (size != img.get_size()) {
          FAIL(log, "Sizes are not the same.");
        }

        queue queue;
        queue.submit([&](handler& cgh) {
          auto img_acc =
              img.template get_access<float4, cl::sycl::access::mode::write>(
                  cgh);
          auto myRange = nd_range<1>(range<1>(4 * size), range<1>(4 * size));
          auto myKernel =
              ([=](item<1> item) { img_acc[item.get_global(0)] = 0.2; });
        });

        queue.wait_and_throw();

        if (p) {
          unsigned int l_channels_type_size = get_channel_type_size(
              g_test_set[l_order_itter].m_type[l_type_itter]);

          util::UNIQUE_PTR<char[]> image_host(
              new char[size * l_channels_type_size * l_channels_count]);

          for (int ii = 0; ii < size * l_channels_count; ii++) {
            for (int ij = 0; ij < l_channels_type_size; ij++) {
              image_host[ii * l_channels_type_size + ij] = ij + 1;
            }
          }

          {
            image<dims> img(
                (void*)image_host.get(), g_test_set[l_order_itter].m_order,
                g_test_set[l_order_itter].m_type[l_type_itter], r, *p);
            if (r.get(0) != img.get_range()[0] ||
                r.get(1) != img.get_range()[1] ||
                r.get(2) != img.get_range()[2]) {
              FAIL(log, "Ranges are not the same.");
            }

            if (*p != img.get_pitch()) {
              FAIL(log, "Pitchs are not the same.");
            }

            if (size * 4 != img.get_size()) {
              FAIL(log, "Sizes are not the same.");
            }
          }
        }

        // white image
        for (int i = 0; i < size; i++) {
          CHECK_VALUE(log, image_host.get()[i], 0.2f, i);
        }
        std::cout << std::endl;
      }
    }
  }
};

/**
 * test cl::sycl::buffer initialization
 */
class TEST_NAME : public util::test_base_opencl {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger& log) override {
    try {
      const int size = 1;
      range<1> range_1d(size);
      range<2> range_2d(size, size);
      range<3> range_3d(size, size, size);

      {
        image_ctors<1, size> img_1d;
        image_ctors<2, size * size> img_2d;
        image_ctors<3, size * size * size> img_3d;
        img_1d(log, range_1d);
        img_2d(log, range_2d);
        img_3d(log, range_3d);
      }
      {
        range<1> pitch_1d(size);
        range<2> pitch_2d(size, size);

        image_ctors<2, size * size> img_2d;
        image_ctors<3, size * size * size> img_3d;
        img_2d(log, range_2d, &pitch_1d);
        img_3d(log, range_3d, &pitch_2d);
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace image_api__ */
