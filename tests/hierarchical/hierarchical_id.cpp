/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_id

namespace hierarchical_id__ {

static const int g_items_1d = 8;
static const int g_items_2d = 4;
static const int g_items_3d = 2;
static const int l_items_1d = 4;
static const int l_items_2d = 2;
static const int l_items_3d = 1;
static const int gr_range_1d = (g_items_1d / l_items_1d);
static const int gr_range_2d = (g_items_2d / l_items_2d);
static const int gr_range_3d = (g_items_3d / l_items_3d);
static const int g_items_total = (g_items_1d * g_items_2d * g_items_3d);
static const int l_items_total = (l_items_1d * l_items_2d * l_items_3d);
static const int gr_range_total = (g_items_total / l_items_total);

using namespace sycl_cts;
using namespace cl::sycl;

/** test cl::sycl::range::get(int index) return size_t
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
      cl::sycl::default_selector sel;
      cl::sycl::queue testQueue(sel);

      int4 gl_id_data[g_items_total];
      int4 gl_size_data[g_items_total];
      for (int i = 0; i < g_items_total; i++) {
        gl_id_data[i] = int4(0, 0, 0, 0);
        gl_size_data[i] = int4(0, 0, 0, 0);
      }

      int4 l_id_data[g_items_total];
      int4 l_size_data[g_items_total];
      for (int i = 0; i < g_items_total; i++) {
        l_id_data[i] = int4(0, 0, 0, 0);
        l_size_data[i] = int4(0, 0, 0, 0);
      }

      int4 gr_id_data[l_items_total];
      int4 gr_range_data[l_items_total];
      for (int i = 0; i < l_items_total; i++) {
        gr_id_data[i] = int4(0, 0, 0, 0);
        gr_range_data[i] = int4(0, 0, 0, 0);
      }

      {
        cl::sycl::buffer<int4, 1> gl_id_buffer(gl_id_data,
                                               range<1>(g_items_total));
        cl::sycl::buffer<int4, 1> gl_size_buffer(gl_size_data,
                                                 range<1>(g_items_total));
        cl::sycl::buffer<int4, 1> l_id_buffer(l_id_data,
                                              range<1>(g_items_total));
        cl::sycl::buffer<int4, 1> l_size_buffer(l_size_data,
                                                range<1>(g_items_total));
        cl::sycl::buffer<int4, 1> gr_id_buffer(gr_id_data,
                                               range<1>(l_items_total));
        cl::sycl::buffer<int4, 1> gr_range_buffer(gr_range_data,
                                                  range<1>(l_items_total));

        testQueue.submit([&](handler &cgh) {

          auto my_range =
              nd_range<3>(range<3>(g_items_1d, g_items_2d, g_items_3d),
                          range<3>(l_items_1d, l_items_2d, l_items_3d));

          auto gl_id_ptr =
              gl_id_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
          auto gl_size_ptr =
              gl_size_buffer.get_access<cl::sycl::access::mode::read_write>(
                  cgh);
          auto l_id_ptr =
              l_id_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
          auto l_size_ptr =
              l_size_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
          auto gr_id_ptr =
              gr_id_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
          auto gr_range_ptr =
              gr_range_buffer.get_access<cl::sycl::access::mode::read_write>(
                  cgh);

          cgh.parallel_for_work_group<class hierarchical_id>(
              my_range, [=](group<3> group) {

                int gr_id_0 = group.get(0);
                int gr_id_1 = group.get(1);
                int gr_id_2 = group.get(2);

                int gr_id_L = group.get_group_linear_id();

                int gr_range_0 = group.get_num_groups(0);
                int gr_range_1 = group.get_num_groups(1);
                int gr_range_2 = group.get_num_groups(2);
                int gr_range_L = group.get_num_groups_linear();

                gl_id_ptr[gr_id_L] = int4(gr_id_0, gr_id_1, gr_id_2, gr_id_L);
                gr_range_ptr[gr_id_L] =
                    int4(gr_range_0, gr_range_1, gr_range_2, gr_range_L);

                parallel_for_work_item(group, [&](item<3> item_id) {
                  int gl_id_0 = item_id.get_global(0);
                  int gl_id_1 = item_id.get_global(1);
                  int gl_id_2 = item_id.get_global(2);

                  int gl_id_L = item_id.get_global_linear_id();

                  int gl_size_0 = item_id.get_global_range(0);
                  int gl_size_1 = item_id.get_global_range(1);
                  int gl_size_2 = item_id.get_global_range(2);
                  int gl_size_L = gl_size_0 * gl_size_1 * gl_size_2;

                  int l_id_0 = item_id.get_local(0);
                  int l_id_1 = item_id.get_local(1);
                  int l_id_2 = item_id.get_local(2);

                  int l_id_L = item_id.get_local_linear_id();

                  int l_size_0 = item_id.get_local_range(0);
                  int l_size_1 = item_id.get_local_range(1);
                  int l_size_2 = item_id.get_local_range(2);
                  int l_size_L = item_id.get_local_linear_range();

                  gl_id_ptr[gl_id_L] = int4(gl_id_0, gl_id_1, gl_id_2, gl_id_L);
                  gl_size_ptr[gl_id_L] =
                      int4(gl_size_0, gl_size_1, gl_size_2, gl_size_L);
                  l_id_ptr[gl_id_L] = int4(l_id_0, l_id_1, l_id_2, l_id_L);
                  l_size_ptr[gl_id_L] =
                      int4(l_size_0, l_size_1, l_size_2, l_size_L);
                });

              });

        });
      }

      bool fail = false;

      for (int k = 0; k < g_items_3d; k++) {
        for (int j = 0; j < g_items_2d; j++) {
          for (int i = 0; i < g_items_1d; i++) {
            int linear_index =
                ((k * g_items_2d * g_items_1d) + (j * g_items_1d) + i);
            if (gl_id_data[linear_index].x() != i) {
              fail = true;
            }
            if (gl_id_data[linear_index].y() != j) {
              fail = true;
            }
            if (gl_id_data[linear_index].z() != k) {
              fail = true;
            }
            if (gl_id_data[linear_index].w() != linear_index) {
              fail = true;
            }
            if (gl_size_data[linear_index].x() != g_items_1d) {
              fail = true;
            }
            if (gl_size_data[linear_index].y() != g_items_2d) {
              fail = true;
            }
            if (gl_size_data[linear_index].z() != g_items_3d) {
              fail = true;
            }
            if (gl_size_data[linear_index].w() != g_items_total) {
              fail = true;
            }
            if (l_id_data[linear_index].x() != (i % l_items_1d)) {
              fail = true;
            }
            if (l_id_data[linear_index].y() != (j % l_items_2d)) {
              fail = true;
            }
            if (l_id_data[linear_index].z() != (k % l_items_3d)) {
              fail = true;
            }
            if (l_id_data[linear_index].w() != (linear_index % l_items_total)) {
              fail = true;
            }
            if (l_size_data[linear_index].x() != l_items_1d) {
              fail = true;
            }
            if (l_size_data[linear_index].y() != l_items_2d) {
              fail = true;
            }
            if (l_size_data[linear_index].z() != l_items_3d) {
              fail = true;
            }
            if (l_size_data[linear_index].w() != l_items_total) {
              fail = true;
            }
          }
        }
      }

      for (int k = 0; k < gr_range_1d; k++) {
        for (int j = 0; j < gr_range_2d; j++) {
          for (int i = 0; i < gr_range_3d; i++) {
            int linear_index =
                ((k * gr_range_2d * gr_range_1d) + (j * gr_range_1d) + i);
            if (gr_id_data[linear_index].x() != i) {
              fail = true;
            }
            if (gr_id_data[linear_index].y() != j) {
              fail = true;
            }
            if (gr_id_data[linear_index].z() != k) {
              fail = true;
            }
            if (gr_id_data[linear_index].w() != linear_index) {
              fail = true;
            }
            if (gr_range_data[linear_index].x() != gr_range_1d) {
              fail = true;
            }
            if (gr_range_data[linear_index].y() != gr_range_2d) {
              fail = true;
            }
            if (gr_range_data[linear_index].z() != gr_range_3d) {
              fail = true;
            }
            if (gr_range_data[linear_index].w() != gr_range_total) {
              fail = true;
            }
          }
        }
      }

      if (fail) {
        FAIL(log, " One of fail statements has been triggered. ");
      }

      testQueue.wait_and_throw();

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace id_api__ */
