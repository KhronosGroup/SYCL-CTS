/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_implicit_conditional

namespace hierarchical_implicit_conditional__ {

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
      int item_ids[gr_range_total];

      cl::sycl::default_selector sel;
      cl::sycl::queue testQueue(sel);

      for (int i = 0; i < gr_range_total; i++) {
        item_ids[i] = 0;
      }

      int output_data[g_items_total];
      for (int i = 0; i < g_items_total; i++) {
        output_data[i] = 0;
      }

      {
        buffer<int, 1> item_ids_buffer(item_ids, range<1>(gr_range_total));
        buffer<int, 1> output_buffer(output_data, range<1>(g_items_total));

        testQueue.submit([&](handler &cgh) {

          auto my_range =
              nd_range<3>(range<3>(g_items_1d, g_items_2d, g_items_3d),
                          range<3>(l_items_1d, l_items_2d, l_items_3d));

          auto item_ids_ptr =
              item_ids_buffer.get_access<cl::sycl::access::mode::read_write>(
                  cgh);
          auto output_ptr =
              output_buffer.get_access<cl::sycl::access::mode::write>(cgh);

          cgh.parallel_for_work_group<class hierarchical_implicit_conditional>(
              my_range, [=](group<3> group) {

                // Create a local variable to store the work item id.
                int work_item_id;

                parallel_for_work_item(group, [&](item<3> item) {
                  // Assign the work item id to a local variable.
                  int global_id = item.get_global_linear();
                  work_item_id = global_id;
                });

                // Assign a value for the work item stored. Although this is not
                // recommened behaviour for the hierarchical API as there is a
                // data race
                // on the itemIds accessor and there is no gaurentee which work
                // item id
                // will be taken, this test makes sure that the assigment is
                // only being
                // done once.
                output_ptr[work_item_id] = 2;

              });
        });
      }

      int result = 0;
      for (int i = 0; i < g_items_total; i++) {
        result += output_data[i];
      }

      if (result != 16) {
        FAIL(log, "Result not as expected.");
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
