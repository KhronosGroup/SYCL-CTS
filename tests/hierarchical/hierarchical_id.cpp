/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_id

namespace TEST_NAMESPACE {

template <int dim> class kernel;

static const int g_items_1d = 8;
static const int g_items_2d = 4;
static const int g_items_3d = 2;
static const int l_items_1d = 4;
static const int l_items_2d = 2;
static const int l_items_3d = 1;
static const int gr_range_1d = (g_items_1d / l_items_1d);
static const int gr_range_2d = (g_items_2d / l_items_2d);
static const int gl_items_total = (g_items_1d * g_items_2d * g_items_3d);
static const int l_items_total = (l_items_1d * l_items_2d * l_items_3d);
static const int gr_range_total = (gl_items_total / l_items_total);

using namespace sycl_cts;

template <int dim> void check_dim(util::logger &log) {
  const int check_g_items_1d = (dim > 1) ? g_items_1d : gl_items_total;
  const int check_g_items_2d =
      (dim > 1) ? ((dim > 2) ? g_items_2d : gl_items_total / g_items_1d) : 1;
  const int check_g_items_3d = (dim > 2) ? g_items_3d : 1;
  const int check_l_items_1d = (dim > 1) ? l_items_1d : l_items_total;
  const int check_l_items_2d =
      (dim > 1) ? ((dim > 2) ? l_items_2d : l_items_total / l_items_1d) : 1;
  const int check_l_items_3d = (dim > 2) ? l_items_3d : 1;
  const int check_gr_range_1d = (check_g_items_1d / check_l_items_1d);
  const int check_gr_range_2d = (check_g_items_2d / check_l_items_2d);
  const int check_gr_range_3d = (check_g_items_3d / check_l_items_3d);

  try {
    cl::sycl::int4 localIdData[gl_items_total];
    cl::sycl::int4 localSizeData[gl_items_total];
    for (int i = 0; i < gl_items_total; i++) {
      localIdData[i] = cl::sycl::int4(-1, -1, -1, -1);
      localSizeData[i] = cl::sycl::int4(-1, -1, -1, -1);
    }

    cl::sycl::int4 groupIdData[l_items_total];
    cl::sycl::int4 groupRangeData[l_items_total];
    for (int i = 0; i < l_items_total; i++) {
      groupIdData[i] = cl::sycl::int4(-1, -1, -1, -1);
      groupRangeData[i] = cl::sycl::int4(-1, -1, -1, -1);
    }

    {
      cl::sycl::buffer<cl::sycl::int4, 1> localIdBuffer(
          localIdData, cl::sycl::range<1>(gl_items_total));
      cl::sycl::buffer<cl::sycl::int4, 1> localSizeBuffer(
          localSizeData, cl::sycl::range<1>(gl_items_total));
      cl::sycl::buffer<cl::sycl::int4, 1> groupIdBuffer(
          groupIdData, cl::sycl::range<1>(l_items_total));
      cl::sycl::buffer<cl::sycl::int4, 1> groupRangeBuffer(
          groupRangeData, cl::sycl::range<1>(l_items_total));

      cl::sycl::queue myQueue(util::get_cts_object::queue());

      myQueue.submit([&](cl::sycl::handler &cgh) {

        auto localIdPtr =
            localIdBuffer.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto localSizePtr =
            localSizeBuffer.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto groupIdPtr =
            groupIdBuffer.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto groupRangePtr =
            groupRangeBuffer.get_access<cl::sycl::access::mode::read_write>(
                cgh);

        auto gr_range =
            sycl_cts::util::get_cts_object::range<dim>::template get_fixed_size<
                gr_range_total>(gr_range_1d, gr_range_2d);
        auto l_range =
            sycl_cts::util::get_cts_object::range<dim>::template get_fixed_size<
                l_items_total>(l_items_1d, l_items_2d);

        cgh.parallel_for_work_group<kernel<dim>>(
            gr_range, l_range, [=](cl::sycl::group<dim> group) {

              int groupId0 = group.get_id(0);
              int groupId1 = (dim > 1) ? group.get_id(1) : 0;
              int groupId2 = (dim > 2) ? group.get_id(2) : 0;
              int groupIdL = group.get_linear_id();

              int groupRange0 = group.get_group_range(0);
              int groupRange1 = (dim > 1) ? group.get_group_range(1) : 0;
              int groupRange2 = (dim > 2) ? group.get_group_range(2) : 0;
              int groupRangeL = group.get_group_range().size();

              // Assign work-group id and range size
              groupIdPtr[groupIdL] =
                  cl::sycl::int4(groupId0, groupId1, groupId2, groupIdL);
              groupRangePtr[groupIdL] = cl::sycl::int4(
                  groupRange0, groupRange1, groupRange2, groupRangeL);

              group.parallel_for_work_item([&](cl::sycl::h_item<dim> itemID) {

                int localId0 = itemID.get_local_id(0);
                int localId1 = (dim > 1) ? itemID.get_local_id(1) : 0;
                int localId2 = (dim > 2) ? itemID.get_local_id(2) : 0;
                int localIdL = itemID.get_local().get_linear_id();

                int localSize0 = itemID.get_local_range(0);
                int localSize1 = (dim > 1) ? itemID.get_local_range(1) : 0;
                int localSize2 = (dim > 2) ? itemID.get_local_range(2) : 0;
                int localSizeL = itemID.get_local_range(0);
                if (dim > 1) {
                  localSizeL *= itemID.get_local_range(1);
                  if (dim > 2)
                    localSizeL *= itemID.get_local_range(2);
                }

                int globalId0 = group.get_id(0) * localSize0 + localId0;
                int globalId1 =
                    (dim > 1) ? group.get_id(1) * localSize1 + localId1 : 0;
                int globalId2 =
                    (dim > 2) ? group.get_id(2) * localSize2 + localId2 : 0;
                int globalIdL =
                    ((globalId0 * check_g_items_2d * check_g_items_3d) +
                     (globalId1 * check_g_items_3d) + globalId2);

                // Assign local id and range size to check with corresponding
                // global id
                localIdPtr[globalIdL] =
                    cl::sycl::int4(localId0, localId1, localId2, localIdL);
                localSizePtr[globalIdL] = cl::sycl::int4(
                    localSize0, localSize1, localSize2, localSizeL);
              });
            });
      });
    }

    for (int k = 0; k < check_g_items_3d; k++) {
      for (int j = 0; j < check_g_items_2d; j++) {
        for (int i = 0; i < check_g_items_1d; i++) {
          int gLinearIndex = ((i * check_g_items_2d * check_g_items_3d) +
                              (j * check_g_items_3d) + k);
          int linearIndex =
              (((i % check_l_items_1d) * check_l_items_2d * check_l_items_3d) +
               ((j % check_l_items_2d) * check_l_items_3d) +
               k % check_l_items_3d);
          CHECK_VALUE(log, static_cast<int>(localIdData[gLinearIndex].x()),
                      (i % check_l_items_1d), gLinearIndex);

          CHECK_VALUE(log, static_cast<int>(localSizeData[gLinearIndex].x()),
                      check_l_items_1d, gLinearIndex);

          if (dim > 2) {
            CHECK_VALUE(log, static_cast<int>(localIdData[gLinearIndex].y()),
                        (j % l_items_2d), gLinearIndex);
            CHECK_VALUE(log, static_cast<int>(localSizeData[gLinearIndex].y()),
                        l_items_2d, gLinearIndex);
            if (dim > 3) {
              CHECK_VALUE(log, static_cast<int>(localIdData[gLinearIndex].z()),
                          (k % l_items_3d), gLinearIndex);
              CHECK_VALUE(log,
                          static_cast<int>(localSizeData[gLinearIndex].z()),
                          l_items_3d, gLinearIndex);
            }
          }

          CHECK_VALUE(log, static_cast<int>(localIdData[gLinearIndex].w()),
                      linearIndex, gLinearIndex);
          CHECK_VALUE(log, static_cast<int>(localSizeData[gLinearIndex].w()),
                      l_items_total, gLinearIndex);
        }
      }
    }

    for (int k = 0; k < check_gr_range_3d; k++) {
      for (int j = 0; j < check_gr_range_2d; j++) {
        for (int i = 0; i < check_gr_range_1d; i++) {
          int linearIndex = ((i * check_gr_range_2d * check_gr_range_3d) +
                             (j * check_gr_range_3d) + k);
          CHECK_VALUE(log, static_cast<int>(groupIdData[linearIndex].x()), i,
                      linearIndex);
          CHECK_VALUE(log, static_cast<int>(groupRangeData[linearIndex].x()),
                      check_gr_range_1d, linearIndex);
          if (dim > 2) {
            CHECK_VALUE(log, static_cast<int>(groupIdData[linearIndex].y()), j,
                        linearIndex);
            CHECK_VALUE(log, static_cast<int>(groupRangeData[linearIndex].y()),
                        check_gr_range_2d, linearIndex);
            if (dim > 3) {
              CHECK_VALUE(log, static_cast<int>(groupIdData[linearIndex].z()),
                          k, linearIndex);
              CHECK_VALUE(log,
                          static_cast<int>(groupRangeData[linearIndex].z()),
                          check_gr_range_3d, linearIndex);
            }
          }
          CHECK_VALUE(log, static_cast<int>(groupIdData[linearIndex].w()),
                      linearIndex, linearIndex);
          CHECK_VALUE(log, static_cast<int>(groupRangeData[linearIndex].w()),
                      gr_range_total, linearIndex);
        }
      }
    }
  } catch (const cl::sycl::exception &e) {
    log_exception(log, e);
    cl::sycl::string_class errorMsg =
        "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
    FAIL(log, errorMsg.c_str());
  }
}

/** test cl::sycl::range::get(int index) return size_t
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
    check_dim<1>(log);
    check_dim<2>(log);
    check_dim<3>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace id_api__ */
