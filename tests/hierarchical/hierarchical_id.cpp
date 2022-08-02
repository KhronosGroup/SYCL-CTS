/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_id

namespace TEST_NAMESPACE {

static const int g_items_1d = 8;
static const int g_items_2d = 4;
static const int g_items_3d = 2;
static const int l_items_1d = 4;
static const int l_items_2d = 2;
static const int l_items_3d = 1;
static const int gr_range_1d = (g_items_1d / l_items_1d);
static const int gr_range_2d = (g_items_2d / l_items_2d);
static const int gr_range_3d = (g_items_3d / l_items_3d);
static const int gl_items_total = (g_items_1d * g_items_2d * g_items_3d);
static const int l_items_total = (l_items_1d * l_items_2d * l_items_3d);
static const int gr_range_total = (gl_items_total / l_items_total);

using namespace sycl_cts;

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
              localSizeBuffer.get_access<cl::sycl::access::mode::read_write>(
                  cgh);
          auto groupIdPtr =
              groupIdBuffer.get_access<cl::sycl::access::mode::read_write>(cgh);
          auto groupRangePtr =
              groupRangeBuffer.get_access<cl::sycl::access::mode::read_write>(
                  cgh);

          cgh.parallel_for_work_group<class hierarchical_id>(
              cl::sycl::range<3>(gr_range_1d, gr_range_2d, gr_range_3d),
              cl::sycl::range<3>(l_items_1d, l_items_2d, l_items_3d),
              [=](cl::sycl::group<3> group) {

                int groupId0 = group.get_id(0);
                int groupId1 = group.get_id(1);
                int groupId2 = group.get_id(2);
                int groupIdL = group.get_linear_id();

                int groupRange0 = group.get_group_range(0);
                int groupRange1 = group.get_group_range(1);
                int groupRange2 = group.get_group_range(2);
                int groupRangeL = group.get_group_range().size();

                groupIdPtr[groupIdL] =
                    cl::sycl::int4(groupId0, groupId1, groupId2, groupIdL);
                groupRangePtr[groupIdL] = cl::sycl::int4(
                    groupRange0, groupRange1, groupRange2, groupRangeL);

                group.parallel_for_work_item([&](cl::sycl::h_item<3> itemID) {

                  int localId0 = itemID.get_local_id(0);
                  int localId1 = itemID.get_local_id(1);
                  int localId2 = itemID.get_local_id(2);
                  int localIdL = itemID.get_local().get_linear_id();

                  int localSize0 = itemID.get_local_range(0);
                  int localSize1 = itemID.get_local_range(1);
                  int localSize2 = itemID.get_local_range(2);
                  int localSizeL = itemID.get_local_range(0) *
                                   itemID.get_local_range(1) *
                                   itemID.get_local_range(2);

                  int globalId0 = group.get_id(0) * localSize0 + localId0;
                  int globalId1 = group.get_id(1) * localSize1 + localId1;
                  int globalId2 = group.get_id(2) * localSize2 + localId2;
                  int globalIdL = ((globalId0 * g_items_2d * g_items_3d) +
                                   (globalId1 * g_items_3d) + globalId2);

                  localIdPtr[globalIdL] =
                      cl::sycl::int4(localId0, localId1, localId2, localIdL);
                  localSizePtr[globalIdL] = cl::sycl::int4(
                      localSize0, localSize1, localSize2, localSizeL);
                });
              });
        });

        myQueue.wait_and_throw();
      }

      for (int k = 0; k < g_items_3d; k++) {
        for (int j = 0; j < g_items_2d; j++) {
          for (int i = 0; i < g_items_1d; i++) {
            int gLinearIndex =
                ((i * g_items_2d * g_items_3d) + (j * g_items_3d) + k);
            int linearIndex =
                (((i % l_items_1d) * l_items_2d * l_items_3d) +
                 ((j % l_items_2d) * l_items_3d) + k % l_items_3d);
            CHECK_VALUE(log, static_cast<int>(localIdData[gLinearIndex].x()),
                        (i % l_items_1d), gLinearIndex);
            CHECK_VALUE(log, static_cast<int>(localIdData[gLinearIndex].x()),
                        (i % l_items_1d), gLinearIndex);
            CHECK_VALUE(log, static_cast<int>(localIdData[gLinearIndex].y()),
                        (j % l_items_2d), gLinearIndex);
            CHECK_VALUE(log, static_cast<int>(localIdData[gLinearIndex].z()),
                        (k % l_items_3d), gLinearIndex);
            CHECK_VALUE(log, static_cast<int>(localIdData[gLinearIndex].w()),
                        linearIndex, gLinearIndex);

            CHECK_VALUE(log, static_cast<int>(localSizeData[gLinearIndex].x()),
                        l_items_1d, gLinearIndex);

            CHECK_VALUE(log, static_cast<int>(localSizeData[gLinearIndex].y()),
                        l_items_2d, gLinearIndex);

            CHECK_VALUE(log, static_cast<int>(localSizeData[gLinearIndex].z()),
                        l_items_3d, gLinearIndex);

            CHECK_VALUE(log, static_cast<int>(localSizeData[gLinearIndex].w()),
                        l_items_total, gLinearIndex);
          }
        }
      }

      for (int k = 0; k < gr_range_1d; k++) {
        for (int j = 0; j < gr_range_2d; j++) {
          for (int i = 0; i < gr_range_3d; i++) {
            int linearIndex =
                ((i * gr_range_2d * gr_range_3d) + (j * gr_range_3d) + k);
            CHECK_VALUE(log, static_cast<int>(groupIdData[linearIndex].x()), i,
                        linearIndex);

            CHECK_VALUE(log, static_cast<int>(groupIdData[linearIndex].y()), j,
                        linearIndex);

            CHECK_VALUE(log, static_cast<int>(groupIdData[linearIndex].z()), k,
                        linearIndex);

            CHECK_VALUE(log, static_cast<int>(groupIdData[linearIndex].w()),
                        linearIndex, linearIndex);

            CHECK_VALUE(log, static_cast<int>(groupRangeData[linearIndex].x()),
                        gr_range_1d, linearIndex);

            CHECK_VALUE(log, static_cast<int>(groupRangeData[linearIndex].y()),
                        gr_range_2d, linearIndex);

            CHECK_VALUE(log, static_cast<int>(groupRangeData[linearIndex].z()),
                        gr_range_3d, linearIndex);

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
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace id_api__ */
