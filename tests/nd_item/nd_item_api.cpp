/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

#include <algorithm>

#define TEST_NAME nd_item_api

namespace test_nd_item__ {
using namespace sycl_cts;

size_t getIndex(sycl::id<1> Id, sycl::range<1> Range) { return Id.get(0); }

size_t getIndex(sycl::id<2> Id, sycl::range<2> Range) {
  return Id.get(1) + Id.get(0) * Range.get(1);
}

size_t getIndex(sycl::id<3> Id, sycl::range<3> Range) {
  return Id.get(2) + Id.get(1) * Range.get(0) +
         Id.get(0) * Range.get(0) * Range.get(1);
}

template <int dimensions>
class kernel_nd_item {
 protected:
  using t_readAccess = sycl::accessor<int, dimensions, sycl::access_mode::read,
                                      sycl::target::device>;
  using t_writeAccess =
      sycl::accessor<int, dimensions, sycl::access_mode::write,
                     sycl::target::device>;

  t_readAccess m_globalID;
  t_readAccess m_localID;
  t_writeAccess m_out;
  t_writeAccess m_out_deprecated;

 public:
  kernel_nd_item(t_readAccess inG_, t_readAccess inL_, t_writeAccess out_,
                 t_writeAccess out_deprecated_)
      : m_globalID(inG_),
        m_localID(inL_),
        m_out(out_),
        m_out_deprecated(out_deprecated_) {}

  void operator()(sycl::nd_item<dimensions> myitem) const {
    bool all_correct = true;

    /* test global ID */
    sycl::id<dimensions> global_id = myitem.get_global_id();
    for (int i = 0; i < dimensions; ++i) {
      all_correct &= myitem.get_global_id(i) == global_id.get(i);
    }

    /* test local ID */
    sycl::id<dimensions> local_id = myitem.get_local_id();
    for (int i = 0; i < dimensions; ++i) {
      all_correct &= myitem.get_local_id(i) == local_id.get(i);
    }

    /* test group ID */
    sycl::group<dimensions> group_id = myitem.get_group();
    for (int i = 0; i < dimensions; ++i) {
      all_correct &= myitem.get_group(i) == group_id.get_id(i);
    }

    /* test range */
    sycl::range<dimensions> globalRange = myitem.get_global_range();
    for (int i = 0; i < dimensions; ++i) {
      all_correct &= myitem.get_global_range(i) == globalRange.get(i);
    }

    size_t globalIndex = getIndex(global_id, globalRange);
    all_correct &= m_globalID[global_id] == globalIndex;

    sycl::range<dimensions> localRange = myitem.get_local_range();
    for (int i = 0; i < dimensions; ++i) {
      all_correct &= myitem.get_local_range(i) == localRange.get(i);
    }

    size_t localIndex = getIndex(local_id, localRange);
    all_correct &= m_localID[local_id] == localIndex;

    for (int i = 0; i < dimensions; ++i) {
      size_t ratio = global_id.get(i) / localRange.get(i);
      all_correct &= group_id.get_id(i) == ratio;
    }

    /* test number of groups */
    sycl::id<dimensions> num_groups = myitem.get_group_range();
    for (int i = 0; i < dimensions; ++i) {
      all_correct &= myitem.get_group_range(i) == num_groups.get(i);
    }

    for (int i = 0; i < dimensions; ++i) {
      size_t ratio = globalRange.get(i) / localRange.get(i);
      all_correct &= ratio == num_groups.get(i);
    }

    /* test NDrange */
    sycl::nd_range<dimensions> NDRange = myitem.get_nd_range();
    sycl::range<dimensions> ndGlobal = NDRange.get_global_range();
    sycl::range<dimensions> ndLocal = NDRange.get_local_range();

    for (int i = 0; i < dimensions; ++i) {
      all_correct &= globalRange.get(i) == ndGlobal.get(i);
      all_correct &= localRange.get(i) == ndLocal.get(i);
    }

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    /* test offset */
    sycl::id<dimensions> offset = myitem.get_offset();
    sycl::id<dimensions> ndOffset = NDRange.get_offset();
    bool deprecated_all_correct = true;
    for (int i = 0; i < dimensions; ++i) {
      deprecated_all_correct &= offset.get(i) == ndOffset.get(i);
    }
    m_out_deprecated[global_id] = deprecated_all_correct;
#endif

    /* test linear_id */
    size_t glid = myitem.get_global_linear_id();
    size_t llid = myitem.get_local_linear_id();
    size_t grlid = myitem.get_group_linear_id();
    size_t groupIndex = getIndex(group_id.get_id(), myitem.get_group_range());

    all_correct &= glid == globalIndex;
    all_correct &= llid == localIndex;
    all_correct &= grlid == groupIndex;

    /* write back whether all checks were successful */
    m_out[global_id] = all_correct;
  }
};

/* Fill buffers with global and local work item ids*/
void populate(int *globalBuf, int *localBuf, const int *localSize,
              const int *globalSize) {
  for (int k = 0; k < globalSize[2]; k++) {
    for (int j = 0; j < globalSize[1]; j++) {
      for (int i = 0; i < globalSize[0]; i++) {
        const int globIndex =
            (k * globalSize[0] * globalSize[1]) + (j * globalSize[0]) + i;
        globalBuf[globIndex] = globIndex;

        int local_i = i % localSize[0];
        int local_j = j % localSize[1];
        int local_k = k % localSize[2];

        const int locIndex = (local_k * localSize[0] * localSize[1]) +
                             (local_j * localSize[0]) + local_i;
        localBuf[globIndex] = locIndex;
      }
    }
  }
}

void test_item(util::logger &log, sycl::queue &queue) {
  /* set sizes*/
  const int globalSize[3] = {16, 16, 16};
  const int localSize[3] = {4, 4, 4};
  const int nSize = globalSize[0] * globalSize[1] * globalSize[2];

  /* allocate and set host buffers */
  std::vector<int> globalIDs(nSize);
  std::vector<int> localIDs(nSize);
  populate(globalIDs.data(), localIDs.data(), localSize, globalSize);
  std::vector<int> dataOut(nSize);
  std::vector<int> dataOutDeprecated(nSize);

  /* test 1 Dimension*/
  {
    std::fill(dataOut.begin(), dataOut.end(), 0);
    std::fill(dataOutDeprecated.begin(), dataOutDeprecated.end(), 0);

    /* create ranges*/
    sycl::range<1> globalRange(globalSize[0]);
    sycl::range<1> localRange(localSize[0]);
    sycl::nd_range<1> dataRange(globalRange, localRange);

    {
      sycl::buffer<int, 1> bufGlob(globalIDs.data(), globalRange);
      sycl::buffer<int, 1> bufLoc(localIDs.data(), globalRange);
      sycl::buffer<int, 1> bufOut(dataOut.data(), globalRange);
      sycl::buffer<int, 1> bufOutDeprecated(dataOutDeprecated.data(),
                                            globalRange);

      queue.submit([&](sycl::handler &cgh) {
        kernel_nd_item<1> kernel_1d(
            bufGlob.template get_access<sycl::access_mode::read>(cgh),
            bufLoc.template get_access<sycl::access_mode::read>(cgh),
            bufOut.template get_access<sycl::access_mode::write>(cgh),
            bufOutDeprecated.template get_access<sycl::access_mode::write>(
                cgh));
        cgh.parallel_for<kernel_nd_item<1>>(dataRange, kernel_1d);
      });
    }

    CHECK(std::all_of(dataOut.begin(), dataOut.begin() + globalRange.size(),
                      [](int val) { return val; }));

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    CHECK(std::all_of(dataOutDeprecated.begin(),
                      dataOutDeprecated.begin() + globalRange.size(),
                      [](int val) { return val; }));
#endif

    STATIC_CHECK_FALSE(std::is_default_constructible_v<sycl::nd_item<1>>);
  }

  /* test 2 Dimensions */
  {
    std::fill(dataOut.begin(), dataOut.end(), 0);
    std::fill(dataOutDeprecated.begin(), dataOutDeprecated.end(), 0);

    /* create ranges*/
    sycl::range<2> globalRange(globalSize[0], globalSize[1]);
    sycl::range<2> localRange(localSize[0], localSize[1]);
    sycl::nd_range<2> dataRange(globalRange, localRange);

    {
      sycl::buffer<int, 2> bufGlob(globalIDs.data(), globalRange);
      sycl::buffer<int, 2> bufLoc(localIDs.data(), globalRange);
      sycl::buffer<int, 2> bufOut(dataOut.data(), globalRange);
      sycl::buffer<int, 2> bufOutDeprecated(dataOutDeprecated.data(),
                                            globalRange);

      queue.submit([&](sycl::handler &cgh) {
        kernel_nd_item<2> kernel_2d(
            bufGlob.template get_access<sycl::access_mode::read>(cgh),
            bufLoc.template get_access<sycl::access_mode::read>(cgh),
            bufOut.template get_access<sycl::access_mode::write>(cgh),
            bufOutDeprecated.template get_access<sycl::access_mode::write>(
                cgh));
        cgh.parallel_for<kernel_nd_item<2>>(dataRange, kernel_2d);
      });
    }

    CHECK(std::all_of(dataOut.begin(), dataOut.begin() + globalRange.size(),
                      [](int val) { return val; }));

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    CHECK(std::all_of(dataOutDeprecated.begin(),
                      dataOutDeprecated.begin() + globalRange.size(),
                      [](int val) { return val; }));
#endif

    STATIC_CHECK_FALSE(std::is_default_constructible_v<sycl::nd_item<2>>);
  }

  /* test 3 Dimensions */
  {
    std::fill(dataOut.begin(), dataOut.end(), 0);
    std::fill(dataOutDeprecated.begin(), dataOutDeprecated.end(), 0);

    /* create ranges*/
    sycl::range<3> globalRange(globalSize[0], globalSize[1], globalSize[2]);
    sycl::range<3> localRange(localSize[0], localSize[1], localSize[2]);
    sycl::nd_range<3> dataRange(globalRange, localRange);

    {
      sycl::buffer<int, 3> bufGlob(globalIDs.data(), globalRange);
      sycl::buffer<int, 3> bufLoc(localIDs.data(), globalRange);
      sycl::buffer<int, 3> bufOut(dataOut.data(), globalRange);
      sycl::buffer<int, 3> bufOutDeprecated(dataOutDeprecated.data(),
                                            globalRange);

      queue.submit([&](sycl::handler &cgh) {
        kernel_nd_item<3> kernel_3d(
            bufGlob.template get_access<sycl::access_mode::read>(cgh),
            bufLoc.template get_access<sycl::access_mode::read>(cgh),
            bufOut.template get_access<sycl::access_mode::write>(cgh),
            bufOutDeprecated.template get_access<sycl::access_mode::write>(
                cgh));
        cgh.parallel_for<kernel_nd_item<3>>(dataRange, kernel_3d);
      });
    }

    CHECK(std::all_of(dataOut.begin(), dataOut.begin() + globalRange.size(),
                      [](int val) { return val; }));

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    CHECK(std::all_of(dataOutDeprecated.begin(),
                      dataOutDeprecated.begin() + globalRange.size(),
                      [](int val) { return val; }));
#endif

    STATIC_CHECK_FALSE(std::is_default_constructible_v<sycl::nd_item<3>>);
  }
}

/** test sycl::nd_item
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   *  @param info, test_base::info structure as output
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   *  @param log, test transcript logging class
   */
  void run(util::logger &log) override {
    {
      auto cmd_queue = util::get_cts_object::queue();

      test_item(log, cmd_queue);

      cmd_queue.wait_and_throw();
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace test_nd_item__ */
