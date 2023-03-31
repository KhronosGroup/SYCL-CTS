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

struct getter {
  enum class methods : size_t {
    global_id = 0,
    local_id,
    group_id,
    range,
    number_of_groups,
    nd_range,
    linear_id,
    methods_count
  };

  static constexpr auto method_cnt = to_integral(methods::methods_count);

  static const char *method_name(methods method) {
    switch (method) {
      case methods::global_id:
        return "global_id";
      case methods::local_id:
        return "local_id";
      case methods::group_id:
        return "group_id";
      case methods::range:
        return "range";
      case methods::number_of_groups:
        return "number_of_groups";
      case methods::nd_range:
        return "nd_range,";
      case methods::linear_id:
        return "linear_id";
      case methods::methods_count:
        return "invalid enum value";
    }
  }
};

template <int dimensions>
class kernel_nd_item {
 protected:
  using t_readAccess = sycl::accessor<int, dimensions, sycl::access_mode::read,
                                      sycl::target::device>;
  using t_errorAccess =
      sycl::accessor<int, 1, sycl::access_mode::write, sycl::target::device>;
  using t_writeAccess =
      sycl::accessor<int, dimensions, sycl::access_mode::write,
                     sycl::target::device>;

  t_readAccess m_globalID;
  t_readAccess m_localID;
  t_errorAccess m_out;
  t_writeAccess m_out_deprecated;

 public:
  kernel_nd_item(t_readAccess inG_, t_readAccess inL_, t_errorAccess out_,
                 t_writeAccess out_deprecated_)
      : m_globalID(inG_),
        m_localID(inL_),
        m_out(out_),
        m_out_deprecated(out_deprecated_) {}

  void operator()(sycl::nd_item<dimensions> myitem) const {
    bool global_id_res = true;
    bool local_id_res = true;
    bool group_id_res = true;
    bool range_res = true;
    bool number_of_groups_res = true;
    bool nd_range_res = true;
    bool offset_res = true;
    bool linear_id_res = true;

    /* test global ID */
    sycl::id<dimensions> global_id = myitem.get_global_id();
    for (int i = 0; i < dimensions; ++i) {
      global_id_res &= myitem.get_global_id(i) == global_id.get(i);
    }

    /* test local ID */
    sycl::id<dimensions> local_id = myitem.get_local_id();
    for (int i = 0; i < dimensions; ++i) {
      local_id_res &= myitem.get_local_id(i) == local_id.get(i);
    }

    /* test group ID */
    sycl::group<dimensions> group_id = myitem.get_group();
    for (int i = 0; i < dimensions; ++i) {
      group_id_res &= myitem.get_group(i) == group_id.get_id(i);
    }

    /* test range */
    sycl::range<dimensions> globalRange = myitem.get_global_range();
    for (int i = 0; i < dimensions; ++i) {
      range_res &= myitem.get_global_range(i) == globalRange.get(i);
    }

    size_t globalIndex = getIndex(global_id, globalRange);
    range_res &= m_globalID[global_id] == globalIndex;

    sycl::range<dimensions> localRange = myitem.get_local_range();
    for (int i = 0; i < dimensions; ++i) {
      range_res &= myitem.get_local_range(i) == localRange.get(i);
    }

    size_t localIndex = getIndex(local_id, localRange);
    range_res &= m_localID[local_id] == localIndex;

    for (int i = 0; i < dimensions; ++i) {
      size_t ratio = global_id.get(i) / localRange.get(i);
      range_res &= group_id.get_id(i) == ratio;
    }

    /* test number of groups */
    sycl::id<dimensions> num_groups = myitem.get_group_range();
    for (int i = 0; i < dimensions; ++i) {
      number_of_groups_res &= myitem.get_group_range(i) == num_groups.get(i);
    }

    for (int i = 0; i < dimensions; ++i) {
      size_t ratio = globalRange.get(i) / localRange.get(i);
      number_of_groups_res &= ratio == num_groups.get(i);
    }

    /* test NDrange */
    sycl::nd_range<dimensions> NDRange = myitem.get_nd_range();
    sycl::range<dimensions> ndGlobal = NDRange.get_global_range();
    sycl::range<dimensions> ndLocal = NDRange.get_local_range();

    for (int i = 0; i < dimensions; ++i) {
      nd_range_res &= globalRange.get(i) == ndGlobal.get(i);
      nd_range_res &= localRange.get(i) == ndLocal.get(i);
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

    linear_id_res &= glid == globalIndex;
    linear_id_res &= llid == localIndex;
    linear_id_res &= grlid == groupIndex;

    /* write back whether all checks were successful */
    m_out[to_integral(getter::methods::global_id)] = global_id_res;
    m_out[to_integral(getter::methods::local_id)] = local_id_res;
    m_out[to_integral(getter::methods::group_id)] = group_id_res;
    m_out[to_integral(getter::methods::range)] = range_res;
    m_out[to_integral(getter::methods::number_of_groups)] =
        number_of_groups_res;
    m_out[to_integral(getter::methods::nd_range)] = nd_range_res;
    m_out[to_integral(getter::methods::linear_id)] = linear_id_res;
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

template <int dims>
void test_item() {
  auto queue = util::get_cts_object::queue();

  /* set sizes*/
  const int globalSize[3] = {16, 16, 16};
  const int localSize[3] = {4, 4, 4};
  const int nSize = globalSize[0] * globalSize[1] * globalSize[2];
  const int nErrorSize = getter::method_cnt;

  /* allocate and set host buffers */
  std::vector<int> globalIDs(nSize);
  std::vector<int> localIDs(nSize);
  populate(globalIDs.data(), localIDs.data(), localSize, globalSize);
  std::vector<int> dataOut(nErrorSize, true);
  std::vector<int> dataOutDeprecated(nSize);

  {
    std::fill(dataOutDeprecated.begin(), dataOutDeprecated.end(), 0);

    /* create ranges*/
    auto globalRange = util::get_cts_object::range<dims>::get(
        globalSize[0], globalSize[1], globalSize[2]);
    auto localRange = util::get_cts_object::range<dims>::get(
        localSize[0], localSize[1], localSize[2]);
    sycl::range<1> errorRange(nErrorSize);
    sycl::nd_range<dims> dataRange(globalRange, localRange);

    {
      sycl::buffer<int, dims> bufGlob(globalIDs.data(), globalRange);
      sycl::buffer<int, dims> bufLoc(localIDs.data(), globalRange);
      sycl::buffer<int, 1> bufOut(dataOut.data(), errorRange);
      sycl::buffer<int, dims> bufOutDeprecated(dataOutDeprecated.data(),
                                               globalRange);

      queue.submit([&](sycl::handler &cgh) {
        kernel_nd_item<dims> kernel_(
            bufGlob.template get_access<sycl::access_mode::read>(cgh),
            bufLoc.template get_access<sycl::access_mode::read>(cgh),
            bufOut.template get_access<sycl::access_mode::write>(cgh),
            bufOutDeprecated.template get_access<sycl::access_mode::write>(
                cgh));
        cgh.parallel_for<kernel_nd_item<dims>>(dataRange, kernel_);
      });
    }

    // check api call results
    for (int i = 0; i < nErrorSize; i++) {
      INFO("Dimensions: " << std::to_string(dims));
      INFO("nd_item " << getter::method_name(static_cast<getter::methods>(i))
                      << " tests fails");
      CHECK(dataOut[i] != 0);
    }

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    CHECK(std::all_of(dataOutDeprecated.begin(),
                      dataOutDeprecated.begin() + globalRange.size(),
                      [](int val) { return val; }));
#endif

    STATIC_CHECK_FALSE(std::is_default_constructible_v<sycl::nd_item<dims>>);
  }
  queue.wait_and_throw();
}

TEST_CASE("nd_item_1d API", "[nd_item]") { test_item<1>(); }

TEST_CASE("nd_item_2d API", "[nd_item]") { test_item<2>(); }

TEST_CASE("nd_item_3d API", "[nd_item]") { test_item<3>(); }
} /* namespace test_nd_item__ */
