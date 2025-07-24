/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
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

#include "../../util/array.h"
#include "../common/common.h"
#include "../common/range_index_space_id.h"

#include <cstddef>
#include <limits>
#include <sstream>

using namespace sycl_cts;

constexpr size_t GROUP_RANGE[3] = {2, 4, 8};
constexpr size_t DEFAULT_LOCAL_RANGE[3] = {4, 3, 2};
constexpr size_t GROUP_LINEAR_RANGE =
    GROUP_RANGE[0] * GROUP_RANGE[1] * GROUP_RANGE[2];
constexpr size_t DEFAULT_LOCAL_LINEAR_RANGE =
    DEFAULT_LOCAL_RANGE[0] * DEFAULT_LOCAL_RANGE[1] * DEFAULT_LOCAL_RANGE[2];

class getter {
 public:
  enum class method : size_t {
    group_id = 0,
    group_id_dims,
    local_id,
    local_id_dims,
    local_range,
    local_range_dims,
    group_range,
    group_range_dims,
    max_local_range,
    subscript,
    group_linear_id,
    local_linear_id,
    group_linear_range,
    local_linear_range,
    leader,
    method_count  // defines size, should be last
  };

  static constexpr auto method_cnt = to_integral(method::method_count);

  static const char* name(getter::method getterMethod) {
    switch (getterMethod) {
      case method::group_id:
        return "get_group_id()";
      case method::group_id_dims:
        return "get_group_id(int)";
      case method::local_id:
        return "get_local_id()";
      case method::local_id_dims:
        return "get_local_id(int)";
      case method::local_range:
        return "get_local_range()";
      case method::local_range_dims:
        return "get_local_range(int)";
      case method::group_range:
        return "get_group_range()";
      case method::group_range_dims:
        return "get_group_range(int)";
      case method::max_local_range:
        return "get_max_local_range";
      case method::subscript:
        return "operator[](int)";
      case method::group_linear_id:
        return "get_group_linear_id()";
      case method::local_linear_id:
        return "get_local_linear_id()";
      case method::group_linear_range:
        return "get_group_linear_range()";
      case method::local_linear_range:
        return "get_local_linear_range()";
      case method::leader:
        return "get_leader()";
      case method::method_count:
        return "invalid enum value";
    }
    // no default case to allow for compiler warning
    return nullptr;
  }
};

template <int dimensions>
class test_kernel;

template <int dimensions>
class test_helper {
 private:
  /** Maximum size. */
  static constexpr int NUM_RESULTS =
      GROUP_LINEAR_RANGE * DEFAULT_LOCAL_LINEAR_RANGE * getter::method_cnt;
  struct call_result_t {
    bool hasValidType;
    sycl_cts::util::array<size_t, dimensions> values;
  };
  std::vector<call_result_t> m_callResults{};
  sycl::range<dimensions> m_globalRange;
  sycl::range<dimensions> m_localRange;

  /**
   * Convert a linear group id, a linear local range, a linear local id,
   * and a getter method to an index in \p m_callResults. */
  static inline size_t get_index(size_t glid, size_t llrange, size_t llid,
                                 getter::method getterMethod) {
    const auto offset = to_integral(getterMethod);
    return glid * llrange * getter::method_cnt + llid * getter::method_cnt +
           offset;
  }

 public:
  test_helper(sycl::range<dimensions> globalRange,
              sycl::range<dimensions> localRange)
      : m_callResults(NUM_RESULTS),
        m_globalRange(globalRange),
        m_localRange(localRange) {
    for (size_t i = 0; i < NUM_RESULTS; i++) {
      auto& callResult = m_callResults[i];
      callResult.hasValidType = false;
      for (size_t j = 0; j < dimensions; j++) {
        // special value to verify that a value was written
        callResult.values[j] = std::numeric_limits<size_t>::max();
      }
    }
  }

  /** Set the call result if the return type is a one-dimensional value. */
  template <typename data_type>
  static void set_val(call_result_t& call_result, data_type val) {
    call_result.values[0] = val;
  }

  /** Set the call result if the return type is a multi-dimensional id. */
  static void set_id(call_result_t& call_result, sycl::id<dimensions> id) {
    for (size_t i = 0; i < dimensions; ++i) {
      call_result.values[i] = id.get(i);
    }
  }

  /**
   * Set the call result if the return type provides a \p get method
   * for each index. */
  template <typename function>
  static void set_for_dim(call_result_t& call_result, function func) {
    for (size_t i = 0; i < dimensions; ++i) {
      call_result.values[i] = func(i);
    }
  }

  /**
   * Launch a kernel to collect the results of calling the group's member
   * functions. */
  void collect_results(sycl::queue& queue) {
    sycl::buffer<call_result_t, 1> buffer(m_callResults.data(),
                                          sycl::range<1>(NUM_RESULTS));

    const size_t local_linear_range = m_localRange.size();
    queue.submit([&](sycl::handler& cgh) {
      auto accessor_device =
          buffer.template get_access<sycl::access_mode::read_write>(cgh);
      // use parallel_for(nd_range) to execute a kernel and control the
      // number of work-groups and work-items
      cgh.parallel_for<test_kernel<dimensions>>(
          sycl::nd_range<dimensions>(m_globalRange * m_localRange,
                                     m_localRange),
          [=](sycl::nd_item<dimensions> item) {
            // obtain indices independently of group api
            const size_t glid = item.get_group_linear_id();
            const size_t llrange = local_linear_range;
            const size_t llid = item.get_local_linear_id();
            sycl::group group = item.get_group();

            // helper function to obtain call result struct
            const auto get_res =
                [=](const getter::method& method) -> call_result_t& {
              return accessor_device[get_index(glid, llrange, llid, method)];
            };

            {  // get_group_id()
              call_result_t& res = get_res(getter::method::group_id);
              set_id(res, group.get_group_id());
              res.hasValidType = std::is_same_v<sycl::id<dimensions>,
                                                decltype(group.get_group_id())>;
            }
            {  // get_group_id(int)
              call_result_t& res = get_res(getter::method::group_id_dims);
              set_for_dim(res, [=](size_t i) { return group.get_group_id(i); });
              res.hasValidType =
                  std::is_same_v<size_t, decltype(group.get_group_id(0))>;
            }
            {  // get_local_id()
              call_result_t& res = get_res(getter::method::local_id);
              set_id(res, group.get_local_id());
              res.hasValidType = std::is_same_v<sycl::id<dimensions>,
                                                decltype(group.get_local_id())>;
            }
            {  // get_local_id(int)
              call_result_t& res = get_res(getter::method::local_id_dims);
              set_for_dim(res, [=](size_t i) { return group.get_local_id(i); });
              res.hasValidType =
                  std::is_same_v<size_t, decltype(group.get_local_id(0))>;
            }
            {  // get_local_range()
              call_result_t& res = get_res(getter::method::local_range);
              set_id(res, group.get_local_range());
              res.hasValidType =
                  std::is_same_v<sycl::range<dimensions>,
                                 decltype(group.get_local_range())>;
            }
            {  // get_local_range(int)
              call_result_t& res = get_res(getter::method::local_range_dims);
              set_for_dim(res,
                          [=](size_t i) { return group.get_local_range(i); });
              res.hasValidType =
                  std::is_same_v<size_t, decltype(group.get_local_range(0))>;
            }
            {  // get_group_range()
              call_result_t& res = get_res(getter::method::group_range);
              set_id(res, group.get_group_range());
              res.hasValidType =
                  std::is_same_v<sycl::range<dimensions>,
                                 decltype(group.get_group_range())>;
            }
            {  // get_group_range(int)
              call_result_t& res = get_res(getter::method::group_range_dims);
              set_for_dim(res,
                          [=](size_t i) { return group.get_group_range(i); });
              res.hasValidType =
                  std::is_same_v<size_t, decltype(group.get_group_range(0))>;
            }
            {  // get_max_local_range
              call_result_t& res = get_res(getter::method::max_local_range);
              set_id(res, group.get_max_local_range());
              res.hasValidType =
                  std::is_same_v<sycl::range<dimensions>,
                                 decltype(group.get_max_local_range())>;
            }
            {  // operator[]
              call_result_t& res = get_res(getter::method::subscript);
              set_for_dim(res, [=](size_t i) { return group[i]; });
              res.hasValidType = std::is_same_v<size_t, decltype(group[0])>;
            }
            {  // get_group_linear_id
              call_result_t& res = get_res(getter::method::group_linear_id);
              set_val(res, group.get_group_linear_id());
              res.hasValidType =
                  std::is_same_v<size_t, decltype(group.get_group_linear_id())>;
            }
            {  // get_group_linear_range
              call_result_t& res = get_res(getter::method::group_linear_range);
              set_val(res, group.get_group_linear_range());
              res.hasValidType =
                  std::is_same_v<size_t,
                                 decltype(group.get_group_linear_range())>;
            }
            {  // get_local_linear_id
              call_result_t& res = get_res(getter::method::local_linear_id);
              set_val(res, group.get_local_linear_id());
              res.hasValidType =
                  std::is_same_v<size_t, decltype(group.get_local_linear_id())>;
            }
            {  // get_local_linear_range
              call_result_t& res = get_res(getter::method::local_linear_range);
              set_val(res, group.get_local_linear_range());
              res.hasValidType =
                  std::is_same_v<size_t,
                                 decltype(group.get_local_linear_range())>;
            }
            {  // leader
              call_result_t& res = get_res(getter::method::leader);
              set_val(res, group.leader());
              res.hasValidType = std::is_same_v<bool, decltype(group.leader())>;
            }
          });
    });
  }

  static std::string format_range(sycl::range<dimensions> range) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < dimensions; i++) {
      ss << range[i] << (i + 1 < dimensions ? ", " : "]");
    }
    return ss.str();
  }

  /** Validates the results obtained by \p collect_results. */
  void validate_results() const {
    INFO("group range: " << format_range(m_globalRange)
                         << " local range: " << format_range(m_localRange));

    // for each work-group and each work-item
    validate_results(std::integral_constant<int, dimensions>{});
  }

 private:
  void validate_results(std::integral_constant<int, 1> loopSelector) const {
    static_cast<void>(loopSelector);
    for (size_t gid0 = 0; gid0 < m_globalRange[0]; ++gid0) {
      size_t glid = linearize({m_globalRange[0]}, {gid0});
      for (size_t lid0 = 0; lid0 < m_localRange[0]; ++lid0) {
        size_t llid = linearize({m_localRange[0]}, {lid0});
        validate_results_impl(glid, {gid0}, llid, {lid0});
      }
    }
  }

  void validate_results(std::integral_constant<int, 2> loopSelector) const {
    static_cast<void>(loopSelector);
    for (size_t gid0 = 0; gid0 < m_globalRange[0]; ++gid0) {
      for (size_t gid1 = 0; gid1 < m_globalRange[1]; ++gid1) {
        size_t glid =
            linearize({m_globalRange[0], m_globalRange[1]}, {gid0, gid1});
        for (size_t lid0 = 0; lid0 < m_localRange[0]; ++lid0) {
          for (size_t lid1 = 0; lid1 < m_localRange[1]; ++lid1) {
            size_t llid =
                linearize({m_localRange[0], m_localRange[1]}, {lid0, lid1});
            validate_results_impl(glid, {gid0, gid1}, llid, {lid0, lid1});
          }
        }
      }
    }
  }

  void validate_results(std::integral_constant<int, 3> loopSelector) const {
    static_cast<void>(loopSelector);
    for (size_t gid0 = 0; gid0 < m_globalRange[0]; ++gid0) {
      for (size_t gid1 = 0; gid1 < m_globalRange[1]; ++gid1) {
        for (size_t gid2 = 0; gid2 < m_globalRange[2]; ++gid2) {
          size_t glid =
              linearize({m_globalRange[0], m_globalRange[1], m_globalRange[2]},
                        {gid0, gid1, gid2});
          for (size_t lid0 = 0; lid0 < m_localRange[0]; ++lid0) {
            for (size_t lid1 = 0; lid1 < m_localRange[1]; ++lid1) {
              for (size_t lid2 = 0; lid2 < m_localRange[2]; ++lid2) {
                size_t llid = linearize(
                    {m_localRange[0], m_localRange[1], m_localRange[2]},
                    {lid0, lid1, lid2});
                validate_results_impl(glid, {gid0, gid1, gid2}, llid,
                                      {lid0, lid1, lid2});
              }
            }
          }
        }
      }
    }
  }

  void validate_results_impl(size_t glid, std::array<size_t, dimensions> gid,
                             size_t llid,
                             std::array<size_t, dimensions> lid) const {
    {  // operator[], get_group_id(), get_group_id(int)
      const std::array<size_t, dimensions>& expected = gid;
      check(glid, llid, getter::method::subscript, expected);
      check(glid, llid, getter::method::group_id, expected);
      check(glid, llid, getter::method::group_id_dims, expected);
    }
    {  // get_local_id(), get_local_id(int)
      const std::array<size_t, dimensions> expected = lid;
      check(glid, llid, getter::method::local_id, expected);
      check(glid, llid, getter::method::local_id_dims, expected);
    }
    {  // get_local_range(), get_local_range(int)
      const std::array<size_t, dimensions>& expected = get_local_range_values();
      check(glid, llid, getter::method::local_range, expected);
      check(glid, llid, getter::method::local_range_dims, expected);
    }
    {  // get_group_range(), get_group_range(int)
      std::array<size_t, dimensions> expected;
      for (size_t i = 0; i < dimensions; i++) {
        expected[i] = GROUP_RANGE[i];
      }
      check(glid, llid, getter::method::group_range, expected);
      check(glid, llid, getter::method::group_range_dims, expected);
    }
    {  // get_max_local_range()
      size_t expected = 0;
      for (size_t i = 0; i < dimensions; i++) {
        expected = std::max(expected, m_localRange.get(i));
      }
      check(glid, llid, getter::method::max_local_range, expected);
    }
    {  // get_group_linear_id()
      const size_t expected = glid;
      check(glid, llid, getter::method::group_linear_id, expected);
    }
    {  // get_local_linear_id()
      const size_t expected = llid;
      check(glid, llid, getter::method::local_linear_id, expected);
    }
    {  // get_group_linear_range()
      size_t expected = 1;
      for (size_t i = 0; i < dimensions; i++) {
        expected *= m_globalRange.get(i);
      }
      check(glid, llid, getter::method::group_linear_range, expected);
    }
    {  // get_local_linear_range()
      size_t expected = 1;
      for (size_t i = 0; i < dimensions; i++) {
        expected *= m_localRange.get(i);
      }
      check(glid, llid, getter::method::local_linear_range, expected);
    }
    {  // leader()
      const size_t expected = llid == 0;
      check(glid, llid, getter::method::leader, expected);
    }
  }

  /** Checks the result for functions that return a multi-dimensional value. */
  template <size_t expectedDimensions>
  void check(size_t glid, size_t llid, getter::method getterMethod,
             const std::array<size_t, expectedDimensions>& expected) const {
    static_assert(expectedDimensions >= dimensions, "Invalid call for check");
    INFO("linear group id: " << glid << ", linear local id: " << llid);
    INFO("" << getter::name(getterMethod));

    const size_t llrange = m_localRange.size();
    const call_result_t& callResult =
        m_callResults[get_index(glid, llrange, llid, getterMethod)];
    for (size_t dim = 0; dim < dimensions; ++dim) {
      INFO("dim: " << dim);
      INFO("actual: " << callResult.values[dim]
                      << " expected: " << expected[dim]);
      CHECK((callResult.values[dim] == expected[dim]));
    }
    CHECK(callResult.hasValidType);
  }

  /** Checks the result for functions that return a one-dimensional value. */
  void check(size_t glid, size_t llid, getter::method getterMethod,
             size_t expected) const {
    INFO("linear group id: " << glid << ", linear local id: " << llid);
    INFO("" << getter::name(getterMethod));

    const size_t llrange = m_localRange.size();
    const call_result_t& callResult =
        m_callResults[get_index(glid, llrange, llid, getterMethod)];

    INFO("actual: " << callResult.values[0] << " expected: " << expected);
    CHECK((callResult.values[0] == expected));
    CHECK(callResult.hasValidType);
  }

  std::array<size_t, dimensions> get_local_range_values() const {
    std::array<size_t, dimensions> result{};
    int i = 0;
    std::generate(result.begin(), result.end(),
                  [this, &i]() mutable { return m_localRange.get(i++); });
    return result;
  }
};

/**
 * Checks a work-group size against the maximum as defined by the device
 * associated with \p queue. */
template <int dimensions>
bool wg_size_too_large(sycl::queue& queue,
                       sycl::range<dimensions> local_range) {
  using k_name = test_kernel<dimensions>;
  auto ctx = queue.get_context();
  auto kb =
      sycl::get_kernel_bundle<k_name, sycl::bundle_state::executable>(ctx);
  auto kernel = kb.get_kernel(sycl::get_kernel_id<k_name>());
  auto device = queue.get_device();

  auto work_group_size_limit =
      device.template get_info<sycl::info::device::max_work_group_size>();

  size_t wg_size = 1;
  for (size_t dim = 0; dim < dimensions; ++dim) {
    wg_size *= local_range[dim];
  }
  return wg_size > work_group_size_limit;
}

TEST_CASE("group api", "[group]") {
  auto queue = util::get_cts_object::queue();

  // validate for dimensions 1, 2, and 3
  {
    sycl::range<1> local_range(DEFAULT_LOCAL_RANGE[0]);
    if (wg_size_too_large(queue, local_range)) {
      WARN("cannot run with default local range, running with range [1]");
      local_range = sycl::range<1>(1);
    }
    auto helper = test_helper(sycl::range<1>(GROUP_RANGE[0]), local_range);
    helper.collect_results(queue);
    helper.validate_results();
  }
  {
    sycl::range<2> local_range(DEFAULT_LOCAL_RANGE[0], DEFAULT_LOCAL_RANGE[1]);
    if (wg_size_too_large(queue, local_range)) {
      WARN("cannot run with default local range, running with range [1, 1]");
      local_range = sycl::range<2>(1, 1);
    }
    auto helper = test_helper(sycl::range<2>(GROUP_RANGE[0], GROUP_RANGE[1]),
                              local_range);
    helper.collect_results(queue);
    helper.validate_results();
  }
  {
    sycl::range<3> local_range(DEFAULT_LOCAL_RANGE[0], DEFAULT_LOCAL_RANGE[1],
                               DEFAULT_LOCAL_RANGE[2]);
    if (wg_size_too_large(queue, local_range)) {
      WARN("cannot run with default local range, running with range [1, 1, 1]");
      local_range = sycl::range<3>(1, 1, 1);
    }
    auto helper = test_helper(
        sycl::range<3>(GROUP_RANGE[0], GROUP_RANGE[1], GROUP_RANGE[2]),
        local_range);
    helper.collect_results(queue);
    helper.validate_results();
  }
}
