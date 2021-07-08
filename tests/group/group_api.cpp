/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "../../util/array.h"

#define TEST_NAME group_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

static const size_t GROUP_RANGE_1D = 2;
static const size_t GROUP_RANGE_2D = 4;
static const size_t GROUP_RANGE_3D = 8;
static const size_t DEFAULT_LOCAL_RANGE[3] = {4, 3, 2};
static const size_t NUM_GROUPS =
    GROUP_RANGE_1D * GROUP_RANGE_2D * GROUP_RANGE_3D;
static const size_t NUM_METHODS = 9;

class getter
{
public:
  enum class method : size_t {
    get = 0,
    get_dims = 1,
    local_range = 2,
    local_range_dims = 3,
    global_range = 4,
    global_range_dims = 5,
    group_range = 6,
    group_range_dims = 7,
    subscript = 8,
  };

  static inline size_t get_index(size_t groupLinearID,
                                 getter::method getterMethod) {
    const auto offset = to_integral(getterMethod);
    return (groupLinearID * NUM_METHODS) + offset;
  }

  static const char *name(getter::method getterMethod) {
    switch (getterMethod) {
      case method::get:
        return "get()";
      case method::get_dims:
        return "get(int)";
      case method::local_range:
        return "get_local_range()";
      case method::local_range_dims:
        return "get_local_range(int)";
      case method::global_range:
        return "get_global_range()";
      case method::global_range_dims:
        return "get_global_range(int)";
      case method::group_range:
        return "get_group_range()";
      case method::group_range_dims:
        return "get_group_range(int)";
      case method::subscript:
        return "operator[](int)";
      default:
        return "__unknown__";
    }
  }
};

template<int dimensions>
class test_kernel;

template <int dimensions>
class test_helper {
public:
  using range_t = sycl::range<dimensions>;

private:
  static constexpr int NUM_RESULTS = NUM_GROUPS * NUM_METHODS;
  struct call_result_t
  {
    bool hasValidType;
    sycl_cts::util::array<size_t, dimensions> values;
  };
  sycl::vector_class<call_result_t> m_callResults;
  sycl::range<dimensions>           m_globalRange;
  sycl::range<dimensions>           m_localRange;

public:
  test_helper(sycl::range<dimensions> globalRange,
              sycl::range<dimensions> localRange):
    m_callResults(NUM_RESULTS),
    m_globalRange(globalRange),
    m_localRange(localRange) {
      for (size_t i = 0; i < NUM_RESULTS; i++) {
        auto& callResult = m_callResults.data()[i];
        callResult.hasValidType = false;
        for (auto& value: callResult.values)
          value = 0;
      }
  }

  void collect_group_indicies(sycl::queue& queue) {
    sycl::buffer<call_result_t, 1> buf(m_callResults.data(),
                                           sycl::range<1>(NUM_RESULTS));

    queue.submit([&](sycl::handler &cgh) {
      auto a_dev =
          buf.template get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for_work_group<test_kernel<dimensions>>(
                m_globalRange,
                m_localRange,
                [=](sycl::group<dimensions> my_group) {

          const size_t groupLinearID = my_group.get_linear_id();

          // get()
          {
            call_result_t& callResult =
                a_dev[getter::get_index(groupLinearID, getter::method::get)];

            auto m_get_group = my_group.get_id();
            for (size_t i = 0; i < dimensions; ++i)
              callResult.values[i] = m_get_group.get(i);

            using expected_t = sycl::id<dimensions>;
            callResult.hasValidType =
                std::is_same<expected_t, decltype(my_group.get_id())>::value;
          }

          // get(int)
          {
            call_result_t& callResult =
                a_dev[getter::get_index(groupLinearID, getter::method::get_dims)];

            for (size_t i = 0; i < dimensions; ++i)
              callResult.values[i] = my_group.get_id(i);

            using expected_t = size_t;
            callResult.hasValidType =
                std::is_same<expected_t, decltype(my_group.get_id(0))>::value;
          }

          // get_local_range()
          {
            call_result_t& callResult = a_dev[getter::get_index(groupLinearID, getter::method::local_range)];

            auto m_get_local_range = my_group.get_local_range();
            for (size_t i = 0; i < dimensions; ++i)
              callResult.values[i] = m_get_local_range.get(i);

            using expected_t = sycl::range<dimensions>;
            callResult.hasValidType =
                std::is_same<expected_t,
                             decltype(my_group.get_local_range())>::value;
          }

          // get_local_range(int)
          {
            call_result_t& callResult = a_dev[getter::get_index(groupLinearID, getter::method::local_range_dims)];

            for (size_t i = 0; i < dimensions; ++i)
              callResult.values[i] = my_group.get_local_range(i);

            using expected_t = size_t;
            callResult.hasValidType =
                std::is_same<expected_t,
                             decltype(my_group.get_local_range(0))>::value;
          }

          // get_global_range()
          {
            call_result_t& callResult = a_dev[getter::get_index(groupLinearID, getter::method::global_range)];

            auto m_get_global_range = my_group.get_global_range();
            for (size_t i = 0; i < dimensions; ++i)
              callResult.values[i] = m_get_global_range.get(i);

            using expected_t = sycl::range<dimensions>;
            callResult.hasValidType =
                std::is_same<expected_t,
                             decltype(my_group.get_global_range())>::value;
          }

          // get_global_range(int)
          {
            call_result_t& callResult = a_dev[getter::get_index(groupLinearID, getter::method::global_range_dims)];

            for (size_t i = 0; i < dimensions; ++i)
              callResult.values[i] = my_group.get_global_range(i);

            using expected_t = size_t;
            callResult.hasValidType =
                std::is_same<expected_t,
                             decltype(my_group.get_global_range(0))>::value;
          }

          // get_group_range()
          {
            call_result_t& callResult = a_dev[getter::get_index(groupLinearID, getter::method::group_range)];

            auto m_get_group_range = my_group.get_group_range();
            for (size_t i = 0; i < dimensions; ++i)
              callResult.values[i] = m_get_group_range.get(i);

            using expected_t = sycl::range<dimensions>;
            callResult.hasValidType =
                std::is_same<expected_t,
                             decltype(my_group.get_group_range())>::value;
          }

          // get_group_range(int)
          {
            call_result_t& callResult = a_dev[getter::get_index(groupLinearID, getter::method::group_range_dims)];

            for (size_t i = 0; i < dimensions; ++i)
              callResult.values[i] = my_group.get_group_range(i);

            using expected_t = size_t;
            callResult.hasValidType =
                std::is_same<expected_t,
                             decltype(my_group.get_group_range(0))>::value;
          }

          // operator[]
          {
            call_result_t& callResult = a_dev[getter::get_index(groupLinearID, getter::method::subscript)];

            for (size_t i = 0; i < dimensions; ++i)
              callResult.values[i] = my_group[i];

            using expected_t = size_t;
            callResult.hasValidType =
                std::is_same<expected_t,
                             decltype(my_group[0])>::value;
          }
      });
    });
  }

  void validate_group_indicies(util::logger &log) const
  {
    // For each work item
    validate_group_indicies(log, std::integral_constant<int, dimensions>{});
  }

private:
  void validate_group_indicies(
      util::logger &log, std::integral_constant<int, 1> loopSelector) const
  {
    static_cast<void>(loopSelector);
    for (size_t groupID0 = 0; groupID0 < m_globalRange[0]; ++groupID0) {
      validate_group_indicies(log, groupID0, std::array<size_t, 1>{groupID0});
    }
  }

  void validate_group_indicies(
      util::logger &log, std::integral_constant<int, 2> loopSelector) const
  {
    static_cast<void>(loopSelector);
    for (size_t groupID0 = 0; groupID0 < m_globalRange[0]; ++groupID0) {
      for (size_t groupID1 = 0; groupID1 < m_globalRange[1]; ++groupID1) {
        const size_t groupLinearID = groupID1 + (groupID0 * GROUP_RANGE_2D);
        validate_group_indicies(
            log, groupLinearID, std::array<size_t, 2>{groupID0, groupID1});
      }
    }
  }

  void validate_group_indicies(
      util::logger &log, std::integral_constant<int, 3> loopSelector) const
  {
    static_cast<void>(loopSelector);
    for (size_t groupID0 = 0; groupID0 <  m_globalRange[0]; ++groupID0) {
      for (size_t groupID1 = 0; groupID1 <  m_globalRange[1]; ++groupID1) {
        for (size_t groupID2 = 0; groupID2 <  m_globalRange[2]; ++groupID2) {
          const size_t groupLinearID =
            (groupID2 + (groupID1 * GROUP_RANGE_3D) +
             (groupID0 * GROUP_RANGE_3D * GROUP_RANGE_2D));
          validate_group_indicies(
              log,
              groupLinearID,
              std::array<size_t, 3>{groupID0, groupID1, groupID2});
        }
      }
    }
  }

  void validate_group_indicies(util::logger &log, size_t groupLinearId,
                               std::array<size_t, dimensions> groupId) const
  {
    // get(), get(int), operator[]
    {
      const auto& expected = groupId;
      check_indices(log, groupLinearId, getter::method::get, expected);
      check_indices(log, groupLinearId, getter::method::get_dims, expected);
      check_indices(log, groupLinearId, getter::method::subscript, expected);
    }

    // get_local_range(), get_local_range(int)
    {
      const auto& expected = get_local_range_values();
      check_indices(log, groupLinearId, getter::method::local_range, expected);
      check_indices(log, groupLinearId, getter::method::local_range_dims,
                    expected);
    }

    // get_global_range(), get_global_range(int)
    {
      const auto& expected = get_global_range_values();
      check_indices(log, groupLinearId, getter::method::global_range, expected);
      check_indices(log, groupLinearId, getter::method::global_range_dims,
                    expected);
    }

    // get_group_range(), get_group_range(int)
    {
      const std::array<size_t, 3> expected {GROUP_RANGE_1D, GROUP_RANGE_2D,
                                            GROUP_RANGE_3D};
      check_indices(log, groupLinearId, getter::method::group_range, expected);
      check_indices(log, groupLinearId, getter::method::group_range_dims,
                    expected);
    }
  }

  template <size_t expectedDimensions>
  void check_indices(
      util::logger &log, size_t groupLinearId,
      getter::method getterMethod,
      const std::array<size_t, expectedDimensions>& expected) const
  {
    static_assert(expectedDimensions >= dimensions,
                  "Invalid call for check_indices");

    const auto& callResult = m_callResults[getter::get_index(groupLinearId,
                                                             getterMethod)];
    for (size_t dim = 0; dim < dimensions; ++dim)
    {
      if (!CHECK_VALUE(log, callResult.values[dim], expected[dim],
                       static_cast<int>(dim))) {
        log.note("  -> group %d: %s", groupLinearId,
                 getter::name(getterMethod));
      };
    }
    if (!callResult.hasValidType) {
      FAIL(log, "Invalid return value type");
      log.note("  -> group %d: %s", groupLinearId,
               getter::name(getterMethod));
    }
  }

  std::array<size_t, dimensions> get_local_range_values() const
  {
    std::array<size_t, dimensions> result{};
    int i = 0;
    std::generate(result.begin(), result.end(), [this, &i] () mutable {
        return m_localRange.get(i++);
      });
    return result;
  }
  std::array<size_t, dimensions> get_global_range_values() const
  {
    std::array<size_t, dimensions> result{};
    int i = 0;
    std::generate(result.begin(), result.end(), [this, &i] () mutable {
        const auto index = i++;
        return m_globalRange.get(index) * m_localRange.get(index);
      });
    return result;
  }
};

template <int dimensions>
bool reduce_size(sycl::queue& queue) {
  bool res = true;
  // Check if default work group size is supported
  sycl::program program(queue.get_context());
  if (is_compiler_available(program.get_devices()) &&
      is_linker_available(program.get_devices())) {
    program.build_with_kernel_type<test_kernel<dimensions>>();
    auto kernel = program.get_kernel<test_kernel<dimensions>>();
    auto device = queue.get_device();

    auto work_group_size_limit = kernel.template get_work_group_info<
        sycl::info::kernel_work_group::work_group_size>(device);

    size_t default_wg_size = 1;
    for (size_t dim = 0; dim < dimensions; ++dim) {
      default_wg_size *= DEFAULT_LOCAL_RANGE[dim];
    }
    res = default_wg_size > work_group_size_limit;
  }
  return res;
}

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
      auto queue = util::get_cts_object::queue();

      // Validate for each dimension possible
      {
        bool reduce_wg_size = reduce_size<1>(queue);
        size_t LOCAL_RANGE_1D = reduce_wg_size ? 1 : DEFAULT_LOCAL_RANGE[0];
        auto validator = test_helper<1>(
            sycl::range<1>(GROUP_RANGE_1D),
            sycl::range<1>(LOCAL_RANGE_1D));

        validator.collect_group_indicies(queue);
        validator.validate_group_indicies(log);
      }
      {
        // Adjust work-group size
        bool reduce_wg_size = reduce_size<2>(queue);
        size_t LOCAL_RANGE_1D = reduce_wg_size ? 1 : DEFAULT_LOCAL_RANGE[0];
        size_t LOCAL_RANGE_2D = reduce_wg_size ? 1 : DEFAULT_LOCAL_RANGE[1];
        auto validator = test_helper<2>(
            sycl::range<2>(GROUP_RANGE_1D, GROUP_RANGE_2D),
            sycl::range<2>(LOCAL_RANGE_1D, LOCAL_RANGE_2D));

        validator.collect_group_indicies(queue);
        validator.validate_group_indicies(log);
      }
      {
        // Adjust work-group size
        bool reduce_wg_size = reduce_size<3>(queue);
        size_t LOCAL_RANGE_1D = reduce_wg_size ? 1 : DEFAULT_LOCAL_RANGE[0];
        size_t LOCAL_RANGE_2D = reduce_wg_size ? 1 : DEFAULT_LOCAL_RANGE[1];
        size_t LOCAL_RANGE_3D = reduce_wg_size ? 1 : DEFAULT_LOCAL_RANGE[2];
        auto validator = test_helper<3>(
            sycl::range<3>(GROUP_RANGE_1D, GROUP_RANGE_2D, GROUP_RANGE_3D),
            sycl::range<3>(LOCAL_RANGE_1D, LOCAL_RANGE_2D, LOCAL_RANGE_3D));

        validator.collect_group_indicies(queue);
        validator.validate_group_indicies(log);
      }
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      sycl::string_class errorMsg =
          "a SYCL exception was caught: " + sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
