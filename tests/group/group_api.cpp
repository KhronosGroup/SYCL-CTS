/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME group_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

static const size_t GROUP_RANGE_1D = 2;
static const size_t GROUP_RANGE_2D = 4;
static const size_t GROUP_RANGE_3D = 8;
static const size_t NUM_DIMENSIONS = 3;
static const size_t NUM_GROUPS =
    GROUP_RANGE_1D * GROUP_RANGE_2D * GROUP_RANGE_3D;
static const size_t NUM_METHODS = 7;

enum class getter : size_t {
  get = 0,
  get_dims = 1,
  global_range = 2,
  global_range_dims = 3,
  group_range = 4,
  group_range_dims = 5,
  subscript = 6,
};

inline size_t get_index(size_t groupLinearID, getter getterMethod) {
  return ((groupLinearID * NUM_METHODS * NUM_DIMENSIONS) +
          (static_cast<size_t>(getterMethod) * NUM_DIMENSIONS));
}

const char *getter_name(getter getterMethod) {
  switch (getterMethod) {
    case getter::get:
      return "get()";
    case getter::get_dims:
      return "get(int)";
    case getter::global_range:
      return "get_global_range()";
    case getter::global_range_dims:
      return "get_global_range(int)";
    case getter::group_range:
      return "get_group_range()";
    case getter::group_range_dims:
      return "get_group_range(int)";
    case getter::subscript:
      return "operator[](int)";
    default:
      return "__unknown__";
  }
}

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
    using namespace cl::sycl;

    try {
      auto queue = util::get_cts_object::queue();
      // Prepare the kernel and get the required info
      cl::sycl::kernel kernel =
          util::get_cts_object::kernel::prebuilt<class TEST_NAME>(queue);
      auto optimalLocalWorkGroupSize =
          kernel.get_work_group_info<cl::sycl::info::kernel_work_group::
                                         preferred_work_group_size_multiple>(
              util::get_cts_object::device());

      size_t resultSize = NUM_GROUPS * NUM_METHODS * NUM_DIMENSIONS;
      cl::sycl::vector_class<size_t> result(resultSize);

      for (size_t i = 0; i < resultSize; i++) {
        result.data()[i] = 0;
      }

      // Collect the group indices
      {
        buffer<size_t, 1> buf(result.data(), range<1>(resultSize));
        queue.submit([&](handler &cgh) {
          auto a_dev = buf.get_access<cl::sycl::access::mode::read_write>(cgh);

          cgh.parallel_for_work_group<class TEST_NAME>(
              range<3>(GROUP_RANGE_1D, GROUP_RANGE_2D, GROUP_RANGE_3D),
              [=](group<3> my_group) {
                const size_t groupLinearID = my_group.get_linear();

                // get()
                {
                  auto m_get_group = my_group.get();
                  auto indexBase = get_index(groupLinearID, getter::get);
                  a_dev[indexBase + 0] = m_get_group.get(0);
                  a_dev[indexBase + 1] = m_get_group.get(1);
                  a_dev[indexBase + 2] = m_get_group.get(2);
                }

                // get(int)
                {
                  auto indexBase = get_index(groupLinearID, getter::get_dims);
                  a_dev[indexBase + 0] = my_group.get(0);
                  a_dev[indexBase + 1] = my_group.get(1);
                  a_dev[indexBase + 2] = my_group.get(2);
                }

                // get_global_range()
                {
                  range<3> m_get_global_range = my_group.get_global_range();
                  auto indexBase =
                      get_index(groupLinearID, getter::global_range);
                  a_dev[indexBase + 0] = m_get_global_range.get(0);
                  a_dev[indexBase + 1] = m_get_global_range.get(1);
                  a_dev[indexBase + 2] = m_get_global_range.get(2);
                }

                // get_global_range(int)
                {
                  auto indexBase =
                      get_index(groupLinearID, getter::global_range_dims);
                  a_dev[indexBase + 0] = my_group.get_global_range(0);
                  a_dev[indexBase + 1] = my_group.get_global_range(1);
                  a_dev[indexBase + 2] = my_group.get_global_range(2);
                }

                // get_group_range()
                {
                  range<3> m_get_group_range = my_group.get_group_range();
                  auto indexBase =
                      get_index(groupLinearID, getter::group_range);
                  a_dev[indexBase + 0] = m_get_group_range.get(0);
                  a_dev[indexBase + 1] = m_get_group_range.get(1);
                  a_dev[indexBase + 2] = m_get_group_range.get(2);
                }

                // get_group_range(int)
                {
                  auto indexBase =
                      get_index(groupLinearID, getter::group_range_dims);
                  a_dev[indexBase + 0] = my_group.get_group_range(0);
                  a_dev[indexBase + 1] = my_group.get_group_range(1);
                  a_dev[indexBase + 2] = my_group.get_group_range(2);
                }

                // operator[]
                {
                  auto indexBase = get_index(groupLinearID, getter::subscript);
                  a_dev[indexBase + 0] = my_group[0];
                  a_dev[indexBase + 1] = my_group[1];
                  a_dev[indexBase + 2] = my_group[2];
                }
              });
        });
      }

      queue.wait_and_throw();

      // Check the group indices
      for (size_t groupID0 = 0; groupID0 < GROUP_RANGE_1D; ++groupID0) {
        for (size_t groupID1 = 0; groupID1 < GROUP_RANGE_2D; ++groupID1) {
          for (size_t groupID2 = 0; groupID2 < GROUP_RANGE_3D; ++groupID2) {
            // Calculate the row-major linear ID
            const size_t groupLinearID =
                (groupID2 + (groupID1 * GROUP_RANGE_3D) +
                 (groupID0 * GROUP_RANGE_3D * GROUP_RANGE_2D));

            auto check_indices = [&](getter getterMethod, size_t offset,
                                     size_t expected) {
              auto indexBase = get_index(groupLinearID, getterMethod);
              if (!CHECK_VALUE(log, result[indexBase + offset], expected,
                               static_cast<int>(offset))) {
                log.note("  -> group %d: %s", groupLinearID,
                         getter_name(getterMethod));
              };
            };

            // get()
            {
              check_indices(getter::get, 0, groupID0);
              check_indices(getter::get, 1, groupID1);
              check_indices(getter::get, 2, groupID2);
            }

            // get(int)
            {
              check_indices(getter::get_dims, 0, groupID0);
              check_indices(getter::get_dims, 1, groupID1);
              check_indices(getter::get_dims, 2, groupID2);
            }

            // get_global_range()
            {
              check_indices(getter::global_range, 0,
                            GROUP_RANGE_1D * optimalLocalWorkGroupSize);
              check_indices(getter::global_range, 1, GROUP_RANGE_2D);
              check_indices(getter::global_range, 2, GROUP_RANGE_3D);
            }

            // get_global_range(int)
            {
              check_indices(getter::global_range_dims, 0,
                            GROUP_RANGE_1D * optimalLocalWorkGroupSize);
              check_indices(getter::global_range_dims, 1, GROUP_RANGE_2D);
              check_indices(getter::global_range_dims, 2, GROUP_RANGE_3D);
            }

            // get_group_range()
            {
              check_indices(getter::group_range, 0, GROUP_RANGE_1D);
              check_indices(getter::group_range, 1, GROUP_RANGE_2D);
              check_indices(getter::group_range, 2, GROUP_RANGE_3D);
            }

            // get_group_range(int)
            {
              check_indices(getter::group_range_dims, 0, GROUP_RANGE_1D);
              check_indices(getter::group_range_dims, 1, GROUP_RANGE_2D);
              check_indices(getter::group_range_dims, 2, GROUP_RANGE_3D);
            }

            // operator[]
            {
              check_indices(getter::subscript, 0, groupID0);
              check_indices(getter::subscript, 1, groupID1);
              check_indices(getter::subscript, 2, groupID2);
            }
          }
        }
      }

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
