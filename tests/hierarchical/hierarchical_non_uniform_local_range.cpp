/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
//  Test description:
//
//    This test makes sure that:
//      - Implementations handles correctly a non-uniform range passed to
//        group::parallel_for_work_item;
//      - Implementations handles correctly call to
//        group::parallel_for_work_item with a 0 logical local range.
//
*******************************************************************************/

#include "../../util/math_vector.h"
#include "../common/common.h"

#define TEST_NAME hierarchical_non_uniform_local_range

namespace TEST_NAMESPACE {

template <int dim>
class kernel;

using namespace sycl_cts;

void check_expected(const std::vector<sycl::int3> &data, unsigned local_id,
                    unsigned idx, int dim, bool set, util::logger &log) {
  int expected = set ? local_id : -1;
  if (getElement(data[idx], dim - 1) != expected) {
    std::string errorMessage =
        std::string("Value for global id ") + std::to_string(idx) +
        std::string(" for dim = ") + std::to_string(dim) +
        std::string(" was not correct (") +
        std::to_string(getElement(data[idx], dim - 1)) +
        std::string(" instead of ") + std::to_string(expected) + ")";
    FAIL(log, errorMessage);
  }
}

template <int dim>
void check_dim(util::logger &log) {
  constexpr unsigned group_range = 4;
  constexpr unsigned local_range = 3;
  constexpr unsigned group_range_total = 64;
  constexpr unsigned local_range_total = 27;
  constexpr unsigned global_range = group_range_total * local_range_total;
  std::vector<sycl::int3> data(global_range);

  const unsigned group_range_1d = (dim > 1) ? group_range : group_range_total;
  const unsigned group_range_2d =
      (dim > 1) ? ((dim > 2) ? group_range : group_range_total / group_range)
                : 1;
  const unsigned group_range_3d = (dim > 2) ? group_range : 1;
  const unsigned local_range_1d = (dim > 1) ? local_range : local_range_total;
  const unsigned local_range_2d =
      (dim > 1) ? ((dim > 2) ? local_range : local_range_total / local_range)
                : 1;
  const unsigned local_range_3d = (dim > 2) ? local_range : 1;
  // Set element of the vector with -1 to represent unset data.
  std::fill(data.begin(), data.end(), sycl::int3(-1, -1, -1));

  auto myQueue = util::get_cts_object::queue();
  // using this scope we ensure that the buffer will update the host values
  // after the wait_and_throw
  {
    sycl::buffer<sycl::int3, 1> buf(data.data(), sycl::range<1>(data.size()));

    myQueue.submit([&](sycl::handler &cgh) {
      sycl::stream os(2048, 80, cgh);
      auto accessor =
          buf.template get_access<sycl::access_mode::read_write>(cgh);

      auto groupRange =
          sycl_cts::util::get_cts_object::range<dim>::template get_fixed_size<
              group_range_total>(group_range, group_range);
      auto localRange =
          sycl_cts::util::get_cts_object::range<dim>::template get_fixed_size<
              local_range_total>(local_range, local_range);
      cgh.parallel_for_work_group<kernel<dim>>(
          groupRange, localRange, [=](sycl::group<dim> group_pid) {
            group_pid.parallel_for_work_item(
                sycl::range<dim>(group_pid.get_id()),
                [&](sycl::h_item<dim> item_id) {
                  unsigned physical_local_d1 = item_id.get_physical_local()[0];
                  unsigned physical_local_d2 =
                      (dim > 1) ? item_id.get_physical_local()[1] : 0;
                  unsigned physical_local_d3 =
                      (dim > 2) ? item_id.get_physical_local()[2] : 0;

                  unsigned globalId1 = item_id.get_global()[0];
                  unsigned globalId2 = (dim > 1) ? item_id.get_global()[1] : 0;
                  unsigned globalId3 = (dim > 2) ? item_id.get_global()[2] : 0;
                  unsigned global_range2 = group_range_2d * local_range_2d;
                  unsigned global_range3 = group_range_3d * local_range_3d;
                  unsigned globalIdL =
                      ((globalId1 * global_range2 * global_range3) +
                       (globalId2 * global_range3) + globalId3);
                  // Assign local item work-item’s position in the local range
                  // but in new non-uniform logical local range that depends
                  // on a work-group id
                  accessor[globalIdL] = sycl::int3(
                      physical_local_d1, physical_local_d2, physical_local_d3);
                });
          });
    });
  }

  unsigned idx = 0;

  for (unsigned group_id1 = 0; group_id1 < group_range_1d; group_id1++)
    for (unsigned local_id1 = 0; local_id1 < local_range_1d; local_id1++)
      for (unsigned group_id2 = 0; group_id2 < group_range_2d; group_id2++)
        for (unsigned local_id2 = 0; local_id2 < local_range_2d; local_id2++)
          for (unsigned group_id3 = 0; group_id3 < group_range_3d; group_id3++)
            for (unsigned local_id3 = 0; local_id3 < local_range_3d;
                 local_id3++) {
              bool set = (local_id1 < group_id1) &&
                         (local_id2 < group_id2 || dim < 2) &&
                         (local_id3 < group_id3 || dim < 3);
              check_expected(data, local_id1, idx, 1, set, log);
              if (dim > 1) {
                check_expected(data, local_id2, idx, 2, set, log);
                if (dim > 2) {
                  check_expected(data, local_id3, idx, 3, set, log);
                }
              }
              idx++;
            }
}

/** test sycl::range::get(int index) return size_t
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

}  // namespace TEST_NAMESPACE
