/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide checks that ranged accessor still creates a requisite for the entire
//  underlying buffer
//
*******************************************************************************/

#include "../common/disabled_for_test_case.h"
#include "../common/get_cts_object.h"
#include "catch2/catch_test_macros.hpp"

// FIXME: re-enable when sycl::accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP
#include "accessor_common.h"
#endif

namespace accessor_requisite_entire_buffer {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("requisite for the entire underlying buffer for sycl::accessor ",
 "[accessor]")({
  auto q = sycl_cts::util::get_cts_object::queue();

  constexpr size_t buffer_size = 10;
  constexpr size_t offset_size = 7;
  int data[buffer_size];
  std::iota(data, (data + buffer_size), 0);
  {
    sycl::buffer<int, 1> data_buf(data, sycl::range(buffer_size));
    // create two commands that uses ranaged accessors that access to
    // non-overlapping regions of the same buffer since ranged accessor still
    // creates a requisite for the entire underlying buffer second sommand
    // should execute only after first finishes to check it both commands assign
    // some data to usm check_data and it is expected that second command will
    // do it only after first command even though first command will do it after
    // delay via loop
    int* check_data = sycl::malloc_shared<int>(1, q);

    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::target::device>
          acc(data_buf, cgh, sycl::range<1>(offset_size));

      cgh.single_task([=] {
        // to delay assigning expected_val to check_data use a loop that will
        // take some time
        for (int i = 0; i < 1000000; i++) {
          int s = sycl::sqrt(float(i));
          acc[s % offset_size] = accessor_tests_common::expected_val;
        }
        *check_data = accessor_tests_common::expected_val;
      });
    });

    q.submit([&](sycl::handler& cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read, sycl::target::device> acc(
          data_buf, cgh, sycl::range<1>(1), sycl::id<1>(offset_size));
      cgh.single_task([=] {
        *check_data = accessor_tests_common::changed_val;
        auto v = acc[offset_size];
      });
    });
    q.wait_and_throw();
    CHECK(*check_data == accessor_tests_common::changed_val);
  }
});

}  // namespace accessor_requisite_entire_buffer
