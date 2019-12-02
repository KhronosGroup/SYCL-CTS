/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME group_async_work_group_copy

namespace TEST_NAMESPACE {
using namespace sycl_cts;

static constexpr size_t GROUP_RANGE_1D = 2;
static constexpr size_t GROUP_RANGE_2D = 4;
static constexpr size_t GROUP_RANGE_3D = 8;
static constexpr size_t BUFFER_SIZE = 128;

class group_async_work_group_copy_1d;
class group_async_work_group_copy_2d;
class group_async_work_group_copy_3d;

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

      {
        auto buf = cl::sycl::buffer<size_t, 1>(cl::sycl::range<1>(BUFFER_SIZE));

        queue.submit([&](cl::sycl::handler &cgh) {
          auto accGlobal =
              buf.get_access<cl::sycl::access::mode::read_write>(cgh);

          auto accLocal =
              cl::sycl::accessor<size_t, 1, cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::local>(
                  cl::sycl::range<1>(BUFFER_SIZE), cgh);

          // Test 1D
          cgh.parallel_for_work_group<class group_async_work_group_copy_1d>(
              cl::sycl::range<1>(GROUP_RANGE_1D),
              [=](cl::sycl::group<1> my_group) {
                auto ptrGlobal = accGlobal.get_pointer();
                auto ptrLocal = accLocal.get_pointer();

                my_group.async_work_group_copy(ptrLocal, ptrGlobal,
                                               BUFFER_SIZE);
                my_group.async_work_group_copy(ptrGlobal, ptrLocal,
                                               BUFFER_SIZE);

                constexpr size_t stride = 2;
                constexpr size_t numElements = BUFFER_SIZE / stride;
                my_group.async_work_group_copy(ptrLocal, ptrGlobal, numElements,
                                               stride);
                my_group.async_work_group_copy(ptrGlobal, ptrLocal, numElements,
                                               stride);
              });
        });

        queue.submit([&](cl::sycl::handler &cgh) {
          auto accGlobal =
              buf.get_access<cl::sycl::access::mode::read_write>(cgh);

          auto accLocal =
              cl::sycl::accessor<size_t, 1, cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::local>(
                  cl::sycl::range<1>(BUFFER_SIZE), cgh);

          // Test 2D
          cgh.parallel_for_work_group<class group_async_work_group_copy_2d>(
              cl::sycl::range<2>(GROUP_RANGE_1D, GROUP_RANGE_2D),
              [=](cl::sycl::group<2> my_group) {
                auto ptrGlobal = accGlobal.get_pointer();
                auto ptrLocal = accLocal.get_pointer();

                my_group.async_work_group_copy(ptrLocal, ptrGlobal,
                                               BUFFER_SIZE);
                my_group.async_work_group_copy(ptrGlobal, ptrLocal,
                                               BUFFER_SIZE);

                constexpr size_t stride = 2;
                constexpr size_t numElements = BUFFER_SIZE / stride;
                my_group.async_work_group_copy(ptrLocal, ptrGlobal, numElements,
                                               stride);
                my_group.async_work_group_copy(ptrGlobal, ptrLocal, numElements,
                                               stride);
              });
        });

        queue.submit([&](cl::sycl::handler &cgh) {
          auto accGlobal =
              buf.get_access<cl::sycl::access::mode::read_write>(cgh);

          auto accLocal =
              cl::sycl::accessor<size_t, 1, cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::local>(
                  cl::sycl::range<1>(BUFFER_SIZE), cgh);

          // Test 3D
          cgh.parallel_for_work_group<class group_async_work_group_copy_3d>(
              cl::sycl::range<3>(GROUP_RANGE_1D, GROUP_RANGE_2D,
                                 GROUP_RANGE_3D),
              [=](cl::sycl::group<3> my_group) {
                auto ptrGlobal = accGlobal.get_pointer();
                auto ptrLocal = accLocal.get_pointer();

                my_group.async_work_group_copy(ptrLocal, ptrGlobal,
                                               BUFFER_SIZE);
                my_group.async_work_group_copy(ptrGlobal, ptrLocal,
                                               BUFFER_SIZE);

                constexpr size_t stride = 2;
                constexpr size_t numElements = BUFFER_SIZE / stride;
                my_group.async_work_group_copy(ptrLocal, ptrGlobal, numElements,
                                               stride);
                my_group.async_work_group_copy(ptrGlobal, ptrLocal, numElements,
                                               stride);
              });
        });
      }

      queue.wait_and_throw();

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

}  // namespace TEST_NAMESPACE
