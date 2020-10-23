/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef SYCL_1_2_1_TESTS_GROUP_ASYNC_WORK_GROUP_COPY_COMMON_H
#define SYCL_1_2_1_TESTS_GROUP_ASYNC_WORK_GROUP_COPY_COMMON_H

#include "../common/common.h"

namespace {
  inline cl::sycl::queue makeQueueOnce() {
    static cl::sycl::queue q = sycl_cts::util::get_cts_object::queue();
    return q;
  }

  using namespace sycl_cts;

  static constexpr size_t GROUP_RANGE_1D = 2;
  static constexpr size_t GROUP_RANGE_2D = 4;
  static constexpr size_t GROUP_RANGE_3D = 8;
  static constexpr size_t BUFFER_SIZE = 128;

  template<typename T, int dim>
  class test_kernel;

  template<typename T, int dim>
  void check_dim(cl::sycl::queue &queue, cl::sycl::buffer<T, 1> &buf, cl::sycl::range<dim> range) {
    queue.submit([&](cl::sycl::handler &cgh) {
    auto accGlobal =
        buf.template get_access<cl::sycl::access::mode::read_write>(cgh);

    auto accLocal =
        cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                            cl::sycl::access::target::local>(
            cl::sycl::range<1>(BUFFER_SIZE), cgh);

    cgh.parallel_for_work_group<test_kernel<T, dim>>(range,
        [=](cl::sycl::group<dim> my_group) {
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

  template<typename T>
  void check_type(util::logger &log) {
    try {
      auto queue = makeQueueOnce();

      {
        auto buf = cl::sycl::buffer<T, 1>(cl::sycl::range<1>(BUFFER_SIZE));
        check_dim<T, 1>(queue, buf, cl::sycl::range<1>(BUFFER_SIZE));
        check_dim<T, 2>(queue, buf, cl::sycl::range<2>(GROUP_RANGE_1D, GROUP_RANGE_2D));
        check_dim<T, 3>(queue, buf, cl::sycl::range<3>(GROUP_RANGE_1D, GROUP_RANGE_2D,
                                 GROUP_RANGE_3D));
      }

      queue.wait_and_throw();

    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }

  template<typename T>
  void check_type_and_vec(util::logger &log) {
    check_type<T>(log);
    check_type<cl::sycl::vec<T, 1>>(log);
    check_type<cl::sycl::vec<T, 2>>(log);
    check_type<cl::sycl::vec<T, 3>>(log);
    check_type<cl::sycl::vec<T, 4>>(log);
    check_type<cl::sycl::vec<T, 8>>(log);
    check_type<cl::sycl::vec<T, 16>>(log);
  }

}  // namespace

#endif // SYCL_1_2_1_TESTS_GROUP_ASYNC_WORK_GROUP_COPY_COMMON_H
