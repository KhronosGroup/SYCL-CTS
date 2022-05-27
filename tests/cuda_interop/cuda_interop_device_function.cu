/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
//
*******************************************************************************/

#include "cuda_helper.hpp"

using namespace sycl_cts::util;

template <typename T>
SYCL_EXTERNAL __device__ T func(T *i) {
  return *i + 1;
}

/** check device function can be called with the backend interop type `T *`
converted from an `accessor`
  */
template <typename T>
void test_device_function_buffer(sycl::queue &queue,
                                 const std::string &typeName) {
  size_t constexpr size = 1;
  T data[size];
  data[0] = 0;
  bool result[size] = {false};
  {
    sycl::buffer<T> buff(data, sycl::range<1>(size));
    sycl::buffer<bool> result_buf(result, sycl::range<1>(size));

    queue.submit([&](sycl::handler &cgh) {
      auto acc = buff.template get_access<sycl::access::mode::read>(cgh);
      auto result_acc = result_buf.get_access<sycl::access::mode::write>(cgh);

      cgh.single_task([=] {
        result_acc[0] =
            func<T>(sycl::get_native<sycl::backend::cuda>(acc)) == 1;
      });
    });
  }

  INFO("Checking for CUDA device function interop with accessor for \"" +
       typeName + "\" type");
  CHECK(result[0]);
}

/** check device function can be called with the backend interop type `T *`
converted from an `local_accessor`
  */
template <typename T>
void test_device_function_local(sycl::queue &queue,
                                const std::string &typeName) {
  size_t constexpr size = 1;
  bool result[size] = {false};
  {
    sycl::buffer<bool> result_buf(result, sycl::range<1>(size));

    queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T> acc(size, cgh);
      auto result_acc = result_buf.get_access<sycl::access::mode::write>(cgh);

      cgh.single_task([=] {
        acc[0] = 0;
        result_acc[0] =
            func<T>(sycl::get_native<sycl::backend::cuda>(acc)) == 1;
      });
    });
  }

  INFO("Checking for CUDA device function interop with local_accessor for \"" +
       typeName + "\" type");
  CHECK(result[0]);
}

/** tests the get_native() methods for CUDA inter-op
 */
TEST_CASE("CUDA interop device function test") {
#ifdef SYCL_BACKEND_CUDA
  auto queue = get_cts_object::queue();

  INFO("Checking queue is using CUDA backend");
  REQUIRE(queue.get_backend() == sycl::backend::cuda);

  test_device_function_buffer<int>(queue, "int");
  test_device_function_local<int>(queue, "int");
#else
  INFO("The test is skipped because CUDA back-end is not supported");
#endif  // SYCL_BACKEND_CUDA
}
