/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:  (c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA
#include "../../util/test_base_cuda.h"

#endif

#define TEST_NAME cuda_interop_device_fucntion

namespace cuda_interop_device_fucntion__ {

using namespace sycl_cts;

template <typename T>
SYCL_EXTERNAL __device__ T func(T *i) {
  return *i + 1;
}

/** check device fucntion can be called with the backend interop type `T *`
converted from an `accessor`
  */
template <typename T>
void test_device_function_buffer(sycl::queue &queue,
                                 sycl_cts::util::logger &log,
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

      cgh.single_task([=]() {
        result_acc[0] =
            func<T>(sycl::get_native<sycl::backend::ext_oneapi_cuda>(acc)) == 1;
      });
    });
  }

  if (!result[0]) {
    log.note(
        "Test for CUDA device function interop with accessor failed for \"" +
        typeName + "\" type");
  }
  assert(result[0]);
}

/** check device fucntion can be called with the backend interop type `T *`
converted from an `local_accessor`
  */
template <typename T>
void test_device_function_local(sycl::queue &queue, sycl_cts::util::logger &log,
                                const std::string &typeName) {
  size_t constexpr size = 1;
  bool result[size] = {false};
  {
    sycl::buffer<bool> result_buf(result, sycl::range<1>(size));

    queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T> acc(size, cgh);
      auto result_acc = result_buf.get_access<sycl::access::mode::write>(cgh);

      cgh.single_task([=]() {
        acc[0] = 0;
        result_acc[0] =
            func<T>(sycl::get_native<sycl::backend::ext_oneapi_cuda>(acc)) == 1;
      });
    });
  }

  if (!result[0]) {
    log.note(
        "Test for CUDA device function interop with local_accessor failed for "
        "\"" +
        typeName + "\" type");
  }
  assert(result[0]);
}

/** tests the get_native() methods for CUDA inter-op
 */
class TEST_NAME :
#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA
    public sycl_cts::util::test_base_cuda
#else
    public util::test_base
#endif
{
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA
    {
      auto queue = util::get_cts_object::queue();
      if (queue.get_backend() != sycl::backend::ext_oneapi_cuda) {
        WARN(
            "CUDA interoperability part is not supported on non-CUDA "
            "backend types");
        return;
      }
      cts_selector ctsSelector;

      test_device_function_buffer<int>(queue, log, "int");
      test_device_function_local<int>(queue, log, "int");
    }
#else
    log.note("The test is skipped because CUDA back-end is not supported");
#endif  // SYCL_EXT_ONEAPI_BACKEND_CUDA
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

}  // namespace cuda_interop_device_fucntion__
