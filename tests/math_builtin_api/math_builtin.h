/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef CL_SYCL_CTS_MATH_BUILTIN_API_MATH_BUILTIN_H
#define CL_SYCL_CTS_MATH_BUILTIN_API_MATH_BUILTIN_H

#include "../../util/math_reference.h"
#include <cfloat>
#include <limits>

template <int T>
class kernel;

inline cl::sycl::queue makeQueueOnce() {
  static cl::sycl::queue q = sycl_cts::util::get_cts_object::queue();
  return q;
}

template <typename T> T eps() { return std::numeric_limits<T>::epsilon(); }

template <> inline cl::sycl::half eps<cl::sycl::half>() {
  return static_cast<cl::sycl::half>(FLT_EPSILON * powf(2.0f, 13.0f));
}

template <typename T> T min_t() { return std::numeric_limits<T>::min(); }

template <> inline cl::sycl::half min_t<cl::sycl::half>() {
  return static_cast<cl::sycl::half>(powf(2.0f, -14.0f));
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value ||
                            std::is_same<cl::sycl::half, T>::value,
                        bool>::type
verify(sycl_cts::util::logger &log, T a, T b) {
  // if result is undefined according to spec,
  // reference function for float numbers returns NAN
  return std::isnan(b) || std::fabs(a - b) <= eps<T>() * std::fabs(a + b) ||
         std::fabs(a - b) < min_t<T>();
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
verify(sycl_cts::util::logger &log, T a, T b) {
  return a == b;
}

template <typename T, int N>
bool verify(sycl_cts::util::logger &log, cl::sycl::vec<T, N> a,
            cl::sycl::vec<T, N> b) {
  for (int i = 0; i < N; i++)
    if (!verify(log, getElement(a, i), getElement(b, i)))
      return false;
  return true;
}

template <int N, typename returnT, typename funT>
void check_function(sycl_cts::util::logger &log, funT fun, returnT ref) {
  cl::sycl::range<1> ndRng(1);
  returnT kernelResult;
  auto testQueue = makeQueueOnce();
  try {
    cl::sycl::buffer<returnT, 1> buffer(&kernelResult, ndRng);
    testQueue.submit([&](cl::sycl::handler &h) {
      auto resultPtr =
          buffer.template get_access<cl::sycl::access::mode::write>(h);
      h.single_task<kernel<N>>([=]() { resultPtr[0] = fun(); });
    });

  } catch (const cl::sycl::exception &e) {
    log_exception(log, e);
    cl::sycl::string_class errorMsg = "tests case: " + std::to_string(N) +
                                      " a SYCL exception was caught: " +
                                      cl::sycl::string_class(e.what());
    FAIL(log, errorMsg.c_str());
  }

  if (!verify(log, kernelResult, ref))
    FAIL(log,
         "tests case: " + std::to_string(N) + ". Correctness check failed.");
}

template <int T, typename returnT, typename funT>
void test_function(funT fun) {
  cl::sycl::range<1> ndRng(1);
  returnT *kernelResult = new returnT[1];
  auto testQueue = makeQueueOnce();
  {
    cl::sycl::buffer<returnT, 1> buffer(kernelResult, ndRng);
    testQueue.submit([&](cl::sycl::handler &h) {
      auto resultPtr = buffer.template get_access<cl::sycl::access::mode::write>(h);
        h.single_task<kernel<T>>([=](){
          resultPtr[0] = fun();
        });
    });
  }
  testQueue.wait_and_throw();
  delete[] kernelResult;
}

template <int T, typename returnT, typename funT, typename argT>
void test_function_multi_ptr_global(funT fun, argT arg) {
  cl::sycl::range<1> ndRng(1);
  returnT *kernelResult = new returnT[1];
  auto testQueue = makeQueueOnce();
  {
    cl::sycl::buffer<returnT, 1> buffer(kernelResult, ndRng);
    cl::sycl::buffer<argT, 1> ptrBuffer(&arg, ndRng);
    testQueue.submit([&](cl::sycl::handler &h) {
      auto resultPtr = buffer.template get_access<cl::sycl::access::mode::write>(h);
      cl::sycl::accessor<argT, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer> globalAccessor(ptrBuffer, h);
        h.single_task<kernel<T>>([=](){
          resultPtr[0] = fun(globalAccessor);
        });
    });
  }
  testQueue.wait_and_throw();
  delete[] kernelResult;
}

template <int T, typename returnT, typename funT, typename argT>
void test_function_multi_ptr_local(funT fun, argT arg) {
  cl::sycl::range<1> ndRng(1);
  returnT *kernelResult = new returnT[1];
  auto testQueue = makeQueueOnce();
  {
    cl::sycl::buffer<returnT, 1> buffer(kernelResult, ndRng);
    testQueue.submit([&](cl::sycl::handler &h) {
      auto resultPtr = buffer.template get_access<cl::sycl::access::mode::write>(h);
      cl::sycl::accessor<argT, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> localAccessor(1, h);
        h.single_task<kernel<T>>([arg, localAccessor, resultPtr, fun](){
          localAccessor[0] = arg;
          resultPtr[0] = fun(localAccessor);
        });
    });
  }
  testQueue.wait_and_throw();
  delete[] kernelResult;
}

#endif // CL_SYCL_CTS_MATH_BUILTIN_API_MATH_BUILTIN_H
