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
#include "../../util/accuracy.h"
#include <cfloat>
#include <limits>

template <int T>
class kernel;

inline cl::sycl::queue makeQueueOnce() {
  static cl::sycl::queue q = sycl_cts::util::get_cts_object::queue();
  return q;
}

template <typename returnT, typename ArgT> struct privatePtrCheck {
  returnT res;
  ArgT resArg;
  privatePtrCheck(returnT res_t, ArgT resArg_t)
      : res(res_t), resArg(resArg_t) {}
};

template <typename T> struct base;
template <> struct base<float> { using type = std::uint32_t; };
template <> struct base<double> { using type = std::uint64_t; };
template <> struct base<cl::sycl::half> { using type = std::uint16_t; };

template <typename T> std::string printable(T value) {
  const auto representation =
      *reinterpret_cast<typename base<T>::type *>(&value);
  std::ostringstream out;
  out.precision(64);
  out << value << " [" << std::hex << representation << "]";
  return out.str();
}

template <typename T> T min_t() { return std::numeric_limits<T>::min(); }

template <> inline cl::sycl::half min_t<cl::sycl::half>() {
  return static_cast<cl::sycl::half>(powf(2.0f, -14.0f));
}

template <typename T>
bool verify(sycl_cts::util::logger &log, T a, T b, int accuracy);

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value ||
                            std::is_same<cl::sycl::half, T>::value,
                        bool>::type
verify(sycl_cts::util::logger &log, T value, resultRef<T> r, int accuracy) {
  const T reference = r.res;

  if (!r.undefined.empty())
    return true; // result is undefined according to spec
  if (std::isnan(value) && std::isnan(reference))
    return true; // NaN can have any nancode within
  if (value == reference)
    return true; // bitwise equal numeric or infinity value

  if (!std::isnan(value) && !std::isnan(reference) && !std::isinf(value) &&
      !std::isinf(reference)) {
    if (accuracy < 0)
      return true; // Implementation-defined or infinite ULP according to spec

    if ((std::fabs(value) < min_t<T>()) && (std::fabs(reference) < min_t<T>()))
      return true; // Subnormal numbers are the lower border for comparison

    const auto ulpsExpected = static_cast<unsigned int>(accuracy);
    const T difference = static_cast<T>(std::fabs(value - reference));
    const T differenceExpected = ulpsExpected * get_ulp_std(reference);

    if (difference <= differenceExpected)
      return true;
  }

  log.note("value: " + printable(value) + ", reference: " +
           printable(reference));
  return false;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
verify(sycl_cts::util::logger &log, T value, resultRef<T> r, int) {
  bool result = value == r.res || !r.undefined.empty();
  if (!result)
    log.note("value: " + std::to_string(value) + ", reference: " +
             std::to_string(r.res));
  return result;
}

template <typename T, int N>
bool verify(sycl_cts::util::logger &log, cl::sycl::vec<T, N> a,
            resultRef<cl::sycl::vec<T, N>> r, int accuracy) {
  cl::sycl::vec<T, N> b = r.res;
  for (int i = 0; i < sycl_cts::math::numElements(a); i++)
    if (r.undefined.find(i) == r.undefined.end() &&
        !verify(log, getElement(a, i), getElement(b, i), accuracy))
      return false;
  return true;
}

template <typename T>
bool verify(sycl_cts::util::logger &log, T a, T b, int accuracy) {
  return verify(log, a, resultRef<T>(b), accuracy);
}

template <int N, typename returnT, typename funT>
void check_function(sycl_cts::util::logger &log, funT fun,
                    resultRef<returnT> ref, int accuracy = 0) {
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

  if (!verify(log, kernelResult, ref, accuracy))
    FAIL(log,
         "tests case: " + std::to_string(N) + ". Correctness check failed.");
}

template <int N, typename returnT, typename funT, typename argT>
void check_function_multi_ptr_private(sycl_cts::util::logger &log, funT fun,
                                      resultRef<returnT> ref, argT ptrRef,
                                      int accuracy = 0) {
  cl::sycl::range<1> ndRng(1);
  returnT kernelResult;
  argT kernelResultArg;
  auto testQueue = makeQueueOnce();
  try {
    cl::sycl::buffer<returnT, 1> buffer(&kernelResult, ndRng);
    cl::sycl::buffer<argT, 1> bufferArg(&kernelResultArg, ndRng);
    testQueue.submit([&](cl::sycl::handler &h) {
      auto resultPtr =
          buffer.template get_access<cl::sycl::access::mode::write>(h);
      auto resultPtrArg =
          bufferArg.template get_access<cl::sycl::access::mode::write>(h);
      h.single_task<kernel<N>>([=]() {
        privatePtrCheck<returnT, argT> result = fun();
        resultPtr[0] = result.res;
        resultPtrArg[0] = result.resArg;
      });
    });
  } catch (const cl::sycl::exception &e) {
    log_exception(log, e);
    cl::sycl::string_class errorMsg = "tests case: " + std::to_string(N) +
                                      " a SYCL exception was caught: " +
                                      cl::sycl::string_class(e.what());
    FAIL(log, errorMsg.c_str());
  }

  if (!verify(log, kernelResult, ref, accuracy))
    FAIL(log,
         "tests case: " + std::to_string(N) + ". Correctness check failed.");
  if (!verify(log, kernelResultArg, ptrRef, accuracy))
    FAIL(log, "tests case: " + std::to_string(N) +
                  ". Correctness check for ptr failed.");
}

template <int N, typename returnT, typename funT, typename argT>
void check_function_multi_ptr_global(sycl_cts::util::logger &log, funT fun,
                                     argT arg, resultRef<returnT> ref,
                                     argT ptrRef, int accuracy = 0) {
  cl::sycl::range<1> ndRng(1);
  returnT kernelResult;
  auto testQueue = makeQueueOnce();
  try {
    cl::sycl::buffer<returnT, 1> buffer(&kernelResult, ndRng);
    cl::sycl::buffer<argT, 1> ptrBuffer(&arg, ndRng);
    testQueue.submit([&](cl::sycl::handler &h) {
      auto resultPtr =
          buffer.template get_access<cl::sycl::access::mode::write>(h);
      cl::sycl::accessor<argT, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>
          globalAccessor(ptrBuffer, h);
      h.single_task<kernel<N>>([=]() { resultPtr[0] = fun(globalAccessor); });
    });
  } catch (const cl::sycl::exception &e) {
    log_exception(log, e);
    cl::sycl::string_class errorMsg = "tests case: " + std::to_string(N) +
                                      " a SYCL exception was caught: " +
                                      cl::sycl::string_class(e.what());
    FAIL(log, errorMsg.c_str());
  }

  if (!verify(log, kernelResult, ref, accuracy))
    FAIL(log,
         "tests case: " + std::to_string(N) + ". Correctness check failed.");
  if (!verify(log, arg, ptrRef, accuracy))
    FAIL(log, "tests case: " + std::to_string(N) +
                  ". Correctness check for ptr failed.");
}

template <int N, typename returnT, typename funT, typename argT>
void check_function_multi_ptr_local(sycl_cts::util::logger &log, funT fun,
                                    argT arg, resultRef<returnT> ref,
                                    argT ptrRef, int accuracy = 0) {
  cl::sycl::range<1> ndRng(1);
  returnT kernelResult;
  auto testQueue = makeQueueOnce();
  try {
    cl::sycl::buffer<returnT, 1> buffer(&kernelResult, ndRng);
    cl::sycl::buffer<argT, 1> bufferArg(&arg, ndRng);
    testQueue.submit([&](cl::sycl::handler &h) {
      auto resultPtr =
          buffer.template get_access<cl::sycl::access::mode::write>(h);
      auto resultPtrArg =
          bufferArg.template get_access<cl::sycl::access::mode::write>(h);
      cl::sycl::accessor<argT, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local> localAccessor(1, h);
      h.single_task<kernel<N>>(
          [arg, localAccessor, resultPtr, resultPtrArg, fun]() {
            localAccessor[0] = arg;
            resultPtr[0] = fun(localAccessor);
            resultPtrArg[0] = localAccessor[0];
          });
    });
  } catch (const cl::sycl::exception &e) {
    log_exception(log, e);
    cl::sycl::string_class errorMsg = "tests case: " + std::to_string(N) +
                                      " a SYCL exception was caught: " +
                                      cl::sycl::string_class(e.what());
    FAIL(log, errorMsg.c_str());
  }

  if (!verify(log, kernelResult, ref, accuracy))
    FAIL(log,
         "tests case: " + std::to_string(N) + ". Correctness check failed.");
  if (!verify(log, arg, ptrRef, accuracy))
    FAIL(log, "tests case: " + std::to_string(N) +
                  ". Correctness check for ptr failed.");
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
