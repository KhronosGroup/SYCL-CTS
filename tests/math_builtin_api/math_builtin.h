/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#ifndef CL_SYCL_CTS_MATH_BUILTIN_API_MATH_BUILTIN_H
#define CL_SYCL_CTS_MATH_BUILTIN_API_MATH_BUILTIN_H

#include "../../util/accuracy.h"
#include "../../util/math_reference.h"
#include "../../util/sycl_exceptions.h"
#include "../common/once_per_unit.h"
#include <cfloat>
#include <limits>

template <int T>
class kernel;

inline sycl::queue makeQueueOnce() {
  static sycl::queue q = sycl_cts::util::get_cts_object::queue();
  return q;
}

template <typename returnT, typename ArgT>
struct privatePtrCheck {
  returnT res;
  ArgT resArg;
  privatePtrCheck(returnT res_t, ArgT resArg_t)
      : res(res_t), resArg(resArg_t) {}
};

template <typename T>
struct base;
template <>
struct base<float> {
  using type = std::uint32_t;
};
template <>
struct base<double> {
  using type = std::uint64_t;
};
template <>
struct base<sycl::half> {
  using type = std::uint16_t;
};

template <typename T>
std::string printable(T value) {
  const auto representation = sycl::bit_cast<typename base<T>::type>(value);
  std::ostringstream out;
  out.precision(64);
  out << value << " [" << std::hex << representation << "]";
  return out.str();
}

template <typename T>
T min_t() {
  return std::numeric_limits<T>::min();
}

template <>
inline sycl::half min_t<sycl::half>() {
  return static_cast<sycl::half>(powf(2.0f, -14.0f));
}

template <typename T>
bool verify(sycl_cts::util::logger& log, T a, T b, int accuracy,
            const std::string& comment);

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value ||
                            std::is_same<sycl::half, T>::value,
                        bool>::type
verify(sycl_cts::util::logger& log, T value, sycl_cts::resultRef<T> r,
       int accuracy, const std::string& comment) {
  const T reference = r.res;

  if (!r.undefined.empty())
    return true;  // result is undefined according to spec
  if (std::isnan(value) && std::isnan(reference))
    return true;  // NaN can have any nancode within
  if (value == reference)
    return true;  // bitwise equal numeric or infinity value

  if (!std::isnan(value) && !std::isnan(reference) && !std::isinf(value) &&
      !std::isinf(reference)) {
    if (accuracy < 0)
      return true;  // Implementation-defined or infinite ULP according to spec

    if ((std::fabs(value) < min_t<T>()) && (std::fabs(reference) < min_t<T>()))
      return true;  // Subnormal numbers are the lower border for comparison

    const auto ulpsExpected = static_cast<unsigned int>(accuracy);
    const T difference = static_cast<T>(std::fabs(value - reference));
    const T differenceExpected = ulpsExpected * get_ulp_std(reference);

    if (difference <= differenceExpected) return true;
  }

  log.note("value: " + printable(value) +
           ", reference: " + printable(reference));
  std::string msg = "Expected accuracy in ULP: " + std::to_string(accuracy);
  if (!comment.empty()) msg += ", " + comment;
  log.note(msg);
  return false;
}

template <typename T>
typename std::enable_if_t<std::is_integral_v<T>, bool> verify(
    sycl_cts::util::logger& log, T value, sycl_cts::resultRef<T> r, int,
    const std::string&) {
  bool result = value == r.res || !r.undefined.empty();
  if (!result)
    log.note("value: " + std::to_string(value) +
             ", reference: " + std::to_string(r.res));
  return result;
}

template <typename T, int N>
bool verify(sycl_cts::util::logger& log, sycl::vec<T, N> a,
            sycl_cts::resultRef<sycl::vec<T, N>> r, int accuracy,
            const std::string& comment) {
  sycl::vec<T, N> b = r.res;
  for (int i = 0; i < sycl_cts::math::numElements(a); i++)
    if (r.undefined.find(i) == r.undefined.end() &&
        !verify(log, a[i], b[i], accuracy, comment))
      return false;
  return true;
}

template <typename T, size_t N>
bool verify(sycl_cts::util::logger& log, sycl::marray<T, N> a,
            sycl_cts::resultRef<sycl::marray<T, N>> r, int accuracy,
            const std::string& comment) {
  sycl::marray<T, N> b = r.res;
  for (size_t i = 0; i < N; i++)
    if (r.undefined.find(i) == r.undefined.end() &&
        !verify(log, a[i], b[i], accuracy, comment))
      return false;
  return true;
}

template <typename T>
bool verify(sycl_cts::util::logger& log, T a, T b, int accuracy,
            const std::string& comment) {
  return verify(log, a, sycl_cts::resultRef<T>(b), accuracy, comment);
}

template <int N, typename returnT, typename funT>
void check_function(sycl_cts::util::logger& log, funT fun,
                    sycl_cts::resultRef<returnT> ref, int accuracy = 0,
                    const std::string& comment = {}) {
  sycl::range<1> ndRng(1);
  returnT kernelResult;
  auto&& testQueue = once_per_unit::get_queue();
  try {
    sycl::buffer<returnT, 1> buffer(&kernelResult, ndRng);
    testQueue.submit([&](sycl::handler& h) {
      auto resultPtr = buffer.template get_access<sycl::access_mode::write>(h);
      h.single_task<kernel<N>>(
          [=]() { value_operations::assign(resultPtr[0], fun()); });
    });
  } catch (const sycl::exception& e) {
    log_exception(log, e);
    std::string errorMsg = "tests case: " + std::to_string(N) +
                           " a SYCL exception was caught: " + e.what();
    FAIL(log, errorMsg.c_str());
  }

  if (!verify(log, kernelResult, ref, accuracy, comment))
    FAIL(log,
         "tests case: " + std::to_string(N) + ". Correctness check failed.");

  // host check
  auto hostRes = fun();
  INFO("tests case: " + std::to_string(N) +
       ". Correctness check failed on host.");
  // SYCL 2020 specification sets no requirements for math built-ins accuracy
  // on host, hence passing negative value to 'verify' helper to indicate that.
  CHECK(verify(log, hostRes, ref, -1, comment));
}

template <int N, typename returnT, typename funT, typename argT>
void check_function_multi_ptr_private(sycl_cts::util::logger& log, funT fun,
                                      sycl_cts::resultRef<returnT> ref,
                                      argT ptrRef, int accuracy = 0,
                                      const std::string& comment = {}) {
  sycl::range<1> ndRng(1);
  returnT kernelResult;
  argT kernelResultArg;
  auto&& testQueue = once_per_unit::get_queue();
  try {
    sycl::buffer<returnT, 1> buffer(&kernelResult, ndRng);
    sycl::buffer<argT, 1> bufferArg(&kernelResultArg, ndRng);
    testQueue.submit([&](sycl::handler& h) {
      auto resultPtr = buffer.template get_access<sycl::access_mode::write>(h);
      auto resultPtrArg =
          bufferArg.template get_access<sycl::access_mode::write>(h);
      h.single_task<kernel<N>>([=]() {
        privatePtrCheck<returnT, argT> result = fun();
        resultPtr[0] = result.res;
        resultPtrArg[0] = result.resArg;
      });
    });
  } catch (const sycl::exception& e) {
    log_exception(log, e);
    std::string errorMsg = "tests case: " + std::to_string(N) +
                           " a SYCL exception was caught: " + e.what();
    FAIL(log, errorMsg.c_str());
  }

  if (!verify(log, kernelResult, ref, accuracy, comment))
    FAIL(log,
         "tests case: " + std::to_string(N) + ". Correctness check failed.");
  if (!verify(log, kernelResultArg, ptrRef, accuracy, comment))
    FAIL(log, "tests case: " + std::to_string(N) +
                  ". Correctness check for ptr failed.");

  // host check
  privatePtrCheck<returnT, argT> hostRes = fun();
  {
    INFO("tests case: " + std::to_string(N) +
         ". Correctness check failed on host.");
    CHECK(verify(log, hostRes.res, ref, accuracy, comment));
  }
  {
    INFO("tests case: " + std::to_string(N) +
         ". Correctness check for ptr failed on host.");
    CHECK(verify(log, hostRes.resArg, ptrRef, accuracy, comment));
  }
}

template <int N, typename returnT, typename funT, typename argT>
void check_function_multi_ptr_global(sycl_cts::util::logger& log, funT fun,
                                     argT arg, sycl_cts::resultRef<returnT> ref,
                                     argT ptrRef, int accuracy = 0,
                                     const std::string& comment = {}) {
  sycl::range<1> ndRng(1);
  returnT kernelResult;
  auto&& testQueue = once_per_unit::get_queue();
  try {
    sycl::buffer<returnT, 1> buffer(&kernelResult, ndRng);
    sycl::buffer<argT, 1> ptrBuffer(&arg, ndRng);
    testQueue.submit([&](sycl::handler& h) {
      auto resultPtr = buffer.template get_access<sycl::access_mode::write>(h);
      sycl::accessor<argT, 1, sycl::access_mode::read_write,
                     sycl::target::device>
          globalAccessor(ptrBuffer, h);
      h.single_task<kernel<N>>([=]() { resultPtr[0] = fun(globalAccessor); });
    });
  } catch (const sycl::exception& e) {
    log_exception(log, e);
    std::string errorMsg = "tests case: " + std::to_string(N) +
                           " a SYCL exception was caught: " + e.what();
    FAIL(log, errorMsg.c_str());
  }

  if (!verify(log, kernelResult, ref, accuracy, comment))
    FAIL(log,
         "tests case: " + std::to_string(N) + ". Correctness check failed.");
  if (!verify(log, arg, ptrRef, accuracy, comment))
    FAIL(log, "tests case: " + std::to_string(N) +
                  ". Correctness check for ptr failed.");
}

template <int N, typename returnT, typename funT, typename argT>
void check_function_multi_ptr_local(sycl_cts::util::logger& log, funT fun,
                                    argT arg, sycl_cts::resultRef<returnT> ref,
                                    argT ptrRef, int accuracy = 0,
                                    const std::string& comment = {}) {
  sycl::range<1> ndRng(1);
  returnT kernelResult;
  auto&& testQueue = once_per_unit::get_queue();
  try {
    sycl::buffer<returnT, 1> buffer(&kernelResult, ndRng);
    sycl::buffer<argT, 1> bufferArg(&arg, ndRng);
    testQueue.submit([&](sycl::handler& h) {
      auto resultPtr = buffer.template get_access<sycl::access_mode::write>(h);
      auto resultPtrArg =
          bufferArg.template get_access<sycl::access_mode::write>(h);
      sycl::local_accessor<argT, 1> localAccessor(1, h);
      h.parallel_for<kernel<N>>(
          sycl::nd_range<1>{ndRng, ndRng},
          [arg, localAccessor, resultPtr, resultPtrArg, fun](sycl::nd_item<1>) {
            localAccessor[0] = arg;
            resultPtr[0] = fun(localAccessor);
            resultPtrArg[0] = localAccessor[0];
          });
    });
  } catch (const sycl::exception& e) {
    log_exception(log, e);
    std::string errorMsg = "tests case: " + std::to_string(N) +
                           " a SYCL exception was caught: " + e.what();
    FAIL(log, errorMsg.c_str());
  }

  if (!verify(log, kernelResult, ref, accuracy, comment))
    FAIL(log,
         "tests case: " + std::to_string(N) + ". Correctness check failed.");
  if (!verify(log, arg, ptrRef, accuracy, comment))
    FAIL(log, "tests case: " + std::to_string(N) +
                  ". Correctness check for ptr failed.");
}

template <int T, typename returnT, typename funT>
void test_function(funT fun) {
  sycl::range<1> ndRng(1);
  returnT* kernelResult = new returnT[1];
  auto&& testQueue = once_per_unit::get_queue();
  {
    sycl::buffer<returnT, 1> buffer(kernelResult, ndRng);
    testQueue.submit([&](sycl::handler& h) {
      auto resultPtr = buffer.template get_access<sycl::access_mode::write>(h);
      h.single_task<kernel<T>>([=]() { resultPtr[0] = fun(); });
    });
  }
  testQueue.wait_and_throw();
  delete[] kernelResult;
}

template <int T, typename returnT, typename funT, typename argT>
void test_function_multi_ptr_global(funT fun, argT arg) {
  sycl::range<1> ndRng(1);
  returnT* kernelResult = new returnT[1];
  auto&& testQueue = once_per_unit::get_queue();
  {
    sycl::buffer<returnT, 1> buffer(kernelResult, ndRng);
    sycl::buffer<argT, 1> ptrBuffer(&arg, ndRng);
    testQueue.submit([&](sycl::handler& h) {
      auto resultPtr = buffer.template get_access<sycl::access_mode::write>(h);
      sycl::accessor<argT, 1, sycl::access_mode::read_write,
                     sycl::target::device>
          globalAccessor(ptrBuffer, h);
      h.single_task<kernel<T>>([=]() { resultPtr[0] = fun(globalAccessor); });
    });
  }
  testQueue.wait_and_throw();
  delete[] kernelResult;
}

template <int T, typename returnT, typename funT, typename argT>
void test_function_multi_ptr_local(funT fun, argT arg) {
  sycl::range<1> ndRng(1);
  returnT* kernelResult = new returnT[1];
  auto&& testQueue = once_per_unit::get_queue();
  {
    sycl::buffer<returnT, 1> buffer(kernelResult, ndRng);
    testQueue.submit([&](sycl::handler& h) {
      auto resultPtr = buffer.template get_access<sycl::access_mode::write>(h);
      sycl::accessor<argT, 1, sycl::access_mode::read_write,
                     sycl::target::local>
          localAccessor(1, h);
      h.single_task<kernel<T>>([arg, localAccessor, resultPtr, fun]() {
        localAccessor[0] = arg;
        resultPtr[0] = fun(localAccessor);
      });
    });
  }
  testQueue.wait_and_throw();
  delete[] kernelResult;
}

template <typename T>
struct ImplicitlyConvertibleType {
  operator T() const { return {}; }
};

#endif  // CL_SYCL_CTS_MATH_BUILTIN_API_MATH_BUILTIN_H
