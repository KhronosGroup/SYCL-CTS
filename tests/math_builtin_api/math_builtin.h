/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef CL_SYCL_CTS_MATH_BUILTIN_API_MATH_BUILTIN_H
#define CL_SYCL_CTS_MATH_BUILTIN_API_MATH_BUILTIN_H

template <int T>
class kernel;

inline cl::sycl::queue makeQueueOnce() {
  static cl::sycl::queue q = sycl_cts::util::get_cts_object::queue();
  return q;
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
