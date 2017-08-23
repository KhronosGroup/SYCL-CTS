/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

template <int T>
class kernel;

template <int T, typename returnT, typename funT>
void test_function(funT fun) {
  cl::sycl::range<1> ndRng(1);
  returnT *kernelResult = new returnT[1];
  auto testQueue = sycl_cts::util::get_cts_object::queue();
  {
    cl::sycl::buffer<returnT, 1> buffer(kernelResult, ndRng);
    testQueue.submit([&](cl::sycl::handler &h) {
      auto ptr = buffer.template get_access<cl::sycl::access::mode::write>(h);
        h.single_task<class kernel<T> >([=](){
          ptr[0] = fun();
          });
    });
  }
  testQueue.wait_and_throw();
  delete[] kernelResult;
}
