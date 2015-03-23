/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME pointer_api

namespace pointer_api__ {
using namespace sycl_cts;

template <typename T>
class pointer_apis {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue) {
    const int size = 64;
    cl::sycl::range<1> range(size);
    cl::sycl::unique_ptr<T[]> data(new T[size]);
    cl::sycl::buffer<T, 1> buffer(data.get(), range);

    queue.submit([&](cl::sycl::handler &handler) {
      cl::sycl::accessor<cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>
          globalAccessor(buffer, handler);
      cl::sycl::accessor<cl::sycl::access::mode::read,
                         cl::sycl::access::target::constant_buffer>
          constantAccessor(buffer, handler);
      cl::sycl::accessor<cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          constantAccessor(size, handler);

      handler.single_task([=]() {
        T privateData[1];

        /** check operator[int]() methods
        */
        {
          cl::sycl::global_ptr<T> globalPtr(&globalAccessor[0]);
          cl::sycl::constant_ptr<T> constantPtr(&constantAccessor[0]);
          cl::sycl::local_ptr<T> localPtr(&localAccessor[0]);
          cl::sycl::private_ptr<T> provatePtr(privateData);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::global_space>
              globalMultiPtr(&globalAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::constant_space>
              constantMultiPtr(&constantAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::local_space>
              localMultiPtr(&localAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::private_space>
              privateMultiPtr(privateData);

          globalPtr[0] = static_cast<T>(0);
          localPtr[0] = static_cast<T>(0);
          privatePtr[0] = static_cast<T>(0);
          globalMultiPtr[0] = static_cast<T>(0);
          localMultiPtr[0] = static_cast<T>(0);
          privateMultiPtr[0] = static_cast<T>(0);

          T &globalElem = globalPtr[0];
          T &constantElem = constantPtr[0];
          T &localElem = localPtr[0];
          T &privateElem = privatePtr[0];
          T &globalElem = globalMultiPtr[0];
          T &constantElem = constantMultiPtr[0];
          T &localElem = localMultiPtr[0];
          T &privateElem = privateMultiPtr[0];
        }

        /** check operator*() methods
        */
        {
          cl::sycl::global_ptr<T> globalPtr(&globalAccessor[0]);
          cl::sycl::constant_ptr<T> constantPtr(&constantAccessor[0]);
          cl::sycl::local_ptr<T> localPtr(&localAccessor[0]);
          cl::sycl::private_ptr<T> provatePtr(privateData);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::global_space>
              globalMultiPtr(&globalAccessor[0]);
          cl::sycl::multi_ptr<T,
                              cl::sycl::cl::sycl::address_space::constant_space>
              constantMultiPtr(&constantAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::local_space>
              localMultiPtr(&localAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::private_space>
              privateMultiPtr(privateData);

          (*globalMultiPtr) = static_cast<T>(0);
          (*localMultiPtr) = static_cast<T>(0);
          (*privateMultiPtr) = static_cast<T>(0);

          T &globalMultiPtr = (*globalPtr);
          T &constantMultiPtr = (*constantPtr);
          T &localMultiPtr = (*localPtr);
          T &privateMultiPtr = (*privatePtr);
        }

        /** check pointer() methods
        */
        {
          cl::sycl::multi_ptr<T, cl::sycl::address_space::global_space>
              globalMultiPtr(&globalAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::constant_space>
              constantMultiPtr(&constantAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::local_space>
              localMultiPtr(&localAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::private_space>
              privateMultiPtr(privateData);

          cl::sycl::global_ptr<T> globalPtr = globalMultiPtr.pointer();
          cl::sycl::constant_ptr<T> constantPtr = constantMultiPtr.pointer();
          cl::sycl::local_ptr<T> constantPtr = localMultiPtr.pointer();
          cl::sycl::private_ptr<T> privatePtr = privateMultiPtr.pointer();
        }

        /** check make_ptr() function
        */
        {
          cl::sycl::multi_ptr<T, cl::sycl::address_space::global_space>
              globalMultiPtr =
                  make_ptr<T, cl::sycl::address_space::global_space>(
                      &globalAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::constant_space>
              constantMultiPtr =
                  make_ptr<T, cl::sycl::address_space::constant_space>(
                      &constantAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::local_space>
              localMultiPtr = make_ptr<T, cl::sycl::address_space::local_space>(
                  &localAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::private_space>
              privateMultiPtr =
                  make_ptr<T, cl::sycl::address_space::private_space>(
                      privateData);
        }

      });
    });
  }
};

struct user_struct {
  float a;
  int b;
  char c;
};

/** tests the api for explicit pointers
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
  */
  virtual void run(util::logger &log) override {
    try {
      cts_selector selector;
      cl::sycl::queue queue(selector);

      pointer_apis<int> intTests;
      intTests(log, queue);

      pointer_apis<long long> longLongTests;
      longLongTests(log, queue);

      pointer_apis<float> floatTests;
      floatTests(log, queue);

      pointer_apis<double> doubleTests;
      doubleTests(log, queue);

      pointer_apis<user_struct> userStructTests;
      userStructTests(log, queue);

      queue.wait_and_throw();
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "a sycl exception was caught");
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace pointer_constructors__ */
