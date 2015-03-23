/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME pointer_constructors

namespace pointer_constructors__ {
using namespace sycl_cts;

template <typename T>
class pointer_apis {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue) {
    const int size = 64;
    cl::sycl::range<1> range(size);
    cl::sycl::unique_ptr<T> data(new T[size]);
    cl::sycl::buffer<T, 1> buffer(data.get(), range);

    queue.submit([&](cl::sycl::handler &handler) {
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>
          globalAccessor(buffer, handler);
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::constant_buffer>
          constantAccessor(buffer, handler);
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          localAccessor(size, handler);

      handler.single_task([=]() {
        T privateData[1];

        /** check (elementType *) constructors and destructors
        */
        {
          cl::sycl::global_ptr<T> globalPtr(&globalAccessor[0]);
          cl::sycl::constant_ptr<T> constantPtr(&constantAccessor[0]);
          cl::sycl::local_ptr<T> localPtr(&localAccessor[0]);
          cl::sycl::private_ptr<T> privatePtr(privateData);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::global_space>
              globalMultiPtr(&globalAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::constant_space>
              constantMultiPtr(&constantAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::local_space>
              localMultiPtr(&localAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::private_space>
              privateMultiPtr(privateData);
        }

        /** check (accessor) constructors
        */
        {
          cl::sycl::global_ptr<T> globalPtr(globalAccessor);
          cl::sycl::constant_ptr<T> constantPtr(constantAccessor);
          cl::sycl::local_ptr<T> localPtr(localAccessor);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::global_space>
              globalMultiPtr(globalAccessor);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::constant_space>
              constantMultiPtr(constantAccessor);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::private_space>
              privateMultiPtr(privateData);
        }

        /** check copy constructors
        */
        {
          cl::sycl::global_ptr<T> globalPtrA(&globalAccessor[0]);
          cl::sycl::constant_ptr<T> constantPtrA(&constantAccessor[0]);
          cl::sycl::local_ptr<T> localPtrA(&localAccessor[0]);
          cl::sycl::private_ptr<T> privatePtrA(privateData);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::global_space>
              globalMultiPtrA(&globalAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::constant_space>
              constantMultiPtrA(&constantAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::local_space>
              localMultiPtrA(&localAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::private_space>
              privateMultiPtrA(privateData);

          cl::sycl::global_ptr<T> globalPtrB(globalPtrA);
          cl::sycl::constant_ptr<T> constantPtrB(constantPtrA);
          cl::sycl::local_ptr<T> localPtrB(localPtrA);
          cl::sycl::private_ptr<T> provatePtrB(provatePtrA);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::global_space>
              globalMultiPtrB(globalMultiPtrA);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::constant_space>
              constantMultiPtrB(constantMultiPtrA);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::local_space>
              localMultiPtrB(localMultiPtrA);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::private_space>
              privateMultiPtrB(privateMultiPtrA);
        }

        /** check assignment operators
        */
        {
          cl::sycl::global_ptr<T> globalPtrA(&globalAccessor[0]);
          cl::sycl::constant_ptr<T> constantPtrA(&constantAccessor[0]);
          cl::sycl::local_ptr<T> localPtrA(&localAccessor[0]);
          cl::sycl::private_ptr<T> privatePtrA(privateData);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::global_space>
              globalMultiPtrA(&globalAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::constant_space>
              constantMultiPtrA(&constantAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::local_space>
              localMultiPtrA(&localAccessor[0]);
          cl::sycl::multi_ptr<T, cl::sycl::address_space::private_space>
              privateMultiPtrA(privateData);

          cl::sycl::global_ptr<T> globalPtrB = globalPtrA;
          cl::sycl::constant_ptr<T> constantPtrB = constantPtrA;
          cl::sycl::local_ptr<T> localPtrB = localPtrA;
          cl::sycl::private_ptr<T> privatePtrB = provatePtrA;
          cl::sycl::multi_ptr<T, cl::sycl::address_space::global_space>
              globalMultiPtrB = globalMultiPtrA;
          cl::sycl::multi_ptr<T, cl::sycl::address_space::constant_space>
              constantMultiPtrB = constantMultiPtrA;
          cl::sycl::multi_ptr<T, cl::sycl::address_space::local_space>
              localMultiPtrB = localMultiPtrA;
          cl::sycl::multi_ptr<T, cl::sycl::address_space::private_space>
              privateMultiPtrB = privateMultiPtrA;

          cl::sycl::global_ptr<T> globalPtrC = &globalAccessor[0];
          cl::sycl::constant_ptr<T> constantPtrC = &constantAccessor[0];
          cl::sycl::local_ptr<T> localPtrC = &localAccessor[0];
          cl::sycl::private_ptr<T> privatePtrC = privateData;
          cl::sycl::multi_ptr<T, cl::sycl::address_space::global_space>
              globalMultiPtrC = &globalAccessor[0];
          ;
          cl::sycl::multi_ptr<T, cl::sycl::address_space::constant_space>
              constantMultiPtrC = &constantAccessor[0];
          cl::sycl::multi_ptr<T, cl::sycl::address_space::local_space>
              localMultiPtrC = &localAccessor[0];
          cl::sycl::multi_ptr<T, cl::sycl::address_space::private_space>
              privateMultiPtrC = privateData;
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

/** tests the constructors for explicit pointers
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
