/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME queue_constructors

namespace queue_constructors__ {
using namespace sycl_cts;

/** tests the constructors for cl::sycl::queue
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
      /** check default constructor and destructor
      */
      {
        cl::sycl::queue queue;

        // the resulting platform is implementation defined and cannot be tested
      }

      /** check (functor_class) constructor
      */
      {
        cts_async_handler asyncHandler;
        cl::sycl::queue queue(asyncHandler);

        // the resulting platform is implementation defined and cannot be tested
      }

      /** check (host_selector, functor_class = {}) constructor
      */
      {
        cl::sycl::host_selector selector;
        cl::sycl::queue queue(selector);

        if (!queue.is_host()) {
          FAIL(log,
               "queue with host_selector was not constructed correctly "
               "(is_host)");
        }
      }

      /** check (device_selector, functor_class = {}) constructor
      */
      {
        cts_selector selector;
        cl::sycl::queue queue(selector);

        if (queue.is_host() != selector.is_host()) {
          FAIL(log,
               "queue with device_selector was not constructed correctly "
               "(is_host)");
        }

        if (!selector.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with device_selector was not constructed correctly "
                 "(get)");
          }
        }
      }

      /** check (device_selector, functor_class) constructor
      */
      {
        cts_selector selector;
        cts_async_handler asyncHandler;
        cl::sycl::queue queue(selector, asyncHandler);

        if (queue.is_host() != selector.is_host()) {
          FAIL(log,
               "queue with device_selector was not constructed correctly "
               "(is_host)");
        }

        if (!selector.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with device_selector was not constructed correctly "
                 "(get)");
          }
        }
      }

      /** check (device, functor_class = {}) constructor
      */
      {
        cl::sycl::device device = util::get_cts_object::device();

        cl::sycl::queue queue(device);

        if (queue.is_host() != device.is_host()) {
          FAIL(log,
               "queue with device was not constructed correctly (is_host)");
        }

        if (!device.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log, "queue with device was not constructed correctly (get)");
          }
        }
      }

      /** check (device, functor_class) constructor
      */
      {
        cl::sycl::device device = util::get_cts_object::device();
        cts_async_handler asyncHandler;

        cl::sycl::queue queue(device, asyncHandler);

        if (queue.is_host() != device.is_host()) {
          FAIL(log,
               "queue with device was not constructed correctly (is_host)");
        }

        if (!device.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log, "queue with device was not constructed correctly (get)");
          }
        }
      }

      /** check (context, device_selector)
       * constructor
      */
      {
        cts_selector selector;
        auto context = util::get_cts_object::context(selector);

        cl::sycl::queue queue(context, selector);

        if (queue.is_host() != selector.is_host()) {
          FAIL(log,
               "queue with context and selector was not constructed correctly "
               "(is_host)");
        }

        if (!selector.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with context and selector was not constructed "
                 "correctly (get)");
          }
        }
      }

      /** check (context, host_selector)
      * constructor
      */
      {
        cl::sycl::host_selector selector;
        auto context = util::get_cts_object::context(selector);

        cl::sycl::queue queue(context, selector);

        if (!queue.is_host()) {
          FAIL(log,
               "queue with context and selector was not constructed correctly "
               "(is_host)");
        }

        if (!context.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with context and selector was not constructed "
                 "correctly (get)");
          }
        }
      }

      /** check (context, device) constructor
      */
      {
        cts_selector selector;
        cts_async_handler asyncHandler;
        auto device = util::get_cts_object::device(selector);
        cl::sycl::context context(device, false, asyncHandler);

        cl::sycl::queue queue(context, device);

        if (queue.is_host() != selector.is_host()) {
          FAIL(log,
               "queue with context and device was not constructed correctly "
               "(is_host)");
        }

        if (!selector.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with context and device was not constructed correctly "
                 "(get)");
          }
        }
      }

      /** check (context, device, info::queue_profiling = false) constructor
      */
      {
        cts_selector selector;
        cts_async_handler asyncHandler;
        auto device = util::get_cts_object::device(selector);
        cl::sycl::context context(device, false, asyncHandler);

        cl::sycl::queue queue(
            context, device,
            static_cast<cl::sycl::info::queue_profiling>(false));

        if (queue.is_host() != selector.is_host()) {
          FAIL(log,
               "queue with context, device, info::queue_profiling was not "
               "constructed correctly (is_host)");
        }

        if (!selector.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with context, device, info::queue_profiling was not "
                 "constructed correctly (get)");
          }
        }
      }

      /** check (context, device, info::queue_profiling = true) constructor
      */
      {
        cts_selector selector;
        cts_async_handler asyncHandler;
        auto device = util::get_cts_object::device(selector);
        cl::sycl::context context(device, false, asyncHandler);

        cl::sycl::queue queue(
            context, device,
            static_cast<cl::sycl::info::queue_profiling>(true));

        if (queue.is_host() != selector.is_host()) {
          FAIL(log,
               "queue with context, device, info::queue_profiling was not "
               "constructed correctly (is_host)");
        }

        if (!selector.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with context, device, info::queue_profiling was not "
                 "constructed correctly (get)");
          }
        }
      }

      /** check copy constructor
      */
      {
        cts_selector selector;
        auto queueA = util::get_cts_object::queue(selector);
        cl::sycl::queue queueB(queueA);

        if (queueA.is_host() != selector.is_host()) {
          FAIL(log, "queue source after copy construction failed (is_host)");
        }

        if (!selector.is_host()) {
          if (queueA.get() == nullptr) {
            FAIL(log, "queue source after copy construction failed (get)");
          }
        }

        if (!selector.is_host()) {
          if (queueB.get() == nullptr) {
            FAIL(log,
                 "queue destination was not copy constructed correctly "
                 "(is_host)");
          }
        }

        if (!selector.is_host()) {
          if (queueA.get() != queueB.get()) {
            FAIL(log,
                 "queue destination was not copy constructed correctly (get)");
          }
        }
      }

      /** check assignment operator
      */
      {
        cts_selector selector;
        auto queueA = util::get_cts_object::queue(selector);
        cl::sycl::queue queueB;
        queueB = queueA;

        if (queueA.is_host() != selector.is_host()) {
          FAIL(log, "queue source after copy assignment (=) failed (is_host)");
        }

        if (!selector.is_host()) {
          if (queueA.get() == nullptr) {
            FAIL(log, "queue source after copy assignment (=) failed (get)");
          }
        }

        if (queueA.is_host() != queueB.is_host()) {
          FAIL(log,
               "queue destination was not copy assigned (=) correctly "
               "(is_host)");
        }

        if (!selector.is_host()) {
          if (queueA.get() != queueB.get()) {
            FAIL(log,
                 "queue destination was not copy assigned (=) correctly "
                 "(get)");
          }
        }
      }

      /** check move constructor
      */
      {
        cts_selector selector;
        auto queueA = util::get_cts_object::queue(selector);
        cl::sycl::queue queueB(std::move(queueA));

        if (queueB.is_host() != selector.is_host()) {
          FAIL(log, "queue was not move constructed correctly (is_host)");
        }

        if (!selector.is_host()) {
          if (queueB.get() == nullptr) {
            FAIL(log, "queue was not move constructed correctly (get)");
          }
        }
      }

      /** check move assignment operator
      */
      {
        cts_selector selector;
        auto queueA = util::get_cts_object::queue(selector);

        cl::sycl::queue queueB;
        queueB = std::move(queueA);

        if (queueB.is_host() != selector.is_host()) {
          FAIL(log, "queue was not move assigned (=) correctly (is_host)");
        }

        if (!selector.is_host()) {
          if (queueB.get() == nullptr) {
            FAIL(log, "queue was not move assigned (=) correctly. (get)");
          }
        }
      }

      /* check equality operator
      */
      {
        cts_selector selector;
        auto queueA = util::get_cts_object::queue(selector);
        cl::sycl::queue queueB(queueA);
        cl::sycl::queue queueC = queueA;

        if (!(queueA == queueB)) {
          FAIL(log,
               "device equality does not work correctly (copy constructed)");
        }
        if (!(queueA == queueC)) {
          FAIL(log, "device equality does not work correctly (copy assigned)");
        }
      }

      /** Test Hash
      */
      {
        cts_selector selector;
        auto queueA = util::get_cts_object::queue(selector);
        cl::sycl::queue queueB(queueA);
        cl::sycl::hash_class<cl::sycl::queue> hasher;

        if (hasher(queueA) != hasher(queueB)) {
          FAIL(log,
               "device hasher does not work correctly (hashing of equal "
               "failed)");
        }
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace queue_constructors__ */
