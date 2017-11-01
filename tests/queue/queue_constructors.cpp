/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME queue_constructors

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** check the constructors for cl::sycl::queue
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
    try {
      /** check default constructor and destructor
      */
      { cl::sycl::queue queue; }

      /** check (property_list) constructor
      */
      {
        cts_async_handler asyncHandler;
        cl::sycl::queue queue({cl::sycl::property::queue::enable_profiling()});

        if (!queue
                 .has_property<cl::sycl::property::queue::enable_profiling>()) {
          FAIL(log,
               "queue with property_list was not constructed correctly "
               "(has_property)");
        }
      }

      /** check (async_handler) constructor
      */
      {
        cts_async_handler asyncHandler;
        cl::sycl::queue queue(asyncHandler);
      }

      /** check (async_handler, property_list) constructor
      */
      {
        cts_async_handler asyncHandler;
        cl::sycl::queue queue(asyncHandler,
                              {cl::sycl::property::queue::enable_profiling()});

        if (!queue
                 .has_property<cl::sycl::property::queue::enable_profiling>()) {
          FAIL(log,
               "queue with async_handler and property_list was not "
               "constructed correctly (has_property)");
        }
      }

      /** check (device_selector) constructor
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
                 "queue with device_selector was not constructed "
                 "correctly (get)");
          }
        }
      }

      /** check (device_selector, property_list) constructor
      */
      {
        cts_selector selector;
        cl::sycl::queue queue(selector,
                              {cl::sycl::property::queue::enable_profiling()});

        if (queue.is_host() != selector.is_host()) {
          FAIL(log,
               "queue with device_selector and property list was not "
               "constructed correctly (is_host)");
        }

        if (!selector.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with device_selector and property_list was not "
                 "constructed correctly (get)");
          }
        }

        if (!queue
                 .has_property<cl::sycl::property::queue::enable_profiling>()) {
          FAIL(log,
               "queue with device_selector and property_list was not "
               "constructed correctly (has_property)");
        }
      }

      /** check (device_selector, async_handler) constructor
      */
      {
        cts_selector selector;
        cts_async_handler asyncHandler;
        cl::sycl::queue queue(selector, asyncHandler);

        if (queue.is_host() != selector.is_host()) {
          FAIL(log,
               "queue with device_selector and async_handler was not "
               "constructed correctly (is_host)");
        }

        if (!selector.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with device_selector and async_handler was not "
                 "constructed correctly (get)");
          }
        }
      }

      /** check (device_selector, async_handler, property_list) constructor
      */
      {
        cts_selector selector;
        cts_async_handler asyncHandler;
        cl::sycl::queue queue(selector, asyncHandler,
                              {cl::sycl::property::queue::enable_profiling()});

        if (queue.is_host() != selector.is_host()) {
          FAIL(log,
               "queue with device_selector, async_handler and "
               "property_list was not constructed correctly (is_host)");
        }

        if (!selector.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with device_selector, async_handler and "
                 "property_list was not constructed correctly (get)");
          }
        }

        if (!queue
                 .has_property<cl::sycl::property::queue::enable_profiling>()) {
          FAIL(log,
               "queue with device_selector, async_handler and "
               "property_list was not constructed correctly (has_property)");
        }
      }

      /** check (device) constructor
      */
      {
        cl::sycl::device device = util::get_cts_object::device();
        cl::sycl::queue queue(device);

        if (queue.is_host() != device.is_host()) {
          FAIL(log,
               "queue with device was not constructed correctly "
               "(is_host)");
        }

        if (!device.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log, "queue with device was not constructed correctly (get)");
          }
        }
      }

      /** check (device, property_list) constructor
      */
      {
        cl::sycl::device device = util::get_cts_object::device();
        cl::sycl::queue queue(device,
                              {cl::sycl::property::queue::enable_profiling()});

        if (queue.is_host() != device.is_host()) {
          FAIL(log,
               "queue with device was not constructed correctly "
               "(is_host)");
        }

        if (!device.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log, "queue with device was not constructed correctly (get)");
          }
        }

        if (!queue
                 .has_property<cl::sycl::property::queue::enable_profiling>()) {
          FAIL(log,
               "queue with device and property_list was not constructed "
               "correctly (has_property)");
        }
      }

      /** check (device, async_handler) constructor
      */
      {
        cl::sycl::device device = util::get_cts_object::device();
        cts_async_handler asyncHandler;
        cl::sycl::queue queue(device, asyncHandler);

        if (queue.is_host() != device.is_host()) {
          FAIL(log,
               "queue with device and async_hander was not constructed "
               "correctly (is_host)");
        }

        if (!device.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with device and async_hander was not constructed "
                 "correctly (get)");
          }
        }
      }

      /** check (device, async_handler, property_list) constructor
      */
      {
        cl::sycl::device device = util::get_cts_object::device();
        cts_async_handler asyncHandler;
        cl::sycl::queue queue(device, asyncHandler,
                              {cl::sycl::property::queue::enable_profiling()});

        if (queue.is_host() != device.is_host()) {
          FAIL(log,
               "queue with device and async_hander was not constructed "
               "correctly (is_host)");
        }

        if (!device.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with device and async_hander was not constructed "
                 "correctly (get)");
          }
        }

        if (!queue
                 .has_property<cl::sycl::property::queue::enable_profiling>()) {
          FAIL(log,
               "queue with device, async_handler and property_list was "
               "not constructed correctly (has_property)");
        }
      }

      /** check (context, device_selector) constructor
      */
      {
        cts_selector selector;
        auto context = util::get_cts_object::context(selector);
        cl::sycl::queue queue(context, selector);

        if (queue.is_host() != selector.is_host()) {
          FAIL(log,
               "queue with context and device_selector was not "
               "constructed correctly (is_host)");
        }

        if (!selector.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with context and device_selector was not "
                 "constructed correctly (get)");
          }
        }
      }

      /** check (context, device_selector, property_list) constructor
      */
      {
        cts_selector selector;
        auto context = util::get_cts_object::context(selector);
        cl::sycl::queue queue(context, selector,
                              {cl::sycl::property::queue::enable_profiling()});

        if (queue.is_host() != selector.is_host()) {
          FAIL(log,
               "queue with context and device_selector was not "
               "constructed correctly (is_host)");
        }

        if (!selector.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with context and device_selector was not "
                 "constructed correctly (get)");
          }
        }

        if (!queue
                 .has_property<cl::sycl::property::queue::enable_profiling>()) {
          FAIL(log,
               "queue with context, device_selector and property_list was "
               "not constructed correctly (has_property)");
        }
      }

      /** check (context, device_selector, async_handler) constructor
      */
      {
        cts_selector selector;
        auto context = util::get_cts_object::context(selector);
        cts_async_handler asyncHandler;
        cl::sycl::queue queue(context, selector, asyncHandler);

        if (queue.is_host() != selector.is_host()) {
          FAIL(log,
               "queue with context, device_selector and async_handler was "
               "not constructed correctly (is_host)");
        }

        if (!selector.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with context, device_selector and async_handler "
                 "was not constructed correctly (get)");
          }
        }
      }

      /** check (context, device_selector, async_handler, property_list)
      constructor
      */
      {
        cts_selector selector;
        auto context = util::get_cts_object::context(selector);
        cts_async_handler asyncHandler;
        cl::sycl::queue queue(context, selector, asyncHandler,
                              {cl::sycl::property::queue::enable_profiling()});

        if (queue.is_host() != selector.is_host()) {
          FAIL(log,
               "queue with context, device_selector, async_handler and "
               "property_list was not constructed correctly (is_host)");
        }

        if (!selector.is_host()) {
          if (queue.get() == nullptr) {
            FAIL(log,
                 "queue with context, device_selector, async_handler and "
                 "property_list was not constructed correctly (get)");
          }
        }

        if (!queue
                 .has_property<cl::sycl::property::queue::enable_profiling>()) {
          FAIL(log,
               "queue with context, device_selector, async_handler and "
               "property_list was not constructed correctly (has_property)");
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
               "queue equality does not work correctly (copy constructed)");
        }
        if (!(queueA == queueC)) {
          FAIL(log, "queue equality does not work correctly (copy assigned)");
        }
      }

      /** check hashing
      */
      {
        cts_selector selector;
        auto queueA = util::get_cts_object::queue(selector);
        cl::sycl::queue queueB(queueA);
        cl::sycl::hash_class<cl::sycl::queue> hasher;

        if (hasher(queueA) != hasher(queueB)) {
          FAIL(log,
               "queue hasher does not work correctly (hashing of equal "
               "failed)");
        }
      }
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
