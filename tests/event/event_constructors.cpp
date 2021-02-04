/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME event_constructors

namespace TEST_NAMESPACE {

using namespace sycl_cts;

class event_constructors_0;
class event_constructors_1;
class event_constructors_2;
class event_constructors_3;
class event_constructors_4;
class event_constructors_5;
class event_constructors_6;
class event_constructors_7;
class event_constructors_8;
class event_constructors_9;

/** test the constructors for cl::sycl::event
 */
class TEST_NAME : public util::test_base {
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      /** check default constructor and destructor
      */
      {
        cl::sycl::event event;

        if (!event.is_host()) {
          FAIL(log, "event was not constructed correctly (is_host).");
        }
      }

      /** check copy constructor
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto eventA = get_queue_event<class event_constructors_0>(queue);
        cl::sycl::event eventB(eventA);

#ifdef SYCL_CTS_TEST_OPENCL_INTEROP
        if (!selector.is_host() && (eventA.get() != eventB.get())) {
          FAIL(log, "event was not copied correctly.");
        }
#endif

        queue.wait_and_throw();
      }

      /** check assignment operator
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto eventA = get_queue_event<class event_constructors_1>(queue);
        auto eventB = get_queue_event<class event_constructors_2>(queue);
        eventB = eventA;

#ifdef SYCL_CTS_TEST_OPENCL_INTEROP
        if (!selector.is_host() && (eventA.get() != eventB.get())) {
          FAIL(log, "event was not assigned correctly.");
        }
#endif

        queue.wait_and_throw();
      }

      /** check move constructor
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto eventA = get_queue_event<class event_constructors_3>(queue);
        cl::sycl::event eventB(std::move(eventA));

        queue.wait_and_throw();
      }

      /** check move assignment operator
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto eventA = get_queue_event<class event_constructors_4>(queue);
        auto eventB = get_queue_event<class event_constructors_5>(queue);
        eventB = std::move(eventA);

        queue.wait_and_throw();
      }

      /** check equality operator
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto eventA = get_queue_event<class event_constructors_6>(queue);
        cl::sycl::event eventB(eventA);
        auto eventC = get_queue_event<class event_constructors_7>(queue);
        eventC = eventA;
        auto eventD = get_queue_event<class event_constructors_8>(queue);

        if (!(eventA == eventB)) {
          FAIL(log,
               "event equality does not work correctly (copy constructed)");
        }
        if (!(eventA == eventC)) {
          FAIL(log, "event equality does not work correctly (copy assigned)");
        }
        if (eventA != eventB) {
          FAIL(log,
               "event non-equality does not work correctly"
               "(copy constructed)");
        }
        if (eventA != eventC) {
          FAIL(log,
               "event non-equality does not work correctly"
               "(copy assigned)");
        }
        if (eventC == eventD) {
          FAIL(log,
               "event equality does not work correctly"
               "(comparing same)");
        }
        if (!(eventC != eventD)) {
          FAIL(log,
               "event non-equality does not work correctly"
               "(comparing same)");
        }

        queue.wait_and_throw();
      }

      /** check hashing
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto eventA = get_queue_event<class event_constructors_9>(queue);
        cl::sycl::event eventB = eventA;
        cl::sycl::hash_class<cl::sycl::event> hasher;

        if (hasher(eventA) != hasher(eventB)) {
          FAIL(log,
               "event hashing does not work correctly (hashing of equals "
               "failed)");
        }

        queue.wait_and_throw();
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
