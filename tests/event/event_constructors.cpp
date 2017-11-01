/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME event_constructors

namespace TEST_NAMESPACE {

using namespace sycl_cts;

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
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

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

        if (!selector.is_host() && (eventA.get() != eventB.get())) {
          FAIL(log, "event was not copied correctly.");
        }
      }

      /** check assignment operator
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto eventA = get_queue_event<class event_constructors_1>(queue);
        auto eventB = get_queue_event<class event_constructors_2>(queue);
        eventB = eventA;

        if (!selector.is_host() && (eventA.get() != eventB.get())) {
          FAIL(log, "event was not assigned correctly.");
        }
      }

      /** check move constructor
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto eventA = get_queue_event<class event_constructors_3>(queue);
        cl::sycl::event eventB(std::move(eventA));

        if (!selector.is_host() && (eventB.get() == nullptr)) {
          FAIL(log, "event was not move constructed correctly. (get)");
        }
      }

      /** check move assignment operator
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto eventA = get_queue_event<class event_constructors_4>(queue);
        auto eventB = get_queue_event<class event_constructors_5>(queue);
        eventB = std::move(eventA);

        if (!selector.is_host() && (eventB.get() == nullptr)) {
          FAIL(log, "event was not move assigned correctly. (get)");
        }
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

        if (!(eventA == eventB)) {
          FAIL(log,
               "event equality does not work correctly (copy constructed)");
        }
        if (!(eventA == eventC)) {
          FAIL(log, "event equality does not work correctly (copy assigned)");
        }
      }

      /** check hashing
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto eventA = get_queue_event<class event_constructors_8>(queue);
        cl::sycl::event eventB = eventA;
        cl::sycl::hash_class<cl::sycl::event> hasher;

        if (hasher(eventA) != hasher(eventB)) {
          FAIL(log,
               "event hashing does not work correctly (hashing of equals "
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
