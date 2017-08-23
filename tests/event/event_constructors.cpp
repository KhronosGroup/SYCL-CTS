/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME event_constructors

namespace event_constructors__ {
using namespace sycl_cts;

inline cl::sycl::event get_queue_event(cl::sycl::queue &queue) {
  return queue.submit([&](cl::sycl::handler &handler) {
    handler.single_task<class TEST_NAME>([=]() {});
  });
}

/** tests the constructors for cl::sycl::event
 */
class TEST_NAME : public util::test_base {
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      cts_selector selector;
      auto queue = util::get_cts_object::queue(selector);

      /** check copy constructor
      */
      {
        auto eventA = get_queue_event(queue);
        cl::sycl::event eventB(eventA);

        if (!selector.is_host() && (eventA.get() != eventB.get())) {
          FAIL(log, "event was not copied correctly.");
        }
      }

      /** check assignment operator
      */
      {
        auto eventA = get_queue_event(queue);
        auto eventB = get_queue_event(queue);
        eventB = eventA;

        if (!selector.is_host() && (eventA.get() != eventB.get())) {
          FAIL(log, "event was not assigned correctly.");
        }
      }

      /** check move constructor
      */
      {
        auto eventA = get_queue_event(queue);
        cl::sycl::event eventB(std::move(eventA));

        if (!selector.is_host() && (eventB.get() == nullptr)) {
          FAIL(log, "event was not move constructed correctly. (get)");
        }
      }

      /** check move assignment operator
      */
      {
        auto eventA = get_queue_event(queue);
        auto eventB = get_queue_event(queue);
        eventB = std::move(eventA);

        if (!selector.is_host() && (eventB.get() == nullptr)) {
          FAIL(log, "event was not move assigned correctly. (get)");
        }
      }

      /** check equality operator
      */
      {
        auto eventA = get_queue_event(queue);
        cl::sycl::event eventB(eventA);
        auto eventC = get_queue_event(queue);
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
        auto eventA = get_queue_event(queue);
        cl::sycl::event eventB = eventA;
        cl::sycl::hash_class<cl::sycl::event> hasher;

        if (hasher(eventA) != hasher(eventB)) {
          FAIL(log,
               "event hashing does not work correctly (hashing of equals "
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

} /* namespace event_constructors__ */
