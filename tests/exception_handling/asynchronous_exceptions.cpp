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

#include "../../util/sycl_exceptions.h"
#include "../common/common.h"

#define TEST_NAME asynchronous_exceptions

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/**
 * @brief Checks exception_list for aliased types
 */
void check_exception_list_types() {
  using value_type = sycl::exception_list::value_type;
  static_assert(std::is_same<value_type, std::exception_ptr>::value,
                "exception_list::value_type is of wrong type");

  { check_type_existence<sycl::exception_list::reference>(); }
  { check_type_existence<sycl::exception_list::const_reference>(); }

  using size_type = sycl::exception_list::size_type;
  static_assert(std::is_same<size_type, std::size_t>::value,
                "exception_list::size_type is of wrong type");

  { check_type_existence<sycl::exception_list::iterator>(); }
  { check_type_existence<sycl::exception_list::const_iterator>(); }
}

/**
 * @brief Checks exception_list for member functions
 * @param exceptionList List to check
 */
void check_exception_list_members(const sycl::exception_list &exceptionList) {
  auto size = exceptionList.size();
  auto beginIt = exceptionList.begin();
  auto endIt = exceptionList.end();
  // Silent warnings
  (void)size;
  (void)beginIt;
  (void)endIt;
}

class TEST_NAME_2;
class TEST_NAME_3;

/**
 */
class TEST_NAME : public util::test_base {
 public:
  struct async_handler_functor {
    std::vector<std::exception_ptr> excps;
    void operator()(sycl::exception_list l) {
      for (auto &e : l) {
        excps.push_back(e);
      }
    }
  };

  static void async_handler_function(sycl::exception_list l) {
    /*no access to logger at this point*/
    for (auto &e : l) {
      throw e;
    }
  }

  /** log the exceptions.
   */
  void check_exceptions(util::logger &log,
                        std::vector<std::exception_ptr> &excps) const {
    for (auto &e : excps) {
      try {
        throw e;
      } catch (const sycl::exception &e) {
        // Check methods
        std::string sc = e.what();
        if (e.has_context()) {
          sycl::context c = e.get_context();
          (void)c;
        }
        std::error_code ci = e.code();
        (void)ci;

        log_exception(log, e);
        FAIL(log, "An exception should not really have been thrown");
      }
    }
  }

  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    /*test lambda async handler*/
    {
      std::vector<std::exception_ptr> excps;
      std::function<void(sycl::exception_list)> asyncHandlerLambda =
          [&excps](sycl::exception_list l) {
            // Check the exception list interface
            check_exception_list_types();
            check_exception_list_members(l);

            for (auto &e : l) {
              excps.push_back(e);
            }
          };

      sycl::queue q(cts_selector, asyncHandlerLambda);

      q.submit([&](sycl::handler &cgh) {
        cgh.single_task<class TEST_NAME>([=]() {});
      });

      // Should not throw exceptions
      q.wait();

      check_exceptions(log, excps);
    }

    /*test functor async handler*/
    {
      async_handler_functor asyncHandlerFunctor;

      sycl::queue q(cts_selector, asyncHandlerFunctor);

      q.submit([&](sycl::handler &cgh) {
        cgh.single_task<class TEST_NAME_2>([=]() {});
      });

      // Should not throw exceptions
      q.wait();

      check_exceptions(log, asyncHandlerFunctor.excps);
    }

    /*test function async handler*/
    {
      sycl::queue q(cts_selector, async_handler_function);

      q.submit([&](sycl::handler &cgh) {
        cgh.single_task<class TEST_NAME_3>([=]() {});
      });

      // Should not throw exceptions
      q.wait();
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
