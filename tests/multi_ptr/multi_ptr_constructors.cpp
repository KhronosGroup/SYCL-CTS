/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "../common/type_list.h"
#include "multi_ptr_constructors_common.h"

#include <string>

#define TEST_NAME multi_ptr_constructors

namespace TEST_NAMESPACE {
using namespace multi_ptr_constructors_common;
using namespace sycl_cts;

/** tests the constructors for explicit pointers
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
      auto queue = util::get_cts_object::queue();

      auto types = named_type_pack<bool, float, double, char,   // types grouped
                                   signed char, unsigned char,  // by sign
                                   short, unsigned short,       //
                                   int, unsigned int,           //
                                   long, unsigned long,         //
                                   long long, unsigned long long>{
          "bool",        "float",
          "double",      "char",
          "signed char", "unsigned char",
          "short",       "unsigned short",
          "int",         "unsigned int",
          "long",        "unsigned long",
          "long long",   "unsigned long long"};

      for_all_types<check_void_pointer_ctors>(types, queue);
      for_all_types<check_pointer_ctors>(types, queue);

      check_pointer_ctors<user_struct>{}(queue, "user_struct");

      queue.wait_and_throw();
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      auto errorMsg = std::string("a SYCL exception was caught: ") + e.what();
      FAIL(log, errorMsg);
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
