/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#pragma once

#include "test_base.h"

namespace sycl_cts {
namespace util {

// test harness function to register a given test
// defined in collection.cpp
extern void register_test(test_base *test);

/** test proxy class
 *  this class is used to register tests with the test harness at compile time.
 */
template <typename T>
class test_proxy {
 public:
  /** test_proxy constructor
   */
  test_proxy() {
    // use an externed function to cut dependency on the collection
    register_test(new T());
  }
};

}  // namespace util
}  // namespace sycl_cts
