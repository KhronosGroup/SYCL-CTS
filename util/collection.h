/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
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

#ifndef __SYCLCTS_UTIL_COLLECTION_H
#define __SYCLCTS_UTIL_COLLECTION_H

#include "stl.h"
#include "test_base.h"
#include "singleton.h"

namespace sycl_cts {
namespace util {

/** this class is a central repository of tests
 */
class collection : public singleton<collection> {
 public:
  /** test structure
   */
  struct test_info {
    test_base *m_test;
    bool m_skip;
    int m_timeout;
  };

  /** constructor
   */
  collection();

  /** add a test to the collection
   *  @param test, the test to be added
   */
  void add_test(test_base *test);

  /** run all tests in the collection
   */
  void run_all();

  /** release all registered tests
   */
  void release();

  /** list all tests in the collection
   */
  void list();

  /** load a test filter (csv file)
   *  @param csvPath, the csv file path fir filtering the tests
   */
  bool filter_tests_csv(const std::string &csvPath);

  /** filter tests by name
   */
  bool filter_tests_name(const std::string &testName);

  /** get the total number of tests in this collection
   */
  int32_t get_test_count() const;

  /** return a specific test
   */
  test_info &get_test(int32_t index);

  /** prepare the list of tests for execution
   */
  void prepare();

 protected:
  /** set the skip status of a test by name
   */
  void set_test_skip(const std::string &testName, bool skip);

  // the test collection itself
  std::vector<test_info> m_tests;
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_COLLECTION_H