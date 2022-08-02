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

#include "../common/common.h"
#include "../common/common_semantics.h"

#define TEST_NAME range_constructors

namespace range_constructors__ {
using namespace sycl_cts;

/** test sycl::range initialization
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   *  @param info, test_base::info structure as output
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   *  @param log, test transcript logging class
   */
  void run(util::logger &log) override {
    {
      // use across all the dimensions
      size_t sizes[] = {16, 8, 4};

      // construct from a range, explicit dimensions perform deep copy and a
      // move

      // dim 1
      {
        sycl::range<1> range_explicit(sizes[0]);
        if ((range_explicit[0] != sizes[0]) ||
            (range_explicit.get(0) != sizes[0])) {
          FAIL(log,
               "range with size_t was not constructed correctly for dim = 1");
        }

        sycl::range<1> range_deep(
            const_cast<const sycl::range<1> &>(range_explicit));
        if ((range_deep[0] != sizes[0]) || (range_deep.get(0) != sizes[0])) {
          FAIL(log,
               "range with range was not constructed correctly for dim = 1");
        }

        sycl::range<1> range_moved_constr(std::move(range_explicit));
        if ((range_moved_constr[0] != sizes[0]) ||
            (range_moved_constr.get(0) != sizes[0])) {
          FAIL(log,
               "range with range was not move constructed correctly for dim = "
               "1");
        }

        sycl::range<1> range_move_assign{0};
        range_move_assign = std::move(range_deep);
        if ((range_move_assign[0] != sizes[0]) ||
            (range_move_assign.get(0) != sizes[0])) {
          FAIL(log,
               "range with range was not move assigned correctly for dim = 1");
        }

        common_semantics::check_on_host(log, range_explicit,
                                        std::string("range"));
      }

      // dim 2
      {
        sycl::range<2> range_explicit(sizes[0], sizes[1]);
        if ((range_explicit[0] != sizes[0]) ||
            (range_explicit.get(0) != sizes[0]) ||
            (range_explicit[1] != sizes[1]) ||
            (range_explicit.get(1) != sizes[1])) {
          FAIL(log,
               "range with size_t was not constructed correctly for dim = 2");
        }

        sycl::range<2> range_deep(
            const_cast<const sycl::range<2> &>(range_explicit));
        if ((range_deep[0] != sizes[0]) || (range_deep.get(0) != sizes[0]) ||
            (range_deep[1] != sizes[1]) || (range_deep.get(1) != sizes[1])) {
          FAIL(log,
               "range with range was not constructed correctly for dim = 2");
        }

        sycl::range<2> range_moved_constr(std::move(range_explicit));
        if ((range_moved_constr[0] != sizes[0]) ||
            (range_moved_constr.get(0) != sizes[0]) ||
            (range_moved_constr[1] != sizes[1]) ||
            (range_moved_constr.get(1) != sizes[1])) {
          FAIL(log,
               "range with range was not move constructed correctly for dim = "
               "2");
        }

        sycl::range<2> range_move_assign{0, 0};
        range_move_assign = std::move(range_deep);
        if ((range_move_assign[0] != sizes[0]) ||
            (range_move_assign.get(0) != sizes[0]) ||
            (range_move_assign[1] != sizes[1]) ||
            (range_move_assign.get(1) != sizes[1])) {
          FAIL(log,
               "range with range was not move assigned correctly for dim = 2");
        }

        common_semantics::check_on_host(log, range_explicit,
                                        std::string("range"));
      }

      // dim 3
      {
        sycl::range<3> range_explicit(sizes[0], sizes[1], sizes[2]);
        if ((range_explicit[0] != sizes[0]) ||
            (range_explicit.get(0) != sizes[0]) ||
            (range_explicit[1] != sizes[1]) ||
            (range_explicit.get(1) != sizes[1]) ||
            (range_explicit[2] != sizes[2]) ||
            (range_explicit.get(2) != sizes[2])) {
          FAIL(log,
               "range with size_t was not constructed correctly for dim = 3");
        }

        sycl::range<3> range_deep(
            const_cast<const sycl::range<3> &>(range_explicit));
        if ((range_deep[0] != sizes[0]) || (range_deep.get(0) != sizes[0]) ||
            (range_deep[1] != sizes[1]) || (range_deep.get(1) != sizes[1]) ||
            (range_deep[2] != sizes[2]) || (range_deep.get(2) != sizes[2])) {
          FAIL(log,
               "range with range was not constructed correctly for dim = 3");
        }

        sycl::range<3> range_moved_constr(std::move(range_explicit));
        if ((range_moved_constr[0] != sizes[0]) ||
            (range_moved_constr.get(0) != sizes[0]) ||
            (range_moved_constr[1] != sizes[1]) ||
            (range_moved_constr.get(1) != sizes[1]) ||
            (range_moved_constr[2] != sizes[2]) ||
            (range_moved_constr.get(2) != sizes[2])) {
          FAIL(log,
               "range with range was not move constructed correctly for dim = "
               "3");
        }

        sycl::range<3> range_move_assign{0, 0, 0};
        range_move_assign = std::move(range_deep);
        if ((range_move_assign[0] != sizes[0]) ||
            (range_move_assign.get(0) != sizes[0]) ||
            (range_move_assign[1] != sizes[1]) ||
            (range_move_assign.get(1) != sizes[1]) ||
            (range_move_assign[2] != sizes[2]) ||
            (range_move_assign.get(2) != sizes[2])) {
          FAIL(log,
               "range with range was not move assigned correctly for dim = 3");
        }

        common_semantics::check_on_host(log, range_explicit,
                                        std::string("range"));
      }
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace range_constructors__ */
