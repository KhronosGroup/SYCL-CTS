/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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
#include "../common/semantics_reference.h"
#include "event.h"

struct storage {
  std::size_t wait_list_size;

  explicit storage(const sycl::event& event)
      // get_wait_list is non-const
      : wait_list_size(sycl::event(event).get_wait_list().size()) {}

  bool check(const sycl::event& event) {
    sycl::event copy(event);
    return copy.get_wait_list().size() == wait_list_size;
  }
};

TEST_CASE("event common reference semantics", "[event]") {
  sycl::event event_0{};
  sycl::event event_1{};

  common_reference_semantics::check_host<storage>(event_0, event_1, "event");
}

TEST_CASE("event common reference semantics, mutation", "[event]") {
  resolvable_host_event dependent_event;
  resolvable_host_event rhe_t0{{dependent_event.get_sycl_event()}};

  sycl::event t1(rhe_t0.get_sycl_event());  // make copy of t0
  dependent_event.resolve();                // allow dependent event to clear
  dependent_event.get_sycl_event().wait();  // wait for dependent event to clear
  // Sanity check: require that when queried via t0, the dependency is resolved.
  // Note that it is implementation defined if completed events remain in the
  // wait list.
  const std::vector<sycl::event> wait_list_t0 =
      rhe_t0.get_sycl_event().get_wait_list();
  REQUIRE((wait_list_t0.empty() ||
           sycl::info::event_command_status::complete ==
               wait_list_t0[0]
                   .get_info<sycl::info::event::command_execution_status>()));
  // Perform the actual check.
  const std::vector<sycl::event> wait_list_t1 = t1.get_wait_list();
  CHECK((wait_list_t1.empty() ||
         sycl::info::event_command_status::complete ==
             wait_list_t1[0]
                 .get_info<sycl::info::event::command_execution_status>()));

  // Note: the mutation is not tied to an instance, rather through outside
  // influence. So only one check is needed here.
}
