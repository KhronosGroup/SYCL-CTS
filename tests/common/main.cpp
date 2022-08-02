/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
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

#include <stdio.h>
#include <stdlib.h>

#include "./../../util/test_manager.h"
#include "./../../util/collection.h"

using namespace sycl_cts::util;

/** program exit callback
 */
void exit_handler() {
  // inform the test manager of program exit
  get<test_manager>().on_exit();
}

/** test suite entry point
 */
int main(int argc, const char **args) {
  // register an exit handler
  atexit(exit_handler);

  // prepare the test collection for use
  get<collection>().prepare();

  // get a handle to the test manager instance
  test_manager &testManager = get<test_manager>();

  // inform the test manager the cts has launched
  testManager.on_start();

  // parse the command line
  if (!testManager.parse(argc, args)) {
    return -1;
  }

  // Dump device info
  testManager.dump_device_info();

  // if the test harness will execute
  if (testManager.will_execute()) {
    // run all of the specified tests
    if (!testManager.run()) {
      return -1;
    }
  }

  return 0;
}
