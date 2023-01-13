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

#define TEST_NAME stream_constructors

namespace TEST_NAMESPACE {

using namespace sycl_cts;

struct stream_kernel {
  void operator()() const {}
};

/** tests the constructors for sycl::stream
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** check equality of two stream objects. Returns true on equal, false
   * otherwise
   */
  bool areEqual(sycl::stream &osA, sycl::stream &osB) {
    if (osA.get_work_item_buffer_size() == osB.get_work_item_buffer_size() ||
        osA.get_precision() == osB.get_precision() ||
        osA.size() == osB.size() ||
        osA.get_stream_mode() == osB.get_stream_mode())
      return false;
    return true;
  }
  /** execute the test
   */
  void run(util::logger &log) override {
    {
      auto queue = util::get_cts_object::queue();
      size_t bufferSize = 2048;
      size_t workItemBufferSize = 80;

      /** check (size_t, size_t, sycl::handler&) constructor
       */
      {
        queue.submit([&](sycl::handler &handler) {
          sycl::stream os(bufferSize, workItemBufferSize, handler);

          if (os.size() != bufferSize) {
            FAIL(log, "sycl::context::size() returned an incorrect value.");
          }
          if (os.get_work_item_buffer_size() != workItemBufferSize) {
            FAIL(log,
                 "sycl::context::get_work_item_buffer_size() returned an  "
                 "incorrect value.");
          }

          handler.single_task(stream_kernel{});
        });
      }

      /** check (size_t, size_t, sycl::handler&, sycl::property_list&)
       * constructor
       */
      {
        queue.submit([&](sycl::handler &handler) {
          sycl::property_list property_list{};
          sycl::stream os(bufferSize, workItemBufferSize, handler,
                          property_list);

          if (os.size() != bufferSize) {
            FAIL(log, "sycl::context::size() returned an incorrect value.");
          }
          if (os.get_work_item_buffer_size() != workItemBufferSize) {
            FAIL(log,
                 "sycl::context::get_work_item_buffer_size() returned an  "
                 "incorrect value.");
          }

          handler.single_task(stream_kernel{});
        });
      }

      /** check copy constructor
       */
      {
        queue.submit([&](sycl::handler &handler) {
          sycl::stream osA(bufferSize, workItemBufferSize, handler);
          sycl::stream osB(osA);

          if (osA.get_work_item_buffer_size() !=
              osB.get_work_item_buffer_size()) {
            FAIL(log,
                 "stream is not copy constructed correctly. "
                 "(get_work_item_buffer_size)");
          }
          if (osA.get_precision() != osB.get_precision()) {
            FAIL(log,
                 "stream is not copy constructed correctly. (get_precision)");
          }
          if (osA.size() != osB.size()) {
            FAIL(log, "stream is not copy constructed correctly. (size)");
          }
          if (osA.get_stream_mode() != osB.get_stream_mode()) {
            FAIL(log,
                 "stream is not copy constructed correctly. (get_stream_mode)");
          }
          if (osB.size() != bufferSize) {
            FAIL(log,
                 "sycl::context::size() returned an incorrect value "
                 "after copy constructed.");
          }
          if (osB.get_work_item_buffer_size() != workItemBufferSize) {
            FAIL(log,
                 "sycl::context::get_work_item_buffer_size() returned an  "
                 "incorrect value after copy constructed.");
          }

          handler.single_task(stream_kernel{});
        });
      }

      /** check assignment operator
       */
      {
        queue.submit([&](sycl::handler &handler) {
          sycl::stream osA(bufferSize, workItemBufferSize, handler);
          sycl::stream osB(bufferSize / 2, workItemBufferSize / 2, handler);
          osB = osA;

          if (osA.get_work_item_buffer_size() !=
              osB.get_work_item_buffer_size()) {
            FAIL(log,
                 "stream is not copy constructed correctly. "
                 "(get_work_item_buffer_size)");
          }
          if (osA.get_precision() != osB.get_precision()) {
            FAIL(log,
                 "stream is not copy constructed correctly. (get_precision)");
          }
          if (osA.size() != osB.size()) {
            FAIL(log, "stream is not copy constructed correctly. (size)");
          }
          if (osA.get_stream_mode() != osB.get_stream_mode()) {
            FAIL(log,
                 "stream is not copy constructed correctly. (get_stream_mode)");
          }
          if (osB.size() != bufferSize) {
            FAIL(log,
                 "sycl::context::size() returned an incorrect value "
                 "after copy assigned.");
          }
          if (osB.get_work_item_buffer_size() != workItemBufferSize) {
            FAIL(log,
                 "sycl::context::get_work_item_buffer_size() returned an  "
                 "incorrect value after copy assigned.");
          }

          handler.single_task(stream_kernel{});
        });
      }

      /* check move constructor
       */
      {
        queue.submit([&](sycl::handler &handler) {
          sycl::stream osA(bufferSize, workItemBufferSize, handler);
          sycl::stream osB(std::move(osA));

          if (osB.size() != bufferSize) {
            FAIL(log,
                 "sycl::context::size() returned an incorrect value "
                 "after move constructed.");
          }
          if (osB.get_work_item_buffer_size() != workItemBufferSize) {
            FAIL(log,
                 "sycl::context::get_work_item_buffer_size() returned an  "
                 "incorrect value after move constructed.");
          }

          handler.single_task(stream_kernel{});
        });
      }

      /* check move assignment operator
       */
      {
        queue.submit([&](sycl::handler &handler) {
          sycl::stream osA(bufferSize, workItemBufferSize, handler);
          sycl::stream osB(bufferSize / 2, workItemBufferSize / 2, handler);
          osB = std::move(osA);

          if (osB.size() != bufferSize) {
            FAIL(log,
                 "sycl::context::size() returned an incorrect value "
                 "after move assigned.");
          }
          if (osB.get_work_item_buffer_size() != workItemBufferSize) {
            FAIL(log,
                 "sycl::context::get_work_item_buffer_size() returned an  "
                 "incorrect value after move assigned.");
          }

          handler.single_task(stream_kernel{});
        });
      }

      /** check equality operator
       */
      {
        queue.submit([&](sycl::handler &handler) {
          sycl::stream osA(bufferSize, workItemBufferSize, handler);
          sycl::stream osB(osA);
          sycl::stream osC(bufferSize * 2, workItemBufferSize * 2, handler);
          osC = osA;
          sycl::stream osD(bufferSize * 2, workItemBufferSize * 2, handler);

          if (!(osA == osB) && areEqual(osA, osB)) {
            FAIL(log,
                 "stream equality does not work correctly (copy constructed)");
          }

          if (!(osA == osC) && areEqual(osA, osC)) {
            FAIL(log,
                 "stream equality does not work correctly (copy assigned)");
          }
          if (osA != osB) {
            FAIL(log,
                 "stream non-equality does not work correctly"
                 "(copy constructed)");
          }
          if (osA != osC) {
            FAIL(log,
                 "stream non-equality does not work correctly"
                 "(copy assigned)");
          }
          if (osC == osD) {
            FAIL(log,
                 "stream equality does not work correctly"
                 "(comparing same)");
          }
          if (!(osC != osD)) {
            FAIL(log,
                 "stream non-equality does not work correctly"
                 "(comparing same)");
          }

          handler.single_task(stream_kernel{});
        });
      }

      /** check hashing
       */
      {
        queue.submit([&](sycl::handler &handler) {
          sycl::stream osA(bufferSize, workItemBufferSize, handler);
          sycl::stream osB = osA;

          std::hash<sycl::stream> hasher;

          if (hasher(osA) != hasher(osB)) {
            FAIL(log,
                 "stream hashing does not work correctly (hashing of equal "
                 "failed)");
          }
          handler.single_task(stream_kernel{});
        });
      }

      queue.wait_and_throw();
    }
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
