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

#include "../common/common.h"

#define TEST_NAME hierarchical_implicit_reduce

namespace TEST_NAMESPACE {

static constexpr int globalItems1d = 26;
static constexpr int globalItems2d = 33;
static constexpr int globalItems3d = 35;
static constexpr int localItems1d = 2;
static constexpr int localItems2d = 3;
static constexpr int localItems3d = 5;
static constexpr int groupItems1d = (globalItems1d / localItems1d);
static constexpr int groupItems2d = (globalItems2d / localItems2d);
static constexpr int groupItems3d = (globalItems3d / localItems3d);
static constexpr int globalItemsTotal =
    (globalItems1d * globalItems2d * globalItems3d);
static constexpr int localItemsTotal =
    (localItems1d * localItems2d * localItems3d);
static constexpr int numGroups = (globalItemsTotal / localItemsTotal);

static constexpr int inputSize = 59;

using namespace sycl_cts;

template <typename T>
class sth {};

template <typename T>
class sth_else {};

template <typename T>
T reduce(T input[inputSize], cl::sycl::device_selector* selector) {
  T mGroupSums[numGroups];

  auto myQueue = util::get_cts_object::queue(*selector);
  cl::sycl::buffer<T, 1> input_buf(input, cl::sycl::range<1>(inputSize));
  cl::sycl::buffer<T, 1> group_sums_buf(mGroupSums,
                                        cl::sycl::range<1>(numGroups));

  myQueue.submit([&](cl::sycl::handler& cgh) {
    cl::sycl::accessor<T, 1, cl::sycl::access::mode::read,
                       cl::sycl::access::target::global_buffer>
        input_ptr(input_buf, cgh);
    cl::sycl::accessor<T, 1, cl::sycl::access::mode::write,
                       cl::sycl::access::target::global_buffer>
        groupSumsPtr(group_sums_buf, cgh);
        cgh.parallel_for_work_group<class sth<T>>(
                    cl::sycl::range<3>( groupItems1d, groupItems2d, groupItems3d ),
                    cl::sycl::range<3>( localItems1d, localItems2d, localItems3d ),
                    [=]( cl::sycl::group<3> group )
        {
          T localSums[localItemsTotal];

          // process items in each work item
          group.parallel_for_work_item([=,
                                        &localSums](cl::sycl::h_item<3> item) {
            int globalId = item.get_global().get_linear_id();
            int localId = item.get_local().get_linear_id();
            /* Split the array into work-group-size different arrays */
            int valuesPerItem = (inputSize / globalItemsTotal);
            valuesPerItem = (valuesPerItem == 0) ? 1 : valuesPerItem;
            int idStart = valuesPerItem * globalId;
            int idEnd = valuesPerItem * (globalId + 1);

            /* Handle the case where the number of input values is not divisible
             * by the number of items. */
            if (idEnd > inputSize - 1) {
              idEnd = inputSize;
            }

            localSums[localId] = T{};
            for (int i = idStart; i < idEnd; i++) {
              localSums[localId].increment(input_ptr[i]);
            }
          });

          /* Sum items in each work group */
          int groupId = group.get_linear_id();
          groupSumsPtr[groupId] = T{};
          for (int i = 0; i < localItemsTotal; i++) {
            groupSumsPtr[groupId].increment(localSums[i]);
          }
      });
  });

  T mTotal;
  {
    cl::sycl::buffer<T, 1> total_buf(&mTotal, cl::sycl::range<1>(1));
    myQueue.submit([&](cl::sycl::handler& cgh) {
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>
          groupSumsPtr(group_sums_buf, cgh);
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>
          totalPtr(total_buf, cgh);

        cgh.single_task<class sth_else<T> >([=]() {
          /* Sum items in all work groups */
          totalPtr[0] = T{};
          for (int i = 0; i < numGroups; i++) {
            totalPtr[0].increment(groupSumsPtr[i]);
          }
        });
    });
  }

  return mTotal;
}

class Adder {
 public:
  using type = std::int64_t;

  constexpr Adder(type val = 0) : value{val} {}

  Adder& increment(const Adder& rhs) noexcept {
    value += rhs.value;
    return *this;
  }

  type value;
};

class Multiplier {
 public:
  using type = std::int64_t;

  constexpr Multiplier(type val = 1) : value{val} {}

  Multiplier& increment(const Multiplier& rhs) noexcept {
    value *= rhs.value;
    return *this;
  }

  type value;
};

/** test cl::sycl::range::get(int index) return size_t
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger& log) override {
    try {
      cts_selector sel;
      {
        Adder data[inputSize];
        for (int i = 0; i < inputSize; i++) data[i] = Adder(2);

        Adder result = reduce<Adder>(data, &sel);

        int expectedResult = inputSize * 2;

        if (result.value != expectedResult) {
          const auto msg =
              "Incorrect result in Adder: " + std::to_string(result.value) +
              " != " + std::to_string(expectedResult);
          FAIL(log, msg);
        }
      }

      {
        Multiplier data[inputSize];
        for (int i = 0; i < inputSize; i++) data[i] = Multiplier(2);

        Multiplier result = reduce<Multiplier>(data, &sel);

        auto expectedResult = std::int64_t{1} << inputSize;

        if (result.value != expectedResult) {
          const auto msg = "Incorrect result in Multiplier: " +
                           std::to_string(result.value) +
                           " != " + std::to_string(expectedResult);
          FAIL(log, msg);
        }
      }
    } catch (const cl::sycl::exception& e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
