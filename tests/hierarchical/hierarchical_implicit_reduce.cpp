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
static constexpr int globalItemsTotal =
    (globalItems1d * globalItems2d * globalItems3d);
static constexpr int localItemsTotal =
    (localItems1d * localItems2d * localItems3d);
static constexpr int numGroups = (globalItemsTotal / localItemsTotal);

static constexpr int inputSize = 59;

using namespace sycl_cts;

template <typename T, int dim> class sth {};

template <typename T, int dim> class sth_else {};

template <typename T, int dim, class DeviceSelector>
T reduce(T input[inputSize], DeviceSelector* selector) {
  T mGroupSums[numGroups];

  auto myQueue = util::get_cts_object::queue(*selector);
  sycl::buffer<T, 1> input_buf(input, sycl::range<1>(inputSize));
  sycl::buffer<T, 1> group_sums_buf(mGroupSums,
                                        sycl::range<1>(numGroups));

  myQueue.submit([&](sycl::handler& cgh) {
    sycl::accessor<T, 1, sycl::access_mode::read,
                       sycl::target::device>
        input_ptr(input_buf, cgh);
    sycl::accessor<T, 1, sycl::access_mode::write,
                       sycl::target::device>
        groupSumsPtr(group_sums_buf, cgh);
    auto groupRange = sycl_cts::util::get_cts_object::range<
        dim>::template get_fixed_size<numGroups>(groupItems1d, groupItems2d);
    auto localRange =
        sycl_cts::util::get_cts_object::range<dim>::template get_fixed_size<
            localItemsTotal>(localItems1d, localItems2d);
        cgh.parallel_for_work_group<class sth<T, dim>>(
                    groupRange, localRange,
                    [=]( sycl::group<dim> group )
        {
          T localSums[localItemsTotal];

          // process items in each work item
          group.parallel_for_work_item(
              [=, &localSums](sycl::h_item<dim> item) {
                int globalId = item.get_global().get_linear_id();
                int localId = item.get_local().get_linear_id();
                /* Split the array into work-group-size different arrays */
                int valuesPerItem = (inputSize / globalItemsTotal);
                valuesPerItem = (valuesPerItem == 0) ? 1 : valuesPerItem;
                int idStart = valuesPerItem * globalId;
                int idEnd = valuesPerItem * (globalId + 1);

                /* Handle the case where the number of input values is not
                 * divisible
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
    sycl::buffer<T, 1> total_buf(&mTotal, sycl::range<1>(1));
    myQueue.submit([&](sycl::handler& cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read,
                         sycl::target::device>
          groupSumsPtr(group_sums_buf, cgh);
      sycl::accessor<T, 1, sycl::access_mode::write,
                         sycl::target::device>
          totalPtr(total_buf, cgh);

        cgh.single_task<class sth_else<T, dim> >([=]() {
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

template <int dim> void check_dim(util::logger &log) {
  {
      cts_selector sel;
      {
        Adder data[inputSize];
        for (int i = 0; i < inputSize; i++) data[i] = Adder(2);

        Adder result = reduce<Adder, dim>(data, &sel);

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

        Multiplier result = reduce<Multiplier, dim>(data, &sel);

        auto expectedResult = std::int64_t{1} << inputSize;

        if (result.value != expectedResult) {
          const auto msg = "Incorrect result in Multiplier: " +
                           std::to_string(result.value) +
                           " != " + std::to_string(expectedResult);
          FAIL(log, msg);
        }
      }
    }
}

/** test sycl::range::get(int index) return size_t
 */
class TEST_NAME : public util::test_base {
public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    check_dim<1>(log);
    check_dim<2>(log);
    check_dim<3>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
