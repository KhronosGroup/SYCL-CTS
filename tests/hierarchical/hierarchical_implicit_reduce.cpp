/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_reduce

namespace TEST_NAMESPACE {

static const int groupItems1d = 2;
static const int localItems1d = 2;
static const int groupItemsTotal = groupItems1d * groupItems1d * groupItems1d;
static const int localItemsTotal = localItems1d * localItems1d * localItems1d;
static const int numGroups = groupItemsTotal / localItemsTotal;

static const int inputSize = 32;

using namespace sycl_cts;

template <typename T>
class sth {};

template <typename T>
class sth_else {};

template <typename T>
T reduce(T input[inputSize], cl::sycl::device_selector *selector) {
  T mTotal;
  T mGroupSums[numGroups];

  auto myQueue = util::get_cts_object::queue(*selector);
  cl::sycl::buffer<T, 1> input_buf(input, cl::sycl::range<1>(inputSize));
  cl::sycl::buffer<T, 1> group_sums_buf(mGroupSums,
                                        cl::sycl::range<1>(numGroups));
  cl::sycl::buffer<T, 1> total_buf(&mTotal, cl::sycl::range<1>(1));

  myQueue.submit([&](cl::sycl::handler &cgh) {
    cl::sycl::accessor<T, 1, cl::sycl::access::mode::read,
                       cl::sycl::access::target::global_buffer>
        input_ptr(input_buf, cgh);
    cl::sycl::accessor<T, 1, cl::sycl::access::mode::read,
                       cl::sycl::access::target::global_buffer>
        groupSumsPtr(group_sums_buf, cgh);
    cl::sycl::accessor<T, 1, cl::sycl::access::mode::write,
                       cl::sycl::access::target::global_buffer>
        totalPtr(total_buf, cgh);
        cgh.parallel_for_work_group<class sth<T>>(
                    cl::sycl::range<3>( groupItems1d, groupItems1d, groupItems1d ),
                    cl::sycl::range<3>( localItems1d, localItems1d, localItems1d ),
                    [=]( cl::sycl::group<3> group )
        {
          T localSums[localItemsTotal];

          // process items in each work item
          group.parallel_for_work_item([=,
                                        &localSums](cl::sycl::h_item<3> item) {
            int localId = item.get_local().get_linear_id();
            /* Split the array into work-group-size different arrays */
            int valuesPerItem = (inputSize / numGroups) / localItemsTotal;
            int idStart = 0;
            int idEnd = valuesPerItem * localId;

            /* Handle the case where the number of input values is not divisible
            * by
            * the number of items. */
            if (idEnd > inputSize - 1) {
              idEnd = inputSize - 1;
            }

            for (int i = idStart; i < idEnd; i++) {
              localSums[i].increment(input_ptr[i]);
            }
          });

          /* Sum items in each work group */
          for (int i = 0; i < localItemsTotal; i++) {
            groupSumsPtr[group.get_id(0)].increment(localSums[i]);
          }
        });
  });

  myQueue.submit([&](cl::sycl::handler &cgh) {
    cl::sycl::accessor<T, 1, cl::sycl::access::mode::read,
                       cl::sycl::access::target::global_buffer>
        groupSumsPtr(group_sums_buf, cgh);
    cl::sycl::accessor<T, 1, cl::sycl::access::mode::write,
                       cl::sycl::access::target::global_buffer>
        totalPtr(total_buf, cgh);

        cgh.single_task<class sth_else<T>>([=]()
        {
          /* Sum items in all work groups */
          for (int i = 0; i < numGroups; i++) {
            totalPtr[0].value = totalPtr[i].value + groupSumsPtr[i].value;
          }
        });
  });

  myQueue.wait_and_throw();

  return mTotal;
}

class Adder {
 public:
  Adder() { value = 0; }
  Adder(int val) { value = val; }

  static Adder default_value() { return Adder(0); }

  Adder increment(Adder rhs) {
    value += rhs.value;
    return *this;
  }

  int value;
};

class Multiplier {
 public:
  Multiplier() { value = 1; }
  Multiplier(int val) { value = val; }

  static Multiplier default_value() { return Multiplier(0); }

  Multiplier increment(Multiplier rhs) {
    value *= rhs.value;
    return *this;
  }

  int value;
};

/** test cl::sycl::range::get(int index) return size_t
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
    return;
    try {
      cts_selector sel;
      {
        Adder data[inputSize];
        for (int i = 0; i < inputSize; i++) data[i] = Adder(2);

        Adder result = reduce<Adder>(data, &sel);

        int expectedResult = inputSize * 2;

        if (result.value != expectedResult) {
          FAIL(log, "Incorrect result in Adder");
        }
      }

      {
        Multiplier data[inputSize];
        for (int i = 0; i < inputSize; i++) data[i] = Multiplier(2);

        Multiplier result = reduce<Multiplier>(data, &sel);

        int expectedResult = 1;
        for (int i = 0; i < inputSize; ++i) expectedResult *= 2;

        if (result.value != expectedResult) {
          FAIL(log, "Incorrect result in Multiplier");
        }
      }
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace hierarchical_reduce__ */
