/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

#define SYCL_SIMPLE_SWIZZLES

#include "../common/type_coverage.h"
#include "../stream/stream_api_common.h"

#define TEST_NAME stream_api_core

namespace TEST_NAMESPACE {

using namespace sycl_cts;

class test_kernel_0;
class test_kernel_1;

template <int dims>
class test_kernel_2;

template <int dims>
class test_kernel_3;

template <int dims>
class test_kernel_4;

template <typename multi_ptr_t>
class test_kernel_ptr;

/**
 * Function that creates a sycl::stream object and streams nd_item.
 */
template <int dims>
void check_nd_item_dims(sycl::range<dims>& range1, sycl::range<dims>& range2) {
  auto testQueue = util::get_cts_object::queue();
  testQueue.submit([&](sycl::handler& cgh) {
    sycl::stream os(2048, 80, cgh);

    cgh.parallel_for<class test_kernel_2<dims>>(
        sycl::nd_range<dims>(range1, range2), [=](sycl::nd_item<dims> ndItem) {
          /** check stream operator for nd_item
           */
          check_type(os, ndItem);

          // check stream operator for nd_range
          check_type(os, ndItem.get_nd_range());
        });
  });

  testQueue.wait_and_throw();
}

/**
 * Function that creates a sycl::stream object and streams item.
 */
template <int dims>
void check_item_dims(sycl::range<dims>& range) {
  auto testQueue = util::get_cts_object::queue();
  testQueue.submit([&](sycl::handler& cgh) {
    sycl::stream os(2048, 80, cgh);

    cgh.parallel_for<class test_kernel_3<dims>>(range,
                                                [=](sycl::item<dims> it) {
                                                  /** check stream operator for
                                                   * item
                                                   */
                                                  check_type(os, it);
                                                });
  });

  testQueue.wait_and_throw();
}

/**
 * Function that creates a sycl::stream object and streams group and h_item.
 */
template <int dims>
void check_group_h_item_dims(sycl::range<dims>& range1,
                             sycl::range<dims>& range2) {
  auto testQueue = util::get_cts_object::queue();
  testQueue.submit([&](sycl::handler& cgh) {
    sycl::stream os(2048, 80, cgh);

    cgh.parallel_for_work_group<class test_kernel_4<dims>>(
        range1, range2, [=](const sycl::group<dims> gp) {
          /** check stream operator for sycl::group
           */
          check_type(os, gp);

          gp.parallel_for_work_item([&](sycl::h_item<dims> hit) {
            /** check stream operator for sycl::h_item
             */
            check_type(os, hit);
          });
        });
  });

  testQueue.wait_and_throw();
}

/**
 * Functor that creates a sycl::stream object and streams pointers.
 */
template <typename multi_ptr_t>
class check_multi_ptr {
  static constexpr sycl::access::address_space space =
      multi_ptr_t::address_space;
  static constexpr sycl::access::decorated decorated =
      multi_ptr_t::is_decorated ? sycl::access::decorated::yes
                                : sycl::access::decorated::no;

 public:
  void operator()() {
    int value = 42;
    auto testQueue = util::get_cts_object::queue();
    {
      sycl::buffer<int> val_buffer(&value, sycl::range(1));
      testQueue.submit([&](sycl::handler& cgh) {
        sycl::stream os(2048, 80, cgh);

        using kernel_name = test_kernel_ptr<multi_ptr_t>;
        if constexpr (space == sycl::access::address_space::local_space) {
          sycl::local_accessor<int> acc_for_multi_ptr{sycl::range(1), cgh};
          cgh.parallel_for<kernel_name>(
              sycl::nd_range({1}, {1}), [=](sycl::nd_item<1> item) {
                value_operations::assign(acc_for_multi_ptr, value);
                sycl::group_barrier(item.get_group());
                const multi_ptr_t multi_ptr(acc_for_multi_ptr);
                check_type(os, multi_ptr);
              });
        } else if constexpr (space ==
                             sycl::access::address_space::private_space) {
          cgh.single_task<kernel_name>([=] {
            int priv_val = value;
            sycl::multi_ptr<int, sycl::access::address_space::private_space,
                            decorated>
                priv_val_mptr = sycl::address_space_cast<
                    sycl::access::address_space::private_space, decorated>(
                    &priv_val);
            check_type(os, priv_val_mptr);
          });
        } else {
          auto acc_for_multi_ptr =
              val_buffer.template get_access<sycl::access_mode::read_write>(
                  cgh);
          cgh.single_task<kernel_name>([=] {
            const multi_ptr_t multi_ptr(acc_for_multi_ptr);
            check_type(os, multi_ptr);
          });
        }
      });
      testQueue.wait_and_throw();
    }
  }
};

/** test sycl::stream interface
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
    {
      /** check sycl::stream_manipulator
       */
      check_enum_class_value(sycl::stream_manipulator::dec);
      check_enum_class_value(sycl::stream_manipulator::hex);
      check_enum_class_value(sycl::stream_manipulator::oct);
      check_enum_class_value(sycl::stream_manipulator::noshowbase);
      check_enum_class_value(sycl::stream_manipulator::showbase);
      check_enum_class_value(sycl::stream_manipulator::noshowpos);
      check_enum_class_value(sycl::stream_manipulator::showpos);
      check_enum_class_value(sycl::stream_manipulator::endl);
      check_enum_class_value(sycl::stream_manipulator::fixed);
      check_enum_class_value(sycl::stream_manipulator::scientific);
      check_enum_class_value(sycl::stream_manipulator::hexfloat);
      check_enum_class_value(sycl::stream_manipulator::defaultfloat);
      check_enum_class_value(sycl::stream_manipulator::flush);

      /** Check stream interface
       */
      {
        auto testQueue = util::get_cts_object::queue();
        testQueue.submit([&](sycl::handler& cgh) {
          sycl::stream os(2048, 80, cgh);

          /** check get_size()
           */
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
          {
            auto size = os.get_size();
            check_return_type<size_t>(log, size, "sycl::stream::get_size()");
          }
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS

          /** check size()
           */
          {
            auto size = os.size();
            check_return_type<size_t>(log, size, "sycl::stream::size()");
            CHECK(noexcept(os.size()));
          }

          /** get_max_statement_size()
           */
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
          {
            auto maxStatementSize = os.get_max_statement_size();
            check_return_type<size_t>(log, maxStatementSize,
                                      "sycl::stream::get_max_statement_size()");
          }
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS

          /** check get_work_item_buffer_size()
           */
          {
            auto workItemBufferSize = os.get_work_item_buffer_size();
            check_return_type<size_t>(
                log, workItemBufferSize,
                "sycl::stream::get_work_item_buffer_size()");
          }

          cgh.single_task<class test_kernel_0>([=]() {});
        });

        testQueue.wait_and_throw();
      }

      /** check stream operator for supported types
       */
      {
        auto testQueue = util::get_cts_object::queue();
        testQueue.submit([&](sycl::handler& cgh) {
          sycl::stream os(2048, 80, cgh);

          cgh.single_task<class test_kernel_1>([=]() {
            /** check stream operator for basic types
             */
            check_type(os, "hello world!");
            check_type(os, const_cast<char*>("hello world!"));
            check_all_vec_dims(os, char('c'));
            check_all_vec_dims(os, static_cast<signed char>('c'));
            check_all_vec_dims(os, static_cast<unsigned char>('c'));
            check_all_vec_dims(os, int(5));
            check_all_vec_dims(os, static_cast<unsigned int>(5));
            check_all_vec_dims(os, short(5));
            check_all_vec_dims(os, static_cast<unsigned short>(5));
            check_all_vec_dims(os, long(5));
            check_all_vec_dims(os, static_cast<unsigned long>(5));
            check_all_vec_dims(os, static_cast<long long>(5));
            check_all_vec_dims(os, static_cast<unsigned long long>(5));
            check_all_vec_dims(os, float(5.5f));
            check_type(os, true);
            check_type(os, size_t(5));

            // check stream operator for pointers
            int a = 5;
            int* aPtr = &a;
            check_type(os, aPtr);
            const int* aConstPtr = &a;
            check_type(os, aConstPtr);

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
            // multi_ptr decorated::legacy
            check_type(os, sycl::global_ptr<int>{});
            check_type(os, sycl::private_ptr<int>{});
            check_type(os, sycl::constant_ptr<int>{});
            check_type(os, sycl::local_ptr<int>{});
#endif

            /** check stream operator for sycl types
             */
            check_type(os, sycl::id<1>(1));
            check_type(os, sycl::id<2>(1, 2));
            check_type(os, sycl::id<3>(1, 2, 3));

            check_type(os, sycl::range<1>(1));
            check_type(os, sycl::range<2>(1, 2));
            check_type(os, sycl::range<3>(1, 2, 3));

            check_type(os,
                       sycl::nd_range<1>(sycl::range<1>(2), sycl::range<1>(1)));
            check_type(os, sycl::nd_range<2>(sycl::range<2>(2, 4),
                                             sycl::range<2>(1, 2)));
            check_type(os, sycl::nd_range<3>(sycl::range<3>(2, 4, 1),
                                             sycl::range<3>(1, 2, 1)));

            /** check stream operator for manipulators
             */
            os << sycl::endl;
            os << sycl::setprecision(5) << float(5.0f);
            os << sycl::setw(3) << float(5.0f);
            os << sycl::hex << float(5.0f);
            os << sycl::oct << float(5.0f);
            os << sycl::dec << float(5.0f);
            os << sycl::showbase << int(5);
            os << sycl::noshowbase << int(5);
            os << sycl::showpos << int(-5) << int(5);
            os << sycl::noshowpos << int(-5) << int(5);
            os << sycl::fixed << float(5.0f);
            os << sycl::scientific << float(5.0f);
            os << sycl::hexfloat << float(5.0f);
            os << sycl::defaultfloat << float(5.0f);
            os << sycl::flush;
          });
        });

        testQueue.wait_and_throw();
      }

      /** check stream operator for sycl::nd_item
       */
      {
        sycl::range<1> r11(2);
        sycl::range<1> r12(1);
        check_nd_item_dims(r11, r12);

        sycl::range<2> r21(2, 4);
        sycl::range<2> r22(1, 2);
        check_nd_item_dims(r21, r22);

        sycl::range<3> r31(2, 4, 1);
        sycl::range<3> r32(1, 2, 1);
        check_nd_item_dims(r31, r32);
      }

      /** check stream operator for sycl::item
       */
      {
        sycl::range<1> r1(4);
        check_item_dims(r1);

        sycl::range<2> r2(4, 2);
        check_item_dims(r2);

        sycl::range<3> r3(4, 2, 1);
        check_item_dims(r3);
      }

      /** check stream operator for sycl::group and sycl::h_item
       */
      {
        sycl::range<1> r11(4);
        sycl::range<1> r12(1);
        check_group_h_item_dims(r11, r12);

        sycl::range<2> r21(4, 2);
        sycl::range<2> r22(1, 1);
        check_group_h_item_dims(r21, r22);

        sycl::range<3> r31(4, 2, 1);
        sycl::range<3> r32(1, 1, 1);
        check_group_h_item_dims(r31, r32);
      }

      // check stream operator for sycl::multi_ptr
      {
        for_all_types<check_multi_ptr>(
            type_pack<sycl::raw_global_ptr<int>, sycl::raw_private_ptr<int>,
                      sycl::raw_local_ptr<int>, sycl::decorated_global_ptr<int>,
                      sycl::decorated_private_ptr<int>,
                      sycl::decorated_local_ptr<int>>{});
      }
    }
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
