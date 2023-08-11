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
#include "../common/disabled_for_test_case.h"
#include "kernel_attributes.h"

using namespace kernel_attributes;

#define RUN_TEST(K_NAME1, K_NAME2, K_NAME3, VEC_T, FUNC1, FUNC2, FUNC3)       \
  {                                                                           \
    auto queue = sycl_cts::util::get_cts_object::queue();                     \
    VEC_T vec_st;                                                             \
    VEC_T vec;                                                                \
    VEC_T vec_wg;                                                             \
    {                                                                         \
      sycl::buffer<VEC_T, 1> buf_st(&vec_st, range);                          \
      sycl::buffer<VEC_T, 1> buf(&vec, range);                                \
      sycl::buffer<VEC_T, 1> buf_wg(&vec_wg, range);                          \
                                                                              \
      queue.submit([&](sycl::handler& cgh) {                                  \
        auto acc = buf_st.template get_access<sycl::access_mode::write>(cgh); \
        cgh.single_task<K_NAME1>(FUNC1(acc));                                 \
      });                                                                     \
      queue.submit([&](sycl::handler& cgh) {                                  \
        auto acc = buf.template get_access<sycl::access_mode::write>(cgh);    \
        cgh.parallel_for<K_NAME2>(sycl::nd_range{range, range}, FUNC2(acc));  \
      });                                                                     \
      queue.submit([&](sycl::handler& cgh) {                                  \
        auto acc = buf_wg.template get_access<sycl::access_mode::write>(cgh); \
        cgh.parallel_for_work_group<K_NAME3>(range, range, FUNC3(acc));       \
      });                                                                     \
      queue.wait_and_throw();                                                 \
    }                                                                         \
    verify(vec_st);                                                           \
    verify(vec);                                                              \
    verify(vec_wg);                                                           \
  }

template <class vec_t>
class functor {
 public:
  using vector_t = vec_t;
  using acc_t =
      sycl::accessor<vec_t, 1, sycl::access_mode::write, sycl::target::device>;

  functor(acc_t _acc) : acc(_acc) {}
  [[sycl::vec_type_hint(vec_t)]] void operator()() const {
    acc[sycl::id<1>()] = expected_val<1>();
  }

  [[sycl::vec_type_hint(vec_t)]] void operator()(
      sycl::nd_item<1> nd_item) const {
    acc[nd_item.get_local_id()] = expected_val<1>();
  }

  [[sycl::vec_type_hint(vec_t)]] void operator()(sycl::group<1> group) const {
    acc[group.get_group_id()] = expected_val<1>();
  }

 private:
  acc_t acc;
};

template <typename vec_t>
void verify(vec_t& vec) {
  const auto size = vec_t::size();
  bool res = true;
  for (int i = 0; i < size; i++) {
    res &= vec[i] == expected_val<1>();
  }
  INFO(
      "Check that kernel is executed without any "
      "exception and have expected result using sycl::vec<"
      << typeid(typename vec_t::element_type).name() << ", " << size << ">");
  CHECK(res);
}

template <class functor>
void check_functor() {
  using k_name1 = kernel_functor_st<functor>;
  using k_name2 = kernel_functor<functor>;
  using k_name3 = kernel_functor_wg<functor>;

  RUN_TEST(k_name1, k_name2, k_name3, typename functor::vector_t, functor,
           functor, functor);
}

template <class vec_t, class acc_t>
const auto get_lambda_st(acc_t& acc) {
  return [=]() [[sycl::vec_type_hint(vec_t)]] {
    acc[sycl::id<1>()] = expected_val<1>();
  };
}

template <class vec_t, class acc_t>
const auto get_lambda(acc_t& acc) {
  return [=](auto nd_item) [[sycl::vec_type_hint(vec_t)]] {
    acc[nd_item.get_local_id()] = expected_val<1>();
  };
}

template <class vec_t, class acc_t>
const auto get_lambda_wg(acc_t& acc) {
  return [=](auto group) [[sycl::vec_type_hint(vec_t)]] {
    acc[group.get_group_id()] = expected_val<1>();
  };
}

template <typename T, int N>
void check_separate_lambda() {
  using vec_t = sycl::vec<T, N>;
  using k_name1 = kernel_separate_lambda_st<vec_t>;
  using k_name2 = kernel_separate_lambda<0, N, T>;
  using k_name3 = kernel_separate_lambda_wg<0, N, T>;

  RUN_TEST(k_name1, k_name2, k_name3, vec_t, get_lambda_st<vec_t>,
           get_lambda<vec_t>, get_lambda_wg<vec_t>);
}

template <typename T, int N>
void check_lambda() {
  using vec_t = sycl::vec<T, N>;

  auto queue = sycl_cts::util::get_cts_object::queue();
  vec_t vec_st;
  vec_t vec;
  vec_t vec_wg;
  {
    sycl::buffer<vec_t, 1> buf_st(&vec_st, range);
    sycl::buffer<vec_t, 1> buf(&vec, range);
    sycl::buffer<vec_t, 1> buf_wg(&vec_wg, range);

    queue.submit([&](sycl::handler& cgh) {
      auto acc = buf_st.template get_access<sycl::access_mode::write>(cgh);
      cgh.single_task<kernel_lambda_st<vec_t>>(
          [=] { acc[sycl::id<1>()] = expected_val<1>(); });
    });
    queue.submit([&](sycl::handler& cgh) {
      auto acc = buf.template get_access<sycl::access_mode::write>(cgh);
      cgh.parallel_for<kernel_lambda<0, N, T>>(
          sycl::nd_range{range, range}, [=](auto nd_item) {
            acc[nd_item.get_local_id()] = expected_val<1>();
          });
    });
    queue.submit([&](sycl::handler& cgh) {
      auto acc = buf_wg.template get_access<sycl::access_mode::write>(cgh);
      cgh.parallel_for_work_group<kernel_lambda_wg<0, N, T>>(
          range, range,
          [=](auto group) { acc[group.get_group_id()] = expected_val<1>(); });
    });
    queue.wait_and_throw();
  }
  verify(vec_st);
  verify(vec);
  verify(vec_wg);
}

template <typename T, int N>
void run_tests_for_size() {
  using vec_t = sycl::vec<T, N>;

  check_functor<functor<vec_t>>();
  check_separate_lambda<T, N>();
  check_lambda<T, N>();
}

template <typename T>
void run_tests_for_type() {
  run_tests_for_size<T, 1>();
  run_tests_for_size<T, 2>();
  run_tests_for_size<T, 3>();
  run_tests_for_size<T, 4>();
  run_tests_for_size<T, 8>();
  run_tests_for_size<T, 16>();
}

DISABLED_FOR_TEST_CASE(hipSYCL)
("Behavior of kernel attribute vec_type_hint", "[kernel]")({
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  run_tests_for_type<int>();
  run_tests_for_type<float>();

#else
  SKIP("Tests for deprecated features are disabled.");
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
})
