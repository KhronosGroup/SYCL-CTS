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
//  Provides tests for kernel parameters
//
*******************************************************************************/
#include "../common/common.h"
#include "../common/type_coverage.h"

namespace kernels_parameters {
constexpr size_t array_size = 5;

constexpr int expected_val = 1;
constexpr int changed_val = 42;

template <typename PrimaryType>
class Base {
 public:
  PrimaryType member;
  Base() = default;
  Base(int value) { member = value_operations::init<PrimaryType>(value); }
  bool operator==(const Base& rhs) const { return member == rhs.member; }
};

template <typename PrimaryType>
class Derived : public Base<PrimaryType> {
 public:
  Derived() = default;
  Derived(int value) : Base<PrimaryType>(value) {}
};

inline auto get_full_primary_type_pack() {
  static const auto types = named_type_pack<
      int, float, bool, std::array<int, 5>, std::array<float, 5>,
      std::array<bool, 5>, std::optional<int>, std::optional<float>,
      std::optional<bool>, std::pair<int, float>, std::tuple<int, float, bool>,
      std::variant<int, float, bool>>::generate("int", "float", "bool",
                                                "std::array<int,5>",
                                                "std::array<float,5>",
                                                "std::array<bool,5>",
                                                "std::optional<int>",
                                                "std::optional<float>",
                                                "std::optional<bool>",
                                                "std::pair<int,float>",
                                                "std::tuple<int,float,bool>",
                                                "std::variant<int,float,bool>");
  return types;
}

inline auto get_lightweight_primary_type_pack() {
  static const auto types = named_type_pack<
      int, float, bool, std::array<int, 5>, std::array<float, 5>,
      std::array<bool, 5>, std::optional<int>, std::optional<float>,
      std::optional<bool>>::generate("int", "float", "bool",
                                     "std::array<int,5>", "std::array<float,5>",
                                     "std::array<bool,5>", "std::optional<int>",
                                     "std::optional<float>",
                                     "std::optional<bool>");
  return types;
}

/**
 * @brief Factory function for getting type_pack with types that depends on full
 *        conformance mode enabling status
 * @return named_type_pack
 */
inline auto get_primary_type_pack() {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return get_full_primary_type_pack();
#else
  return get_lightweight_primary_type_pack();
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
}

template <typename PrimaryType>
inline auto get_derived_type_pack(const std::string& primary_type_name) {
  if constexpr (std::is_arithmetic_v<PrimaryType>) {
    static const auto types = named_type_pack<
        sycl::marray<PrimaryType, 5>, sycl::vec<PrimaryType, 4>,
        PrimaryType[array_size], Base<PrimaryType>,
        Derived<PrimaryType>>::generate("sycl::marray<" + primary_type_name +
                                            ", 5>",
                                        "sycl::vec<" + primary_type_name +
                                            ", 4>",
                                        primary_type_name + "[array_size]",
                                        "Base<" + primary_type_name + ">",
                                        "Derived<" + primary_type_name + ">");
    return types;
  } else {
    static const auto types =
        named_type_pack<PrimaryType[array_size], Base<PrimaryType>,
                        Derived<PrimaryType>>::generate(primary_type_name +
                                                            "[array_size]",
                                                        "Base<" +
                                                            primary_type_name +
                                                            ">",
                                                        "Derived<" +
                                                            primary_type_name +
                                                            ">");
    return types;
  }
}

inline auto get_ranges_type_pack() {
  static const auto types =
      named_type_pack<sycl::range<1>, sycl::range<2>, sycl::range<3>>::generate(
          "sycl::range<1>", "sycl::range<2>", "sycl::range<3>");
  return types;
}

inline auto get_ids_type_pack() {
  static const auto types =
      named_type_pack<sycl::id<1>, sycl::id<2>, sycl::id<3>>::generate(
          "sycl::id<1>", "sycl::id<2>", "sycl::id<3>");
  return types;
}

template <typename T>
void init_data(T& data, int value) {
  if constexpr (std::is_array_v<T>) {
    std::remove_reference_t<decltype(data[0])> tmp;
    for (size_t i = 0; i < array_size; ++i) {
      data[i] = value_operations::init<decltype(tmp)>(value);
    }
  } else {
    data = value_operations::init<T>(value);
  }
}

template <typename AccType, typename KerParDerivedType>
class named_kernel {
  KerParDerivedType ker_par;
  AccType buf_acc;

 public:
  named_kernel(AccType buf_acc) {
    this->buf_acc = buf_acc;
    init_data(ker_par, changed_val);
  }
  void operator()() const {
    if constexpr (!std::is_array_v<KerParDerivedType>)
      buf_acc[0] = ker_par;
    else {
      for (size_t i = 0; i < array_size; ++i) buf_acc[i] = ker_par[i];
    }
  }
};

template <typename T>
class named_kernel_test {
  template <typename BufType>
  void queue_submit_task(BufType buf_expected) {
    queue
        .submit([&](sycl::handler& cgh) {
          auto acc_expected =
              buf_expected.template get_access<sycl::access_mode::read_write>(
                  cgh);
          named_kernel<decltype(acc_expected), T> kernel(acc_expected);
          cgh.single_task(kernel);
        })
        .wait_and_throw();
  }
  sycl::queue queue;

 public:
  named_kernel_test() { queue = sycl_cts::util::get_cts_object::queue(); }
  void operator()(const std::string& type_name) {
    T expected;
    init_data(expected, expected_val);
    T changed;
    init_data(changed, changed_val);
    {
      if constexpr (!std::is_array_v<T>) {
        sycl::buffer<T, 1> buf_expected(&expected, sycl::range<1>(1));
        queue_submit_task(buf_expected);
      } else {
        sycl::buffer<std::remove_reference_t<decltype(expected[0])>, 1>
            buf_expected(expected, sycl::range<1>(array_size));
        queue_submit_task(buf_expected);
      }
    }
    CHECK(value_operations::are_equal(expected, changed));
  }
};

template <typename PrimaryType>
class run_ker_par_test_named {
 public:
  void operator()(const std::string& primary_type_name) {
    const auto derived_types =
        get_derived_type_pack<PrimaryType>(primary_type_name);
    for_all_types<named_kernel_test>(derived_types);
  }
};

template <typename AccType, int Dim>
class named_kernel<AccType, sycl::range<Dim>> {
  sycl::range<Dim> ker_par;
  AccType buf_acc;

 public:
  named_kernel(AccType buf_acc)
      : ker_par(sycl_cts::util::get_cts_object::range<Dim>::get(
            changed_val, changed_val, changed_val)) {
    this->buf_acc = buf_acc;
  }
  void operator()() const { buf_acc[0] = ker_par; }
};

template <int Dim>
class named_kernel_test<sycl::range<Dim>> {
  template <typename BufType>
  void queue_submit_task(BufType buf_expected) {
    queue
        .submit([&](sycl::handler& cgh) {
          auto acc_expected =
              buf_expected.template get_access<sycl::access_mode::read_write>(
                  cgh);
          named_kernel<decltype(acc_expected), sycl::range<Dim>> kernel(
              acc_expected);
          cgh.single_task(kernel);
        })
        .wait_and_throw();
  }
  sycl::queue queue;

 public:
  named_kernel_test() { queue = sycl_cts::util::get_cts_object::queue(); }
  void operator()(const std::string& type_name) {
    auto expected = sycl_cts::util::get_cts_object::range<Dim>::get(
        expected_val, expected_val, expected_val);
    auto changed = sycl_cts::util::get_cts_object::range<Dim>::get(
        changed_val, changed_val, changed_val);
    {
      sycl::buffer<sycl::range<Dim>, 1> buf_expected(&expected,
                                                     sycl::range<1>(1));
      queue_submit_task(buf_expected);
    }
    CHECK(expected == changed);
  }
};

template <typename AccType, int Dim>
class named_kernel<AccType, sycl::id<Dim>> {
  sycl::id<Dim> ker_par;
  AccType buf_acc;

 public:
  named_kernel(AccType buf_acc)
      : ker_par(sycl_cts::util::get_cts_object::id<Dim>::get(
            changed_val, changed_val, changed_val)) {
    this->buf_acc = buf_acc;
  }
  void operator()() const { buf_acc[0] = ker_par; }
};

template <int Dim>
class named_kernel_test<sycl::id<Dim>> {
  template <typename BufType>
  void queue_submit_task(BufType buf_expected) {
    queue
        .submit([&](sycl::handler& cgh) {
          auto acc_expected =
              buf_expected.template get_access<sycl::access_mode::read_write>(
                  cgh);
          named_kernel<decltype(acc_expected), sycl::id<Dim>> kernel(
              acc_expected);
          cgh.single_task(kernel);
        })
        .wait_and_throw();
  }
  sycl::queue queue;

 public:
  named_kernel_test() { queue = sycl_cts::util::get_cts_object::queue(); }
  void operator()(const std::string& type_name) {
    auto expected = sycl_cts::util::get_cts_object::id<Dim>::get(
        expected_val, expected_val, expected_val);
    auto changed = sycl_cts::util::get_cts_object::id<Dim>::get(
        changed_val, changed_val, changed_val);
    {
      sycl::buffer<sycl::id<Dim>, 1> buf_expected(&expected, sycl::range<1>(1));
      queue_submit_task(buf_expected);
    }
    CHECK(expected == changed);
  }
};

template <typename T>
class unnamed_kernel_test {
  template <typename BufType>
  void queue_submit_task(BufType buf_expected, T& changed) {
    queue
        .submit([&](sycl::handler& cgh) {
          auto acc_expected =
              buf_expected.template get_access<sycl::access_mode::read_write>(
                  cgh);
          cgh.single_task([=]() {
            if constexpr (!std::is_array_v<T>)
              acc_expected[0] = changed;
            else {
              for (size_t i = 0; i < array_size; ++i) {
                acc_expected[i] = changed[i];
              }
            }
          });
        })
        .wait_and_throw();
  }
  sycl::queue queue;

 public:
  unnamed_kernel_test() { queue = sycl_cts::util::get_cts_object::queue(); }
  void operator()(const std::string& type_name) {
    T expected;
    init_data(expected, expected_val);
    T changed;
    init_data(changed, changed_val);
    {
      if constexpr (!std::is_array_v<T>) {
        sycl::buffer<T, 1> buf_expected(&expected, sycl::range<1>(1));
        queue_submit_task(buf_expected, changed);
      } else {
        sycl::buffer<std::remove_reference_t<decltype(expected[0])>, 1>
            buf_expected(expected, sycl::range<1>(array_size));
        queue_submit_task(buf_expected, changed);
      }
    }
    CHECK(value_operations::are_equal(expected, changed));
  }
};

template <int Dim>
class unnamed_kernel_test<sycl::range<Dim>> {
 public:
  void operator()(const std::string& type_name) {
    auto queue = sycl_cts::util::get_cts_object::queue();
    auto expected = sycl_cts::util::get_cts_object::range<Dim>::get(
        expected_val, expected_val, expected_val);
    auto changed = sycl_cts::util::get_cts_object::range<Dim>::get(
        changed_val, changed_val, changed_val);
    {
      sycl::buffer<sycl::range<Dim>, 1> buf_expected(&expected,
                                                     sycl::range<1>(1));
      queue
          .submit([&](sycl::handler& cgh) {
            auto acc_expected =
                buf_expected.template get_access<sycl::access_mode::read_write>(
                    cgh);
            cgh.single_task([=]() { acc_expected[0] = changed; });
          })
          .wait_and_throw();
    }
    CHECK(expected == changed);
  }
};

template <int Dim>
class unnamed_kernel_test<sycl::id<Dim>> {
 public:
  void operator()(const std::string& type_name) {
    auto queue = sycl_cts::util::get_cts_object::queue();
    auto expected = sycl_cts::util::get_cts_object::id<Dim>::get(
        expected_val, expected_val, expected_val);
    auto changed = sycl_cts::util::get_cts_object::id<Dim>::get(
        changed_val, changed_val, changed_val);
    {
      sycl::buffer<sycl::id<Dim>, 1> buf_expected(&expected, sycl::range<1>(1));
      queue
          .submit([&](sycl::handler& cgh) {
            auto acc_expected =
                buf_expected.template get_access<sycl::access_mode::read_write>(
                    cgh);
            cgh.single_task([=]() { acc_expected[0] = changed; });
          })
          .wait_and_throw();
    }
    CHECK(expected == changed);
  }
};

template <typename PrimaryType>
class run_ker_par_test_unnamed {
 public:
  void operator()(const std::string& primary_type_name) {
    const auto derived_types =
        get_derived_type_pack<PrimaryType>(primary_type_name);
    for_all_types<unnamed_kernel_test>(derived_types);
  }
};

TEST_CASE("Parameters passing to unnamed kernels", "[kernels]") {
  const auto types = get_primary_type_pack();
  for_all_types<run_ker_par_test_unnamed>(types);
}

TEST_CASE("sycl::range<N> passing to unnamed kernels", "[kernels]") {
  const auto range_types = get_ranges_type_pack();
  for_all_types<unnamed_kernel_test>(range_types);
}

TEST_CASE("sycl::id<N> passing to unnamed kernels", "[kernels]") {
  const auto id_types = get_ids_type_pack();
  for_all_types<unnamed_kernel_test>(id_types);
}

TEST_CASE("Parameters passing to named kernels", "[kernels]") {
  const auto types = get_primary_type_pack();
  for_all_types<run_ker_par_test_named>(types);
}

TEST_CASE("sycl::range<N> passing to named kernels", "[kernels]") {
  const auto range_types = get_ranges_type_pack();
  for_all_types<named_kernel_test>(range_types);
}

TEST_CASE("sycl::id<N> passing to named kernels", "[kernels]") {
  const auto id_types = get_ids_type_pack();
  for_all_types<named_kernel_test>(id_types);
}

}  // namespace kernels_parameters
