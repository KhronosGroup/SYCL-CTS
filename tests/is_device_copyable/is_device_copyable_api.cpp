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
#include "../common/type_list.h"

class kernel_name;

/** @brief User defined class, must satisfy the
 *         requirements of device copyable type
 */
class user_def_type {
 public:
  user_def_type() = default;
  ~user_def_type() = default;

  user_def_type(int _a, float _b, char _c) : a(_a), b(_b), c(_c) {}
  user_def_type(const user_def_type& other)
      : a(other.a), b(other.b), c(other.c) {}

  user_def_type& operator=(const user_def_type& other) {
    a = other.a;
    b = other.b;
    c = other.c;
    return *this;
  }

  friend bool operator==(const user_def_type& lhs, const user_def_type& rhs) {
    return (lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c);
  }

  int a;
  float b;
  char c;
};

template <>
struct sycl::is_device_copyable<user_def_type> : std::true_type {};

template <typename T>
constexpr T expected_val() {
  if constexpr (std::is_same_v<T, int>) {
    return 42;
  }
  if constexpr (std::is_same_v<T, float>) {
    return 21.f;
  }
  if constexpr (std::is_same_v<T, char>) {
    return 'c';
  }
  return 0;
}

TEST_CASE("is_device_copyable UnaryTrait requirements",
          "[is_device_copyable]") {
  using device_copyable_t = sycl::is_device_copyable<user_def_type>;
  using integral_constant_t = std::integral_constant<bool, true>;

  // DefaultConstructible and CopyConstructible
  CHECK(std::is_default_constructible_v<device_copyable_t>);
  CHECK(std::is_copy_constructible_v<device_copyable_t>);

  // Takes one template type parameter
  CHECK(std::is_constructible_v<device_copyable_t>);

  // Publicly and unambiguously derived from a specialization of
  // integral_constant
  constexpr bool is_base_of =
      std::is_base_of_v<integral_constant_t, device_copyable_t>;
  CHECK(is_base_of);

  // The member names of the base characteristic are not hidden and are
  // unambiguously available
  CHECK(std::is_same_v<decltype(integral_constant_t::value),
                       decltype(device_copyable_t::value)>);
  CHECK(std::is_same_v<integral_constant_t::value_type,
                       device_copyable_t::value_type>);
  CHECK(std::is_same_v<decltype(integral_constant_t::value_type()),
                       decltype(device_copyable_t::value_type())>);
  CHECK(std::is_same_v<integral_constant_t::type, device_copyable_t::type>);
  CHECK(integral_constant_t() == device_copyable_t());
}

TEST_CASE("is_device_copyable specialization for user defined class",
          "[is_device_copyable]") {
#if SYCL_DEVICE_COPYABLE == 1
  INFO(
      "User defined type must be device copyable after specialization for "
      "sycl::is_device_copyable<T>");
  REQUIRE(sycl::is_device_copyable<user_def_type>::value);

  auto queue = sycl_cts::util::get_cts_object::queue();
  const user_def_type reference_object(
      expected_val<int>(), expected_val<float>(), expected_val<char>());

  bool host_res;
  user_def_type host_object(reference_object);
  user_def_type device_object;
  {
    sycl::buffer<bool, 1> host_res_buf(&host_res, {1});
    sycl::buffer<user_def_type, 1> host_data_buf(&host_object, {1});
    sycl::buffer<user_def_type, 1> dev_data_buf(&device_object, {1});

    queue.submit([&](sycl::handler& cgh) {
      sycl::accessor<bool, 0, sycl::access_mode::read_write> host_res_acc(
          host_res_buf, cgh);
      sycl::accessor<user_def_type, 0, sycl::access_mode::read_write>
          host_data_acc(host_data_buf, cgh);
      sycl::accessor<user_def_type, 0, sycl::access_mode::read_write>
          dev_data_acc(dev_data_buf, cgh);

      cgh.single_task<kernel_name>([=] {
        const user_def_type& temp_obj =
            host_data_acc;  // to access class members
        host_res_acc = temp_obj.a == expected_val<int>() &&
                       temp_obj.b == expected_val<float>() &&
                       temp_obj.c == expected_val<char>();

        dev_data_acc = user_def_type(expected_val<int>(), expected_val<float>(),
                                     expected_val<char>());
      });
    });
    queue.wait_and_throw();
  }

  {
    INFO("Check that host object has expected values in the kernel");
    CHECK(host_res);
  }
  {
    INFO("Check that host object has not changed");
    CHECK(host_object == reference_object);
  }
  {
    INFO("Check that device object has expected values on the host");
    CHECK(device_object == reference_object);
  }

#else
  SKIP("SYCL_DEVICE_COPYABLE is not defined!");
#endif
}
