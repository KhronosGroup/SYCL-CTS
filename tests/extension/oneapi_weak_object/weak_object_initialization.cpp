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

#include "weak_object_common.h"

namespace weak_object_initialization {
using namespace weak_object_common;

template <typename Optional, typename SYCLObjT>
void check_weak_object(sycl::ext::oneapi::weak_object<SYCLObjT> w,
                       const Optional& object, const std::string& init_type) {
  auto ret = w.try_lock();
  {
    INFO("Check using object_type = SyclObject " + init_type);
    CHECK(std::is_same_v<typename decltype(w)::object_type, SYCLObjT>);
  }

  if constexpr (!std::is_same_v<Optional, std::nullopt_t>) {
    {
      INFO("Check that try_lock returns same type as SYCL object");
      CHECK(std::is_same_v<std::remove_reference_t<decltype(ret.value())>,
                           SYCLObjT>);
    }
    {
      INFO("Check that lock returns same type as SYCL object");
      CHECK(std::is_same_v<decltype(w.lock()), SYCLObjT>);
    }
    {
      INFO("Check that try_lock returns exactly that SYCL object");
      CHECK(ret.value() == object);
    }
    {
      INFO("Check that lock returns exactly that SYCL object");
      CHECK(w.lock() == object);
    }
  } else {
    {
      INFO("Check that empty object has no value");
      CHECK(ret.has_value() == false);
    }

    auto action = [&] { w.lock(); };
    INFO(
        "Implementation has to throw a sycl::exception with "
        "sycl::errc::invalid when empty weak_object tries to return "
        "underlying SYCL object");
    CHECK_THROWS_MATCHES(action(), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }
}

template <typename SYCLObjT>
void test_constructors(SYCLObjT sycl_object) {
  // weak object for copy and move
  auto w_other = sycl::ext::oneapi::weak_object<SYCLObjT>(sycl_object);

  auto w_empty = sycl::ext::oneapi::weak_object<SYCLObjT>();
  auto w_obj = sycl::ext::oneapi::weak_object<SYCLObjT>(sycl_object);
  auto w_copy = sycl::ext::oneapi::weak_object<SYCLObjT>(w_other);
  auto w_move = sycl::ext::oneapi::weak_object<SYCLObjT>(std::move(w_other));

  check_weak_object(w_empty, std::nullopt, "empty constructor");
  check_weak_object(w_obj, sycl_object, "SYCL object constructor");
  check_weak_object(w_copy, sycl_object, "copy constructor");
  check_weak_object(w_move, sycl_object, "move constructor");
}

template <typename SYCLObjT>
void test_assign_operators(SYCLObjT sycl_object) {
  auto w_other = sycl::ext::oneapi::weak_object<SYCLObjT>(sycl_object);

  sycl::ext::oneapi::weak_object<SYCLObjT> w_obj;
  sycl::ext::oneapi::weak_object<SYCLObjT> w_copy;
  sycl::ext::oneapi::weak_object<SYCLObjT> w_move;
  w_obj = sycl_object;

  w_copy = w_other;
  w_move = std::move(w_other);

  check_weak_object(w_obj, sycl_object, "SYCL object assignment");
  check_weak_object(w_copy, sycl_object, "copy assignment");
  check_weak_object(w_move, sycl_object, "move assignment");
}

template <typename SYCLObjT>
void test_swap(SYCLObjT sycl_object) {
  auto w_empty = sycl::ext::oneapi::weak_object<SYCLObjT>();
  auto w_other = sycl::ext::oneapi::weak_object<SYCLObjT>(sycl_object);
  w_empty.swap(w_other);
  check_weak_object(w_empty, sycl_object, "swap() member function");
}

template <typename SYCLObjT>
void test_weak_object_init(const std::string& typeName) {
  INFO(typeName + " type: ");

  auto object = get_sycl_object<SYCLObjT>();
  test_constructors(object);
  test_assign_operators(object);
  test_swap(object);
}

void test_stream() {
  sycl::queue q;
  q.submit([&](sycl::handler& cgh) {
    sycl::stream s(1024, 256, cgh);

    test_constructors(s);
    test_assign_operators(s);
    test_swap(s);
  });
}

void test_local_accessor() {
  sycl::queue q;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int> l_accessor({1}, cgh);

    test_constructors(l_accessor);
    test_assign_operators(l_accessor);
    test_swap(l_accessor);
  });
}

TEST_CASE("weak_object init", "[weak_object]") {
#if !defined SYCL_EXT_ONEAPI_WEAK_OBJECT
  SKIP("SYCL_EXT_ONEAPI_WEAK_OBJECT is not defined");
#else
  test_weak_object_init<sycl::accessor<int> >("accessor");
  test_weak_object_init<sycl::host_accessor<int> >("host_accessor");
  test_weak_object_init<sycl::buffer<int> >("buffer");
  test_weak_object_init<sycl::queue>("queue");
  test_weak_object_init<sycl::context>("context");
  test_weak_object_init<sycl::event>("event");

  test_stream();
  test_local_accessor();
#endif
}
}  // namespace weak_object_initialization
