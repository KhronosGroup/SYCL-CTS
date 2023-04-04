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

#ifndef __SYCLCTS_TESTS_WEAK_OBJECT_COMMON_H
#define __SYCLCTS_TESTS_WEAK_OBJECT_COMMON_H

#include "../../../util/sycl_exceptions.h"
#include "../../common/common.h"
#include "../../common/type_coverage.h"

#ifdef SYCL_EXT_ONEAPI_WEAK_OBJECT

namespace weak_object_common {

template <typename SYCLObjT>
SYCLObjT get_sycl_object() {
  static sycl::buffer<int> buf{{1}};
  static sycl::buffer<int> host_buf{{1}};

  if constexpr (std::is_same_v<SYCLObjT, sycl::buffer<int>>) {
    return {sycl::range{1}};
  } else if constexpr (std::is_same_v<SYCLObjT, sycl::accessor<int>>) {
    return {buf};
  } else if constexpr (std::is_same_v<SYCLObjT, sycl::host_accessor<int>>) {
    return {host_buf};
  } else if constexpr (std::is_same_v<SYCLObjT, sycl::context>) {
    return {};
  } else if constexpr (std::is_same_v<SYCLObjT, sycl::event>) {
    return {};
  } else if constexpr (std::is_same_v<SYCLObjT, sycl::queue>) {
    return {};
  }
}

template <typename SYCLObjT>
class test_weak_object_init {
 public:
  void operator()(const std::string& typeName) {
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

 private:
  template <typename Optional>
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
      CHECK_THROWS_MATCHES(
          action(), sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));
    }
  }

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

  void test_swap(SYCLObjT sycl_object) {
    auto w_empty = sycl::ext::oneapi::weak_object<SYCLObjT>();
    auto w_other = sycl::ext::oneapi::weak_object<SYCLObjT>(sycl_object);
    w_empty.swap(w_other);
    check_weak_object(w_empty, sycl_object, "swap() member function");
  }
};

template <typename SYCLObjT>
class test_weak_object_expiring {
 public:
  void operator()(const std::string& typeName) {
    INFO(typeName + " type: ");
    sycl::ext::oneapi::weak_object<SYCLObjT> w;
    {
      auto sycl_object = get_sycl_object<SYCLObjT>();
      w = sycl_object;
      INFO("Check that weak_object now has a value and not expired");
      CHECK(!w.expired());
    }
    INFO("Check that weak_object was expired");
    CHECK(w.expired());
  }
};

template <typename SYCLObjT>
class test_weak_object_ownership {
 public:
  void operator()(const std::string& typeName) {
    INFO(typeName + " type: ");

    auto object1 = get_sycl_object<SYCLObjT>();
    auto object2 = get_sycl_object<SYCLObjT>();

    test_owner_before_same_objects(object1);
    test_owner_before_empty_objects(object1);
    test_owner_before_different_objects(object1, object2);
    test_owner_less_same_objects(object1);
    test_owner_less_empty_objects(object1);
    test_owner_less_different_objects(object1, object2);
  }

  static void check_type() {
    auto w1 = sycl::ext::oneapi::weak_object<SYCLObjT>();
    auto w2 = sycl::ext::oneapi::weak_object<SYCLObjT>();

    INFO("Check that weak_object owner_before() returns bool");
    CHECK(std::is_same_v<decltype(w1.owner_before(w2)), bool>);
  }

  void test_stream() {
    sycl::queue q;
    q.submit([&](sycl::handler& cgh) {
      sycl::stream stream1(1024, 256, cgh);
      sycl::stream stream2(1024, 256, cgh);

      test_owner_before_same_objects(stream1);
      test_owner_before_empty_objects(stream1);
      test_owner_before_different_objects(stream1, stream2);
      test_owner_less_same_objects(stream1);
      test_owner_less_empty_objects(stream1);
      test_owner_less_different_objects(stream1, stream2);
    });
  }

  void test_local_accessor() {
    sycl::queue q;
    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<int> l_accessor1({1}, cgh);
      sycl::local_accessor<int> l_accessor2({1}, cgh);

      test_owner_before_same_objects(l_accessor1);
      test_owner_before_empty_objects(l_accessor1);
      test_owner_before_different_objects(l_accessor1, l_accessor2);
      test_owner_less_same_objects(l_accessor1);
      test_owner_less_empty_objects(l_accessor1);
      test_owner_less_different_objects(l_accessor1, l_accessor2);
    });
  }

 private:
  sycl::ext::oneapi::owner_less<SYCLObjT> owner_less;

  void test_owner_before_same_objects(SYCLObjT sycl_object) {
    auto w1 = sycl::ext::oneapi::weak_object<SYCLObjT>(sycl_object);
    auto w2 = sycl::ext::oneapi::weak_object<SYCLObjT>(sycl_object);
    {
      bool weak_object_result = !w1.owner_before(w2) && !w2.owner_before(w1) &&
                                !w1.owner_before(sycl_object) &&
                                !w2.owner_before(sycl_object);

      INFO(
          "Verify that owner_before compares equivalent for two weak objects "
          "that both refer to the same underlying SYCL object");
      CHECK(weak_object_result);
    }
    {
      bool sycl_object_result = !sycl_object.ext_oneapi_owner_before(w1) ||
                                !sycl_object.ext_oneapi_owner_before(w2);

      INFO(
          "Verify that ext_oneapi_owner_before compares equivalent for two "
          "weak objects that both refer to the same underlying SYCL object");
      CHECK(sycl_object_result);
    }
  }

  void test_owner_less_same_objects(SYCLObjT sycl_object) {
    auto w1 = sycl::ext::oneapi::weak_object<SYCLObjT>(sycl_object);
    auto w2 = sycl::ext::oneapi::weak_object<SYCLObjT>(sycl_object);

    bool result = owner_less(w1, w2) == false && owner_less(w2, w1) == false &&
                  owner_less(w1, sycl_object) == false &&
                  owner_less(sycl_object, w1) == false &&
                  owner_less(w2, sycl_object) == false &&
                  owner_less(sycl_object, w2) == false;

    INFO(
        "Verify that owner_less compares equivalent for two weak objects that "
        "both refer to the same underlying SYCL object");
    CHECK(result);
  }

  void test_owner_before_empty_objects(SYCLObjT sycl_object) {
    auto w1 = sycl::ext::oneapi::weak_object<SYCLObjT>();
    auto w2 = sycl::ext::oneapi::weak_object<SYCLObjT>();
    {
      bool weak_object_result = !w1.owner_before(w2) && !w2.owner_before(w1);

      INFO(
          "Verify that owner_before compares equivalent for two weak objects "
          "that are both empty");
      CHECK(weak_object_result);
    }
  }

  void test_owner_less_empty_objects(SYCLObjT sycl_object) {
    auto w1 = sycl::ext::oneapi::weak_object<SYCLObjT>();
    auto w2 = sycl::ext::oneapi::weak_object<SYCLObjT>();

    bool result = !owner_less(w1, w2) &&
                  (owner_less(w1, sycl_object) || owner_less(w2, sycl_object));

    INFO(
        "Verify that owner_less compares equivalent for two weak objects that "
        "are both empty");
    CHECK(result);
  }

  void test_owner_before_different_objects(SYCLObjT sycl_object1,
                                           SYCLObjT sycl_object2) {
    auto w1 = sycl::ext::oneapi::weak_object<SYCLObjT>(sycl_object1);
    auto w2 = sycl::ext::oneapi::weak_object<SYCLObjT>(sycl_object2);
    {
      bool weak_object_result = (w1.owner_before(w2) && !w2.owner_before(w1)) ||
                                (!w1.owner_before(w2) && w2.owner_before(w1));

      INFO(
          "Verify that owner_before has some order for two weak object that "
          "refer to different underlying SYCL objects");
      CHECK(weak_object_result);
    }
    {
      bool sycl_object_result = (sycl_object1.ext_oneapi_owner_before(w2) &&
                                 !sycl_object2.ext_oneapi_owner_before(w1)) ||
                                (!sycl_object1.ext_oneapi_owner_before(w2) &&
                                 sycl_object2.ext_oneapi_owner_before(w1));

      INFO(
          "Verify that ext_oneapi_owner_before has some order for two weak "
          "object that refer to different underlying SYCL objects");
      CHECK(sycl_object_result);
    }
    {
      bool weak_object_referred_result =
          w1.owner_before(sycl_object2) == w1.owner_before(w2) &&
          w2.owner_before(sycl_object1) == w2.owner_before(w1);

      INFO(
          "Check that owner_before returns same values with weak object and "
          "underlying SYCL object");
      CHECK(weak_object_referred_result);
    }
    {
      bool sycl_object_referred_result =
          sycl_object1.ext_oneapi_owner_before(sycl_object2) ==
              sycl_object1.ext_oneapi_owner_before(w2) &&
          sycl_object2.ext_oneapi_owner_before(sycl_object1) ==
              sycl_object2.ext_oneapi_owner_before(w1);

      INFO(
          "Check that ext_oneapi_owner_before returns same values with weak "
          "object and underlying SYCL object");
      CHECK(sycl_object_referred_result);
    }
  }

  void test_owner_less_different_objects(SYCLObjT sycl_object1,
                                         SYCLObjT sycl_object2) {
    auto w1 = sycl::ext::oneapi::weak_object<SYCLObjT>(sycl_object1);
    auto w2 = sycl::ext::oneapi::weak_object<SYCLObjT>(sycl_object2);
    {
      bool weak_object_result = (owner_less(w1, w2) && !owner_less(w2, w1)) ||
                                (!owner_less(w1, w2) && owner_less(w2, w1));

      INFO(
          "Verify that owner_less has some order for two weak object that "
          "refer to different underlying SYCL objects");
      CHECK(weak_object_result);
    }
    {
      bool referred_result =
          owner_less(w1, sycl_object2) == owner_less(w1, w2) &&
          owner_less(w2, sycl_object1) == owner_less(w2, w1);

      INFO(
          "Check that owner_less returns same values with weak object and "
          "underlying SYCL object");
      CHECK(referred_result);
    }
  }
};

}  // namespace weak_object_common
#endif  // SYCL_EXT_ONEAPI_WEAK_OBJECT

#endif  // __SYCLCTS_TESTS_WEAK_OBJECT_COMMON_H
