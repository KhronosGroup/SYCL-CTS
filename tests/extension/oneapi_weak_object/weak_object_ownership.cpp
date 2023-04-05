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

namespace weak_object_ownership {
using namespace weak_object_common;

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

TEST_CASE("weak_object_ownership", "[weak_object]") {
#if !defined SYCL_EXT_ONEAPI_WEAK_OBJECT
  SKIP("SYCL_EXT_ONEAPI_WEAK_OBJECT is not defined");
#else
  test_weak_object_ownership<sycl::accessor<int> >{}("accessor");
  test_weak_object_ownership<sycl::buffer<int> >{}("buffer");
  test_weak_object_ownership<sycl::context>{}("context");
  test_weak_object_ownership<sycl::event>{}("event");
  test_weak_object_ownership<sycl::queue>{}("queue");

  test_weak_object_ownership<sycl::queue>::check_type();
  test_weak_object_ownership<sycl::stream>{}.test_stream();
  test_weak_object_ownership<sycl::local_accessor<int> >{}
      .test_local_accessor();
#endif
}

}  // namespace weak_object_ownership
