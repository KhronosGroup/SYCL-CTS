/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
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
#include "../common/type_coverage.h"

#include <string>
#include <type_traits>

template <typename Property>
void check_property_impl() {
  CHECK(std::is_base_of_v<std::true_type, sycl::is_property<Property>>);
  CHECK(sycl::is_property_v<Property>);
}

template <typename Property>
struct check_property {
  void operator()(const std::string& property_name) {
    INFO("property: " << property_name);
    check_property_impl<Property>();
  }
};

template <typename Property, typename SyclObject>
void check_property_object_impl() {
  CHECK(std::is_base_of_v<std::true_type,
                          sycl::is_property_of<Property, SyclObject>>);
  CHECK(sycl::is_property_of_v<Property, SyclObject>);
}

template <typename Property, typename SyclObject>
struct check_property_object {
  void operator()(const std::string& property_name,
                  const std::string& object_name) {
    INFO("object: " << object_name << ", property: " << property_name);
    check_property_impl<Property>();
    check_property_object_impl<Property, SyclObject>();
  }
};

TEST_CASE("property api", "[property]") {
  {
    const auto properties = named_type_pack<
        sycl::property::queue::enable_profiling,
        sycl::property::queue::in_order>::generate("queue::enable_profiling",
                                                   "queue::in_order");
    const auto objects = named_type_pack<sycl::queue>::generate("queue");
    for_all_combinations<check_property_object>(properties, objects);
  }
  {
    const auto properties =
        named_type_pack<sycl::property::buffer::use_host_ptr,
                        sycl::property::buffer::use_mutex,
                        sycl::property::buffer::context_bound>::
            generate("buffer::use_host_ptr", "buffer::use_mutex",
                     "buffer::context_bound");
    // provide any template argument for the types that require it
    const auto objects = named_type_pack<sycl::buffer<int>>::generate("buffer");
    for_all_combinations<check_property_object>(properties, objects);
  }
#if !(defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP))
  {
    const auto properties = named_type_pack<
        sycl::property::image::use_host_ptr, sycl::property::image::use_mutex,
        sycl::property::image::context_bound>::generate("image::use_host_ptr",
                                                        "image::use_mutex",
                                                        "image::context_bound");
    const auto objects =
        named_type_pack<sycl::sampled_image<>,
                        sycl::unsampled_image<>>::generate("sampled_image",
                                                           "unsampled_image");
    for_all_combinations<check_property_object>(properties, objects);
  }
#else
  WARN(
      "Implementation does not define sycl::sampled_image and "
      "sycl::unsampled_image "
      "Skipping the test case.");
#endif
  {
#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    WARN(
        "Implementation does not define sycl::unsampled_image_accessor and "
        "sycl::host_unsampled_image_accessor "
        "Skipping the test case.");
#endif

    const auto properties =
        named_type_pack<sycl::property::no_init>::generate("property::no_init");
    // provide any template argument for the types that require it
    const auto objects =
        named_type_pack<sycl::accessor<int>, sycl::host_accessor<int>
#if !(defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP))
                        ,
                        sycl::unsampled_image_accessor<
                            sycl::int4, 1, sycl::access_mode::read_write>,
                        sycl::host_unsampled_image_accessor<sycl::int4, 1>
#endif
                        >::generate("accessor", "host_accessor"
#if !(defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP))
                                    ,
                                    "unsampled_image_accessor",
                                    "host_unsampled_image_accessor"
#endif
        );
    for_all_combinations<check_property_object>(properties, objects);
  }
  {
    const auto properties =
        named_type_pack<sycl::property::reduction::initialize_to_identity>::
            generate("reduction::initialize_to_identity");
    for_all_combinations<check_property>(properties);
  }
}
