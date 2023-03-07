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
//  Provides buffer deduction guides tests
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_list.h"

namespace buffer_deduction_guides {
using namespace sycl;

// size for container
constexpr int size = 3;

// create property lists
std::mutex mutex;
auto context = sycl_cts::util::get_cts_object::context();
const property_list props{property::buffer::use_mutex(mutex),
                          property::buffer::context_bound(context)};

template <typename T>
class check_buffer_deduction {
 public:
  void operator()(const std::string& type) {
    // create container
    // using this API because of no_cnstr and no_def_cnstr types
    std::array<T, size> arr = {user_def_types::get_init_value_helper<T>(0),
                               user_def_types::get_init_value_helper<T>(0),
                               user_def_types::get_init_value_helper<T>(0)};
    typeName = type;

    test_inputiterator(arr);
    test_container(arr);

    range<1> range1d(size);
    range<2> range2d(size, size);
    range<3> range3d(size, size, size);
    test_type<1>(range1d);
    test_type<2>(range2d);
    test_type<3>(range3d);
  }

 private:
  std::allocator<T> std_alloc;
  buffer_allocator<T> buf_alloc;
  std::string typeName;

  inline void test_inputiterator(std::array<T, size> data) {
    INFO("buffer_ctors_deduction::test_inputiterator() " + typeName);
    // buffer with no allocator and no property list
    {
      buffer buf(data.begin(), data.end());

      INFO("Incorrect deduction with InputIterator");
      CHECK(std::is_same_v<decltype(buf),
                           buffer<typename decltype(data)::value_type, 1>>);

      CHECK(std::is_same_v<typename decltype(buf)::value_type,
                           typename decltype(data)::value_type>);
    }
    // buffer with allocator and no property list
    {
      buffer buf_std_alloc(data.begin(), data.end(), std_alloc);
      buffer buf_buf_alloc(data.begin(), data.end(), buf_alloc);

      INFO("Incorrect deduction with allocator");
      CHECK(std::is_same_v<
            decltype(buf_std_alloc),
            buffer<typename decltype(data)::value_type, 1, std::allocator<T>>>);

      CHECK(std::is_same_v<decltype(buf_buf_alloc),
                           buffer<typename decltype(data)::value_type, 1,
                                  buffer_allocator<T>>>);

      CHECK(std::is_same_v<decltype(buf_std_alloc.get_allocator()),
                           std::allocator<T>>);
      CHECK(std::is_same_v<decltype(buf_buf_alloc.get_allocator()),
                           buffer_allocator<T>>);
    }
    // buffer with property list and no alloccator
    {
      buffer buf_prop(data.begin(), data.end(), props);

      INFO("Incorrect deduction with propery list");
      CHECK(std::is_same_v<decltype(buf_prop),
                           buffer<typename decltype(data)::value_type, 1>>);
    }
    // buffer with allocator and property list
    {
      buffer buf_std_alloc(data.begin(), data.end(), std_alloc, props);
      buffer buf_buf_alloc(data.begin(), data.end(), buf_alloc, props);

      INFO("Incorrect deduction with property list and allocator");
      CHECK(std::is_same_v<
            decltype(buf_std_alloc),
            buffer<typename decltype(data)::value_type, 1, std::allocator<T>>>);

      CHECK(std::is_same_v<decltype(buf_buf_alloc),
                           buffer<typename decltype(data)::value_type, 1,
                                  buffer_allocator<T>>>);
    }
  }

  template <int dims>
  void test_type(const range<dims>& r) {
    const int arr_size = r.size();

    std::unique_ptr<T[]> data(
        new T[size]{user_def_types::get_init_value_helper<T>(0),
                    user_def_types::get_init_value_helper<T>(0),
                    user_def_types::get_init_value_helper<T>(0)});

    INFO("buffer_ctors_deduction::test_type() " + typeName);
    // buffer with no alloccator and no property list
    {
      buffer buf(data.get(), r);
      {
        INFO("Incorrect deduction with T");
        CHECK(std::is_same_v<decltype(buf), buffer<T, dims>>);
        CHECK(std::is_same_v<typename decltype(buf)::value_type, T>);
      }
      {
        INFO("Incorrect deduction with range");
        CHECK(std::is_same_v<decltype(buf.get_range()), range<dims>>);
      }
    }
    // buffer with allocator and no property list
    {
      buffer buf_std_alloc(data.get(), r, std_alloc);
      buffer buf_buf_alloc(data.get(), r, buf_alloc);

      INFO("Incorrect deduction with T");
      CHECK(std::is_same_v<decltype(buf_std_alloc),
                           buffer<T, dims, std::allocator<T>>>);
      CHECK(std::is_same_v<decltype(buf_buf_alloc),
                           buffer<T, dims, buffer_allocator<T>>>);

      CHECK(std::is_same_v<decltype(buf_std_alloc.get_allocator()),
                           std::allocator<T>>);
      CHECK(std::is_same_v<decltype(buf_buf_alloc.get_allocator()),
                           buffer_allocator<T>>);
    }
    // buffer with property list and no alloccator
    {
      buffer buf_prop(data.get(), r, props);
      INFO("Incorrect deductionwith property list");
      CHECK(std::is_same_v<decltype(buf_prop), buffer<T, dims>>);
    }
    // buffer with allocator and property list
    {
      buffer buf_std_alloc(data.get(), r, std_alloc, props);
      buffer buf_buf_alloc(data.get(), r, buf_alloc, props);

      INFO("Incorrect deduction with T");
      CHECK(std::is_same_v<decltype(buf_std_alloc),
                           buffer<T, dims, std::allocator<T>>>);
      CHECK(std::is_same_v<decltype(buf_buf_alloc),
                           buffer<T, dims, buffer_allocator<T>>>);
    }
  }

  inline void test_container(std::array<T, size> data) {
    INFO("buffer_ctors_deduction::test_container() " + typeName);
    // buffer with no alloccator and no property list
    {
      buffer buf(data);

      INFO("Incorrect deduction with InputIterator");
      CHECK(std::is_same_v<decltype(buf),
                           buffer<typename decltype(data)::value_type, 1>>);

      CHECK(std::is_same_v<typename decltype(buf)::value_type,
                           typename decltype(data)::value_type>);

      CHECK(std::is_same_v<decltype(buf.get_range()), range<1>>);
    }
    // buffer with allocator and no property list
    {
      buffer buf_std_alloc(data, std_alloc);
      buffer buf_buf_alloc(data, buf_alloc);

      INFO("Incorrect deduction with allocator");
      CHECK(std::is_same_v<
            decltype(buf_std_alloc),
            buffer<typename decltype(data)::value_type, 1, std::allocator<T>>>);

      CHECK(std::is_same_v<decltype(buf_buf_alloc),
                           buffer<typename decltype(data)::value_type, 1,
                                  buffer_allocator<T>>>);

      CHECK(std::is_same_v<decltype(buf_std_alloc.get_allocator()),
                           std::allocator<T>>);
      CHECK(std::is_same_v<decltype(buf_buf_alloc.get_allocator()),
                           buffer_allocator<T>>);
    }
    // buffer with property list and no alloccator
    {
      buffer buf_prop(data, props);

      INFO("Incorrect deduction with propery list");
      CHECK(std::is_same_v<decltype(buf_prop),
                           buffer<typename decltype(data)::value_type, 1>>);
    }
    // buffer with allocator and property list
    {
      buffer buf_std_alloc(data, std_alloc, props);
      buffer buf_buf_alloc(data, buf_alloc, props);

      INFO("Incorrect deduction with property list and allocator");
      CHECK(std::is_same_v<
            decltype(buf_std_alloc),
            buffer<typename decltype(data)::value_type, 1, std::allocator<T>>>);

      CHECK(std::is_same_v<decltype(buf_buf_alloc),
                           buffer<typename decltype(data)::value_type, 1,
                                  buffer_allocator<T>>>);
    }
  }
};

TEST_CASE("buffer deduction guides", "[buffer]") {
  for_all_types<check_buffer_deduction>(deduction::vector_types);
  for_all_types<check_buffer_deduction>(deduction::scalar_types);
}
}  // namespace buffer_deduction_guides
