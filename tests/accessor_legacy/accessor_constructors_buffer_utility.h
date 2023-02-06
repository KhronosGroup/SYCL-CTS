/*************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
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
//  This file is a common utility for the implementation of
//  accessor_constructors.cpp and accessor_api.cpp.
//
**************************************************************************/
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_BUFFER_UTILITY_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_BUFFER_UTILITY_H

#include "../common/common.h"
#include "accessor_constructors_utility.h"

#ifndef TEST_NAME
#error Invalid test namespace
#endif

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** @brief Creates a buffer accessor and checks all its members for correctness
 */
template <typename accTag, typename ... propertyListT>
class check_accessor_constructor_buffer {
public:
  /** @brief Overload to verify all constructors w/o range and offset
   */
  template <typename allocatorT, typename ... handlerArgsT>
  static void check(sycl::buffer<typename accTag::dataT, accTag::dataDims,
                                     allocatorT> &buffer,
                    sycl_cts::util::logger &log,
                    const std::string& constructorName,
                    const std::string& typeName,
                    const propertyListT& ... properties,
                    handlerArgsT&& ... handler) {
    // construct the accessor
    typename accTag::type accessor(buffer,
                                   std::forward<handlerArgsT>(handler)...,
                                   properties...);
    const auto offset =
        sycl_cts::util::get_cts_object::id<accTag::dataDims>::get(0, 0, 0);

    // check the accessor
    check_accessor_members<accTag>::check(
        log, accessor, constructorName, typeName,
        accessor_members::size{buffer.byte_size()},
        accessor_members::count{buffer.size()},
        accessor_members::offset<accTag::dataDims>{offset},
        accessor_members::range<accTag::dataDims>{buffer.get_range()},
        accessor_members::placeholder{accTag::placeholder});
  }
  /** @brief Overload to verify all constructors with range only
   */
  template <typename allocatorT, typename ... handlerArgsT>
  static void check(sycl::buffer<typename accTag::dataT, accTag::dataDims,
                                     allocatorT> &buffer,
                    sycl::range<accTag::dataDims> range,
                    sycl_cts::util::logger &log,
                    const std::string& constructorName,
                    const std::string& typeName,
                    const propertyListT& ... properties,
                    handlerArgsT&& ... handler) {
    // construct the accessor
    typename accTag::type accessor(buffer,
                                   std::forward<handlerArgsT>(handler)...,
                                   range,
                                   properties...);
    const auto offset =
        sycl_cts::util::get_cts_object::id<accTag::dataDims>::get(0, 0, 0);

    // check the accessor
    check_accessor_members<accTag>::check(
        log, accessor, constructorName, typeName,
        accessor_members::size{range.size() * sizeof(typename accTag::dataT)},
        accessor_members::count{range.size()},
        accessor_members::offset<accTag::dataDims>{offset},
        accessor_members::range<accTag::dataDims>{range},
        accessor_members::placeholder{accTag::placeholder});
  }
  /** @brief Overload to verify all constructors with range and offset
   */
  template <typename allocatorT, typename ... handlerArgsT>
  static void check(sycl::buffer<typename accTag::dataT, accTag::dataDims,
                                     allocatorT> &buffer,
                    sycl::range<accTag::dataDims> range,
                    sycl::id<accTag::dataDims> offset,
                    sycl_cts::util::logger &log,
                    const std::string& constructorName,
                    const std::string& typeName,
                    const propertyListT& ... properties,
                    handlerArgsT&& ... handler) {
    // construct the accessor
    typename accTag::type accessor(buffer,
                                   std::forward<handlerArgsT>(handler)...,
                                   range, offset,
                                   properties...);

    // check the accessor
    check_accessor_members<accTag>::check(
        log, accessor, constructorName, typeName,
        accessor_members::size{range.size() * sizeof(typename accTag::dataT)},
        accessor_members::count{range.size()},
        accessor_members::offset<accTag::dataDims>{offset},
        accessor_members::range<accTag::dataDims>{range},
        accessor_members::placeholder{accTag::placeholder});
  }
};

/** @brief Checks all constructors available for non-zero dimensions
 */
template <typename T, size_t dims, sycl::target target,
          sycl::access::placeholder placeholder>
class check_all_accessor_constructors_buffer {
public:
  template <sycl::access_mode mode, typename allocatorT,
            typename ... handlerArgsT>
  static void check(sycl::buffer<T, dims, allocatorT> &buffer,
                    sycl::range<dims> range,
                    sycl::id<dims> offset,
                    sycl_cts::util::logger &log,
                    const std::string& typeName,
                    handlerArgsT&& ... handler) {
    // Run verification for accessors with dim > 0
    using accTag = accessor_type_info<T, dims, mode, target, placeholder>;

    constexpr bool usesHander = sizeof...(handlerArgsT) != 0;
    {
      using verifier = check_accessor_constructor_buffer<accTag>;
      {
      const auto constructorName = usesHander ?
          "constructor(buffer, handler)" :
          "constructor(buffer)";
      verifier::check(buffer,
                      log, constructorName, typeName, handler...);
      }
      {
        const auto constructorName = usesHander ?
            "constructor(buffer, handler, range)" :
            "constructor(buffer, range)";
        verifier::check(buffer, range,
                        log, constructorName, typeName, handler...);
      }
      {
        const auto constructorName = usesHander ?
            "constructor(buffer, handler, range, offset)" :
            "constructor(buffer, range, offset)";
        verifier::check(buffer, range, offset,
                        log, constructorName, typeName, handler...);
      }
    }
    {
      using property_list = sycl::property_list;
      using verifier = check_accessor_constructor_buffer<accTag, property_list>;

      auto context = util::get_cts_object::context();
      property_list properties {
          sycl::property::buffer::context_bound(context)};

      {
        const auto constructorName = usesHander ?
            "constructor(buffer, handler, property_list)" :
            "constructor(buffer, property_list)";
        verifier::check(buffer,
                        log, constructorName, typeName, properties, handler...);
      }
      {
        const auto constructorName = usesHander ?
            "constructor(buffer, handler, range, property_list)" :
            "constructor(buffer, range, property_list)";
        verifier::check(buffer, range,
                        log, constructorName, typeName, properties, handler...);
      }
      {
        const auto constructorName = usesHander ?
            "constructor(buffer, handler, range, offset, property_list)" :
            "constructor(buffer, range, offset, property_list)";
        verifier::check(buffer, range, offset,
                        log, constructorName, typeName, properties, handler...);
      }
    }
  }
};
/** @brief Checks all constructors available for 0 dimension
 */
template <typename T, sycl::target target,
          sycl::access::placeholder placeholder>
class check_all_accessor_constructors_buffer<T, 0, target, placeholder> {
  static constexpr size_t dims = 0;
public:
  template <sycl::access_mode mode, typename allocatorT,
            typename ... handlerArgsT>
  static void check(sycl::buffer<T, 1, allocatorT> &buffer,
                    sycl::range<1> range,
                    sycl::id<1> offset,
                    sycl_cts::util::logger &log,
                    const std::string& typeName,
                    handlerArgsT&& ... handler) {
    // Run verification for accessors with dim == 0
    static_cast<void>(range);
    static_cast<void>(offset);
    using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
    using verifier = check_accessor_constructor_buffer<accTag>;

    constexpr bool usesHander = sizeof...(handlerArgsT) != 0;
    const auto constructorName = usesHander ?
        "constructor(buffer, handler)" :
        "constructor(buffer)";
    verifier::check(buffer, log, constructorName, typeName, handler...);
  }
};

/** @brief Check common-by-reference semantics for non-zero dimensions
 */
template <typename T, size_t dims, sycl::target target,
          sycl::access::placeholder placeholder>
class check_accessor_common_by_reference_buffer {
public:
  template <sycl::access_mode mode, typename allocatorT,
            typename ... handlerArgsT>
  static void check(sycl::buffer<T, dims, allocatorT> &buffer,
                    sycl::range<dims> range,
                    sycl::id<dims> offset,
                    sycl_cts::util::logger &log,
                    const std::string& typeName,
                    handlerArgsT&& ... handler) {
    // Run verification for accessors with dim > 0
    using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
    {
      using verifier = check_accessor_copy_constructable<accTag>;

      typename accTag::type srcAccessor(buffer, handler..., range, offset);

      verifier::check(srcAccessor, log, typeName);
    }
    {
      using verifier = check_accessor_copy_assignable<accTag>;

      typename accTag::type srcAccessor(buffer, handler..., range, offset);
      typename accTag::type dstAccessor(buffer, handler...);

      verifier::check(srcAccessor, dstAccessor, log, typeName);
    }
    {
      using verifier = check_accessor_move_constructable<accTag>;

      typename accTag::type srcAccessor(buffer, handler...);

      verifier::check(srcAccessor, log, typeName);
    }
    {
      using verifier = check_accessor_move_assignable<accTag>;

      typename accTag::type srcAccessor(buffer, handler...);
      typename accTag::type dstAccessor(buffer, handler..., range, offset);

      verifier::check(srcAccessor, dstAccessor, log, typeName);
    }
  }
};
/** @brief Check common-by-reference semantics for 0 dimension
 */
template <typename T, sycl::target target,
          sycl::access::placeholder placeholder>
class check_accessor_common_by_reference_buffer<T, 0, target, placeholder> {
  static constexpr size_t dims = 0;
public:
  template <sycl::access_mode mode, typename allocatorT,
            typename ... handlerArgsT>
  static void check(sycl::buffer<T, 1, allocatorT> &buffer,
                    sycl::range<1> range,
                    sycl::id<1> offset,
                    sycl_cts::util::logger &log,
                    const std::string& typeName,
                    handlerArgsT&& ... handler) {
    // Neither range nor offset available for dim == 0
    static_cast<void>(range);
    static_cast<void>(offset);
    using accTag = accessor_type_info<T, dims, mode, target, placeholder>;
    {
      using verifier = check_accessor_copy_constructable<accTag>;

      typename accTag::type srcAccessor(buffer, handler...);

      verifier::check(srcAccessor, log, typeName);
    }
    {
      using verifier = check_accessor_copy_assignable<accTag>;

      typename accTag::type srcAccessor(buffer, handler...);
      typename accTag::type dstAccessor(buffer, handler...);

      verifier::check(srcAccessor, dstAccessor, log, typeName);
    }
    {
      using verifier = check_accessor_move_constructable<accTag>;

      typename accTag::type srcAccessor(buffer, handler...);

      verifier::check(srcAccessor, log, typeName);
    }
    {
      using verifier = check_accessor_move_assignable<accTag>;

      typename accTag::type srcAccessor(buffer, handler...);
      typename accTag::type dstAccessor(buffer, handler...);

      verifier::check(srcAccessor, dstAccessor, log, typeName);
    }
  }
};

/** @brief enum used to denote that the buffer_accessor_dims specialization
 *         performs checks either only for host_buffer or for any other buffer
 */
enum is_host_buffer : bool { false_t = false, true_t = true };

/** @brief Provide uniform way to initialize input data for different accessors
 */
template <typename T, size_t dims, typename ... allocatorT>
class buffer_accesor_input_data {
public:
  static constexpr size_t dataDims = dims;
  static constexpr int rangeSize = 32;

  using range_t = sycl::range<dataDims>;
  using offset_t = sycl::id<dataDims>;
  using data_t = std::vector<sycl::cl_uchar>;
  using buffer_t = sycl::buffer<T, dataDims, allocatorT...>;

public:
  buffer_accesor_input_data():
    dataRange(sycl_cts::util::get_cts_object::range<dataDims>::get(rangeSize,
                                                                   rangeSize,
                                                                   rangeSize)),
    data(dataRange.size() * sizeof(T)),
    range(dataRange / 2),
    offset(dataRange / 2) {
    std::iota(std::begin(data), std::end(data), 0);
  }

  inline T* getDataPointer() {
    return reinterpret_cast<T *>(data.data());
  }
  inline const range_t& getDataRange() const {
    return dataRange;
  }
  inline const range_t& getRange() const {
    return range;
  }
  inline const offset_t& getOffset() const {
    return offset;
  }
private:
  range_t dataRange;
  data_t data;
  range_t range;
  offset_t offset;
};
/** @brief Specialization to implement input data initialization for 0 dimension
 *         in an uniform way with the non-zero dimensions
 */
template <typename T, typename ... allocatorT>
class buffer_accesor_input_data<T, 0, allocatorT...> {
public:
  static constexpr size_t dataDims = 1;
  static constexpr int rangeSize = 1;

  using range_t = sycl::range<dataDims>;
  using offset_t = sycl::id<dataDims>;
  using data_t = std::vector<sycl::cl_uchar>;
  using buffer_t = sycl::buffer<T, dataDims, allocatorT...>;

public:
  buffer_accesor_input_data():
    dataRange(rangeSize),
    data(sizeof(T), 0),
    range(dataRange),
    offset(0) {
  }

  inline T* getDataPointer() {
    return reinterpret_cast<T *>(data.data());
  }
  inline const range_t& getDataRange() const {
    return dataRange;
  }
  inline const range_t& getRange() const {
    return range;
  }
  inline const offset_t& getOffset() const {
    return offset;
  }
private:
  range_t dataRange;
  data_t data;
  range_t range;
  offset_t offset;
};

/** @brief Used to test the buffer accessor combinations for device and
 *         constant_buffer
 */
template <typename T, typename kernelName, size_t dims,
          is_host_buffer isHostBuffer = false_t,
          sycl::access::placeholder placeholder =
              sycl::access::placeholder::false_t,
          typename ... allocatorT>
class buffer_accessor_dims {
public:
  static void check(util::logger &log, sycl::queue &queue,
                    const std::string& typeName) {
    using input_type = buffer_accesor_input_data<T, dims>;
    input_type input;

    const auto dataRange = input.getDataRange();
    typename input_type::buffer_t buffer(input.getDataPointer(), dataRange);

    const auto r = input.getRange();
    const auto offset = input.getOffset();

    /** check buffer accessor constructors for device
     */
    {
      constexpr auto target = sycl::target::device;
      using verifier =
          check_all_accessor_constructors_buffer<T, dims, target, placeholder>;
      using semantics_verifier =
          check_accessor_common_by_reference_buffer<T, dims, target,
                                                    placeholder>;

      queue.submit([&](sycl::handler &h) {
        /** check device constructors for different modes
         */
        {
          constexpr auto mode = sycl::access_mode::read;
          verifier::template check<mode>(buffer, r, offset, log, typeName, h);
        }
        {
          constexpr auto mode = sycl::access_mode::write;
          verifier::template check<mode>(buffer, r, offset, log, typeName, h);
        }
        {
          constexpr auto mode = sycl::access_mode::read_write;
          verifier::template check<mode>(buffer, r, offset, log, typeName, h);
        }
        {
          constexpr auto mode = sycl::access_mode::discard_write;
          verifier::template check<mode>(buffer, r, offset, log, typeName, h);
        }
        {
          constexpr auto mode = sycl::access_mode::discard_read_write;
          verifier::template check<mode>(buffer, r, offset, log, typeName, h);
        }
        {
          constexpr auto mode = sycl::access_mode::atomic;
          verifier::template check<mode>(buffer, r, offset, log, typeName, h);
        }
        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = sycl::access_mode::discard_read_write;
          semantics_verifier::template check<mode>(buffer, r, offset, log, typeName, h);
        }
        /** dummy kernel as no kernel is required for these checks
         */
        using dummy =
            dummy_functor<kernelName, sycl::target::device>;
        h.single_task<dummy>(dummy{});
      });
      queue.wait_and_throw();
    }

    /** check buffer accessor constructors for constant_buffer
     */
    {
      constexpr auto target = sycl::target::constant_buffer;
      using verifier =
          check_all_accessor_constructors_buffer<T, dims, target, placeholder>;
      using semantics_verifier =
          check_accessor_common_by_reference_buffer<T, dims, target,
                                                    placeholder>;

      queue.submit([&](sycl::handler &h) {
        /** check constant_buffer constructors for different modes
         */
        {
          constexpr auto mode = sycl::access_mode::read;
          verifier::template check<mode>(buffer, r, offset, log, typeName, h);
        }
        /** check common-by-reference semantics
         */
        {
          constexpr auto mode = sycl::access_mode::read;
          semantics_verifier::template check<mode>(buffer, r, offset, log, typeName, h);
        }
        /** dummy kernel as no kernel is required for these checks
         */
        using dummy =
            dummy_functor<kernelName, sycl::target::constant_buffer>;
        h.single_task<dummy>(dummy{});

      });
      queue.wait_and_throw();
    }
  }
};

/** @brief Specialization of buffer_accessor_dims for host_buffer
 */
template <typename T, typename kernelName, size_t dims,
          sycl::access::placeholder placeholder, typename ... allocatorT>
class buffer_accessor_dims<T, kernelName, dims, is_host_buffer::true_t,
                           placeholder, allocatorT...> {
 public:
  static void check(util::logger &log, sycl::queue &queue,
                    const std::string& typeName) {
    using input_type = buffer_accesor_input_data<T, dims>;
    input_type input;

    const auto dataRange = input.getDataRange();
    typename input_type::buffer_t buffer(input.getDataPointer(), dataRange);

    const auto r = input.getRange();
    const auto offset = input.getOffset();

    /** check buffer accessor constructors for host_buffer
     */
    {
      constexpr auto target = sycl::target::host_buffer;
      using verifier =
          check_all_accessor_constructors_buffer<T, dims, target, placeholder>;
      using semantics_verifier =
          check_accessor_common_by_reference_buffer<T, dims, target,
                                                    placeholder>;

      /** check host_buffer constructors for different modes
       */
      {
        constexpr auto mode = sycl::access_mode::read;
        verifier::template check<mode>(buffer, r, offset, log, typeName);
      }
      {
        constexpr auto mode = sycl::access_mode::write;
        verifier::template check<mode>(buffer, r, offset, log, typeName);
      }
      {
        constexpr auto mode = sycl::access_mode::read_write;
        verifier::template check<mode>(buffer, r, offset, log, typeName);
      }
      {
        constexpr auto mode = sycl::access_mode::discard_write;
        verifier::template check<mode>(buffer, r, offset, log, typeName);
      }
      {
        constexpr auto mode = sycl::access_mode::discard_read_write;
        verifier::template check<mode>(buffer, r, offset, log, typeName);
      }
      /** check common-by-reference semantics
       */
      {
        constexpr auto mode = sycl::access_mode::read;
        semantics_verifier::template check<mode>(buffer, r, offset, log, typeName);
      }
    }
  }
};

/** @brief Used to test the buffer accessor combinations for placeholder
 *         device and placeholder constant_buffer
 */
template <typename T, typename kernelName, size_t dims, typename ... allocatorT>
class buffer_accessor_dims<T, kernelName, dims, is_host_buffer::false_t,
                           sycl::access::placeholder::true_t,
                           allocatorT...> {
  static constexpr auto placeholder = sycl::access::placeholder::true_t;
public:
  static void check(util::logger &log, sycl::queue &queue,
                    const std::string& typeName) {
    using input_type = buffer_accesor_input_data<T, dims>;
    input_type input;

    const auto dataRange = input.getDataRange();
    typename input_type::buffer_t buffer(input.getDataPointer(), dataRange);

    const auto r = input.getRange();
    const auto offset = input.getOffset();

    /** check buffer accessor constructors for device
     */
    {
      constexpr auto target = sycl::target::device;
      using verifier =
          check_all_accessor_constructors_buffer<T, dims, target, placeholder>;
      using semantics_verifier =
          check_accessor_common_by_reference_buffer<T, dims, target,
                                                    placeholder>;

      /** check device constructors for different modes
       */
      {
        constexpr auto mode = sycl::access_mode::read;
        verifier::template check<mode>(buffer, r, offset, log, typeName);
      }
      {
        constexpr auto mode = sycl::access_mode::write;
        verifier::template check<mode>(buffer, r, offset, log, typeName);
      }
      {
        constexpr auto mode = sycl::access_mode::read_write;
        verifier::template check<mode>(buffer, r, offset, log, typeName);
      }
      {
        constexpr auto mode = sycl::access_mode::discard_write;
        verifier::template check<mode>(buffer, r, offset, log, typeName);
      }
      {
        constexpr auto mode = sycl::access_mode::discard_read_write;
        verifier::template check<mode>(buffer, r, offset, log, typeName);
      }
      {
        constexpr auto mode = sycl::access_mode::atomic;
        verifier::template check<mode>(buffer, r, offset, log, typeName);
      }
      /** check common-by-reference semantics
       */
      {
        constexpr auto mode = sycl::access_mode::read;
        semantics_verifier::template check<mode>(buffer, r, offset, log, typeName);
      }

      queue.submit([&](sycl::handler &h) {
        /** dummy kernel as no kernel is required for these checks
         */
        using dummy =
            dummy_functor<kernelName, sycl::target::device>;
        h.single_task<dummy>(dummy{});
      });
      queue.wait_and_throw();
    }

    /** check buffer accessor constructors for constant_buffer
     */
    {
      constexpr auto target = sycl::target::constant_buffer;
      using verifier =
          check_all_accessor_constructors_buffer<T, dims, target, placeholder>;
      using semantics_verifier =
          check_accessor_common_by_reference_buffer<T, dims, target,
                                                    placeholder>;

      /** check constant_buffer constructors for different modes
       */
      {
        constexpr auto mode = sycl::access_mode::read;
        verifier::template check<mode>(buffer, r, offset, log, typeName);
      }
      /** check common-by-reference semantics
       */
      {
        constexpr auto mode = sycl::access_mode::read;
        semantics_verifier::template check<mode>(buffer, r, offset, log, typeName);
      }

      queue.submit([&](sycl::handler &h) {
        /** dummy kernel as no kernel is required for these checks
         */
        using dummy =
            dummy_functor<kernelName, sycl::target::constant_buffer>;
        h.single_task<dummy>(dummy{});

      });
      queue.wait_and_throw();
    }
  }
};

/** @brief Run tests for all buffer accessor dimensions
 */
template <typename T, typename kernelName, is_host_buffer isHostBuffer,
          sycl::access::placeholder placeholder,
          typename ... allocatorT, typename ... argsT>
void buffer_accessor_all_dims(argsT&& ... args) {
  buffer_accessor_dims<T, kernelName, 0, isHostBuffer, placeholder,
                       allocatorT...>::check(std::forward<argsT>(args)...);
  buffer_accessor_dims<T, kernelName, 1, isHostBuffer, placeholder,
                       allocatorT...>::check(std::forward<argsT>(args)...);
  buffer_accessor_dims<T, kernelName, 2, isHostBuffer, placeholder,
                       allocatorT...>::check(std::forward<argsT>(args)...);
  buffer_accessor_dims<T, kernelName, 3, isHostBuffer, placeholder,
                       allocatorT...>::check(std::forward<argsT>(args)...);
}

/** @brief Run tests for all non-placeholder buffer accessors
 */
template <typename T, typename /*extensionTag*/, typename kernelName>
class buffer_accessor_type {
public:
  template <typename ... argsT>
  void operator()(argsT&& ... args) {
    constexpr auto placeholder = sycl::access::placeholder::false_t;
    using user_allocator = std::allocator<T>;
    {
      constexpr auto isHostBuffer = is_host_buffer::false_t;

      buffer_accessor_all_dims<T, kernelName, isHostBuffer, placeholder>(
          std::forward<argsT>(args)...);
      buffer_accessor_all_dims<T, kernelName, isHostBuffer, placeholder,
                               user_allocator>(
          std::forward<argsT>(args)...);
    }
    {
      constexpr auto isHostBuffer = is_host_buffer::true_t;

      buffer_accessor_all_dims<T, kernelName, isHostBuffer, placeholder>(
          std::forward<argsT>(args)...);
      buffer_accessor_all_dims<T, kernelName, isHostBuffer, placeholder,
                               user_allocator>(
          std::forward<argsT>(args)...);
    }
  }
};
/** @brief Run tests for all placeholder buffer accessors
 */
template <typename T, typename /*extensionTag*/, typename kernelName>
class buffer_accessor_type_placeholder {
public:
  template <typename ... argsT>
  void operator()(argsT&& ... args) {
    constexpr auto placeholder = sycl::access::placeholder::true_t;
    constexpr auto isHostBuffer = is_host_buffer::false_t;
    using user_allocator = std::allocator<T>;

    buffer_accessor_all_dims<T, kernelName, isHostBuffer, placeholder>(
        std::forward<argsT>(args)...);
    buffer_accessor_all_dims<T, kernelName, isHostBuffer, placeholder,
                             user_allocator>(
        std::forward<argsT>(args)...);
  }
};

}  // namespace accessor_utility__

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_BUFFER_UTILITY_H
