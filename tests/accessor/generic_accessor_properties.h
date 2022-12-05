/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for the properties of generic sycl::accessor
//
*******************************************************************************/
#ifndef SYCL_CTS_GENERIC_ACCESSOR_PROPERTIES_H
#define SYCL_CTS_GENERIC_ACCESSOR_PROPERTIES_H
#include "accessor_common.h"

namespace generic_accessor_properties {
using namespace sycl_cts;
using namespace accessor_tests_common;

constexpr accessor_type AccType = accessor_type::generic_accessor;

template <typename DataT, int Dimension, sycl::access_mode AccessMode,
          sycl::target Target>
void test_constructor_with_no_init(const std::string& type_name,
                                   const std::string& access_mode_name,
                                   const std::string& target_name) {
  const auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
  const auto offset = sycl::id<Dimension>();
  const sycl::property_list prop_list(sycl::no_init);

  auto section_name = get_section_name<Dimension>(
      type_name, access_mode_name, target_name,
      "From buffer constructor with no_init property");

  SECTION(section_name) {
    auto get_acc_functor = [&prop_list](
                               sycl::buffer<DataT, Dimension>& data_buf,
                               sycl::handler& cgh) {
      return sycl::accessor<DataT, Dimension, AccessMode, Target>(data_buf, cgh,
                                                                  prop_list);
    };

    check_no_init_prop<AccType, DataT, Dimension, AccessMode, Target>(
        get_acc_functor, r);
  }

  section_name = get_section_name<Dimension>(
      type_name, access_mode_name, target_name,
      "From buffer and range constructor with no_init property");

  SECTION(section_name) {
    auto get_acc_functor = [&prop_list, r](
                               sycl::buffer<DataT, Dimension>& data_buf,
                               sycl::handler& cgh) {
      return sycl::accessor<DataT, Dimension, AccessMode, Target>(data_buf, cgh,
                                                                  r, prop_list);
    };
    check_no_init_prop<AccType, DataT, Dimension, AccessMode, Target>(
        get_acc_functor, r);
  }

  section_name = get_section_name<Dimension>(
      type_name, access_mode_name, target_name,
      "From buffer,range and offset constructor with no_init property");

  SECTION(section_name) {
    auto get_acc_functor = [&prop_list, r, offset](
                               sycl::buffer<DataT, Dimension>& data_buf,
                               sycl::handler& cgh) {
      return sycl::accessor<DataT, Dimension, AccessMode, Target>(
          data_buf, cgh, r, offset, prop_list);
    };
    check_no_init_prop<AccType, DataT, Dimension, AccessMode, Target>(
        get_acc_functor, r);
  }
}

template <typename DataT, int Dimension, sycl::target Target>
void test_exception(const std::string& type_name,
                    const std::string& target_name) {
  const auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
  const auto offset = sycl::id<Dimension>();
  const sycl::property_list prop_list(sycl::no_init);

  auto section_name = get_section_name<Dimension>(
      type_name, "access_mode::read", target_name,
      "Expecting exception when attempting to construct accessor from buffer "
      "with no_init property and access_mode::read");
  SECTION(section_name) {
    auto construct_acc = [&prop_list](sycl::queue& queue,
                                      sycl::buffer<DataT, Dimension> data_buf) {
      queue
          .submit([&](sycl::handler& cgh) {
            sycl::accessor<DataT, Dimension, sycl::access_mode::read, Target>(
                data_buf, cgh, prop_list);
          })
          .wait();
    };
    check_no_init_prop_exception<AccType, DataT, Dimension, Target>(
        construct_acc, r);
  }

  section_name = get_section_name<Dimension>(
      type_name, "access_mode::read", target_name,
      "Expecting exception when attempting to construct accessor from buffer "
      "and range with no_init property and access_mode::read");
  SECTION(section_name) {
    auto construct_acc = [&prop_list, r](
                             sycl::queue& queue,
                             sycl::buffer<DataT, Dimension> data_buf) {
      queue
          .submit([&](sycl::handler& cgh) {
            sycl::accessor<DataT, Dimension, sycl::access_mode::read, Target>(
                data_buf, cgh, r, prop_list);
          })
          .wait();
    };
    check_no_init_prop_exception<AccType, DataT, Dimension, Target>(
        construct_acc, r);
  }

  section_name = get_section_name<Dimension>(
      type_name, "access_mode::read", target_name,
      "Expecting exception when attempting to construct accessor from "
      "buffer, "
      "range and offset with no_init property and access_mode::read");
  SECTION(section_name) {
    auto construct_acc = [&prop_list, r, offset](
                             sycl::queue& queue,
                             sycl::buffer<DataT, Dimension> data_buf) {
      queue
          .submit([&](sycl::handler& cgh) {
            sycl::accessor<DataT, Dimension, sycl::access_mode::read, Target>(
                data_buf, cgh, r, offset, prop_list);
          })
          .wait();
    };
    check_no_init_prop_exception<AccType, DataT, Dimension, Target>(
        construct_acc, r);
  }
}

template <typename DataT, int Dimension, sycl::access_mode AccessMode,
          sycl::target Target>
void test_property_member_functions_without_no_init(
    const std::string& type_name, const std::string& access_mode_name,
    const std::string& target_name) {
  const auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
  const auto offset = sycl::id<Dimension>();

  {
    auto get_acc_functor = [](
                               sycl::buffer<DataT, Dimension>& data_buf,
                               [[maybe_unused]]sycl::handler& cgh) {
      return sycl::accessor<DataT, Dimension, AccessMode, Target, sycl::access::placeholder::true_t>(data_buf);
    };

    auto section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting false == accessor.has_property<property::no_init>() "
        "for acc constructed with buffer constructor without handler and no_init property");

    SECTION(section_name) {
      check_has_property_member_without_no_init<AccType, DataT, Dimension, AccessMode, Target>(
          get_acc_functor, r);
    }
    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception for call accessor.get_property<property::no_init>() "
        "for acc constructed with buffer constructor without handler and no_init property");

    SECTION(section_name) {
      check_get_property_member_without_no_init<AccType, DataT, Dimension, AccessMode, Target>(
          get_acc_functor, r);
    }
  }
  {
    auto get_acc_functor = [](sycl::buffer<DataT, Dimension>& data_buf,
                              sycl::handler& cgh) {
      return sycl::accessor<DataT, Dimension, AccessMode, Target>(data_buf,
                                                                  cgh);
    };

    auto section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting false == accessor.has_property<property::no_init>() "
        "for acc constructed with buffer constructor without no_init property");

    SECTION(section_name) {
      check_has_property_member_without_no_init<AccType, DataT, Dimension,
                                                AccessMode, Target>(
          get_acc_functor, r);
    }
    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception for call "
        "accessor.get_property<property::no_init>() "
        "for acc constructed with buffer constructor without no_init property");

    SECTION(section_name) {
      check_get_property_member_without_no_init<AccType, DataT, Dimension,
                                                AccessMode, Target>(
          get_acc_functor, r);
    }
  }
  {
    auto get_acc_functor = [r](sycl::buffer<DataT, Dimension>& data_buf,
                               sycl::handler& cgh) {
      return sycl::accessor<DataT, Dimension, AccessMode, Target>(data_buf, cgh,
                                                                  r);
    };
    auto section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting false == accessor.has_property<property::no_init>() "
        "for acc constructed with buffer and range constructor without no_init "
        "property");

    SECTION(section_name) {
      check_has_property_member_without_no_init<AccType, DataT, Dimension,
                                                AccessMode, Target>(
          get_acc_functor, r);
    }
    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception for call "
        "accessor.get_property<property::no_init>() "
        "for acc constructed with buffer and range constructor without no_init "
        "property");

    SECTION(section_name) {
      check_get_property_member_without_no_init<AccType, DataT, Dimension,
                                                AccessMode, Target>(
          get_acc_functor, r);
    }
  }
  {
    auto get_acc_functor = [r, offset](sycl::buffer<DataT, Dimension>& data_buf,
                                       sycl::handler& cgh) {
      return sycl::accessor<DataT, Dimension, AccessMode, Target>(data_buf, cgh,
                                                                  r, offset);
    };
    auto section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting false == accessor.has_property<property::no_init>() "
        "for acc constructed with buffer,range and offset constructor without "
        "no_init property");

    SECTION(section_name) {
      check_has_property_member_without_no_init<AccType, DataT, Dimension,
                                                AccessMode, Target>(
          get_acc_functor, r);
    }
    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Expecting exception for call "
        "accessor.get_property<property::no_init>() "
        "for acc constructed with buffer,range and offset constructor without "
        "no_init property");

    SECTION(section_name) {
      check_get_property_member_without_no_init<AccType, DataT, Dimension,
                                                AccessMode, Target>(
          get_acc_functor, r);
    }
  }
}

template <typename DataT, int Dimension, sycl::access_mode AccessMode,
          sycl::target Target>
void test_property_member_functions(const std::string& type_name,
                                    const std::string& access_mode_name,
                                    const std::string& target_name) {
  const auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
  const auto offset = sycl::id<Dimension>();
  const sycl::property_list prop_list(sycl::no_init);

  const auto construct_acc =
      [&prop_list](sycl::buffer<DataT, Dimension> data_buf) {
        return sycl::accessor<DataT, Dimension, AccessMode, Target>(data_buf,
                                                                    prop_list);
      };

  auto section_name =
      get_section_name<Dimension>(type_name, access_mode_name, target_name,
                                  "has_property member function invocation");
  SECTION(section_name) {
    check_has_property_member_func<DataT, Dimension, sycl::property::no_init>(
        construct_acc, r);
  }

  section_name =
      get_section_name<Dimension>(type_name, access_mode_name, target_name,
                                  "get_property member function invocation");
  SECTION(section_name) {
    check_get_property_member_func<DataT, Dimension, sycl::property::no_init>(
        construct_acc, r);
  }
}

template <typename T, typename AccessModeT, typename TargetT,
          typename DimensionT>
class run_tests_properties {
  static constexpr sycl::access_mode AccessMode = AccessModeT::value;
  static constexpr int Dimension = DimensionT::value;
  static constexpr sycl::target Target = TargetT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name,
                  const std::string& target_name) {
    // In order not to run again with same parameters
    if constexpr (AccessMode != sycl::access_mode::read) {
      test_constructor_with_no_init<T, Dimension, AccessMode, Target>(
          type_name, access_mode_name, target_name);

      test_property_member_functions<T, Dimension, AccessMode, Target>(
          type_name, access_mode_name, target_name);
      test_property_member_functions_without_no_init<T, Dimension, AccessMode,
                                                     Target>(
          type_name, access_mode_name, target_name);
    } else {
      test_exception<T, Dimension, Target>(type_name, target_name);
    }
  }
};

template <typename T>
class run_generic_properties_tests {
 public:
  void operator()(const std::string& type_name) {
    // Type packs instances have to be const, otherwise for_all_combination
    // will not compile
    const auto access_modes = get_access_modes();
    const auto dimensions = get_dimensions();
    const auto targets = get_targets();

    // To handle cases when class was called from functions
    // like for_all_types_vectors_marray or for_all_device_copyable_std_containers.
    // This will wrap string with type T to string with container<T> if T is
    // an array or other kind of container.
    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_tests_properties, T>(access_modes, targets,
                                                  dimensions, actual_type_name);
  }
};
}  // namespace generic_accessor_properties

#endif  // SYCL_CTS_GENERIC_ACCESSOR_PROPERTIES_H
