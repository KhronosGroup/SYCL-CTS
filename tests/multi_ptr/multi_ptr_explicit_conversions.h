/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides code for tests for multi_ptr explicit conversions
//
*******************************************************************************/
#ifndef __SYCLCTS_TESTS_MULTI_PTR_EXPLICIT_CONVERSIONS_H
#define __SYCLCTS_TESTS_MULTI_PTR_EXPLICIT_CONVERSIONS_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

namespace multi_ptr_explicit_conversions {

constexpr int expected_val = 42;
template <typename T, typename AddrSpaceT, typename IsDecorated>
class run_explicit_convert_tests {
  static constexpr sycl::access::address_space target_space = AddrSpaceT::value;
  static constexpr sycl::access::decorated decorated = IsDecorated::value;
  using multi_ptr_t =
      sycl::multi_ptr<T, sycl::access::address_space::generic_space, decorated>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &target_address_space_name,
                  const std::string &is_decorated_name) {
    auto queue = sycl_cts::util::get_cts_object::queue();
    T value = user_def_types::get_init_value_helper<T>(expected_val);
    auto r = sycl::range(1);
    SECTION(
        section_name("Check multi_ptr<T, target_address_space, IsDecorated>()")
            .with("T", type_name)
            .with("target address_space", target_address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      bool res = false;
      {
        sycl::buffer<bool> res_buf(&res, sycl::range<1>(1));
        sycl::buffer<T> val_buffer(&value, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto res_acc =
              res_buf.template get_access<sycl::access_mode::write>(cgh);
          if constexpr (target_space ==
                        sycl::access::address_space::global_space) {
            auto val_acc =
                val_buffer.template get_access<sycl::access_mode::read>(cgh);
            cgh.single_task([=] {
              multi_ptr_t mptr_in(val_acc);
              auto mptr_out =
                  sycl::multi_ptr<T, target_space, decorated>(mptr_in);

              // Check that second mptr has the same value as first mptr
              res_acc[0] = *(mptr_out.get_raw()) == val_acc[0];
            });
          } else {
            sycl::local_accessor<T> local_acc(r, cgh);
            cgh.parallel_for(
                sycl::nd_range<1>(r, r), [=](sycl::nd_item<1> item) {
                  if constexpr (target_space ==
                                sycl::access::address_space::local_space) {
                    auto ref = local_acc[0];
                    value_operations::assign(ref, expected_val);
                    multi_ptr_t mptr_in(local_acc);

                    auto mptr_out =
                        sycl::multi_ptr<T, target_space, decorated>(mptr_in);
                    res_acc[0] = (*(mptr_out.get()) == ref);
                  } else {
                    T private_val =
                        user_def_types::get_init_value_helper<T>(expected_val);

                    multi_ptr_t mptr_in = sycl::address_space_cast<
                        sycl::access::address_space::generic_space, decorated,
                        T>(&private_val);

                    auto mptr_out =
                        sycl::multi_ptr<T, target_space, decorated>(mptr_in);
                    res_acc[0] = *(mptr_out.get_raw()) == private_val;
                  }
                });
          }
        });
      }
      CHECK(res);
    }

    SECTION(section_name("Check multi_ptr<const T, target_address_space, "
                         "IsDecorated>() const")
                .with("T", type_name)
                .with("target address_space", target_address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      bool res = false;
      {
        sycl::buffer<bool> res_buf(&res, sycl::range<1>(1));
        sycl::buffer<T> val_buffer(&value, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto res_acc =
              res_buf.template get_access<sycl::access_mode::write>(cgh);
          if constexpr (target_space ==
                        sycl::access::address_space::global_space) {
            auto val_acc =
                val_buffer.template get_access<sycl::access_mode::read>(cgh);
            cgh.single_task([=] {
              const multi_ptr_t mptr_in(val_acc);
              auto mptr_out =
                  sycl::multi_ptr<const T, target_space, decorated>(mptr_in);

              // Check that second mptr has the same value as first mptr
              res_acc[0] = *(mptr_out.get_raw()) == val_acc[0];
            });
          } else {
            sycl::local_accessor<T> local_acc(r, cgh);
            cgh.parallel_for(sycl::nd_range<1>(r, r), [=](sycl::nd_item<1>
                                                              item) {
              if constexpr (target_space ==
                            sycl::access::address_space::local_space) {
                auto ref = local_acc[0];
                value_operations::assign(ref, expected_val);
                const multi_ptr_t mptr_in(local_acc);

                auto mptr_out =
                    sycl::multi_ptr<const T, target_space, decorated>(mptr_in);
                res_acc[0] = (*(mptr_out.get()) == ref);
              } else {
                T private_val =
                    user_def_types::get_init_value_helper<T>(expected_val);

                const multi_ptr_t mptr_in = sycl::address_space_cast<
                    sycl::access::address_space::generic_space, decorated, T>(
                    &private_val);

                auto mptr_out =
                    sycl::multi_ptr<const T, target_space, decorated>(mptr_in);
                res_acc[0] = *(mptr_out.get_raw()) == private_val;
              }
            });
          }
        });
      }
      CHECK(res);
    }
  }
};

template <typename T>
bool check_pointer_aliases(const std::string &type_name) {
  SECTION(section_name("Check explicit pointer aliases")
              .with("T", type_name)
              .create()) {
    {
      INFO("raw_global_ptr");
      STATIC_CHECK(std::is_same_v<
                   sycl::raw_global_ptr<T>,
                   sycl::multi_ptr<T, sycl::access::address_space::global_space,
                                   sycl::access::decorated::no>>);
    }
    {
      INFO("raw_local_ptr");
      STATIC_CHECK(std::is_same_v<
                   sycl::raw_local_ptr<T>,
                   sycl::multi_ptr<T, sycl::access::address_space::local_space,
                                   sycl::access::decorated::no>>);
    }
    {
      INFO("raw_private_ptr");
      STATIC_CHECK(
          std::is_same_v<
              sycl::raw_private_ptr<T>,
              sycl::multi_ptr<T, sycl::access::address_space::private_space,
                              sycl::access::decorated::no>>);
    }
    {
      INFO("decorated_global_ptr");
      STATIC_CHECK(std::is_same_v<
                   sycl::decorated_global_ptr<T>,
                   sycl::multi_ptr<T, sycl::access::address_space::global_space,
                                   sycl::access::decorated::yes>>);
    }
    {
      INFO("decorated_local_ptr");
      STATIC_CHECK(std::is_same_v<
                   sycl::decorated_local_ptr<T>,
                   sycl::multi_ptr<T, sycl::access::address_space::local_space,
                                   sycl::access::decorated::yes>>);
    }
    {
      INFO("decorated_private_ptr");
      STATIC_CHECK(
          std::is_same_v<
              sycl::decorated_private_ptr<T>,
              sycl::multi_ptr<T, sycl::access::address_space::private_space,
                              sycl::access::decorated::yes>>);
    }
  }
}

template <typename T>
class check_multi_ptr_explicit_convert_for_type {
 public:
  void operator()(const std::string &type_name) {
    check_pointer_aliases<T>(type_name);

    const auto target_address_spaces = value_pack<
        sycl::access::address_space, sycl::access::address_space::global_space,
        sycl::access::address_space::local_space,
        sycl::access::address_space::private_space>::generate_named();

    const auto is_decorated = multi_ptr_common::get_decorated();

    for_all_combinations<run_explicit_convert_tests, T>(
        target_address_spaces, is_decorated, type_name);
  }
};

}  // namespace multi_ptr_explicit_conversions

#endif  // __SYCLCTS_TESTS_MULTI_PTR_EXPLICIT_CONVERSIONS_H
