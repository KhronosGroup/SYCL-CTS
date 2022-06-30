/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides code for tests for multi_ptr implicit conversions
//
*******************************************************************************/
#ifndef __SYCLCTS_TESTS_MULTI_PTR_IMPLICIT_CONVERSIONS_H
#define __SYCLCTS_TESTS_MULTI_PTR_IMPLICIT_CONVERSIONS_H

#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

namespace multi_ptr_implicit_conversions {

namespace detail {

/**
 * @brief Functor to invoke implicit conversion
 * @tparam SrcAccT Source accessor type, enforced to be stated explicitly
 * @tparam DstAccT Destination accessor type
 * @param src_acc Instance of a source accessor type
 */
template <typename SrcAccT, typename DstAccT>
struct invoke_implicit_conversion {
  auto operator()(const SrcAccT &src_acc) const {
    // We might safely go with returning `src_acc` parameter as `DstAccT` type,
    // still better to make it explicit
    const DstAccT result = src_acc;
    return result;
  }
};

/**
 * @brief Stub functor to use if no implicit conversion is available
 * @param src_acc Reference to the source accessor instance
 * @returns The same input reference with no side effects
 * @details Invocation signature is aligned with `invoke_implicit_conversion`
 */
template <typename AccT>
struct avoid_implicit_conversion {
  const AccT &operator()(const AccT &src_acc) const {
    // Just forward parameter further with no copy/move constructor call
    return src_acc;
  }
};
}  // namespace detail

constexpr int expected_val = 42;
template <typename T, typename AddrSpaceT, typename IsDecoratedT>
class run_implicit_convert_tests {
  using namespace detail;
  static constexpr sycl::access::address_space address_space =
      AddrSpaceT::value;
  static constexpr sycl::access::decorated decorated = IsDecoratedT::value;

  template <typename from_multi_ptr_t, typename dest_multi_ptr_t>
  void preform_implicit_conversion_test() {
    auto queue = sycl_cts::util::get_cts_object::queue();
    T value = user_def_types::get_init_value_helper<T>(expected_val);
    bool res = false;

    constexpr bool has_implicit_conversion_available =
        std::is_convertible_v<from_multi_ptr_t, dest_multi_ptr_t>;

    // We don't want to fail compilation of entire test every time
    // conversion is not available for some of accessor template instantiations
    // So first we check precondition, and only in case of conversion
    // availability we try to go on with functionality check
    {
      INFO("Implicit conversion is not avaliable.");
      REQUIRE(has_implicit_conversion_available);
    }
    // Execution would never pass the 'REQUIRE' call in case there is no
    // implicit conversion available; still it's better to switch off only
    // the minimal part of compilation path for better test support

    using invoke_conversion_t = std::conditional_t<
        has_implicit_conversion_available,
        invoke_implicit_conversion<from_multi_ptr_t, dest_multi_ptr_t>,
        avoid_implicit_conversion<from_multi_ptr_t>>;

    {
      sycl::range r(1);
      sycl::buffer<bool> res_buf(&res, r);
      sycl::buffer<T> expected_val_buffer(&value, r);
      queue.submit([&](sycl::handler &cgh) {
        auto res_acc =
            res_buf.template get_access<sycl::access_mode::write>(cgh);
        auto expected_val_acc =
            expected_val_buffer.template get_access<sycl::access_mode::read>(
                cgh);
        cgh.single_task([=] {
          from_multi_ptr_t mptr_from(expected_val_acc);

          // From cppreference.com
          //  Implicit conversions are performed whenever an
          //  expression of some type T1 is used in context
          //  that does not accept that type, but accepts some
          //  other type T2; in particular:
          //  <...>
          //  - when initializing a new object of type T2, including return
          //  statement in a function returning T2;
          //  <...>
          dest_multi_ptr_t mptr_dest = invoke_conversion_t{}(mptr_from);

          // for cases, when dest_multi_ptr_t equals to multi_ptr<void>
          T value_dest = static_cast<T>(*(mptr_dest.get()));

          res_acc[0] = (value_dest == expected_val);
        });
      });
    }
    CHECK(res);
  }

 public:
  void operator()(const std::string &type_name,
                  const std::string &address_space_str,
                  const std::string &is_decorated_str) {
    SECTION(section_name("Verifying implicit conversion from "
                         "multi_ptr<T,address_space,decorated>")
                .with("T", type_name)
                .with("address_space", address_space_str)
                .with("decorated", is_decorated_str)
                .create()) {
      SECTION("Implicit conversion from multi_ptr<T> to multi_ptr<void>") {
        using from_multi_ptr_t = sycl::multi_ptr<T, address_space, decorated>;

        using dest_multi_ptr_not_decorated_t =
            sycl::multi_ptr<void, address_space, sycl::access::decorated::no>;
        using dest_multi_ptr_decorated_t =
            sycl::multi_ptr<void, address_space, sycl::access::decorated::yes>;

        SECTION("Conversion to multi_ptr<void, decorated::yes>") {
          preform_implicit_conversion_test<from_multi_ptr_t,
                                           dest_multi_ptr_decorated_t>();
        }
        SECTION("Conversion to multi_ptr<void, decorated::no>") {
          preform_implicit_conversion_test<from_multi_ptr_t,
                                           dest_multi_ptr_not_decorated_t>();
        }
      }
      SECTION(
          "Implicit conversion from multi_ptr<const T> to multi_ptr<const "
          "void>") {
        using from_multi_ptr_t =
            sycl::multi_ptr<const T, address_space, decorated>;

        using dest_multi_ptr_not_decorated_t =
            sycl::multi_ptr<const void, address_space,
                            sycl::access::decorated::no>;
        using dest_multi_ptr_decorated_t =
            sycl::multi_ptr<const void, address_space,
                            sycl::access::decorated::yes>;

        SECTION("Conversion to multi_ptr<const void, decorated::yes>") {
          preform_implicit_conversion_test<from_multi_ptr_t,
                                           dest_multi_ptr_decorated_t>();
        }
        SECTION("Conversion to multi_ptr<const void, decorated::no>") {
          preform_implicit_conversion_test<from_multi_ptr_t,
                                           dest_multi_ptr_not_decorated_t>();
        }
      }

      SECTION("Implicit conversion from multi_ptr<T> to multi_ptr<const T>") {
        using from_multi_ptr_t = sycl::multi_ptr<T, address_space, decorated>;

        using dest_multi_ptr_not_decorated_t =
            sycl::multi_ptr<const T, address_space,
                            sycl::access::decorated::no>;
        using dest_multi_ptr_decorated_t =
            sycl::multi_ptr<const T, address_space,
                            sycl::access::decorated::yes>;

        preform_implicit_conversion_test<from_multi_ptr_t, dest_multi_ptr_t>();
        SECTION("Conversion to multi_ptr<const T, decorated::yes>") {
          preform_implicit_conversion_test<from_multi_ptr_t,
                                           dest_multi_ptr_decorated_t>();
        }
        SECTION("Conversion to multi_ptr<const T, decorated::no>") {
          preform_implicit_conversion_test<from_multi_ptr_t,
                                           dest_multi_ptr_not_decorated_t>();
        }
      }

      SECTION("Implicit conversion from multi_ptr<T> to multi_ptr<T>") {
        using from_multi_ptr_t = sycl::multi_ptr<T, address_space, decorated>;

        using dest_multi_ptr_not_decorated_t =
            sycl::multi_ptr<const T, address_space,
                            sycl::access::decorated::no>;
        using dest_multi_ptr_decorated_t =
            sycl::multi_ptr<const T, address_space,
                            sycl::access::decorated::yes>;

        if constexpr (from_multi_ptr_t::is_decorated) {
          SECTION(
              "Conversion from multi_ptr<T, decorated::yes> to multi_ptr<T, "
              "decorated::no>") {
            preform_implicit_conversion_test<from_multi_ptr_t,
                                             dest_multi_ptr_not_decorated_t>();
          }
        } else {
          SECTION(
              "Conversion from multi_ptr<T, decorated::no> to multi_ptr<T, "
              "decorated::yes>") {
            preform_implicit_conversion_test<from_multi_ptr_t,
                                             dest_multi_ptr_decorated_t>();
          }
        }
      }
    }
  }
};

template <typename T>
class check_multi_ptr_implicit_convert_for_type {
 public:
  void operator()(const std::string &type_name) {
    using multi_ptr_common;

    const auto address_spaces_pack = get_address_spaces();
    const auto is_decorated_pack = get_decorated();

    for_all_combinations<run_implicit_convert_tests, T>(
        address_spaces_pack, is_decorated_pack, type_name);
  }
};
}  // namespace multi_ptr_implicit_conversions

#endif  // __SYCLCTS_TESTS_MULTI_PTR_IMPLICIT_CONVERSIONS_H
