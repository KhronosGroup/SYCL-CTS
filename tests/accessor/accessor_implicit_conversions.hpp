/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for SYCL accessor implicit conversions
//  The main idea is to:
//  - check if source accessor is convertible to destination accessor
//  - and re-use generic checks for accessor constructors to verify data remains
//    valid
//
*******************************************************************************/
#ifndef SYCL_CTS_ACCESSOR_IMPLICIT_CONVERSIONS_H
#define SYCL_CTS_ACCESSOR_IMPLICIT_CONVERSIONS_H

#include "accessor_common.h"

#include <type_traits>

namespace accessor_implicit_conversions {
using namespace accessor_tests_common;

namespace detail {
template <typename SrcAccT, typename DstAccT>
struct invoke_implicit_conversion {
  auto operator()(const SrcAccT& srcAcc) const {
    // We might safely go with returning `srcAcc` parameter as `DstAccT` type,
    // still better to make it explicit
    const DstAccT result = srcAcc;
    return result;
  }
};

template <typename AccT>
struct avoid_implicit_conversion {
  const AccT& operator()(const AccT& srcAcc) const {
    // Just forward parameter further with no copy/move constructor call
    return srcAcc;
  }
};

/**
 * @brief Provide section name builder prototype
 */
template <accessor_type AccType, int Dimension>
struct section_name_prototype {
  static auto get(const std::string& type_name) {
    return section_name("implicit conversion for " +
                        Catch::StringMaker<accessor_type>::convert(AccType))
        .with("T", type_name)
        .with("dim", Dimension);
  }
};
}  // namespace detail

/**
 * @brief Validates implicit conversion within command
 */
template <typename DataT, typename DimensionT, typename TargetT>
class check_conversion_generic {
  // TODO: refactor into `check_conversion<AccessorT, ...>` with separate
  // specialization for local accesor once `accessor_tests_common` refactoring
  // is done
  static constexpr auto AccType = accessor_type::generic_accessor;
  static constexpr sycl::target Target = TargetT::value;
  static constexpr int Dimension = DimensionT::value;

  static_assert(!std::is_const_v<DataT>, "No need to pass const type here");
  static_assert((Target == sycl::target::host_task) ||
                    (Target == sycl::target::device),
                "Unsupported target");

  template <typename SrcDataT, sycl::access_mode SrcAccessMode,
            typename DstDataT, sycl::access_mode DstAccessMode>
  void run_check() const {
    constexpr int BufferDimension = (Dimension == 0) ? 1 : Dimension;
    using src_accessor_t =
        sycl::accessor<SrcDataT, Dimension, SrcAccessMode, Target>;
    using dst_accessor_t =
        sycl::accessor<DstDataT, Dimension, DstAccessMode, Target>;
    using src_buffer_t = sycl::buffer<SrcDataT, BufferDimension>;

    constexpr bool has_implicit_conversion_available =
        std::is_convertible_v<src_accessor_t, dst_accessor_t>;

    // We don't want to fail compilation of entire test every time conversion
    // is not available for some of accessor template instantiations
    // So first we check precodition, and only in case of conversion
    // availability we try to go on with functionality check
    REQUIRE(has_implicit_conversion_available);
    // Execution would never pass the 'REQUIRE' call in case there is no
    // implicit conversion available; still it's better to switch off only
    // the minimal part of compilation path for better test support
    using invoke_implicit_conversion_t = std::conditional_t<
        has_implicit_conversion_available,
        detail::invoke_implicit_conversion<src_accessor_t, dst_accessor_t>,
        detail::avoid_implicit_conversion<src_accessor_t>>;

    // We should create source accessor on the host side
    const auto get_acc_functor = [](src_buffer_t& data_buf,
                                    sycl::handler& cgh) {
      const src_accessor_t src_acc(data_buf, cgh);
      return src_acc;
    };
    // Implicit conversion should be verified within command
    const invoke_implicit_conversion_t modify_acc_functor;

    if constexpr (Dimension == 0) {
      check_zero_dim_constructor<AccType, SrcDataT, DstAccessMode, Target>(
          get_acc_functor, modify_acc_functor);
    } else {
      const auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
      check_common_constructor<AccType, SrcDataT, Dimension, DstAccessMode,
                               Target>(r, get_acc_functor, modify_acc_functor);
    }
  }

 public:
  void operator()(const std::string& type_name,
                  const std::string& target_name) const {
    const auto make_section_name = [&](sycl::access_mode SrcAccessMode,
                                       sycl::access_mode DstAccessMode,
                                       const std::string& details) {
      return detail::section_name_prototype<AccType, Dimension>::get(type_name)
          .with("target", Target)
          .with("source access", SrcAccessMode)
          .with("destination access", DstAccessMode)
          .with("case", details)
          .create();
    };

    // From read-only accessor to read-only accessor
    {
      constexpr auto AccessMode = sycl::access_mode::read;

      SECTION((
          make_section_name(AccessMode, AccessMode, "from 'T' to 'const T'"))) {
        using SrcDataT = DataT;
        using DstDataT = const DataT;
        run_check<SrcDataT, AccessMode, DstDataT, AccessMode>();
      }
      SECTION((
          make_section_name(AccessMode, AccessMode, "from 'const T' to 'T'"))) {
        using SrcDataT = const DataT;
        using DstDataT = DataT;
        run_check<SrcDataT, AccessMode, DstDataT, AccessMode>();
      }
    }
    // From read-write accessor to read-only accessor
    {
      constexpr auto SrcAccessMode = sycl::access_mode::read_write;
      constexpr auto DstAccessMode = sycl::access_mode::read;

      SECTION((make_section_name(SrcAccessMode, DstAccessMode,
                                 "from 'T' to 'const T'"))) {
        using SrcDataT = DataT;
        using DstDataT = const DataT;
        run_check<SrcDataT, SrcAccessMode, DstDataT, DstAccessMode>();
      }
      SECTION((make_section_name(SrcAccessMode, DstAccessMode,
                                 "from 'const T' to 'T'"))) {
        using SrcDataT = const DataT;
        using DstDataT = DataT;
        run_check<SrcDataT, SrcAccessMode, DstDataT, DstAccessMode>();
      }
    }
  }
};

/**
 * @brief Validates implicit conversion for local accessor
 */
template <typename DataT, typename DimensionT>
class check_conversion_local {
  static constexpr auto AccType = accessor_type::local_accessor;
  static constexpr int Dimension = DimensionT::value;

  static_assert(!std::is_const_v<DataT>, "No need to pass const type here");

  template <typename SrcDataT, typename DstDataT>
  inline void run_check() const {
    // Workarounds to use generic algorithm
    // TODO: Refactor `check_common_constructor` to make it more generic
    constexpr auto Target = sycl::target::device;
    constexpr auto DstAccessMode = std::is_const_v<DstDataT>
                                       ? sycl::access_mode::read
                                       : sycl::access_mode::read_write;
    constexpr int BufferDimension = (Dimension == 0) ? 1 : Dimension;
    using src_buffer_t = sycl::buffer<SrcDataT, BufferDimension>;
    using src_accessor_t = sycl::local_accessor<SrcDataT, Dimension>;
    using dst_accessor_t = sycl::local_accessor<DstDataT, Dimension>;

    constexpr bool has_implicit_conversion_available =
        std::is_convertible_v<src_accessor_t, dst_accessor_t>;

    // Ensure we don't go further in runtime if conversion is unavailable
    REQUIRE(has_implicit_conversion_available);

    // Still go on with the further compilation to make sure test compiles
    using invoke_implicit_conversion_t = std::conditional_t<
        has_implicit_conversion_available,
        detail::invoke_implicit_conversion<src_accessor_t, dst_accessor_t>,
        detail::avoid_implicit_conversion<src_accessor_t>>;

    // Implicit conversion should be verified within command
    const invoke_implicit_conversion_t modify_acc_functor;

    if constexpr (Dimension == 0) {
      const auto get_acc_functor = [](src_buffer_t&, sycl::handler& cgh) {
        const src_accessor_t src_acc(cgh);
        return src_acc;
      };
      check_zero_dim_constructor<AccType, SrcDataT, DstAccessMode, Target>(
          get_acc_functor, modify_acc_functor);
    } else {
      const auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
      const auto get_acc_functor = [=](src_buffer_t&, sycl::handler& cgh) {
        const src_accessor_t src_acc(r, cgh);
        return src_acc;
      };
      check_common_constructor<AccType, SrcDataT, Dimension, DstAccessMode,
                               Target>(r, get_acc_functor, modify_acc_functor);
    }
  }

 public:
  void operator()(const std::string& type_name) const {
    const auto section_name =
        detail::section_name_prototype<AccType, Dimension>::get(type_name)
            .with("case", "conversion from 'T' to 'const T'")
            .create();
    SECTION(section_name) { run_check<DataT, const DataT>(); }
  }
};

/**
 * @brief Validates implicit conversion for host accessor
 */
template <typename DataT, typename DimensionT>
class check_conversion_host {
  static constexpr auto AccType = accessor_type::host_accessor;
  static constexpr int Dimension = DimensionT::value;

  static_assert(!std::is_const_v<DataT>, "No need to pass const type here");

  template <typename SrcDataT, sycl::access_mode SrcAccessMode,
            typename DstDataT, sycl::access_mode DstAccessMode>
  void run_check() const {
    constexpr int BufferDimension = (Dimension == 0) ? 1 : Dimension;
    using src_accessor_t =
        sycl::host_accessor<SrcDataT, Dimension, SrcAccessMode>;
    using dst_accessor_t =
        sycl::host_accessor<DstDataT, Dimension, DstAccessMode>;
    using src_buffer_t = sycl::buffer<SrcDataT, BufferDimension>;

    constexpr bool has_implicit_conversion_available =
        std::is_convertible_v<src_accessor_t, dst_accessor_t>;

    // We don't want to fail compilation of entire test every time conversion
    // is not available for some of accessor template instantiations
    // So first we check precodition, and only in case of conversion
    // availability we try to go on with functionality check
    REQUIRE(has_implicit_conversion_available);
    // Execution would never pass the 'REQUIRE' call in case there is no
    // implicit conversion available; still it's better to switch off only
    // the minimal part of compilation path for better test support
    using invoke_implicit_conversion_t = std::conditional_t<
        has_implicit_conversion_available,
        detail::invoke_implicit_conversion<src_accessor_t, dst_accessor_t>,
        detail::avoid_implicit_conversion<src_accessor_t>>;

    // We should create source accessor on the host side
    const auto get_acc_functor = [](src_buffer_t& data_buf) {
      const src_accessor_t src_acc(data_buf);
      return src_acc;
    };
    // Implicit conversion should be verified within command
    const invoke_implicit_conversion_t modify_acc_functor;

    if constexpr (Dimension == 0) {
      check_zero_dim_constructor<AccType, SrcDataT, DstAccessMode>(
          get_acc_functor, modify_acc_functor);
    } else {
      const auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
      check_common_constructor<AccType, SrcDataT, Dimension, DstAccessMode>(
          r, get_acc_functor, modify_acc_functor);
    }
  }

 public:
  void operator()(const std::string& type_name) const {
    const auto make_section_name = [&](sycl::access_mode SrcAccessMode,
                                       sycl::access_mode DstAccessMode,
                                       const std::string& details) {
      return detail::section_name_prototype<AccType, Dimension>::get(type_name)
          .with("source access", SrcAccessMode)
          .with("destination access", DstAccessMode)
          .with("case", details)
          .create();
    };

    // From read-only accessor to read-only accessor
    {
      constexpr auto AccessMode = sycl::access_mode::read;

      SECTION((
          make_section_name(AccessMode, AccessMode, "from 'T' to 'const T'"))) {
        using SrcDataT = DataT;
        using DstDataT = const DataT;
        run_check<SrcDataT, AccessMode, DstDataT, AccessMode>();
      }
      SECTION((
          make_section_name(AccessMode, AccessMode, "from 'const T' to 'T'"))) {
        using SrcDataT = const DataT;
        using DstDataT = DataT;
        run_check<SrcDataT, AccessMode, DstDataT, AccessMode>();
      }
    }
    // From read-write accessor to read-only accessor
    {
      constexpr auto SrcAccessMode = sycl::access_mode::read_write;
      constexpr auto DstAccessMode = sycl::access_mode::read;

      SECTION((make_section_name(SrcAccessMode, DstAccessMode,
                                 "from 'T' to 'const T'"))) {
        using SrcDataT = DataT;
        using DstDataT = const DataT;
        run_check<SrcDataT, SrcAccessMode, DstDataT, DstAccessMode>();
      }
      SECTION((make_section_name(SrcAccessMode, DstAccessMode,
                                 "from 'const T' to 'T'"))) {
        using SrcDataT = const DataT;
        using DstDataT = DataT;
        run_check<SrcDataT, SrcAccessMode, DstDataT, DstAccessMode>();
      }
    }
  }
};

/**
 * @brief Run tests for generic sycl::accessor
 * @detail A wrapper around for_all_combinations call to make possible extended
 *         type coverage - e.g. vectors and marrays
 */
template <typename DataT>
struct run_test_generic {
  void operator()(const std::string& type_name) {
    // TODO: make for_all_combinations recognize non-const type packs
    const auto dimensions = get_dimensions();
    const auto targets = get_targets();

    for_all_combinations<check_conversion_generic, DataT>(dimensions, targets,
                                                          type_name);
  }
};

/**
 * @brief Run tests for sycl::local_accessor
 */
template <typename DataT>
struct run_test_local {
  void operator()(const std::string& type_name) {
    const auto dimensions = get_dimensions();

    for_all_combinations<check_conversion_local, DataT>(dimensions, type_name);
  }
};

/**
 * @brief Run tests for sycl::host_accessor
 */
template <typename DataT>
struct run_test_host {
  void operator()(const std::string& type_name) {
    const auto dimensions = get_dimensions();

    for_all_combinations<check_conversion_host, DataT>(dimensions, type_name);
  }
};

}  // namespace accessor_implicit_conversions
#endif  // SYCL_CTS_ACCESSOR_IMPLICIT_CONVERSIONS_H
