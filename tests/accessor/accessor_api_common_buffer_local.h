/*************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
//  This file is a common header for implementing buffer and local accessor
//  tests
//
**************************************************************************/
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_COMMON_BUFFER_LOCAL_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_COMMON_BUFFER_LOCAL_H

#include "../common/common.h"
#include "accessor_api_utility.h"

#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

/** explicit pointer type
*/
template <typename T, sycl::target target>
struct explicit_pointer;

/** explicit pointer type (specialization for local)
*/
template <typename T>
struct explicit_pointer<T, sycl::target::local> {
  using type = sycl::local_ptr<T>;
};

/** explicit pointer type (specialization for device)
*/
template <typename T>
struct explicit_pointer<T, sycl::target::device> {
  using type = sycl::global_ptr<T>;
};

/** explicit pointer type (specialization for constant_buffer)
*/
template <typename T>
struct explicit_pointer<T, sycl::target::constant_buffer> {
  using type = sycl::constant_ptr<T>;
};

/** explicit pointer type (specialization for host_buffer)
*/
template <typename T>
struct explicit_pointer<T, sycl::target::host_buffer> {
  using type = T *;
};

/** explicit pointer alias
 */
template <typename T, sycl::target target>
using explicit_pointer_t = typename explicit_pointer<T, target>::type;

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

namespace accessor_utility {

template <typename T, sycl::target target, sycl::access_mode mode>
struct buffer_accessor_value {
  using dataT = T;
  using elemT = T;
  using elemRefT = T&;

  static inline dataT& get(elemRefT value) {
    return std::forward<dataT&>(value);
  }
  static inline void set(elemRefT dst, const dataT& value) {
    dst = value;
  }
};

template <typename T, sycl::target target>
struct buffer_accessor_value<T, target, sycl::access_mode::read> {
  using dataT = T;
  using elemT = T;

  static inline dataT get(elemT value) {
    return std::forward<dataT>(value);
  }
};

template <typename T, sycl::target target>
struct buffer_accessor_value<T, target, sycl::access_mode::atomic> {
  static constexpr auto addressSpace =
      (target == sycl::target::local) ?
      sycl::access::address_space::local_space :
      sycl::access::address_space::global_space;
  using dataT = T;
  using elemT = sycl::atomic<T, addressSpace>;
  using elemRefT = elemT;

  static inline dataT get(elemRefT value) {
    return value.load();
  }
  static inline void set(elemRefT dst, const dataT& value) {
    dst.store(value);
  }
};

template <typename T, sycl::access_mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder>
T multidim_subscript_read(
    const sycl::accessor<T, 1, mode, target, placeholder>& acc,
    sycl::id<1> idx) {
  return buffer_accessor_value<T, target, mode>::get(acc[idx[0]]);
}

template <typename T, sycl::access_mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder>
T multidim_subscript_read(
    const sycl::accessor<T, 2, mode, target, placeholder>& acc,
    sycl::id<2> idx) {
  return buffer_accessor_value<T, target, mode>::get(acc[idx[0]][idx[1]]);
}

template <typename T, sycl::access_mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder>
T multidim_subscript_read(
    const sycl::accessor<T, 3, mode, target, placeholder>& acc,
    sycl::id<3> idx) {
  return
      buffer_accessor_value<T, target, mode>::get(acc[idx[0]][idx[1]][idx[2]]);
}

template <typename T, sycl::access_mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder>
void multidim_subscript_write(
    const sycl::accessor<T, 1, mode, target, placeholder>& acc,
    sycl::id<1> idx, T value) {
  buffer_accessor_value<T, target, mode>::set(acc[idx[0]], value);
}

template <typename T, sycl::access_mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder>
void multidim_subscript_write(
    const sycl::accessor<T, 2, mode, target, placeholder>& acc,
    sycl::id<2> idx, T value) {
  buffer_accessor_value<T, target, mode>::set(acc[idx[0]][idx[1]], value);
}

template <typename T, sycl::access_mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder>
void multidim_subscript_write(
    const sycl::accessor<T, 3, mode, target, placeholder>& acc,
    sycl::id<3> idx, T value) {
  buffer_accessor_value<T, target, mode>::set(acc[idx[0]][idx[1]][idx[2]],
                                              value);
}

struct buffer_accessor_api_pointer_error_code {
  static constexpr size_t pointer_read_access = 0;
  static constexpr size_t pointer_write_access = 1;
};
struct buffer_accessor_api_subscripts_error_code {
  static constexpr size_t zero_dim_access = 0;
  static constexpr size_t multi_dim_read_id = 0;
  static constexpr size_t multi_dim_read_size_t = 1;
  static constexpr size_t multi_dim_write_id = 2;
  static constexpr size_t multi_dim_write_size_t = 3;
};

template <typename T, int dim>
struct buffer_accessor_expected_value {
  static constexpr size_t write_mul = 2;
  static constexpr size_t write_offset = 1;

  static inline size_t expected_read_data(sycl_id_t<dim> idx,
                                          sycl_range_t<dim> range) {
    return compute_linear_id(idx, range);
  }
  static inline constexpr sycl_id_t<dim> default_id() {
    return sycl_cts::util::get_cts_object::id<dim>::get(0, 0, 0);
  }
  static inline constexpr sycl_range_t<dim> default_range() {
    return sycl_cts::util::get_cts_object::range<dim>::get(1, 1, 1);
  }
  static inline T expected_read(sycl_id_t<dim> idx = default_id(),
                                sycl_range_t<dim> range = default_range()) {
    return static_cast<T>(expected_read_data(idx, range));
  }
  static inline T expected_write(sycl_id_t<dim> idx = default_id(),
                                 sycl_range_t<dim> range = default_range()) {
    const auto value = expected_read_data(idx, range);
    return static_cast<T>(value * write_mul + write_offset);
  }
};

template <typename T>
struct buffer_accessor_expected_value<T, 0> {
  static constexpr size_t write_mul = 2;
  static constexpr size_t write_offset = 1;

  template <typename ... argsT>
  static inline T expected_read(argsT ...) {
    return get_zero_dim_buffer_value<T>();
  }

  template <typename ... argsT>
  static inline T expected_write(argsT ...){
    const auto value = get_zero_dim_buffer_value<size_t>();
    return static_cast<T>(value * write_mul + write_offset);
  }
};

/**
 *  @brief Tests buffer accessors pointer value with read-only data access
 */
template <typename T, int dim>
struct buffer_accessor_get_pointer_r {
  using error_code_t = buffer_accessor_api_pointer_error_code;
  using expected_value_t = buffer_accessor_expected_value<T, dim>;

  template <typename accT, typename errorAccT>
  static void check(const accT& acc, const errorAccT& errorAccessor,
                    const sycl_id_t<dim>& offset) {
    sycl_id_t<dim>
        zero_offset;  // In SYCL 2020, get_pointer() always returns the base
                      // buffer pointer, ignoring any offset.
    const auto expectedRead = expected_value_t::expected_read(zero_offset);
    auto ptr = acc.get_pointer();

    T elem = *ptr;
    if (!check_elems_equal(elem, expectedRead)) {
      errorAccessor[error_code_t::pointer_read_access] = 1;
    }
  }
};
/**
 *  @brief Tests buffer accessors pointer value with write-only data access
 */
template <typename T, int dim>
struct buffer_accessor_get_pointer_w {
  using expected_value_t = buffer_accessor_expected_value<T, dim>;

  template <typename accT, typename errorAccT>
  static void check(const accT& acc, const errorAccT&,
                    const sycl_id_t<dim>& offset) {
    sycl_id_t<dim>
        zero_offset;  // In SYCL 2020, get_pointer() always returns the base
                      // buffer pointer, ignoring any offset.
    const auto value = expected_value_t::expected_write(zero_offset);
    auto ptr = acc.get_pointer();

    *ptr = value;
  }
};
/**
 *  @brief Tests buffer accessors pointer value with read-write data access
 */
template <typename T, int dim, sycl::access_mode mode, sycl::target target>
struct buffer_accessor_get_pointer_rw {
  using error_code_t = buffer_accessor_api_pointer_error_code;
  using expected_value_t = buffer_accessor_expected_value<T, dim>;

  template <sycl::access::placeholder placeholder>
  using acc_t = sycl::accessor<T, dim, mode, target, placeholder>;

  template <sycl::access::placeholder placeholder, typename errorAccT>
  static void check(const acc_t<placeholder>& acc,
                    const errorAccT& errorAccessor,
                    const sycl_id_t<dim>& offset) {
    sycl_id_t<dim>
        zero_offset;  // In SYCL 2020, get_pointer() always returns the base
                      // buffer pointer, ignoring any offset.
    const auto expectedRead = expected_value_t::expected_read(zero_offset);
    const auto expectedWrite = expected_value_t::expected_write(zero_offset);
    constexpr bool noInitExpected =
        (target == sycl::target::local) ||
        (mode == sycl::access_mode::discard_read_write);

    auto ptr = acc.get_pointer();
    T elem;

    if constexpr (!noInitExpected) {
      /** check read syntax
       */
      elem = *ptr;
      if (!check_elems_equal(elem, expectedRead)) {
        errorAccessor[error_code_t::pointer_read_access] = 1;
      }
    }
    /** check write syntax
     */
    *ptr = expectedWrite;

    /** validate write syntax
     */
    elem = *ptr;
    if (!check_elems_equal(elem, expectedWrite)) {
      errorAccessor[error_code_t::pointer_write_access] = 1;
    }
  }
};
/**
 *  @brief Kernel name for buffer_accessor_get_pointer functor
 */
template <typename kernelName, int dim, sycl::access_mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder>
struct buffer_accessor_get_pointer_kernel {};
/**
 *  @brief Tests buffer accessors pointer value with any data access
 */
template <typename T, int dim, sycl::access_mode mode,
          sycl::target target, sycl::target errorTarget,
          sycl::access::placeholder placeholder>
class buffer_accessor_get_pointer {
  using acc_t = sycl::accessor<T, dim, mode, target, placeholder>;
  using error_acc_t = sycl::accessor<int, 1, errorMode, errorTarget>;
  acc_t m_acc;
  error_acc_t m_errorAccessor;
  sycl_id_t<dim> m_offset;

 public:
  buffer_accessor_get_pointer(acc_t acc, error_acc_t errorAccessor,
                              sycl_id_t<dim> offset)
      : m_acc(acc),
        m_errorAccessor(errorAccessor),
        m_offset(offset) {}

  void operator()() const {
    check_get_pointer(acc_mode_tag::get<mode>());
  }
  void operator()(sycl_id_t<dim>) const {
    operator()();
  }

 private:
  void check_get_pointer(acc_mode_tag::generic) const {
    buffer_accessor_get_pointer_rw<T, dim, mode, target>::check(
        m_acc, m_errorAccessor, m_offset);
  }
  void check_get_pointer(acc_mode_tag::write_only) const {
    buffer_accessor_get_pointer_w<T, dim>::check(m_acc, m_errorAccessor,
                                                 m_offset);
  }
  void check_get_pointer(acc_mode_tag::read_only) const {
    buffer_accessor_get_pointer_r<T, dim>::check(m_acc, m_errorAccessor,
                                                 m_offset);
  }
};

/**
 *  @brief Kernel name for buffer_accessor_api_* functors
 */
template <typename kernelName, int dim, sycl::access_mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder>
struct buffer_accessor_api_kernel {};

/** tests buffer accessors reads
*/
template <typename T, int dim, sycl::access_mode mode,
          sycl::target target, sycl::target errorTarget,
          sycl::access::placeholder placeholder>
class buffer_accessor_api_r {
  using acc_t = sycl::accessor<T, dim, mode, target, placeholder>;
  using error_acc_t = sycl::accessor<int, 1, errorMode, errorTarget>;
  using elem_acc_t = buffer_accessor_value<T, target, mode>;
  using error_code_t = buffer_accessor_api_subscripts_error_code;
  using expected_value_t = buffer_accessor_expected_value<T, dim>;

  acc_t m_accIdSyntax;
  acc_t m_accMultiDimSyntax;
  error_acc_t m_errorAccessor;
  sycl_range_t<dim> m_range;
  size_t size;

 public:
  buffer_accessor_api_r(size_t size_, acc_t accIdSyntax,
                        acc_t accMultiDimSyntax, error_acc_t errorAccessor,
                        sycl_range_t<dim> rng)
      : m_accIdSyntax(accIdSyntax),
        m_accMultiDimSyntax(accMultiDimSyntax),
        m_errorAccessor(errorAccessor),
        m_range(rng),
        size(size_) {}

  void operator()(sycl_id_t<dim> idx) const {
    return check_subscripts(idx, is_zero_dim<dim>{});
  }

 private:
  /**
   * @brief Checks reading from an accessor using the subscript operators.
   *        Only executed when (dim != 0).
   * @param idx Work-item ID
   */
  void check_subscripts(sycl_id_t<dim> idx, generic_dim_tag) const {
    const auto expectedRead = expected_value_t::expected_read(idx, m_range);

    /** check id read syntax
    */
    T elem = elem_acc_t::get(m_accIdSyntax[idx]);
    if (!check_elems_equal(elem, expectedRead)) {
      m_errorAccessor[error_code_t::multi_dim_read_id] = 1;
    }

    /** check size_t read syntax
    */
    elem = multidim_subscript_read(m_accMultiDimSyntax, idx);
    if (!check_elems_equal(elem, expectedRead)) {
      m_errorAccessor[error_code_t::multi_dim_read_size_t] = 1;
    }
  };

  /**
   * @brief Checks reading from an accessor using the subscript operators.
   *        Only executed when (dim == 0).
   */
  void check_subscripts(sycl_id_t<dim> /*idx*/, zero_dim_tag) const {
    const auto expectedRead = expected_value_t::expected_read();

    /** check operator dataT&() read syntax
     */
    T elem = elem_acc_t::get(m_accIdSyntax);
    if (!check_elems_equal(elem, expectedRead)) {
      m_errorAccessor[error_code_t::zero_dim_access] = 1;
    }
  };
};

/** tests buffer accessors writes
*/
template <typename T, int dim, sycl::access_mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder>
class buffer_accessor_api_w {
  using acc_t = sycl::accessor<T, dim, mode, target, placeholder>;
  using elem_acc_t = buffer_accessor_value<T, target, mode>;
  using expected_value_t = buffer_accessor_expected_value<T, dim>;

  acc_t m_accIdSyntax;
  acc_t m_accMultiDimSyntax;
  sycl_range_t<dim> m_range;
  size_t size;

 public:
  buffer_accessor_api_w(size_t size_, acc_t accIdSyntax,
                        acc_t accMultiDimSyntax, sycl_range_t<dim> r)
      : m_accIdSyntax(accIdSyntax),
        m_accMultiDimSyntax(accMultiDimSyntax),
        m_range(r),
        size(size_) {}

  void operator()(sycl_id_t<dim> idx) const {
    return check_subscripts(idx, is_zero_dim<dim>{});
  }

 private:
  /**
   * @brief Checks writing to an accessor using the subscript operators.
   *        Only executed when (dim != 0).
   * @param idx Work-item ID
   */
  void check_subscripts(sycl_id_t<dim> idx, generic_dim_tag) const {
    const auto expectedWrite = expected_value_t::expected_write(idx, m_range);

    /** check id write syntax
    */
    elem_acc_t::set(m_accIdSyntax[idx], expectedWrite);

    /** check size_t write syntax
    */
    multidim_subscript_write<T>(m_accMultiDimSyntax, idx, expectedWrite);
  };

  /**
   * @brief Checks writing to an accessor using the subscript operators.
   *        Only executed when (dim == 0).
   */
  void check_subscripts(sycl_id_t<dim> /*idx*/, zero_dim_tag) const {
    const auto expectedWrite = expected_value_t::expected_write();

    /** check operator dataT&() write syntax
     */
    elem_acc_t::set(m_accIdSyntax, expectedWrite);
  };
};

/** tests buffer accessors reads and writes
*/
template <typename T, int dim, sycl::access_mode mode,
          sycl::target target, sycl::target errorTarget,
          sycl::access::placeholder placeholder>
class buffer_accessor_api_rw {
  using acc_t = sycl::accessor<T, dim, mode, target, placeholder>;
  using error_acc_t = sycl::accessor<int, 1, errorMode, errorTarget>;
  using elem_acc_t = buffer_accessor_value<T, target, mode>;
  using error_code_t = buffer_accessor_api_subscripts_error_code;
  using expected_value_t = buffer_accessor_expected_value<T, dim>;

  acc_t m_accIdSyntax;
  acc_t m_accMultiDimSyntax;
  error_acc_t m_errorAccessor;
  sycl_range_t<dim> m_range;
  size_t size;

 public:
  buffer_accessor_api_rw(size_t size_, acc_t accIdSyntax,
                         acc_t accMultiDimSyntax, error_acc_t errorAccessor,
                         sycl_range_t<dim> rng)
      : m_accIdSyntax(accIdSyntax),
        m_accMultiDimSyntax(accMultiDimSyntax),
        m_errorAccessor(errorAccessor),
        m_range(rng),
        size(size_) {}

  void operator()(sycl_id_t<dim> idx) const {
    // We do not need work-item synchronization for atomic mode because of:
    // - load-store consistency within single work-item
    // - access to the different elements from different work-items
    return check_subscripts(idx, is_zero_dim<dim>{});
  }

 private:
  /**
   * @brief Checks writing to an accessor using the subscript operators.
   *        Only executed when (dim != 0).
   * @param idx Work-item ID
   */
  void check_subscripts(sycl_id_t<dim> idx, generic_dim_tag) const {
    T elem;

    constexpr bool noInitExpected =
        (target == sycl::target::local) ||
        (mode == sycl::access_mode::discard_read_write);

    // Ensure we can expect any valid information at this point
    if constexpr (!noInitExpected) {
      const auto expectedRead = expected_value_t::expected_read(idx, m_range);

      /** check id read syntax
      */
      elem = elem_acc_t::get(m_accIdSyntax[idx]);
      if (!check_elems_equal(elem, expectedRead)) {
        m_errorAccessor[error_code_t::multi_dim_read_id] = 1;
      }

      /** check size_t read syntax
      */
      elem = multidim_subscript_read(m_accMultiDimSyntax, idx);
      if (!check_elems_equal(elem, expectedRead)) {
        m_errorAccessor[error_code_t::multi_dim_read_size_t] = 1;
      }
    }

    const auto expectedWrite = expected_value_t::expected_write(idx, m_range);

    /** check id write syntax
    */
    elem_acc_t::set(m_accIdSyntax[idx], expectedWrite);

    /** validate id write syntax
    */
    elem = elem_acc_t::get(m_accIdSyntax[idx]);
    if (!check_elems_equal(elem, expectedWrite)) {
      m_errorAccessor[error_code_t::multi_dim_write_id] = 1;
    }

    /** check size_t write syntax
    */
    multidim_subscript_write<T>(m_accMultiDimSyntax, idx, expectedWrite);

    /** validate size_t write syntax
    */
    elem = multidim_subscript_read(m_accMultiDimSyntax, idx);
    if (!check_elems_equal(elem, expectedWrite)) {
      m_errorAccessor[error_code_t::multi_dim_write_size_t] = 1;
    }
  };

  /**
   * @brief Checks writing to an accessor using the subscript operators.
   *        Only executed when (dim == 0).
   */
  void check_subscripts(sycl_id_t<dim> /*idx*/, zero_dim_tag) const {
    /** check operator dataT&() read syntax, only the interface
     */
    T elem = elem_acc_t::get(m_accIdSyntax);
    (void)elem;

    /** check operator dataT&() write syntax
     */
    const auto expectedWrite = expected_value_t::expected_write();
    elem_acc_t::set(
        static_cast<typename elem_acc_t::elemRefT>(m_accIdSyntax),
        expectedWrite);

    elem = elem_acc_t::get(m_accIdSyntax);
    if (!check_elems_equal(elem, expectedWrite)) {
      m_errorAccessor[error_code_t::zero_dim_access] = 1;
    }
  }
};

}  // namespace accessor_utility

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_COMMON_BUFFER_LOCAL_H
