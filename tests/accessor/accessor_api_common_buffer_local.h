/*************************************************************************
//
//  SYCL Conformance Test Suite
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
#include "accessor_utility.h"
#include <stdexcept>
#include <sstream>
#include <utility>
#include <vector>

namespace {

/** computes the linear id for 1 dimension
*/
size_t compute_linear_id(cl::sycl::id<1> id, cl::sycl::range<1> r) {
  return id[0];
}

/** computes the linear id for 2 dimension
*/
size_t compute_linear_id(cl::sycl::id<2> id, cl::sycl::range<2> r) {
  return id[1] + (id[0] * r[1]);
}

/** computes the linear id for 3 dimension
*/
size_t compute_linear_id(cl::sycl::id<3> id, cl::sycl::range<3> r) {
  return id[2] + (id[1] * r[2]) + (id[0] * r[2] * r[1]);
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

namespace accessor_utility {

template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder>
T multidim_subscript_read(
    cl::sycl::accessor<T, 1, mode, target, placeholder>& acc,
    cl::sycl::id<1> idx) {
  return acc[idx[0]];
}

template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder>
T multidim_subscript_read(
    cl::sycl::accessor<T, 2, mode, target, placeholder>& acc,
    cl::sycl::id<2> idx) {
  return acc[idx[0]][idx[1]];
}

template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder>
T multidim_subscript_read(
    cl::sycl::accessor<T, 3, mode, target, placeholder>& acc,
    cl::sycl::id<3> idx) {
  return acc[idx[0]][idx[1]][idx[2]];
}

template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder>
void multidim_subscript_write(
    cl::sycl::accessor<T, 1, mode, target, placeholder>& acc,
    cl::sycl::id<1> idx, T value) {
  acc[idx[0]] = value;
}

template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder>
void multidim_subscript_write(
    cl::sycl::accessor<T, 2, mode, target, placeholder>& acc,
    cl::sycl::id<2> idx, T value) {
  acc[idx[0]][idx[1]] = value;
}

template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder>
void multidim_subscript_write(
    cl::sycl::accessor<T, 3, mode, target, placeholder>& acc,
    cl::sycl::id<3> idx, T value) {
  acc[idx[0]][idx[1]][idx[2]] = value;
}

/** tests buffer accessors reads
*/
template <typename T, int dim, cl::sycl::access::mode mode,
          cl::sycl::access::target target, cl::sycl::access::target errorTarget,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
class buffer_accessor_api_r {
  using acc_t = cl::sycl::accessor<T, dim, mode, target, placeholder>;
  using error_acc_t = cl::sycl::accessor<int, 1, errorMode, errorTarget>;

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

  void operator()(sycl_id_t<dim> idx) {
    return check_subscripts(idx, is_zero_dim<dim>{});
  }

 private:
  /**
   * @brief Checks reading from an accessor using the subscript operators.
   *        Only executed when (dim != 0).
   * @param idx Work-item ID
   */
  void check_subscripts(sycl_id_t<dim> idx, generic_dim_tag) {
    size_t linearID = compute_linear_id(idx, m_range);
    const auto expectedRead = static_cast<T>(linearID);

    /** check id read syntax
    */
    T elem = m_accIdSyntax[idx];
    if (!check_elems_equal(elem, expectedRead)) {
      m_errorAccessor[0] = 1;
    }

    /** check size_t read syntax
    */
    elem = multidim_subscript_read(m_accMultiDimSyntax, idx);
    if (!check_elems_equal(elem, expectedRead)) {
      m_errorAccessor[1] = 1;
    }
  };

  /**
   * @brief Checks reading from an accessor using the subscript operators.
   *        Only executed when (dim == 0).
   */
  void check_subscripts(sycl_id_t<dim> /*idx*/, zero_dim_tag) {
    const auto expectedRead = get_zero_dim_buffer_value<T>();

    /** check operator dataT&() read syntax
     */
    T elem = m_accIdSyntax;
    if (!check_elems_equal(elem, expectedRead)) {
      m_errorAccessor[0] = 1;
    }
  };
};

/** tests buffer accessors writes
*/
template <typename T, int dim, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
class buffer_accessor_api_w {
  using acc_t = cl::sycl::accessor<T, dim, mode, target, placeholder>;

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

  void operator()(sycl_id_t<dim> idx) {
    return check_subscripts(idx, is_zero_dim<dim>{});
  }

 private:
  /**
   * @brief Checks writing to an accessor using the subscript operators.
   *        Only executed when (dim != 0).
   * @param idx Work-item ID
   */
  void check_subscripts(sycl_id_t<dim> idx, generic_dim_tag) {
    size_t linearID = compute_linear_id(idx, m_range);
    const auto expected = static_cast<T>(linearID);

    /** check id write syntax
    */
    m_accIdSyntax[idx] = expected;

    /** check size_t write syntax
    */
    multidim_subscript_write<T>(m_accMultiDimSyntax, idx, expected);
  };

  /**
   * @brief Checks writing to an accessor using the subscript operators.
   *        Only executed when (dim == 0).
   */
  void check_subscripts(sycl_id_t<dim> /*idx*/, zero_dim_tag) {
    const auto expected = get_zero_dim_buffer_value<T>();

    /** check operator dataT&() write syntax
     */
    static_cast<T&>(m_accMultiDimSyntax) = expected;
  };
};

/** tests buffer accessors reads and writes
*/
template <typename T, int dim, cl::sycl::access::mode mode,
          cl::sycl::access::target target, cl::sycl::access::target errorTarget,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
class buffer_accessor_api_rw {
  using acc_t = cl::sycl::accessor<T, dim, mode, target, placeholder>;
  using error_acc_t = cl::sycl::accessor<int, 1, errorMode, errorTarget>;

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

  void operator()(sycl_id_t<dim> idx) {
    return check_subscripts(idx, is_zero_dim<dim>{});
  }

 private:
  /**
   * @brief Checks writing to an accessor using the subscript operators.
   *        Only executed when (dim != 0).
   * @param idx Work-item ID
   */
  void check_subscripts(sycl_id_t<dim> idx, generic_dim_tag) {
    size_t linearID = compute_linear_id(idx, m_range);

    T elem;

    // A local accessor doesn't have any valid information at this point
    if (target != cl::sycl::access::target::local) {
      const auto expectedRead = static_cast<T>(linearID);

      /** check id read syntax
      */
      elem = m_accIdSyntax[idx];
      if (!check_elems_equal(elem, expectedRead)) {
        m_errorAccessor[0] = 1;
      }

      /** check size_t read syntax
      */
      elem = multidim_subscript_read(m_accMultiDimSyntax, idx);
      if (!check_elems_equal(elem, expectedRead)) {
        m_errorAccessor[1] = 1;
      }
    }

    const auto expectedWrite = static_cast<T>(linearID * 2);

    /** check id write syntax
    */
    m_accIdSyntax[idx] = expectedWrite;

    /** validate id write syntax
    */
    elem = m_accIdSyntax[idx];
    if (!check_elems_equal(elem, expectedWrite)) {
      m_errorAccessor[2] = 1;
    }

    /** check size_t write syntax
    */
    multidim_subscript_write<T>(m_accMultiDimSyntax, idx, expectedWrite);

    /** validate size_t write syntax
    */
    elem = multidim_subscript_read(m_accMultiDimSyntax, idx);
    if (!check_elems_equal(elem, expectedWrite)) {
      m_errorAccessor[3] = 1;
    }
  };

  /**
   * @brief Checks writing to an accessor using the subscript operators.
   *        Only executed when (dim == 0).
   */
  void check_subscripts(sycl_id_t<dim> /*idx*/, zero_dim_tag) {
    /** check operator dataT&() read syntax, only the interface
     */
    T elem = m_accIdSyntax;
    (void)elem;

    /** check operator dataT&() write syntax
     */
    const auto expected = get_zero_dim_buffer_value<T>();
    static_cast<T&>(m_accMultiDimSyntax) = expected;

    if (!check_elems_equal(static_cast<T>(m_accMultiDimSyntax), expected)) {
      m_errorAccessor[0] = 1;
    }
  }
};

}  // namespace accessor_utility

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_COMMON_BUFFER_LOCAL_H
