/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "./../../util/math_helper.h"
#include "accessor_utility.h"

#define TEST_NAME accessor_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** unique dummy_functor per file
 *  this is a hack until the CMake script is fixed; kill both the alias and the
 *  dummy class once it is fixed
 */
class dummy_accessor_api {};
using dummy_functor = ::dummy_functor<dummy_accessor_api>;

/** user defined struct that is used in accessor tests
*/
struct user_struct {
  float a;
  int b;
  char c;

  using element_type = int;

  user_struct() : a(0), b(0), c(0){};

  user_struct(int val) : a(0), b(val), c(0) {}

  element_type operator[](size_t index) const { return b; }
};

/** creates a list of ids
*/
template <int dims>
cl::sycl::vector_class<cl::sycl::id<dims>> create_id_list(
    cl::sycl::range<dims> &r);

/** creates a list of ids (specialization for 1 dimension)
*/
template <>
cl::sycl::vector_class<cl::sycl::id<1>> create_id_list<1>(
    cl::sycl::range<1> &r) {
  cl::sycl::vector_class<cl::sycl::id<1>> ret;
  for (size_t i = 0; i < r[0]; ++i) {
    ret.emplace_back(i);
  }
  return ret;
}

/** creates a list of ids (specialization for 2 dimension)
*/
template <>
cl::sycl::vector_class<cl::sycl::id<2>> create_id_list<2>(
    cl::sycl::range<2> &r) {
  cl::sycl::vector_class<cl::sycl::id<2>> ret;
  for (size_t i = 0; i < r[0]; ++i) {
    for (size_t j = 0; j < r[1]; ++j) {
      ret.emplace_back(i, j);
    }
  }
  return ret;
}

/** creates a list of ids (specialization for 3 dimension)
*/
template <>
cl::sycl::vector_class<cl::sycl::id<3>> create_id_list<3>(
    cl::sycl::range<3> &r) {
  cl::sycl::vector_class<cl::sycl::id<3>> ret;
  for (size_t i = 0; i < r[0]; ++i) {
    for (size_t j = 0; j < r[1]; ++j) {
      for (size_t k = 0; k < r[2]; ++k) {
        ret.emplace_back(i, j, k);
      }
    }
  }
  return ret;
}

/** explicit pointer type
*/
template <typename T, cl::sycl::access::target target>
struct explicit_pointer;

/** explicit pointer type (specialization for global_buffer)
*/
template <typename T>
struct explicit_pointer<T, cl::sycl::access::target::global_buffer> {
  using type = cl::sycl::global_ptr<T>;
};

/** explicit pointer type (specialization for constant_buffer)
*/
template <typename T>
struct explicit_pointer<T, cl::sycl::access::target::constant_buffer> {
  using type = cl::sycl::constant_ptr<T>;
};

/** explicit pointer type (specialization for local)
*/
template <typename T>
struct explicit_pointer<T, cl::sycl::access::target::local> {
  using type = cl::sycl::local_ptr<T>;
};

/** explicit pointer type (specialisation for host_buffer)
*/
template <typename T>
struct explicit_pointer<T, cl::sycl::access::target::host_buffer> {
  using type = T *;
};

/** explicit pointer alias
 */
template <typename T, cl::sycl::access::target target>
using explicit_pointer_t = typename explicit_pointer<T, target>::type;

/** image format channel order and type
*/
template <typename T>
struct image_format_channel;

/** image format channel order and type (specialization for int4)
*/
template <>
struct image_format_channel<cl::sycl::int4> {
  static constexpr cl::sycl::image_channel_type type =
      cl::sycl::image_channel_type::signed_int8;
  static constexpr cl::sycl::image_channel_order order =
      cl::sycl::image_channel_order::rgba;
};

/** image format channel order and type (specialization for uint4)
*/
template <>
struct image_format_channel<cl::sycl::uint4> {
  static constexpr cl::sycl::image_channel_type type =
      cl::sycl::image_channel_type::unsigned_int8;
  static constexpr cl::sycl::image_channel_order order =
      cl::sycl::image_channel_order::rgba;
};

/** image format channel order and type (specialization for float4)
*/
template <>
struct image_format_channel<cl::sycl::float4> {
  static constexpr cl::sycl::image_channel_type type =
      cl::sycl::image_channel_type::unorm_int8;
  static constexpr cl::sycl::image_channel_order order =
      cl::sycl::image_channel_order::rgba;
};

/** specialized struct for defining the normalization coefficient for an image
 * accessor type. 1.0f by default.
*/
template <typename elementT>
struct use_normalization_coefficient {
  static constexpr bool value = false;
};

/** specialized struct for defining the normalization coefficient for an image
 * accessor type. Specializationf or cl::sycl::float4.
*/
template <>
struct use_normalization_coefficient<cl::sycl::float4> {
  static constexpr bool value = true;
};

/** checker for the cl::sycl::vec type
*/
template <typename T>
struct is_sycl_vec : std::integral_constant<bool, false> {};
template <typename T, int dim>
struct is_sycl_vec<cl::sycl::vec<T, dim>> : std::integral_constant<bool, true> {
};

/** tests that two values are equal
*/
template <typename T1, typename T2, REQUIRES(std::is_scalar<T1>::value)>
bool check_element_valid(const T1 &elem, const T2 &correct,
                         bool imageTest = false) {
  return elem == (static_cast<T1>(correct));
}

/** tests that two values are equal (overload for float4 type)
*/
bool check_element_valid(const cl::sycl::float4 &elem, const size_t &correct,
                         bool imageTest = false) {
  float errorMargin = 0.0001f;
  for (int i = 0; i < 4; i++) {
    float error =
        (getElement(elem, i) -
         (((static_cast<float>(correct) * 4.0f) + static_cast<float>(i))));
    if (error > errorMargin) {
      return false;
    }
  }
  return true;
}

/** tests that two values are equal (overload for vec type)
*/
template <typename T1, REQUIRES(is_sycl_vec<T1>::value)>
bool check_element_valid(const T1 &elem, const size_t &correct,
                         bool imageTest = false) {
  if (imageTest) {
    return (getElement(elem, 0) ==
                (static_cast<typename T1::element_type>(correct * 4)) &&
            getElement(elem, 1) ==
                (static_cast<typename T1::element_type>((correct * 4) + 1)) &&
            getElement(elem, 2) ==
                (static_cast<typename T1::element_type>((correct * 4) + 2)) &&
            getElement(elem, 3) ==
                (static_cast<typename T1::element_type>((correct * 4) + 3)));
  } else {
    return (getElement(elem, 0) ==
            (static_cast<typename T1::element_type>(correct)));
  }
}

/** tests that two values are equal (overload for user_struct)
*/
template <typename T2>
bool check_element_valid(const user_struct &elem, const T2 &correct,
                         bool imageTest = false) {
  return elem[0] == (static_cast<int>(correct));
}

/** tests that an array of linear ids is correct
*/
template <typename T>
bool check_linear_index(T *data, size_t size, int mul = 1,
                        bool imageTest = false) {
  for (size_t i = 0; i < size; i++) {
    if (!check_element_valid(data[i], (i * mul), imageTest)) {
      return false;
    }
  }
  return true;
}

/** computes the linear id for 1 dimension
*/
size_t compute_linear_id(cl::sycl::id<1> id, cl::sycl::range<1> r) {
  return id[0];
}

/** computes the linear id for 2 dimension
*/
size_t compute_linear_id(cl::sycl::id<2> id, cl::sycl::range<2> r) {
  return id[0] + (id[1] * r[0]);
}

/** computes the linear id for 3 dimension
*/
size_t compute_linear_id(cl::sycl::id<3> id, cl::sycl::range<3> r) {
  return id[0] + (id[1] * r[0]) + (id[2] * r[0] * r[1]);
}

/** tests accessor multi dim read syntax for 1 dimension
  * specialised for buffer accessors
  */
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t,
          REQUIRES(is_buffer<target>::value)>
T multidim_subscript_read(
    cl::sycl::accessor<T, 1, mode, target, placeholder> &acc,
    cl::sycl::id<1> idx) {
  return acc[idx[0]];
}

/** tests accessor multi dim read syntax for 1 dimension
  * specialised for image accessors
  */
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target, REQUIRES(is_image<target>::value)>
T multidim_subscript_read(cl::sycl::accessor<T, 1, mode, target> &acc,
                          cl::sycl::id<1> idx) {
  return acc.read(idx[0]);
}

/** tests accessor multi dim read syntax for 2 dimensions
  * specialised for buffer accessors
  */
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t,
          REQUIRES(is_buffer<target>::value)>
T multidim_subscript_read(
    cl::sycl::accessor<T, 2, mode, target, placeholder> &acc,
    cl::sycl::id<2> idx) {
  return acc[idx[0]][idx[1]];
}

/** tests accessor multi dim read syntax for 2 dimensions
  * specialised for image accessors
  */
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target, REQUIRES(is_image<target>::value)>
T multidim_subscript_read(cl::sycl::accessor<T, 2, mode, target> &acc,
                          cl::sycl::id<2> idx) {
  return acc.read(cl::sycl::int2(idx[0], idx[1]));
}

/** tests accessor multi dim read syntax for 3 dimensions
  * specialised for buffer accessors
  */
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t,
          REQUIRES(is_buffer<target>::value)>
T multidim_subscript_read(
    cl::sycl::accessor<T, 3, mode, target, placeholder> &acc,
    cl::sycl::id<3> idx) {
  return acc[idx[0]][idx[1]][idx[2]];
}

/** tests accessor multi dim read syntax for 3 dimensions
  * specialised for image accessors
  */
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target, REQUIRES(is_image<target>::value)>
T multidim_subscript_read(cl::sycl::accessor<T, 3, mode, target> &acc,
                          cl::sycl::id<3> idx) {
  return acc.read(cl::sycl::int3(idx[0], idx[1], idx[2]));
}

/** tests accessor multi dim sampled read syntax for 1 dimension
*/
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target, REQUIRES(is_image<target>::value)>
T multidim_subscript_sampled_read(cl::sycl::accessor<T, 1, mode, target> &acc,
                                  cl::sycl::sampler smpl, cl::sycl::id<1> idx) {
  return acc.read(idx[0], smpl);
}

/** tests accessor multi dim sampled read syntax for 2 dimension
*/
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target, REQUIRES(is_image<target>::value)>
T multidim_subscript_sampled_read(cl::sycl::accessor<T, 2, mode, target> &acc,
                                  cl::sycl::sampler smpl, cl::sycl::id<2> idx) {
  return acc.read(cl::sycl::int2(idx[0], idx[1]), smpl);
}

/** tests accessor multi dim sampled read syntax for 3 dimension
*/
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target, REQUIRES(is_image<target>::value)>
T multidim_subscript_sampled_read(cl::sycl::accessor<T, 3, mode, target> &acc,
                                  cl::sycl::sampler smpl, cl::sycl::id<3> idx) {
  return acc.read(cl::sycl::int3(idx[0], idx[1], idx[2]), smpl);
}

/** tests accessor multi dim write syntax for 1 dimension
  * specialised for buffer accessors
*/
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t,
          REQUIRES(is_buffer<target>::value)>
void multidim_subscript_write(
    cl::sycl::accessor<T, 1, mode, target, placeholder> &acc,
    cl::sycl::id<1> idx, T value) {
  acc[idx[0]] = value;
}

/** tests accessor multi dim write syntax for 1 dimension
  * specialised for image accessors
  */
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target, REQUIRES(is_image<target>::value)>
void multidim_subscript_write(cl::sycl::accessor<T, 1, mode, target> &acc,
                              cl::sycl::id<1> idx, T value) {
  acc.write(idx[0], value);
}

/** tests accessor multi dim write syntax for 2 dimension
  * specialised for buffer accessors
  */
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t,
          REQUIRES(is_buffer<target>::value)>
void multidim_subscript_write(
    cl::sycl::accessor<T, 2, mode, target, placeholder> &acc,
    cl::sycl::id<2> idx, T value) {
  acc[idx[0]][idx[1]] = value;
}

/** tests accessor multi dim write syntax for 2 dimension
  * specialised for image accessors
  */
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target, REQUIRES(is_image<target>::value)>
void multidim_subscript_write(cl::sycl::accessor<T, 2, mode, target> &acc,
                              cl::sycl::id<2> idx, T value) {
  acc.write(cl::sycl::int2(idx[0], idx[1]), value);
}

/** tests accessor multi dim write syntax for 3 dimension
  * specialised for buffer accessors
  */
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t,
          REQUIRES(is_buffer<target>::value)>
void multidim_subscript_write(
    cl::sycl::accessor<T, 3, mode, target, placeholder> &acc,
    cl::sycl::id<3> idx, T value) {
  acc[idx[0]][idx[1]][idx[2]] = value;
}

/** tests accessor multi dim write syntax for 3 dimension
  * specialised for image accessors
  */
template <typename T, cl::sycl::access::mode mode,
          cl::sycl::access::target target, REQUIRES(is_image<target>::value)>
void multidim_subscript_write(cl::sycl::accessor<T, 3, mode, target> &acc,
                              cl::sycl::id<3>, T) {
  // CTS not required to test -- this is an extension
}

/** tests buffer accessors reads
*/
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target, cl::sycl::access::target errorTarget,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
class buffer_accessor_api_r {
  cl::sycl::accessor<T, dim, mode, target, placeholder> m_accessorA;
  cl::sycl::accessor<T, dim, mode, target, placeholder> m_accessorB;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, errorTarget,
                     placeholder>
      m_errorAccessor;
  cl::sycl::range<dim> m_range;

 public:
  buffer_accessor_api_r(
      cl::sycl::accessor<T, dim, mode, target, placeholder> accessorA,
      cl::sycl::accessor<T, dim, mode, target, placeholder> accessorB,
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, errorTarget,
                         placeholder>
          errorAccessor,
      cl::sycl::range<dim> rng)
      : m_accessorA(accessorA),
        m_accessorB(accessorB),
        m_errorAccessor(errorAccessor),
        m_range(rng) {}

  void operator()(cl::sycl::id<dim> idx) {
    size_t linearID = compute_linear_id(idx, m_range);

    /** check id read syntax
    */
    T elem = m_accessorA[idx];
    if (!check_element_valid(elem, linearID, false)) {
      m_errorAccessor[0] = 1;
    }

    /** check size_t read syntax
    */
    elem = multidim_subscript_read(m_accessorB, idx);
    if (!check_element_valid(elem, linearID, false)) {
      m_errorAccessor[1] = 1;
    }
  };
};

/** tests buffer accessors writes
*/
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
class buffer_accessor_api_w {
  cl::sycl::accessor<T, dim, mode, target, placeholder> m_accessorA;
  cl::sycl::accessor<T, dim, mode, target, placeholder> m_accessorB;
  cl::sycl::range<dim> m_range;

 public:
  buffer_accessor_api_w(
      cl::sycl::accessor<T, dim, mode, target, placeholder> accessorA,
      cl::sycl::accessor<T, dim, mode, target, placeholder> accessorB,
      cl::sycl::range<dim> r)
      : m_accessorA(accessorA), m_accessorB(accessorB), m_range(r) {}

  void operator()(cl::sycl::id<dim> idx) {
    size_t linearID = compute_linear_id(idx, m_range);

    /** check id write syntax
    */
    m_accessorA[idx] = linearID;

    /** check size_t write syntax
    */
    multidim_subscript_write<T>(m_accessorB, idx, static_cast<T>(linearID));
  };
};

/** tests buffer accessors reads and writes
*/
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target, cl::sycl::access::target errorTarget,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
class buffer_accessor_api_rw {
  cl::sycl::accessor<T, dim, mode, target, placeholder> m_accessorA;
  cl::sycl::accessor<T, dim, mode, target, placeholder> m_accessorB;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, errorTarget,
                     placeholder>
      m_errorAccessor;
  cl::sycl::range<dim> m_range;

 public:
  buffer_accessor_api_rw(
      cl::sycl::accessor<T, dim, mode, target, placeholder> accessorA,
      cl::sycl::accessor<T, dim, mode, target, placeholder> accessorB,
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, errorTarget,
                         placeholder>
          errorAccessor,
      cl::sycl::range<dim> rng)
      : m_accessorA(accessorA),
        m_accessorB(accessorB),
        m_errorAccessor(errorAccessor),
        m_range(rng) {}

  void operator()(cl::sycl::id<dim> idx) {
    size_t linearID = compute_linear_id(idx, m_range);
    T elem;

    /** check id read syntax
    */
    elem = m_accessorA[idx];
    if (!check_element_valid(elem, linearID)) {
      m_errorAccessor[0] = 1;
    }

    /** check size_t read syntax
    */
    elem = multidim_subscript_read(m_accessorB, idx);
    if (!check_element_valid(elem, linearID)) {
      m_errorAccessor[1] = 1;
    }

    /** check id write syntax
    */
    m_accessorA[idx] = (linearID * 2);

    /** check size_t write syntax
    */
    multidim_subscript_write<T>(m_accessorB, idx, static_cast<T>(linearID * 2));

    /** validate id write syntax
    */
    elem = m_accessorA[idx];
    if (!check_element_valid(elem, (linearID * 2))) {
      m_errorAccessor[2] = 1;
    }

    /** validate size_t write syntax
    */
    elem = multidim_subscript_read(m_accessorB, idx);
    if (!check_element_valid(elem, (linearID * 2))) {
      m_errorAccessor[3] = 1;
    }
  };
};

/** tests image accessors reads
*/
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target, cl::sycl::access::target errorTarget>
class image_accessor_api_r {
  cl::sycl::accessor<T, dim, mode, target> m_accessorA;
  cl::sycl::accessor<T, dim, mode, target> m_accessorB;
  cl::sycl::accessor<T, dim, mode, target> accessorC_;
  cl::sycl::accessor<T, dim, mode, target> accessorD_;
  cl::sycl::sampler sampler_;

  cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, errorTarget>
      m_errorAccessor;
  cl::sycl::range<dim> m_range;

 public:
  image_accessor_api_r(
      cl::sycl::accessor<T, dim, mode, target> accessorA,
      cl::sycl::accessor<T, dim, mode, target> accessorB,
      cl::sycl::accessor<T, dim, mode, target> accessorC,
      cl::sycl::accessor<T, dim, mode, target> accessorD,
      cl::sycl::sampler smpl,
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, errorTarget>
          errorAccessor,
      cl::sycl::range<dim> rng)
      : m_accessorA(accessorA),
        m_accessorB(accessorB),
        accessorC_(accessorC),
        accessorD_(accessorD),
        sampler_(smpl),
        m_errorAccessor(errorAccessor),
        m_range(rng) {}

  void operator()(cl::sycl::id<dim> idx) {
    size_t linearID = compute_linear_id(idx, m_range);

    /** check id read syntax
    */
    T elem = m_accessorA.read(idx);
    if (use_normalization_coefficient<T>::value) {
      elem *= 255.f;
    }
    if (!check_element_valid(elem, linearID, true)) {
      m_errorAccessor[0] = 1;
    }

    /** check sampled id read syntax
    */
    elem = m_accessorB.read(idx, sampler_);
    if (use_normalization_coefficient<T>::value) {
      elem *= 255.f;
    }
    if (!check_element_valid(elem, linearID, true)) {
      m_errorAccessor[1] = 1;
    }

    /** check size_t read syntax
    */
    elem = multidim_subscript_read(accessorC_, idx);
    if (use_normalization_coefficient<T>::value) {
      elem *= 255.f;
    }
    if (!check_element_valid(elem, linearID, true)) {
      m_errorAccessor[2] = 1;
    }

    /** check sampled size_t read syntax
    */
    elem = multidim_subscript_sampled_read(accessorD_, sampler_, idx);
    if (use_normalization_coefficient<T>::value) {
      elem *= 255.f;
    }
    if (!check_element_valid(elem, linearID, true)) {
      m_errorAccessor[3] = 1;
    }
  }
};

/** tests image accessors writes
*/
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class image_accessor_api_w {
  cl::sycl::accessor<T, dim, mode, target> m_accessorA;
  cl::sycl::accessor<T, dim, mode, target> m_accessorB;
  cl::sycl::range<dim> m_range;

 public:
  image_accessor_api_w(cl::sycl::accessor<T, dim, mode, target> accessorA,
                       cl::sycl::accessor<T, dim, mode, target> accessorB,
                       cl::sycl::range<dim> rng)
      : m_accessorA(accessorA), m_accessorB(accessorB), m_range(rng) {}

  void operator()(cl::sycl::id<dim> idx) {
    size_t linearID = compute_linear_id(idx, m_range);
    size_t multiplyer = linearID * 4;

    typename T::element_type elem0 =
        static_cast<typename T::element_type>(multiplyer);
    typename T::element_type elem1 =
        static_cast<typename T::element_type>(multiplyer + 1);
    typename T::element_type elem2 =
        static_cast<typename T::element_type>(multiplyer + 2);
    typename T::element_type elem3 =
        static_cast<typename T::element_type>(multiplyer + 3);

    if (use_normalization_coefficient<T>::value) {
      elem0 *= 0.0039215686f;
      elem1 *= 0.0039215686f;
      elem2 *= 0.0039215686f;
      elem3 *= 0.0039215686f;
    }

    /** check id write syntax
    */
    m_accessorA.write(idx, T(elem0, elem1, elem2, elem3));

    /** check size_t write syntax
    */
    multidim_subscript_write(m_accessorB, idx, T(elem0, elem1, elem2, elem3));
  }
};

/** tests buffer accessors methods
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
class check_buffer_accessor_api_methods {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    auto data = [] {
      auto data = std::array<T, count>{};
      std::fill(data.begin(), data.end(), T{});
      return data;
    }();
    cl::sycl::buffer<T, dims> buffer(data, range);

    auto check = [&log, &range](
        const cl::sycl::accessor<T, dims, mode, target, placeholder>
            &accessor) {

      {
        /** check get_count() method
        */
        auto accessorCount = accessor.get_count();
        check_return_type<std::size_t>(log, accessor.get_count(),
                                       "get_count()");
        if (accessorCount != count) {
          FAIL(log, "accessor does not return the correct count");
        }
      }

      {
        /** check get_size() method
        */
        auto accessorSize = accessor.get_size();
        check_return_type<std::size_t>(log, accessor.get_size(), "get_size()");
        if (accessorSize != size) {
          FAIL(log, "accessor does not return the correct size");
        }
      }
      {
        /** check get_pointer() method
        */
        check_return_type<explicit_pointer_t<T, target>>(
            log, accessor.get_pointer(), "get_pointer()");
      }

      {
        /** check get_range() method
        */
        auto accessorRange = accessor.get_range();
        check_return_type<cl::sycl::range<dims>>(log, accessor.get_range(),
                                                 "get_range()");
        if (accessorRange != range) {
          FAIL(log, "accessor does not return the correct range");
        }
      }

      {
        /** check get_offset() method
        */
        auto accessorOffset = accessor.get_offset();
        check_return_type<cl::sycl::id<dims>>(log, accessor.get_offset(),
                                              "get_offset()");
        if (accessorOffset != cl::sycl::id<dims>(range)) {
          FAIL(log, "accessor does not return the correct offset");
        }
      }
    };

    if_constexpr<(target == cl::sycl::access::target::host_buffer)>(
        [&buffer, &check] {
          auto a = make_accessor<T, dims, mode, target, placeholder>(buffer);
          check(a);
        },
        [&queue, &buffer, &range, &check] {
          queue.submit([&](cl::sycl::handler &handler) {
            auto a = make_accessor<T, dims, mode, target, placeholder>(
                buffer, handler, range);
            check(a);

            /** dummy kernel as no kernel is required for these checks
            */
            handler.single_task(dummy_functor());
          });
        });
  }
};

template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target>
class kernel_name;

/** tests local accessor methods
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode>
class check_local_accessor_api_methods {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    queue.submit([&](cl::sycl::handler &h) {
      auto a =
          make_accessor<T, dims, mode, cl::sycl::access::placeholder::local>(
              range, h);
      {
        /** check get_count() method
        */
        auto accessorCount = accessor.get_count();
        check_return_type<std::size_t>(log, accessor.get_count(),
                                       "get_count()");
        if (accessorCount != count) {
          FAIL(log, "accessor does not return the correct count");
        }
      }

      {
        /** check get_size() method
        */
        auto accessorSize = accessor.get_size();
        check_return_type<std::size_t>(log, accessor.get_size(), "get_size()");
        if (accessorSize != size) {
          FAIL(log, "accessor does not return the correct size");
        }
      }

      {
        /** check get_pointer() method
        */
        check_return_type<explicit_pointer_t<T, target>>(
            log, accessor.get_pointer(), "get_pointer()");
      }
      /** dummy kernel, as no kernel is required for these checks
      */
      h.single_task(dummy_functor());
    });
  }
};

/** tests image accessors methods
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target>
class check_image_accessor_api_methods {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    auto data = [] {
      auto data = std::array<char, size>{};
      std::fill(data.begin(), data.end(), static_cast<char>(0));
      return data;
    }();
    auto image = cl::sycl::image<(
        (target == cl::sycl::access::target::image_array) ? (dims + 1) : dims)>(
        data, image_format_channel<T>::order, image_format_channel<T>::type,
        range);

    const auto check = [&log](
        const cl::sycl::accessor<T, dims, mode, target> &accessor) {
      /** check get_count() method
      */
      auto accessorCount = accessor.get_count();
      check_return_type<std::size_t>(log, accessor.get_count(), "get_count");
      if (accessorCount != count) {
        FAIL(log, "accessor does not return the correct count");
      }

      /** check get_size() method
      */
      auto accessorSize = accessor.get_size();
      check_return_type<std::size_t>(log, accessor.get_size(), "get_size");
      if (accessorSize != size) {
        FAIL(log, "accessor is not the correct size");
      }
    };

    if_constexpr<(target == cl::sycl::access::target::host_image)>(
        [&image, &check] {
          check(make_accessor<T, dims, mode, target>(image));
        },
        [&queue, &image, &check] {
          queue.submit([&](cl::sycl::handler &handler) {
            check(make_accessor<T, dims, mode, target>(image, handler));
            /** dummy kernel as no kernel is required for these checks
            */
            handler.single_task(dummy_functor());
          });
        });
  }
};

/** tests buffer accessors reads
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
class check_buffer_accessor_api_reads {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    T dataA[count];
    T dataB[count];
    for (int i = 0; i < count; i++) {
      dataA[i] = T(i);
      dataB[i] = T(i);
    }
    int errors[2] = {0};

    {
      cl::sycl::buffer<T, dims> bufferA(dataA, range);
      cl::sycl::buffer<T, dims> bufferB(dataB, range);
      cl::sycl::buffer<int, 1> errorBuffer(errors, cl::sycl::range<1>(2));

      if_constexpr<(target == cl::sycl::access::target::host_buffer)>(
          [&range, &bufferA, &bufferB, &errorBuffer] {
            auto accessorA = make_accessor<T, dims, mode, target>(bufferA);
            auto accessorB = make_accessor<T, dims, mode, target>(bufferB);
            auto errorAccessor =
                make_accessor<int, 1, cl::sycl::access::mode::write,
                              cl::sycl::access::target::host_buffer>(
                    errorBuffer);

            /** check buffer accessor subscript operators for reads
            */
            auto idList = create_id_list<dims>(range);
            for (cl::sycl::id<dims> id : idList) {
              buffer_accessor_api_r<T, dims, size, mode, target,
                                    cl::sycl::access::target::host_buffer>(
                  accessorA, accessorB, errorAccessor, range)(id);
            }
          },
          [&log, &queue, &range, &bufferA, &bufferB, &errorBuffer] {
            queue.submit([&](cl::sycl::handler &handler) {
              auto accessorA =
                  make_accessor<T, dims, mode, target, placeholder>(
                      bufferA, handler, range);
              auto accessorB =
                  make_accessor<T, dims, mode, target, placeholder>(
                      bufferB, handler, range);
              auto errorAccessor =
                  make_accessor<int, 1, cl::sycl::access::mode::write,
                                cl::sycl::access::target::global_buffer,
                                placeholder>(errorBuffer, handler);

              handler.parallel_for(
                  range,
                  buffer_accessor_api_r<T, dims, size, mode, target,
                                        cl::sycl::access::target::global_buffer,
                                        placeholder>(accessorA, accessorB,
                                                     errorAccessor, range));
            });
          });

      if_constexpr<(placeholder == cl::sycl::access::placeholder::true_t)>([&] {
        check_placeholder_command_group<placeholder>(
            log, queue, bufferA, bufferB, errorBuffer, range);
      });
    }

    if (errors[0] != 0) {
      FAIL(log, "operator[id<N>] did not assign to the correct index");
    }
    if (errors[1] != 0) {
      FAIL(log,
           "operator[size_t][size_t][size_t] did not assign to the correct "
           "index");
    }
  }

 private:
  template <cl::sycl::access::placeholder p,
            REQUIRES(p == cl::sycl::access::placeholder::true_t)>
  void check_placeholder_command_group(util::logger &log,
                                       cl::sycl::queue &queue,
                                       const cl::sycl::buffer<T, dims> &b1,
                                       const cl::sycl::buffer<T, dims> &b2,
                                       cl::sycl::range<dims> range) {
    auto a1 =
        b1.get_access<mode, target, cl::sycl::access::placeholder::true_t>();
    auto a2 =
        b2.get_access<mode, target, cl::sycl::access::placeholder::true_t>();
    auto errorAccessor = make_accessor<int, 1, cl::sycl::access::mode::write,
                                       cl::sycl::access::target::host_buffer>(
        errorBuffer, handler);

    if (!a1.is_placeholder()) {
      FAIL(log, "expected is_placeholder() == true, got false");
    }

    auto reader = buffer_accessor_api_r<T, dims, size, mode, target,
                                        cl::sycl::access::target::global_buffer,
                                        placeholder>{accessorA, accessorB,
                                                     errorAccessor, range};
    queue.submit([&](cl::sycl::handler &h) {
      h.require(a1);
      h.require(a2);
      h.single_task<class Read_placeholder_accessor>([=] {
        auto idList = create_id_list<dims>(range);
        for (cl::sycl::id<dims> id : idList) {
          reader(id);
        }
      });
    });
  }
};

/** tests buffer accessors writes
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
class check_buffer_accessor_api_writes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    auto dataA = [] {
      auto data = std::array<T, count>{};
      std::fill(data.begin(), data.end(), T{});
      return data;
    }();
    auto dataB = dataA;

    {
      cl::sycl::buffer<T, dims> bufferA(dataA, range);
      cl::sycl::buffer<T, dims> bufferB(dataB, range);
      if_constexpr<(target == cl::sycl::access::target::host_buffer)>(
          [&range, &bufferA, &bufferB] {
            auto accessorA = make_accessor<T, dims, mode, target>(bufferA);
            auto accessorB = make_accessor<T, dims, mode, target>(bufferB);

            /** check buffer accessor subscript operators for writes
            */
            auto idList = create_id_list<dims>(range);
            for (cl::sycl::id<dims> id : idList) {
              buffer_accessor_api_w<T, dims, size, mode, target>(
                  accessorA, accessorB, range)(id);
            }
          },
          [&queue, &range, &bufferA, &bufferB] {
            queue.submit([&](cl::sycl::handler &handler) {
              auto accessorA =
                  make_accessor<T, dims, mode, target, placeholder>(
                      bufferA, handler, range);
              auto accessorB =
                  make_accessor<T, dims, mode, target, placeholder>(
                      bufferB, handler, range);

              /** check buffer accessor subscript operators for writes
              */
              handler.parallel_for(
                  range, buffer_accessor_api_w<T, dims, size, mode, target,
                                               placeholder>(accessorA,
                                                            accessorB, range));
            });
          });

      if_constexpr<(placeholder == cl::sycl::access::placeholder::true_t)>(
          [&log, &queue, &bufferA] {
            check_placeholder_command_group<placeholder>(log, queue, bufferA);
          });
    }

    if (!check_linear_index(dataA, count)) {
      FAIL(log, "operator[id<N>] did not assign to the correct index");
    }

    if (!check_linear_index(dataB, count)) {
      FAIL(log,
           "operator[size_t][size_t][size_t] did not assign to the correct "
           "index");
    }
  }

 private:
  template <cl::sycl::access::placeholder p,
            REQUIRES(p == cl::sycl::access::placeholder::true_t)>
  void check_placeholder_command_group(util::logger &log,
                                       cl::sycl::queue &queue,
                                       const cl::sycl::buffer<T, dims> &b1,
                                       const cl::sycl::buffer<T, dims> &b2,
                                       cl::sycl::range<dims> range) {
    auto idList = create_id_list<dims>(range);
    auto a1 =
        b1.get_access<mode, target, cl::sycl::access::placeholder::true_t>();
    auto a2 =
        b2.get_access<mode, target, cl::sycl::access::placeholder::true_t>();

    if (!a1.is_placeholder()) {
      FAIL(log, "expected is_placeholder() == true, got false");
    }

    auto writer = buffer_accessor_api_w<T, dims, size, mode, target>{
        accessorA, accessorB, range};
    queue.submit([&](cl::sycl::handler &h) {
      h.require(a1);
      h.require(a2);
      h.single_task<class Write_placeholder_accessor>([=] {
        auto idList = create_id_list<dims>(range);
        for (cl::sycl::id<dims> id : idList) {
          writer(id);
        }
      }
    });
  });
}
};

/** tests buffer accessors reads and writes
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
class check_buffer_accessor_api_reads_and_writes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    auto dataA = [] {
      auto a = std::array<T, count>{};
      std::transform(a.begin(), a.end(), a.begin(), [](const T &) {
        static int i = 0;
        return T(i);
      });
      return a;
    }();
    auto dataB = dataA;

    constexpr bool isHostBuffer =
        (target == cl::sycl::access::target::host_buffer);
    auto errors = [] {
      auto a = std::array<int, (isHostBuffer ? 2 : 4)>{};
      std::fill(a.begin(), a.end(), 0);
      return a;
    }();

    {
      cl::sycl::buffer<T, dims> bufferA(dataA.data(), range);
      cl::sycl::buffer<T, dims> bufferB(dataB.data(), range);
      cl::sycl::buffer<int, 1> errorBuffer(errors.data(),
                                           cl::sycl::range<1>(errors.size()));
      if_constexpr<isHostBuffer>(
          [&range, &bufferA, &bufferB, &errorBuffer] {
            auto accessorA = make_accessor<T, dims, mode, target>(bufferA);
            auto accessorB = make_accessor<T, dims, mode, target>(bufferB);
            auto errorAccessor =
                make_accessor<int, 1, cl::sycl::access::mode::write,
                              cl::sycl::access::target::host_buffer>(
                    errorBuffer);

            /** check buffer accessor subscript operators for reads and writes
            */
            auto idList = create_id_list<dims>(range);
            for (cl::sycl::id<dims> id : idList) {
              buffer_accessor_api_rw<T, dims, size, mode, target,
                                     cl::sycl::access::target::host_buffer>(
                  accessorA, accessorB, errorAccessor, range)(id);
            }
          },
          [&queue, &range, &bufferA, &bufferB, &errorBuffer] {
            queue.submit([&](cl::sycl::handler &handler) {
              auto accessorA =
                  make_accessor<T, dims, mode, target, placeholder>(
                      bufferA, handler, range);
              auto accessorB =
                  make_accessor<T, dims, mode, target, placeholder>(
                      bufferB, handler, range);
              auto errorAccessor =
                  make_accessor<int, 1, cl::sycl::access::mode::write,
                                cl::sycl::access::target::global_buffer>(
                      errorBuffer, handler);

              /** check buffer accessor subscript operators for reads and writes
              */
              handler.parallel_for(
                  range, buffer_accessor_api_rw<
                             T, dims, size, mode, target,
                             cl::sycl::access::target::global_buffer>(
                             accessorA, accessorB, errorAccessor, range));
            });
          });

      if_constexpr<(placeholder == cl::sycl::access::placeholder::true_t)>([&] {
        check_placeholder_command_group<placeholder>(
            log, queue, bufferA, bufferB, errorBuffer, range);
      });
    }

    if (errors[0] != 0) {
      FAIL(log, "operator[id<N>] did not read from the correct index");
    }
    if (errors[1] != 0) {
      FAIL(log,
           "operator[size_t][size_t][size_t] did not read from the "
           "correct index");
    }
    if (!check_linear_index(dataA, count, 2)) {
      FAIL(log, "operator[id<N>] did not assign to the correct index");
    }

    if (!check_linear_index(dataB, count, 2)) {
      FAIL(log,
           "operator[size_t][size_t][size_t] did not assign to the "
           "correct index");
    }

    if_constexpr<!isHostBuffer>([&log, &errors] {
      if (errors[2] != 0) {
        FAIL(log, "operator[id<N>] did not write to the correct index");
      }
      if (errors[3] != 0) {
        FAIL(log,
             "operator[size_t][size_t][size_t] did not write to the correct "
             "index");
      }
    });
  }

 private:
  template <cl::sycl::access::placeholder p,
            REQUIRES(p == cl::sycl::access::placeholder::true_t)>
  void check_placeholder_command_group(util::logger &log,
                                       cl::sycl::queue &queue,
                                       const cl::sycl::buffer<T, dims> &b1,
                                       const cl::sycl::buffer<T, dims> &b2,
                                       cl::sycl::range<dims> range) {
    auto a1 =
        b1.get_access<mode, target, cl::sycl::access::placeholder::true_t>();
    auto a2 =
        b2.get_access<mode, target, cl::sycl::access::placeholder::true_t>();
    auto errorAccessor = make_accessor<int, 1, cl::sycl::access::mode::write,
                                       cl::sycl::access::target::host_buffer>(
        errorBuffer, handler);

    if (!a1.is_placeholder()) {
      FAIL(log, "expected is_placeholder() == true, got false");
    }

    auto reader_writer =
        buffer_accessor_api_rw<T, dims, size, mode, target,
                               cl::sycl::access::target::global_buffer,
                               placeholder>{accessorA, accessorB, errorAccessor,
                                            range};
    queue.submit([&](cl::sycl::handler &h) {
      h.require(a1);
      h.require(a2);
      h.single_task<class Read_write_placeholder_accessor>([=] {
        auto idList = create_id_list<dims>(range);
        for (cl::sycl::id<dims> id : idList) {
          reader_writer(id);
        }
      });
    });
  }
};

/** tests local accessor reads and writes
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode>
class check_local_accessor_api_reads_and_writes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    auto errors = [] {
      auto a = std::array<int, 4>{};
      std::fill(a.begin(), a.end(), 0);
      return a;
    }();

    cl::sycl::buffer<int, 1> errorBuffer(errors.data(),
                                         cl::sycl::range<1>(errors.size()));
    queue.submit([&](cl::sycl::handler &handler) {
      auto accessorA =
          make_accessor<T, dims, count, size, mode,
                        cl::sycl::access::target::local>(range, handler);
      auto accessorB =
          make_accessor<T, dims, count, size, mode,
                        cl::sycl::access::target::local>(range, handler);
      auto errorAccessor =
          make_accessor<int, 1, cl::sycl::access::mode::write,
                        cl::sycl::access::target::global_buffer>(errorBuffer,
                                                                 handler);
      /** check buffer accessor subscript operators for reads and writes
      */
      handler.parallel_for(
          range,
          buffer_accessor_api_rw<T, dims, size, mode, target,
                                 cl::sycl::access::target::global_buffer>(
              accessorA, accessorB, errorAccessor, range));
    });

    if (errors[0] != 0) {
      FAIL(log, "operator[id<N>] did not read from the correct index");
    }
    if (errors[1] != 0) {
      FAIL(log,
           "operator[size_t][size_t][size_t] did not read from the "
           "correct index");
    }
    if (!check_linear_index(dataA, count, 2)) {
      FAIL(log, "operator[id<N>] did not assign to the correct index");
    }

    if (!check_linear_index(dataB, count, 2)) {
      FAIL(log,
           "operator[size_t][size_t][size_t] did not assign to the "
           "correct index");
    }

    if (errors[2] != 0) {
      FAIL(log, "operator[id<N>] did not write to the correct index");
    }
    if (errors[3] != 0) {
      FAIL(log,
           "operator[size_t][size_t][size_t] did not write to the correct "
           "index");
    }
  }
};
/** tests image accessors reads
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target>
class check_image_accessor_api_reads {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    auto dataA = [] {
      auto data = std::array<char, size>{};
      std::iota(data.begin(), data.end(), static_cast<char>(0));
      return data;
    }();
    auto dataB = dataA;
    auto dataC = dataA;
    auto dataD = dataA;
    auto errors = std::array<int, 4>{};

    {
      constexpr auto isImageArray =
          target == cl::sycl::access::target::image_array;
      cl::sycl::image<(isImageArray ? (dims + 1) : dims)> imageA(
          dataA, image_format_channel<T>::order, image_format_channel<T>::type,
          range);
      cl::sycl::image<(isImageArray ? (dims + 1) : dims)> imageB(
          dataB, image_format_channel<T>::order, image_format_channel<T>::type,
          range);
      cl::sycl::image<(isImageArray ? (dims + 1) : dims)> imageC(
          dataC, image_format_channel<T>::order, image_format_channel<T>::type,
          range);
      cl::sycl::image<(isImageArray ? (dims + 1) : dims)> imageD(
          dataD, image_format_channel<T>::order, image_format_channel<T>::type,
          range);
      cl::sycl::buffer<int, 1> errorBuffer(errors.data(),
                                           cl::sycl::range<1>(4));

      if_constexpr<(target == cl::sycl::access::target::host_image)>(
          [&range, &imageA, &imageB, &imageC, &imageD, &errorBuffer] {
            auto accessorA = make_accessor<T, dims, mode, target>(imageA);
            auto accessorB = make_accessor<T, dims, mode, target>(imageB);
            auto accessorC = make_accessor<T, dims, mode, target>(imageC);
            auto accessorD = make_accessor<T, dims, mode, target>(imageD);
            auto errorAccessor =
                make_accessor<int, 1, cl::sycl::access::mode::write,
                              cl::sycl::access::target::host_buffer>(
                    errorBuffer);
            auto sampler =
                cl::sycl::sampler(false, cl::sycl::addressing_mode::none,
                                  cl::sycl::filtering_mode::nearest);
            /** check image accessor subscript operators for reads
            */
            auto idList = create_id_list<dims>(range);
            for (cl::sycl::id<dims> id : idList) {
              image_accessor_api_r<T, dims, size, mode, target,
                                   cl::sycl::access::target::host_buffer>(
                  accessorA, accessorB, accessorC, accessorD, sampler,
                  errorAccessor, range)(id);
            }
          },
          [&queue, &range, &imageA, &imageB, &imageC, &imageD, &errorBuffer] {
            queue.submit([&](cl::sycl::handler &handler) {
              auto accessorA =
                  make_accessor<T, dims, mode, target>(imageA, handler);
              auto accessorB =
                  make_accessor<T, dims, mode, target>(imageB, handler);
              auto accessorC =
                  make_accessor<T, dims, mode, target>(imageC, handler);
              auto accessorD =
                  make_accessor<T, dims, mode, target>(imageD, handler);
              auto errorAccessor =
                  make_accessor<int, 1, cl::sycl::access::mode::write,
                                cl::sycl::access::target::global_buffer>(
                      errorBuffer, handler);
              auto sampler =
                  cl::sycl::sampler(false, cl::sycl::addressing_mode::none,
                                    cl::sycl::filtering_mode::nearest);

              /** check image accessor subscript operators for reads
              */
              handler.parallel_for(
                  range,
                  image_accessor_api_r<T, dims, size, mode, target,
                                       cl::sycl::access::target::global_buffer>(
                      accessorA, accessorB, accessorC, accessorD, sampler,
                      errorAccessor, range));
            });
          });
    }

    if (errors[0] != 0) {
      FAIL(log, "operator[id<N>] did not assign to the correct index");
    }
    if (errors[1] != 0) {
      FAIL(log, "operator[id<N>](sampler) did not assign to the correct index");
    }
    if (errors[2] != 0) {
      FAIL(log,
           "operator[size_t][size_t][size_t] did not assign to the correct "
           "index");
    }
    if (errors[3] != 0) {
      FAIL(log,
           "operator[size_t][size_t][size_t](sampler) did not assign to the "
           "correct index");
    }
  }
};

/** tests image accessors writes
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target,
          bool hostImage = (target == cl::sycl::access::target::host_image)>
class check_image_accessor_api_writes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    auto dataA = [] {
      auto data = std::array<char, size>{};
      std::fill(data.begin(), data.end(), 0);
    }();
    auto dataB = dataA;

    {
      constexpr auto isImageArray =
          target == cl::sycl::access::target::image_array;
      auto imageA = cl::sycl::image<(isImageArray ? (dims + 1) : dims)>(
          dataA, image_format_channel<T>::order, image_format_channel<T>::type,
          range);
      auto imageB = cl::sycl::image<(isImageArray ? (dims + 1) : dims)>(
          dataB, image_format_channel<T>::order, image_format_channel<T>::type,
          range);

      if_constexpr<(target == cl::sycl::access::target::host_image)>(
          [&range, &imageA, &imageB] {
            auto accessorA = make_accessor<T, dims, mode, target>(imageA);
            auto accessorB = make_accessor<T, dims, mode, target>(imageB);

            /** check image accessor subscript operators for writes
            */
            auto idList = create_id_list<dims>(range);
            for (cl::sycl::id<dims> id : idList) {
              image_accessor_api_w<T, dims, size, mode, target>(
                  accessorA, accessorB, range)(id);
            }
          },
          [&queue, &range, &imageA, &imageB] {
            queue.submit([&](cl::sycl::handler &handler) {
              auto accessorA =
                  make_accessor<T, dims, mode, target>(imageA, handler);
              auto accessorB =
                  make_accessor<T, dims, mode, target>(imageB, handler);

              /** check image accessor subscript operators for writes
              */
              handler.parallel_for(
                  range, image_accessor_api_w<T, dims, size, mode, target>(
                             accessorA, accessorB, range));
            });
          });
    }

    if (!check_linear_index(dataA, count)) {
      FAIL(log, "operator[id<N>] did not assign to the correct index");
    }

    if (!check_linear_index(dataB, count)) {
      FAIL(log,
           "operator[size_t][size_t][size_t] did not assign to the correct "
           "index");
    }
  }
};

/** tests buffer accessors subscript operators
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
void check_buffer_accessor_api_subscripts(util::logger &log,
                                          cl::sycl::queue &queue,
                                          cl::sycl::range<dims> range) {
  /** think of the following as a compile-time conditional operator.
    * the alternative is to have five explicit class specialisations that
    * add duplicate code and decrease readability
    *
    * lambdas serve as compound statements, and return their values so that
    * the compiler can deduce the type of `test`.
    */
  auto test = if_constexpr<(mode == cl::sycl::access::mode::read)>(
      [] {
        check_buffer_accessor_api_reads<T, dims, count, size, mode, target,
                                        placeholder>{};
      },
      [] {
        return if_constexpr<(mode == cl::sycl::access::mode::write ||
                             mode == cl::sycl::access::mode::discard_write)>(
            [] {
              return check_buffer_accessor_api_writes<
                  T, dims, count, size, mode, target, placeholder>{};
            },
            [] {
              return check_buffer_accessor_api_reads_and_writes<
                  T, dims, count, size, mode, target, placeholder>{};
            });
      });

  test(log, queue, range);
}

/** tests local accessor subscript operators
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode>
void check_local_accessor_api_subscripts(util::logger &log,
                                         cl::sycl::queue &queue,
                                         cl::sycl::range<dims> range) {
  check_local_accessor_api_read_and_write<T, dims, count, size, mode>(
      log, queue, range);
}

/** tests buffer accessors with different modes
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
void check_buffer_accessor_api_mode(util::logger &log, cl::sycl::queue &queue,
                                    cl::sycl::range<dims> range) {
  /** check buffer accessor subscript operators
  */
  check_buffer_accessor_api_subscripts<T, dims, count, size, mode, target,
                                       placeholder>(log, queue, range);

  /** check buffer accessor other apis
  */
  check_buffer_accessor_api_methods<T, dims, count, size, mode, target,
                                    placeholder>()(log, queue, range);
}

/** tests local accessors with different modes
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode>
void check_local_accessor_api_mode(util::logger &log, cl::sycl::queue &queue,
                                   cl::sycl::range<dims> range) {
  /** check buffer accessor subscript operators
    */
  check_local_accessor_api_subscripts<T, dims, count, size, mode>(log, queue,
                                                                  range);

  /** check buffer accessor other apis
  */
  check_local_accessor_api_methods<T, dims, count, size, mode>()(log, queue,
                                                                 range);
}

/** tests image accessors with different modes
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target>
void check_image_accessor_api_mode(util::logger &log, cl::sycl::queue &queue,
                                   cl::sycl::range<dims> range) {
  auto test = if_constexpr<(mode == cl::sycl::access::mode::read)>(
      [] {
        return check_image_accessor_api_reads<T, dims, count, size, mode,
                                              target>{};
      },
      [] {
        return check_image_accessor_api_writes<T, dims, count, size, mode,
                                               target>{};
      });

  test(log, queue, range);

  /** check buffer accessor other apis
  */
  check_image_accessor_api_methods<T, dims, count, size, mode, target>
      otherAPITests;
  otherAPITests(log, queue, range);
}

/** tests buffer accessors with different targets
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
void check_buffer_accessor_api_target(util::logger &log, cl::sycl::queue &queue,
                                      cl::sycl::range<dims> range) {
  using cl::sycl::access::mode;

  /** check buffer accessor api for read
  */
  check_buffer_accessor_api_mode<T, dims, count, size, mode::read, target,
                                 placeholder>(log, queue, range);

  if_constexpr<target != cl::sycl::access::target::constant_buffer>([&] {
    /** check buffer accessor api for read_write
    */
    check_buffer_accessor_api_mode<T, dims, count, size, mode::read_write,
                                   target, placeholder>(log, queue, range);

    /** check buffer accessor api for write
    */
    check_buffer_accessor_api_mode<T, dims, count, size, mode::write, target>(
        log, queue, range);
    if_constexpr<target != cl::sycl::access::target::host_buffer>([&] {
      /** check buffer accessor api for discard_write
      */
      check_buffer_accessor_api_mode<T, dims, count, size, mode::discard_write,
                                     target>(log, queue, range);

      /** check buffer accessor api for discard_read_write
      */
      check_buffer_accessor_api_mode<T, dims, count, size,
                                     mode::discard_read_write, target>(
          log, queue, range);
    });
  });
}

/** tests local accessors with different targets
*/
template <typename T, int dims, int count, int size>
void check_local_accessor_api_target(util::logger &log, cl::sycl::queue &queue,
                                     cl::sycl::range<dims> range) {
  using cl::sycl::access::mode;
  check_local_accessor_api_mode<T, dims, count, size, mode::read_write>(
      log, queue, range);
}

/** tests image accessors with different targets
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::target target>
void check_image_accessor_api_target(util::logger &log, cl::sycl::queue &queue,
                                     cl::sycl::range<dims> range) {
  /** check image accessor api for read
  */
  check_image_accessor_api_mode<T, dims, count, size,
                                cl::sycl::access::mode::read, target>(
      log, queue, range);

  /** check image accessor api for write
  */
  check_image_accessor_api_mode<T, dims, count, size,
                                cl::sycl::access::mode::write, target>(
      log, queue, range);
}

/** tests buffer accessors with different placeholder values
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::target target>
void check_buffer_accessor_api_placeholder(util::logger &log,
                                           cl::sycl::queue &queue,
                                           cl::sycl::range<dims> range) {
  check_buffer_accessor_api_target<T, dims, count, size, target>(log, queue,
                                                                 range);
  check_buffer_accessor_api_target<T, dims, count, size, target,
                                   cl::sycl::access::placeholder::true_t>(
      log, queue, range);
}

/** tests buffer accessors with different dimensions
*/
template <typename T, int dims, int count, int size>
void check_buffer_accessor_api_dim(util::logger &log, cl::sycl::queue &queue,
                                   cl::sycl::range<dims> range) {
  /** check buffer accessor api for global_buffer
  */
  check_buffer_accessor_api_placeholder<
      T, dims, count, size, cl::sycl::access::target::global_buffer>(log, queue,
                                                                     range);

  /** check buffer accessor api for constant_buffer
  */
  check_buffer_accessor_api_placeholder<
      T, dims, count, size, cl::sycl::access::target::constant_buffer>(
      log, queue, range);

  /** check buffer accessor api for host_buffer
  */
  check_buffer_accessor_api_target<T, dims, count, size,
                                   cl::sycl::access::target::host_buffer>(
      log, queue, range);
}

/** tests local accessors with different dimensions
*/
template <typename T, int dims, int count, int size>
void check_local_accessor_api_dim(util::logger &log, cl::sycl::queue &queue,
                                  cl::sycl::range<dims> range) {
  check_local_accessor_api_target<T, dims, count, size>(log, queue, range);
}

/** tests image accessors with different dimensions
*/
template <typename T, int dims, int count, int size>
void check_image_accessor_api_dim(util::logger &log, cl::sycl::queue &queue,
                                  cl::sycl::range<dims> range) {
  /** check image accessor api for image
  */
  check_image_accessor_api_target<T, dims, count, size,
                                  cl::sycl::access::target::image>(log, queue,
                                                                   range);

  /** check image accessor api for host_image
  */
  check_image_accessor_api_target<T, dims, count, size,
                                  cl::sycl::access::target::host_image>(
      log, queue, range);
}

/** tests buffer accessors with different types
*/
template <typename T>
class check_buffer_accessor_api_type {
  constexpr auto count = 8;
  constexpr auto size = count * sizeof(T);

 public:
  void operator(util::logger &log, cl::sycl::queue &queue) {
    /** check buffer accessor api for 1 dimension
    */
    check(log, queue, cl::sycl::range<1> range1d(count));

    /** check buffer accessor api for 2 dimension
    */
    check(log, queue, cl::sycl::range<2> range2d(count / 4, 4));

    /** check buffer accessor api for 3 dimension
    */
    check(log, queue, cl::sycl::range<3> range3d(count / 8, 4, 2));
  }

 private:
  template <int dims>
  void check(util::logger &log, cl::sycl::queue &queue,
             cl::sycl::range<dims> r) {
    check_buffer_accessor_api_dim<T, dims, count, size>(log, queue, r);
  }
};

/**
*/
template <typename T>
class check_buffer_accessor_api_type {
  constexpr auto count = 8;
  constexpr auto size = count * sizeof(T);

 public:
  void operator()(util::logger &log, cl::sycl::queue &queue) {
    /** check local accessor api for 1 dimension
    */
    check<1>(log, queue);

    /** check local accessor api for 2 dimensions
    */
    check<2>(log, queue);

    /** check local accessor api for 3 dimensions
    */
    check<3>(log, queue);
  }

 private:
  template <int dims>
  void check(util::logger &log, cl::sycl::queue &queue,
             cl::sycl::range<dims> r) {
    check_local_accessor_api_dim<T, dims, count, size>(log, queue, r);
  }
};

/** tests image accessors with different types
*/
template <typename T>
void check_image_accessor_api_type(util::logger &log, cl::sycl::queue &queue) {
  const int count = 8;
  const int size = count * 4;

  /** check image accessor api for 1 dimension
  */
  cl::sycl::range<1> range1d(count);
  check_image_accessor_api_dim<T, 1, count, size>(log, queue, range1d);

  /** check image accessor api for 2 dimension
  */
  cl::sycl::range<2> range2d(count / 4, 4);
  check_image_accessor_api_dim<T, 2, count, size>(log, queue, range2d);

  /** check image accessor api for 3 dimension
  */
  cl::sycl::range<3> range3d(count / 8, 4, 2);
  check_image_accessor_api_dim<T, 3, count, size>(log, queue, range3d);
}

/** tests the api for cl::sycl::accessor
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
  */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
  */
  void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

      /** check buffer accessors
      */
      accessor_check<check_buffer_accessor_api_type>(log, queue);

      /** check local accessors
      */
      accessor_check<check_local_accessor_api_type>(log, queue);

      /** check image accessor api for int4
      */
      check_image_accessor_api_type<cl::sycl::int4>(log, queue);

      /** check image accessor api for uint4
      */
      check_image_accessor_api_type<cl::sycl::uint4>(log, queue);

      /** check image accessor api for float4
      */
      check_image_accessor_api_type<cl::sycl::float4>(log, queue);

      queue.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }

 private:
  template <template <typename> class A>
  void accessor_check(util::logger &log, cl::sycl::queue &queue) {
    /** check buffer accessor api for int
          */
    A<int>()(log, queue);

    /** check buffer accessor api for float
    */

    A<float>()(log, queue);

    /** check buffer accessor api for double
    */
    A<double>()(log, queue);

    /** check buffer accessor api for char
    */
    A<char>()(log, queue);

    /** check buffer accessor api for vec
    */
    A<cl::sycl::int2>()(log, queue);

    /** check buffer accessor api for user_struct
    */
    A<user_struct>()(log, queue);
  }
};

/** register this test with the test_collection
*/
util::test_proxy<TEST_NAME> proxy;

}  // namespace accessor_api__
