/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "./../../util/math_helper.h"

#define TEST_NAME accessor_api

namespace accessor_api__ {
using namespace sycl_cts;

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
  for (size_t i = 0; i < r[0]; ++i) ret.push_back(cl::sycl::id<1>(i));
  return ret;
}

/** creates a list of ids (specialization for 2 dimension)
*/
template <>
cl::sycl::vector_class<cl::sycl::id<2>> create_id_list<2>(
    cl::sycl::range<2> &r) {
  cl::sycl::vector_class<cl::sycl::id<2>> ret;
  for (size_t i = 0; i < r[0]; ++i)
    for (size_t j = 0; j < r[1]; ++j) ret.push_back(cl::sycl::id<2>(i, j));
  return ret;
}

/** creates a list of ids (specialization for 3 dimension)
*/
template <>
cl::sycl::vector_class<cl::sycl::id<3>> create_id_list<3>(
    cl::sycl::range<3> &r) {
  cl::sycl::vector_class<cl::sycl::id<3>> ret;
  for (size_t i = 0; i < r[0]; ++i)
    for (size_t j = 0; j < r[1]; ++j)
      for (size_t k = 0; k < r[2]; ++k) ret.push_back(cl::sycl::id<3>(i, j, k));
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

/** image format channel order and type
*/
template <typename T>
struct image_format_channel;

/** image format channel order and type (specialization for int4)
*/
template <>
struct image_format_channel<cl::sycl::int4> {
  static const cl::sycl::image_channel_type type =
      cl::sycl::image_channel_type::signed_int8;
  static const cl::sycl::image_channel_order order =
      cl::sycl::image_channel_order::rgba;
};

/** image format channel order and type (specialization for uint4)
*/
template <>
struct image_format_channel<cl::sycl::uint4> {
  static const cl::sycl::image_channel_type type =
      cl::sycl::image_channel_type::unsigned_int8;
  static const cl::sycl::image_channel_order order =
      cl::sycl::image_channel_order::rgba;
};

/** image format channel order and type (specialization for float4)
*/
template <>
struct image_format_channel<cl::sycl::float4> {
  static const cl::sycl::image_channel_type type =
      cl::sycl::image_channel_type::unorm_int8;
  static const cl::sycl::image_channel_order order =
      cl::sycl::image_channel_order::rgba;
};

/** specialized struct for defining the normalization coefficient for an image
 * accessor type. 1.0f by default.
*/
template <typename elementT>
struct use_normalization_coefficient {
  static const bool value = false;
};

/** specialized struct for defining the normalization coefficient for an image
 * accessor type. Specializationf or cl::sycl::float4.
*/
template <>
struct use_normalization_coefficient<cl::sycl::float4> {
  static const bool value = true;
};

/** creates an accessor
*/
template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
struct make_accessor {
  make_accessor() = default;

  cl::sycl::accessor<T, dims, mode, target> operator()(
      cl::sycl::buffer<T, dims> buf, cl::sycl::handler &hand,
      cl::sycl::range<dims> range) {
    return buf.template get_access<mode, target>(hand);
  }

  cl::sycl::accessor<T, dims, mode, target> operator()(
      cl::sycl::image<dims> img, cl::sycl::handler &hand,
      cl::sycl::range<dims> range) {
    return img.template get_access<mode, target>(hand);
  }
};

/** creates an accessor (specialization for local)
*/
template <typename T, int dims, int size, cl::sycl::access::mode mode>
struct make_accessor<T, dims, size, mode, cl::sycl::access::target::local> {
  make_accessor() = default;

  cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::local> operator()(
      cl::sycl::buffer<T, dims> b, cl::sycl::handler &handler,
      cl::sycl::range<dims> range) {
    return cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::local>(
        range, handler);
  }
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
template <typename T1, typename T2,
          typename std::enable_if<std::is_scalar<T1>::value>::type * = nullptr>
bool check_element_valid(const T1 &elem, const T2 &correct,
                         bool imageTest = false) {
  return elem == (static_cast<T1>(correct));
};

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
template <typename T1,
          typename std::enable_if<(is_sycl_vec<T1>::value)>::type * = nullptr>
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
};

/** tests that two values are equal (overload for user_struct)
*/
template <typename T2>
bool check_element_valid(const user_struct &elem, const T2 &correct,
                         bool imageTest = false) {
  return elem[0] == (static_cast<int>(correct));
};

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
*/
template <typename accessorT>
typename accessorT::value_type multidim_subscript_read(accessorT &acc,
                                                       cl::sycl::id<1> idx) {
  return acc[idx[0]];
}

/** tests accessor multi dim read syntax for 2 dimension
*/
template <typename accessorT>
typename accessorT::value_type multidim_subscript_read(accessorT &acc,
                                                       cl::sycl::id<2> idx) {
  return acc[idx[0]][idx[1]];
}

/** tests accessor multi dim read syntax for 3 dimension
*/
template <typename accessorT>
typename accessorT::value_type multidim_subscript_read(accessorT &acc,
                                                       cl::sycl::id<3> idx) {
  return acc[idx[0]][idx[1]][idx[2]];
}

/** tests accessor multi dim sampled read syntax for 1 dimension
*/
template <typename accessorT>
typename accessorT::value_type multidim_subscript_sampled_read(
    accessorT &acc, cl::sycl::sampler smpl, cl::sycl::id<1> idx) {
  return acc(smpl)[idx[0]];
}

/** tests accessor multi dim sampled read syntax for 2 dimension
*/
template <typename accessorT>
typename accessorT::value_type multidim_subscript_sampled_read(
    accessorT &acc, cl::sycl::sampler smpl, cl::sycl::id<2> idx) {
  return acc(smpl)[idx[0]][idx[1]];
}

/** tests accessor multi dim sampled read syntax for 3 dimension
*/
template <typename accessorT>
typename accessorT::value_type multidim_subscript_sampled_read(
    accessorT &acc, cl::sycl::sampler smpl, cl::sycl::id<3> idx) {
  return acc(smpl)[idx[0]][idx[1]][idx[2]];
}

/** tests accessor multi dim write syntax for 1 dimension
*/
template <typename accessorT, typename valueT>
void multidim_subscript_write(accessorT &acc, cl::sycl::id<1> idx,
                              valueT value) {
  acc[idx[0]] = value;
}

/** tests accessor multi dim write syntax for 2 dimension
*/
template <typename accessorT, typename valueT>
void multidim_subscript_write(accessorT &acc, cl::sycl::id<2> idx,
                              valueT value) {
  acc[idx[0]][idx[1]] = value;
}

/** tests accessor multi dim write syntax for 3 dimension
*/
template <typename accessorT, typename valueT>
void multidim_subscript_write(accessorT &acc, cl::sycl::id<3> idx,
                              valueT value) {
  acc[idx[0]][idx[1]][idx[2]] = value;
}

/** tests buffer accessors reads
*/
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target, cl::sycl::access::target errorTarget>
class buffer_accessor_api_r {
  cl::sycl::accessor<T, dim, mode, target> accessorA_;
  cl::sycl::accessor<T, dim, mode, target> accessorB_;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, errorTarget>
      errorAccessor_;
  cl::sycl::range<dim> range_;

 public:
  buffer_accessor_api_r(
      cl::sycl::accessor<T, dim, mode, target> accessorA,
      cl::sycl::accessor<T, dim, mode, target> accessorB,
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, errorTarget>
          errorAccessor,
      cl::sycl::range<dim> rng)
      : accessorA_(accessorA),
        accessorB_(accessorB),
        errorAccessor_(errorAccessor),
        range_(rng) {}

  void operator()(cl::sycl::id<dim> idx) {
    size_t linearID = compute_linear_id(idx, range_);
    T elem;

    /** check id read syntax
    */
    elem = accessorA_[idx];
    if (!check_element_valid(elem, linearID, false)) {
      errorAccessor_[0] = 1;
    }

    /** check size_t read syntax
    */
    elem = multidim_subscript_read(accessorB_, idx);
    if (!check_element_valid(elem, linearID, false)) {
      errorAccessor_[1] = 1;
    }
  };
};

/** tests buffer accessors writes
*/
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class buffer_accessor_api_w {
  cl::sycl::accessor<T, dim, mode, target> accessorA_;
  cl::sycl::accessor<T, dim, mode, target> accessorB_;
  cl::sycl::range<dim> range_;

 public:
  buffer_accessor_api_w(cl::sycl::accessor<T, dim, mode, target> accessorA,
                        cl::sycl::accessor<T, dim, mode, target> accessorB,
                        cl::sycl::range<dim> r)
      : accessorA_(accessorA), accessorB_(accessorB), range_(r) {}

  void operator()(cl::sycl::id<dim> idx) {
    size_t linearID = compute_linear_id(idx, range_);

    /** check id write syntax
    */
    accessorA_[idx] = linearID;

    /** check size_t write syntax
    */
    multidim_subscript_write(accessorB_, idx, linearID);
  };
};

/** tests buffer accessors reads and writes
*/
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target, cl::sycl::access::target errorTarget>
class buffer_accessor_api_rw {
  cl::sycl::accessor<T, dim, mode, target> accessorA_;
  cl::sycl::accessor<T, dim, mode, target> accessorB_;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, errorTarget>
      errorAccessor_;
  cl::sycl::range<dim> range_;

 public:
  buffer_accessor_api_rw(
      cl::sycl::accessor<T, dim, mode, target> accessorA,
      cl::sycl::accessor<T, dim, mode, target> accessorB,
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, errorTarget>
          errorAccessor,
      cl::sycl::range<dim> rng)
      : accessorA_(accessorA),
        accessorB_(accessorB),
        errorAccessor_(errorAccessor),
        range_(rng) {}

  void operator()(cl::sycl::id<dim> idx) {
    size_t linearID = compute_linear_id(idx, range_);
    T elem;

    /** check id read syntax
    */
    elem = accessorA_[idx];
    if (!check_element_valid(elem, linearID)) {
      errorAccessor_[0] = 1;
    }

    /** check size_t read syntax
    */
    elem = multidim_subscript_read(accessorB_, idx);
    if (!check_element_valid(elem, linearID)) {
      errorAccessor_[1] = 1;
    }

    /** check id write syntax
    */
    accessorA_[idx] = (linearID * 2);

    /** check size_t write syntax
    */
    multidim_subscript_write(accessorB_, idx, (linearID * 2));

    /** validate id write syntax
    */
    elem = accessorA_[idx];
    if (!check_element_valid(elem, (linearID * 2))) {
      errorAccessor_[2] = 1;
    }

    /** validate size_t write syntax
    */
    elem = multidim_subscript_read(accessorB_, idx);
    if (!check_element_valid(elem, (linearID * 2))) {
      errorAccessor_[3] = 1;
    }
  };
};

/** tests image accessors reads
*/
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target, cl::sycl::access::target errorTarget>
class image_accessor_api_r {
  cl::sycl::accessor<T, dim, mode, target> accessorA_;
  cl::sycl::accessor<T, dim, mode, target> accessorB_;
  cl::sycl::accessor<T, dim, mode, target> accessorC_;
  cl::sycl::accessor<T, dim, mode, target> accessorD_;
  cl::sycl::sampler sampler_;

  cl::sycl::accessor<int, 1, cl::sycl::access::mode::write, errorTarget>
      errorAccessor_;
  cl::sycl::range<dim> range_;

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
      : accessorA_(accessorA),
        accessorB_(accessorB),
        accessorC_(accessorC),
        accessorD_(accessorD),
        sampler_(smpl),
        errorAccessor_(errorAccessor),
        range_(rng) {}

  void operator()(cl::sycl::id<dim> idx) {
    size_t linearID = compute_linear_id(idx, range_);
    T elem;

    /** check id read syntax
    */
    elem = accessorA_[idx];
    if (use_normalization_coefficient<T>::value) {
      elem *= 255.f;
    }
    if (!check_element_valid(elem, linearID, true)) {
      errorAccessor_[0] = 1;
    }

    /** check sampled id read syntax
    */
    elem = accessorB_(sampler_)[idx];
    if (use_normalization_coefficient<T>::value) {
      elem *= 255.f;
    }
    if (!check_element_valid(elem, linearID, true)) {
      errorAccessor_[1] = 1;
    }

    /** check size_t read syntax
    */
    elem = multidim_subscript_read(accessorC_, idx);
    if (use_normalization_coefficient<T>::value) {
      elem *= 255.f;
    }
    if (!check_element_valid(elem, linearID, true)) {
      errorAccessor_[2] = 1;
    }

    /** check sampled size_t read syntax
    */
    elem = multidim_subscript_sampled_read(accessorD_, sampler_, idx);
    if (use_normalization_coefficient<T>::value) {
      elem *= 255.f;
    }
    if (!check_element_valid(elem, linearID, true)) {
      errorAccessor_[3] = 1;
    }
  }
};

/** tests image accessors writes
*/
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class image_accessor_api_w {
  cl::sycl::accessor<T, dim, mode, target> accessorA_;
  cl::sycl::accessor<T, dim, mode, target> accessorB_;
  cl::sycl::range<dim> range_;

 public:
  image_accessor_api_w(cl::sycl::accessor<T, dim, mode, target> accessorA,
                       cl::sycl::accessor<T, dim, mode, target> accessorB,
                       cl::sycl::range<dim> rng)
      : accessorA_(accessorA), accessorB_(accessorB), range_(rng) {}

  void operator()(cl::sycl::id<dim> idx) {
    size_t linearID = compute_linear_id(idx, range_);
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
    accessorA_[idx] = T(elem0, elem1, elem2, elem3);

    /** check size_t write syntax
    */
    multidim_subscript_write(accessorB_, idx, T(elem0, elem1, elem2, elem3));
  }
};

/** tests buffer accessors methods
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target>
class check_buffer_accessor_api_methods {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    T data[count];
    ::memset(data, 0xFF, sizeof(data));
    cl::sycl::buffer<T, dims> buffer(data, range);

    queue.submit([&](cl::sycl::handler &handler) {
      auto accessor =
          make_accessor<T, dims, size, mode, target>()(buffer, handler, range);

      /** check get_count() method
      */
      auto accessorCount = accessor.get_count();
      if (typeid(accessorCount) != typeid(size_t)) {
        FAIL(log, "get_count() does not return size_t");
      }
      if (accessorCount != count) {
        FAIL(log, "accessor does not return the correct count");
      }

      /** check get_size() method
      */
      auto accessorSize = accessor.get_size();
      if (typeid(accessorSize) != typeid(size_t)) {
        FAIL(log, "get_size() does not return size_t");
      }
      if (accessorSize != size) {
        FAIL(log, "accessor does not return the correct size");
      }

      /** check get_size() method
      */
      auto accessorRange = accessor.get_range();
      if (typeid(accessorRange) != typeid(range<dims>)) {
        FAIL(log, "get_range() does not return size_t");
      }
      if (accessorRange != range) {
        FAIL(log, "accessor does not return the correct range (get_range)");
      }

      /** check get_pointer() method
      */
      auto accessorPointer = accessor.get_pointer();
      if (!std::is_same<decltype(accessorPointer),
                        typename explicit_pointer<T, target>::type>::value) {
        FAIL(log, "get_pointer() does not return the explicit pointer type");
      }

      /** dummy kernel as no kernel is required for these checks
      */
      handler.single_task(dummy_functor());
    });
  }
};

/** tests buffer accessors methods (specialization for host_buffer)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode>
class check_buffer_accessor_api_methods<T, dims, count, size, mode,
                                        cl::sycl::access::target::host_buffer> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    T data[count];
    ::memset(data, 0xFF, sizeof(data));
    cl::sycl::buffer<T, dims> buffer(data, range);

    cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
        accessor(buffer);

    /** check get_count() method
    */
    auto accessorCount = accessor.get_count();
    if (typeid(accessorCount) != typeid(size_t)) {
      FAIL(log, "get_count() does not return size_t");
    }
    if (accessorCount != count) {
      FAIL(log, "accessor does not return the correct count");
    }

    /** check get_size() method
    */
    auto accessorSize = accessor.get_size();
    if (typeid(accessorSize) != typeid(size_t)) {
      FAIL(log, "get_size() does not return size_t");
    }
    if (accessorSize != size) {
      FAIL(log, "accessor does not return the correct size");
    }

    /** check get_event() method
    */
    auto accessorEvent = accessor.get_event();
    if (typeid(accessorEvent) != typeid(cl::sycl::event)) {
      FAIL(log, "get_event() does not return event");
    }

    /** check get_pointer() method
    */
    auto accessorPointer = accessor.get_pointer();
    if (typeid(accessorPointer) != typeid(T *)) {
      FAIL(log, "get_pointer() does not return T*");
    }
  }
};

template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target>
class kernel_name;

/** tests image accessors methods
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target>
class check_image_accessor_api_methods {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    char data[size];
    ::memset(data, 0, sizeof(data));
    cl::sycl::image<dims> image(image_format_channel<T>::order,
                                image_format_channel<T>::type, range);

    queue.submit([&](cl::sycl::handler &handler) {
      cl::sycl::accessor<T, dims, mode, target> accessor(image, handler);

      /** check get_count() method
      */
      auto accessorCount = accessor.get_count();
      if (typeid(accessorCount) != typeid(size_t)) {
        FAIL(log, "get_count() does not return size_t");
      }
      if (accessorCount != count) {
        FAIL(log, "accessor does not return the correct count");
      }

      /** check get_size() method
      */
      auto accessorSize = accessor.get_size();
      if (typeid(accessorSize) != typeid(size_t)) {
        FAIL(log, "get_size() does not return size_t");
      }
      if (accessorSize != size) {
        FAIL(log, "accessor is not the correct size");
      }

      /** check get_event() method
      */
      auto accessorEvent = accessor.get_event();
      if (typeid(accessorEvent) != typeid(cl::sycl::event)) {
        FAIL(log, "get_event() does not return event");
      }

      /** dummy kernel as no kernel is required for these checks
      */
      handler.single_task(dummy_functor());
    });
  }
};

/** tests image accessors methods (specialization for host_image)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode>
class check_image_accessor_api_methods<T, dims, count, size, mode,
                                       cl::sycl::access::target::host_image> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    char data[size];
    ::memset(data, 0, sizeof(data));
    cl::sycl::image<dims> image(data, image_format_channel<T>::order,
                                image_format_channel<T>::type, range);

    cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
        accessor(image);

    /** check get_count() method
    */
    auto accessorCount = accessor.get_count();
    if (typeid(accessorCount) != typeid(size_t)) {
      FAIL(log, "get_count() does not return size_t");
    }
    if (accessorCount != count) {
      FAIL(log, "accessor does not return the correct count");
    }

    /** check get_size() method
    */
    auto accessorSize = accessor.get_size();
    if (typeid(accessorSize) != typeid(size_t)) {
      FAIL(log, "get_size() does not return size_t");
    }
    if (accessorSize != size) {
      FAIL(log, "accessor is not the correct size");
    }

    /** check get_event() method
    */
    auto accessorEvent = accessor.get_event();
    if (typeid(accessorEvent) != typeid(cl::sycl::event)) {
      FAIL(log, "get_event() does not return event");
    }
  }
};

/** tests buffer accessors reads
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target>
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

      queue.submit([&](cl::sycl::handler &handler) {
        auto accessorA = make_accessor<T, dims, size, mode, target>()(
            bufferA, handler, range);
        auto accessorB = make_accessor<T, dims, size, mode, target>()(
            bufferB, handler, range);
        cl::sycl::accessor<int, 1, cl::sycl::access::mode::write,
                           cl::sycl::access::target::global_buffer>
            errorAccessor(errorBuffer, handler);

        /** check buffer accessor subscript operators for reads
        */
        handler.parallel_for(
            range,
            buffer_accessor_api_r<T, dims, size, mode, target,
                                  cl::sycl::access::target::global_buffer>(
                accessorA, accessorB, errorAccessor, range));
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
};

/** tests buffer accessors reads (specialized for host_buffer)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode>
class check_buffer_accessor_api_reads<T, dims, count, size, mode,
                                      cl::sycl::access::target::host_buffer> {
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

      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
          accessorA(bufferA);
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
          accessorB(bufferB);
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::host_buffer>
          errorAccessor(errorBuffer);

      /** check buffer accessor subscript operators for reads
      */
      auto idList = create_id_list<dims>(range);
      for (cl::sycl::id<dims> id : idList) {
        buffer_accessor_api_r<T, dims, size, mode,
                              cl::sycl::access::target::host_buffer,
                              cl::sycl::access::target::host_buffer>(
            accessorA, accessorB, errorAccessor, range)(id);
      }
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
};

/** tests buffer accessors writes
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target>
class check_buffer_accessor_api_writes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    T dataA[count];
    T dataB[count];
    ::memset(dataA, 0xFF, sizeof(dataA));
    ::memset(dataB, 0xFF, sizeof(dataB));

    {
      cl::sycl::buffer<T, dims> bufferA(dataA, range);
      cl::sycl::buffer<T, dims> bufferB(dataB, range);

      queue.submit([&](cl::sycl::handler &handler) {
        auto accessorA = make_accessor<T, dims, size, mode, target>()(
            bufferA, handler, range);
        auto accessorB = make_accessor<T, dims, size, mode, target>()(
            bufferB, handler, range);

        /** check buffer accessor subscript operators for writes
        */
        handler.parallel_for(range,
                             buffer_accessor_api_w<T, dims, size, mode, target>(
                                 accessorA, accessorB, range));
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

/** tests buffer accessors writes (specialized for host_buffer)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode>
class check_buffer_accessor_api_writes<T, dims, count, size, mode,
                                       cl::sycl::access::target::host_buffer> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    T dataA[count];
    T dataB[count];
    ::memset(dataA, 0xFF, sizeof(dataA));
    ::memset(dataB, 0xFF, sizeof(dataB));

    {
      cl::sycl::buffer<T, dims> bufferA(dataA, range);
      cl::sycl::buffer<T, dims> bufferB(dataB, range);

      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
          accessorA(bufferA);
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
          accessorB(bufferB);

      /** check buffer accessor subscript operators for writes
      */
      auto idList = create_id_list<dims>(range);
      for (cl::sycl::id<dims> id : idList) {
        buffer_accessor_api_w<T, dims, size, mode,
                              cl::sycl::access::target::host_buffer>(
            accessorA, accessorB, range)(id);
      }
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

/** tests buffer accessors reads and writes
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target>
class check_buffer_accessor_api_reads_and_writes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    T dataA[count];
    T dataB[count];
    for (int i = 0; i < count; i++) {
      dataA[i] = T(i);
      dataB[i] = T(i);
    }
    int errors[4] = {0};

    {
      cl::sycl::buffer<T, dims> bufferA(dataA, range);
      cl::sycl::buffer<T, dims> bufferB(dataB, range);
      cl::sycl::buffer<int, 1> errorBuffer(errors, cl::sycl::range<1>(4));

      queue.submit([&](cl::sycl::handler &handler) {
        auto accessorA = make_accessor<T, dims, size, mode, target>()(
            bufferA, handler, range);
        auto accessorB = make_accessor<T, dims, size, mode, target>()(
            bufferB, handler, range);
        cl::sycl::accessor<int, 1, cl::sycl::access::mode::write,
                           cl::sycl::access::target::global_buffer>
            errorAccessor(errorBuffer, handler);

        /** check buffer accessor subscript operators for reads and writes
        */
        handler.parallel_for(
            range,
            buffer_accessor_api_rw<T, dims, size, mode, target,
                                   cl::sycl::access::target::global_buffer>(
                accessorA, accessorB, errorAccessor, range));
      });
    }

    /** the initial and final values of the accessors are not tested for local
     * acccessors
    */
    if (target != cl::sycl::access::target::local) {
      if (errors[0] != 0) {
        FAIL(log, "operator[id<N>] did not read from the correct index");
      }
      if (errors[1] != 0) {
        FAIL(log,
             "operator[size_t][size_t][size_t] did not read from the correct "
             "index");
      }
      if (!check_linear_index(dataA, count, 2)) {
        FAIL(log, "operator[id<N>] did not assign to the correct index");
      }

      if (!check_linear_index(dataB, count, 2)) {
        FAIL(log,
             "operator[size_t][size_t][size_t] did not assign to the correct "
             "index");
      }
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

/** tests buffer accessors reads and writes (specialized for host_buffer)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode>
class check_buffer_accessor_api_reads_and_writes<
    T, dims, count, size, mode, cl::sycl::access::target::host_buffer> {
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

      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
          accessorA(bufferA);
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
          accessorB(bufferB);
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::host_buffer>
          errorAccessor(errorBuffer);

      /** check buffer accessor subscript operators for reads and writes
      */
      auto idList = create_id_list<dims>(range);
      for (cl::sycl::id<dims> id : idList) {
        buffer_accessor_api_rw<T, dims, size, mode,
                               cl::sycl::access::target::host_buffer,
                               cl::sycl::access::target::host_buffer>(
            accessorA, accessorB, errorAccessor, range)(id);
      }
    }

    if (errors[0] != 0) {
      FAIL(log, "operator[id<N>] did not assign to the correct index");
    }
    if (errors[1] != 0) {
      FAIL(log,
           "operator[size_t][size_t][size_t] did not assign to the correct "
           "index");
    }

    if (!check_linear_index(dataA, count, 2)) {
      FAIL(log, "operator[id<N>] did not assign to the correct index");
    }

    if (!check_linear_index(dataB, count, 2)) {
      FAIL(log,
           "operator[size_t][size_t][size_t] did not assign to the correct "
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
    char dataA[size] = {0};
    char dataB[size] = {0};
    char dataC[size] = {0};
    char dataD[size] = {0};

    for (size_t i = 0; i < size; i++) {
      dataA[i] = static_cast<char>(i);
      dataB[i] = static_cast<char>(i);
      dataC[i] = static_cast<char>(i);
      dataD[i] = static_cast<char>(i);
    }

    int errors[4] = {0};

    {
      cl::sycl::image<dims> imageA(dataA, image_format_channel<T>::order,
                                   image_format_channel<T>::type, range);
      cl::sycl::image<dims> imageB(dataB, image_format_channel<T>::order,
                                   image_format_channel<T>::type, range);
      cl::sycl::image<dims> imageC(dataC, image_format_channel<T>::order,
                                   image_format_channel<T>::type, range);
      cl::sycl::image<dims> imageD(dataD, image_format_channel<T>::order,
                                   image_format_channel<T>::type, range);
      cl::sycl::buffer<int, 1> errorBuffer(errors, cl::sycl::range<1>(4));

      queue.submit([&](cl::sycl::handler &handler) {
        cl::sycl::accessor<T, dims, mode, target> accessorA(imageA, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorB(imageB, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorC(imageC, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorD(imageD, handler);
        cl::sycl::accessor<int, 1, cl::sycl::access::mode::write,
                           cl::sycl::access::target::global_buffer>
            errorAccessor(errorBuffer, handler);
        cl::sycl::sampler sampler(false, cl::sycl::addressing_mode::none,
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
           "correct "
           "index");
    }
  }
};

/** tests image accessors reads (specialized for host_image)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode>
class check_image_accessor_api_reads<T, dims, count, size, mode,
                                     cl::sycl::access::target::host_image> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    char dataA[size];
    char dataB[size];
    char dataC[size];
    char dataD[size];

    for (size_t i = 0; i < size; i++) {
      dataA[i] = static_cast<char>(i);
      dataB[i] = static_cast<char>(i);
      dataC[i] = static_cast<char>(i);
      dataD[i] = static_cast<char>(i);
    }

    int errors[4] = {0};

    {
      cl::sycl::image<dims> imageA(dataA, image_format_channel<T>::order,
                                   image_format_channel<T>::type, range);
      cl::sycl::image<dims> imageB(dataB, image_format_channel<T>::order,
                                   image_format_channel<T>::type, range);
      cl::sycl::image<dims> imageC(dataC, image_format_channel<T>::order,
                                   image_format_channel<T>::type, range);
      cl::sycl::image<dims> imageD(dataD, image_format_channel<T>::order,
                                   image_format_channel<T>::type, range);
      cl::sycl::buffer<int, 1> errorBuffer(errors, cl::sycl::range<1>(4));

      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
          accessorA(imageA);
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
          accessorB(imageB);
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
          accessorC(imageC);
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
          accessorD(imageD);
      cl::sycl::accessor<int, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::host_buffer>
          errorAccessor(errorBuffer);

      cl::sycl::sampler sampler(false, cl::sycl::addressing_mode::none,
                                cl::sycl::filtering_mode::nearest);

      /** check image accessor subscript operators for reads
      */
      auto idList = create_id_list<dims>(range);
      for (cl::sycl::id<dims> id : idList) {
        image_accessor_api_r<T, dims, size, mode,
                             cl::sycl::access::target::host_image,
                             cl::sycl::access::target::host_buffer>(
            accessorA, accessorB, accessorC, accessorD, sampler, errorAccessor,
            range)(id);
      }
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
           "correct "
           "index");
    }
  }
};

/** tests image accessors writes
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target>
class check_image_accessor_api_writes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    char dataA[size];
    char dataB[size];
    ::memset(dataA, 0, sizeof(dataA));
    ::memset(dataB, 0, sizeof(dataB));

    {
      cl::sycl::image<dims> imageA(dataA, image_format_channel<T>::order,
                                   image_format_channel<T>::type, range);
      cl::sycl::image<dims> imageB(dataB, image_format_channel<T>::order,
                                   image_format_channel<T>::type, range);

      queue.submit([&](cl::sycl::handler &handler) {
        cl::sycl::accessor<T, dims, mode, target> accessorA(imageA, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorB(imageB, handler);

        /** check image accessor subscript operators for writes
        */
        handler.parallel_for(range,
                             image_accessor_api_w<T, dims, size, mode, target>(
                                 accessorA, accessorB, range));
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

/** tests image accessors writes (specialized for host_image)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode>
class check_image_accessor_api_writes<T, dims, count, size, mode,
                                      cl::sycl::access::target::host_image> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    char dataA[size];
    char dataB[size];
    ::memset(dataA, 0, sizeof(dataA));
    ::memset(dataB, 0, sizeof(dataB));

    {
      cl::sycl::image<dims> imageA(dataA, image_format_channel<T>::order,
                                   image_format_channel<T>::type, range);
      cl::sycl::image<dims> imageB(dataB, image_format_channel<T>::order,
                                   image_format_channel<T>::type, range);

      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
          accessorA(imageA);
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
          accessorB(imageB);

      /** check image accessor subscript operators for writes
      */
      auto idList = create_id_list<dims>(range);
      for (cl::sycl::id<dims> id : idList) {
        image_accessor_api_w<T, dims, size, mode,
                             cl::sycl::access::target::host_image>(
            accessorA, accessorB, range)(id);
      }
    }

    if (!check_linear_index(dataA, count)) {
      FAIL(log, "operator[id] did not assign to the correct index");
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
          cl::sycl::access::mode mode, cl::sycl::access::target target>
class check_buffer_accessor_api_subscripts;

/** tests buffer accessors subscript operators (specialization for read)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::target target>
class check_buffer_accessor_api_subscripts<
    T, dims, count, size, cl::sycl::access::mode::read, target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor subscript operators for read
    */
    check_buffer_accessor_api_reads<T, dims, count, size,
                                    cl::sycl::access::mode::read, target>
        readTests;
    readTests(log, queue, range);
  }
};

/** tests buffer accessors subscript operators (specialization for write)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::target target>
class check_buffer_accessor_api_subscripts<
    T, dims, count, size, cl::sycl::access::mode::write, target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor subscript operators for write
    */
    check_buffer_accessor_api_writes<T, dims, count, size,
                                     cl::sycl::access::mode::write, target>
        writeTests;
    writeTests(log, queue, range);
  }
};

/** tests buffer accessors subscript operators (specialization for read_write)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::target target>
class check_buffer_accessor_api_subscripts<
    T, dims, count, size, cl::sycl::access::mode::read_write, target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor subscript operators for read and write
    */
    check_buffer_accessor_api_reads_and_writes<
        T, dims, count, size, cl::sycl::access::mode::read_write, target>
        readWriteTests;
    readWriteTests(log, queue, range);
  }
};

/** tests buffer accessors subscript operators (specialization for
 * discard_write)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::target target>
class check_buffer_accessor_api_subscripts<
    T, dims, count, size, cl::sycl::access::mode::discard_write, target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor subscript operators for write
    */
    check_buffer_accessor_api_writes<
        T, dims, count, size, cl::sycl::access::mode::discard_write, target>
        writeTests;
    writeTests(log, queue, range);
  }
};

/** tests buffer accessors subscript operators (specialization for
 * discard_read_write)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::target target>
class check_buffer_accessor_api_subscripts<
    T, dims, count, size, cl::sycl::access::mode::discard_read_write, target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor subscript operators for read and write
    */
    check_buffer_accessor_api_reads_and_writes<
        T, dims, count, size, cl::sycl::access::mode::discard_read_write,
        target>
        readWriteTests;
    readWriteTests(log, queue, range);
  }
};

/** tests buffer accessors with different modes
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target>
class check_buffer_accessor_api_mode {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor subscript operators
    */
    check_buffer_accessor_api_subscripts<T, dims, count, size, mode, target>
        subscriptTests;
    subscriptTests(log, queue, range);

    /** check buffer accessor other apis
    */
    check_buffer_accessor_api_methods<T, dims, count, size, mode, target>
        otherAPITests;
    otherAPITests(log, queue, range);
  }
};

/** tests image accessors with different modes
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::mode mode, cl::sycl::access::target target>
class check_image_accessor_api_mode;

/** tests image accessors with different modes (specialization for read)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::target target>
class check_image_accessor_api_mode<T, dims, count, size,
                                    cl::sycl::access::mode::read, target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check image accessor subscript operators for read
    */
    check_image_accessor_api_reads<T, dims, count, size,
                                   cl::sycl::access::mode::read, target>
        readTests;
    readTests(log, queue, range);

    /** check buffer accessor other apis
    */
    check_image_accessor_api_methods<T, dims, count, size,
                                     cl::sycl::access::mode::read, target>
        otherAPITests;
    otherAPITests(log, queue, range);
  }
};

/** tests image accessors with different modes (specialization for write)
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::target target>
class check_image_accessor_api_mode<T, dims, count, size,
                                    cl::sycl::access::mode::write, target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check image accessor subscript operators for write
    */
    check_image_accessor_api_writes<T, dims, count, size,
                                    cl::sycl::access::mode::write, target>
        writeTests;
    writeTests(log, queue, range);

    /** check buffer accessor other apis
    */
    check_image_accessor_api_methods<T, dims, count, size,
                                     cl::sycl::access::mode::write, target>
        otherAPITests;
    otherAPITests(log, queue, range);
  }
};

/** tests buffer accessors with different targets
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::target target>
class check_buffer_accessor_api_target {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor api for read
    */
    check_buffer_accessor_api_mode<T, dims, count, size,
                                   cl::sycl::access::mode::read, target>
        readTests;
    readTests(log, queue, range);

    /** check buffer accessor api for write
    */
    check_buffer_accessor_api_mode<T, dims, count, size,
                                   cl::sycl::access::mode::write, target>
        writeTests;
    writeTests(log, queue, range);

    /** check buffer accessor api for read_write
    */
    check_buffer_accessor_api_mode<T, dims, count, size,
                                   cl::sycl::access::mode::read_write, target>
        readWriteTests;
    readWriteTests(log, queue, range);

    /** check buffer accessor api for disccard_write
    */
    check_buffer_accessor_api_mode<
        T, dims, count, size, cl::sycl::access::mode::discard_write, target>
        discardWriteTests;
    discardWriteTests(log, queue, range);

    /** check buffer accessor api for discard_read_write
    */
    check_buffer_accessor_api_mode<T, dims, count, size,
                                   cl::sycl::access::mode::discard_read_write,
                                   target>
        discardReadWriteTests;
    discardReadWriteTests(log, queue, range);
  }
};

/** tests buffer accessors with different targets (specialization for
 * host_buffer)
*/
template <typename T, int dims, int count, int size>
class check_buffer_accessor_api_target<T, dims, count, size,
                                       cl::sycl::access::target::host_buffer> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor api for read
    */
    check_buffer_accessor_api_mode<T, dims, count, size,
                                   cl::sycl::access::mode::read,
                                   cl::sycl::access::target::host_buffer>()(
        log, queue, range);

    /** check buffer accessor api for write
    */
    check_buffer_accessor_api_mode<T, dims, count, size,
                                   cl::sycl::access::mode::write,
                                   cl::sycl::access::target::host_buffer>()(
        log, queue, range);

    /** check buffer accessor api for read_write
    */
    check_buffer_accessor_api_mode<T, dims, count, size,
                                   cl::sycl::access::mode::read_write,
                                   cl::sycl::access::target::host_buffer>()(
        log, queue, range);
  }
};

/** tests buffer accessors with different targets (specialization for
 * constant_buffer)
*/
template <typename T, int dims, int count, int size>
class check_buffer_accessor_api_target<
    T, dims, count, size, cl::sycl::access::target::constant_buffer> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor api for read
    */
    check_buffer_accessor_api_mode<T, dims, count, size,
                                   cl::sycl::access::mode::read,
                                   cl::sycl::access::target::constant_buffer>()(
        log, queue, range);
  }
};

/** tests buffer accessors with different targets (specialization for local)
*/
template <typename T, int dims, int count, int size>
class check_buffer_accessor_api_target<T, dims, count, size,
                                       cl::sycl::access::target::local> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor api for read_write
    */
    check_buffer_accessor_api_mode<T, dims, count, size,
                                   cl::sycl::access::mode::read_write,
                                   cl::sycl::access::target::local>()(
        log, queue, range);
  }
};

/** tests image accessors with different targets
*/
template <typename T, int dims, int count, int size,
          cl::sycl::access::target target>
class check_image_accessor_api_target {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check image accessor api for read
    */
    check_image_accessor_api_mode<T, dims, count, size,
                                  cl::sycl::access::mode::read, target>()(
        log, queue, range);

    /** check image accessor api for write
    */
    check_image_accessor_api_mode<T, dims, count, size,
                                  cl::sycl::access::mode::write, target>()(
        log, queue, range);
  }
};

/** tests buffer accessors with different dimensions
*/
template <typename T, int dims, int count, int size>
class check_buffer_accessor_api_dim {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor api for global_buffer
    */
    check_buffer_accessor_api_target<T, dims, count, size,
                                     cl::sycl::access::target::global_buffer>()(
        log, queue, range);

    /** check buffer accessor api for constant_buffer
    */
    check_buffer_accessor_api_target<
        T, dims, count, size, cl::sycl::access::target::constant_buffer>()(
        log, queue, range);

    /** check buffer accessor api for host_buffer
    */
    check_buffer_accessor_api_target<T, dims, count, size,
                                     cl::sycl::access::target::host_buffer>()(
        log, queue, range);

    /** check buffer accessor api for local
    */
    check_buffer_accessor_api_target<T, dims, count, size,
                                     cl::sycl::access::target::local>()(
        log, queue, range);
  }
};

/** tests image accessors with different dimensions
*/
template <typename T, int dims, int count, int size>
class check_image_accessor_api_dim {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check image accessor api for image
    */
    check_image_accessor_api_target<T, dims, count, size,
                                    cl::sycl::access::target::image>()(
        log, queue, range);

    /** check image accessor api for host_image
    */
    check_image_accessor_api_target<T, dims, count, size,
                                    cl::sycl::access::target::host_image>()(
        log, queue, range);
  }
};

/** tests buffer accessors with different types
*/
template <typename T>
class check_buffer_accessor_api_type {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue) {
    const int count = 32;
    const int size = 32 * sizeof(T);

    /** check buffer accessor api for 1 dimension
    */
    cl::sycl::range<1> range1d(count);
    check_buffer_accessor_api_dim<T, 1, count, size>()(log, queue, range1d);

    /** check buffer accessor api for 2 dimension
    */
    cl::sycl::range<2> range2d(count / 4, 4);
    check_buffer_accessor_api_dim<T, 2, count, size>()(log, queue, range2d);

    /** check buffer accessor api for 3 dimension
    */
    cl::sycl::range<3> range3d(count / 8, 4, 2);
    check_buffer_accessor_api_dim<T, 3, count, size>()(log, queue, range3d);
  }
};

/** tests image accessors with different types
*/
template <typename T>
class check_image_accessor_api_type {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue) {
    const int count = 32;
    const int size = count * 4;

    /** check image accessor api for 1 dimension
    */
    cl::sycl::range<1> range1d(count);
    check_image_accessor_api_dim<T, 1, count, size>()(log, queue, range1d);

    /** check image accessor api for 2 dimension
    */
    cl::sycl::range<2> range2d(count / 4, 4);
    check_image_accessor_api_dim<T, 2, count, size>()(log, queue, range2d);

    /** check image accessor api for 3 dimension
    */
    cl::sycl::range<3> range3d(count / 8, 4, 2);
    check_image_accessor_api_dim<T, 3, count, size>()(log, queue, range3d);
  }
};

/** tests the api for cl::sycl::accessor
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
  */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
  */
  virtual void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

      /** check buffer accessor api for int
      */
      check_buffer_accessor_api_type<int>()(log, queue);

      /** check buffer accessor api for float
      */

      check_buffer_accessor_api_type<float>()(log, queue);

/** check buffer accessor api for double
*/
#ifdef ENABLE_DOUBLE_SUPPORT
      check_buffer_accessor_api_type<double>()(log, queue);
#endif

      /** check buffer accessor api for char
      */
      check_buffer_accessor_api_type<char>()(log, queue);

      /** check buffer accessor api for vec
      */
      check_buffer_accessor_api_type<cl::sycl::int2>()(log, queue);

      /** check buffer accessor api for user_struct
      */
      check_buffer_accessor_api_type<user_struct>()(log, queue);

      /** check image accessor api for int4
      */
      check_image_accessor_api_type<cl::sycl::int4>()(log, queue);

      /** check image accessor api for uint4
      */
      check_image_accessor_api_type<cl::sycl::uint4>()(log, queue);

      /** check image accessor api for float4
      */
      check_image_accessor_api_type<cl::sycl::float4>()(log, queue);

      queue.wait_and_throw();
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

/** register this test with the test_collection
*/
util::test_proxy<TEST_NAME> proxy;

}  // namespace accessor_api__
