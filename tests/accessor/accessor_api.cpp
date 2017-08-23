/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME accessor_api

namespace accessor_api__ {
using namespace sycl_cts;

template <int dims>
cl::sycl::vector_class<cl::sycl::id<dims>> create_id_list(
    cl::sycl::range<dims> &r);

template <>
cl::sycl::vector_class<cl::sycl::id<1>> create_id_list<1>(
    cl::sycl::range<1> &r) {
  cl::sycl::vector_class<cl::sycl::id<1>> ret;
  for (size_t i = 0; i < r[0]; ++i) ret.push_back(cl::sycl::id<1>(i));
  return ret;
}

template <>
cl::sycl::vector_class<cl::sycl::id<2>> create_id_list<2>(
    cl::sycl::range<2> &r) {
  cl::sycl::vector_class<cl::sycl::id<2>> ret;
  for (size_t i = 0; i < r[0]; ++i)
    for (size_t j = 0; j < r[1]; ++j) ret.push_back(cl::sycl::id<2>(i, j));
  return ret;
}

template <>
cl::sycl::vector_class<cl::sycl::id<3>> create_id_list<3>(
    cl::sycl::range<3> &r) {
  cl::sycl::vector_class<cl::sycl::id<3>> ret;
  for (size_t i = 0; i < r[0]; ++i)
    for (size_t j = 0; j < r[1]; ++j)
      for (size_t k = 0; k < r[2]; ++k) ret.push_back(cl::sycl::id<3>(i, j, k));
  return ret;
}

template <typename T, cl::sycl::access::target target>
struct explicit_pointer;

template <typename T>
struct explicit_pointer<T, cl::sycl::access::target::global_buffer> {
  using type = cl::sycl::global_ptr<T>;
};

template <typename T>
struct explicit_pointer<T, cl::sycl::access::target::constant_buffer> {
  using type = cl::sycl::constant_ptr<T>;
};

template <typename T>
struct explicit_pointer<T, cl::sycl::access::target::local> {
  using type = cl::sycl::local_ptr<T>;
};

/* make_accessor.
 * Creates an accessor from the given arguments
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

/* make_accessor.
 * Specialization for locals that don't require the buffer.
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

/**** get_subscript */
template <typename AccessorT, int dims,
          typename std::enable_if<dims == 1>::type * = nullptr>
typename AccessorT::ElementType get_linear_subscript(AccessorT acc,
                                                     size_t index) {
  return typename AccessorT::ElementType(acc[index]);
}

template <typename AccessorT, int dims,
          typename std::enable_if<dims == 2>::type * = nullptr>
typename AccessorT::ElementType get_linear_subscript(AccessorT acc,
                                                     size_t index) {
  /* Empty instance, this is not expected to work */
  return typename AccessorT::ElementType();
}

template <typename AccessorT, int dims,
          typename std::enable_if<dims == 3>::type * = nullptr>
typename AccessorT::ElementType get_linear_subscript(AccessorT acc,
                                                     size_t index) {
  /* Empty instance, this is not expected to work */
  return typename AccessorT::ElementType();
}

template <typename AccessorT>
typename AccessorT::ElementType get_subscript(AccessorT acc,
                                              cl::sycl::id<1> index) {
  return typename AccessorT::ElementType(acc[index[0]]);
}

template <typename AccessorT>
typename AccessorT::ElementType get_subscript(AccessorT acc,
                                              cl::sycl::id<2> index) {
  return typename AccessorT::ElementType(acc[index[0]][index[1]]);
}

template <typename AccessorT>
typename AccessorT::ElementType get_subscript(AccessorT acc,
                                              cl::sycl::id<3> index) {
  return typename AccessorT::ElementType(acc[index[0]][index[1]][index[2]]);
}

/***** check_element_valid */
template <typename T1, typename T2,
          typename std::enable_if<std::is_scalar<T1>::value>::type * = nullptr>
bool check_element_valid(const T1 &elem, const T2 &correct) {
  return elem == (static_cast<T1>(correct));
};

template <
    typename T1, typename T2,
    typename std::enable_if<(!std::is_scalar<T1>::value)>::type * = nullptr>
bool check_element_valid(const T1 &elem, const T2 &correct) {
  return elem[0] == (static_cast<typename T1::element_type>(correct));
};

/**** check_linear_index */
template <typename T>
bool check_linear_index(T *data, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (!check_element_valid(data[i], i)) {
      return false;
    }
  }
  return true;
}

/**** init_linear_index */
template <typename T>
bool init_linear_index(T *data, size_t size) {
  for (size_t i = 0; i < size; i++) {
    data[i] = static_cast<T>(i);
  }
  return true;
}

/** compute linear id */
size_t compute_linear_id(cl::sycl::id<1> id, cl::sycl::range<1> r) {
  return id[0];
}

size_t compute_linear_id(cl::sycl::id<2> id, cl::sycl::range<2> r) {
  return id[0] + (id[1] * r[0]);
}

size_t compute_linear_id(cl::sycl::id<3> id, cl::sycl::range<3> r) {
  return id[0] + (id[1] * r[0]) + (id[2] * r[0] * r[1]);
}

/** set_subscript
 */
template <typename AccessorT>
void set_subscript(AccessorT &acc, cl::sycl::id<1> index, size_t linearID) {
  acc[index[0]] = static_cast<typename AccessorT::ElementType>(linearID);
}

template <typename AccessorT>
void set_subscript(AccessorT &acc, cl::sycl::id<2> index, size_t linearID) {
  acc[index[0]][index[1]] =
      static_cast<typename AccessorT::ElementType>(linearID);
}

template <typename AccessorT>
void set_subscript(AccessorT &acc, cl::sycl::id<3> index, size_t linearID) {
  acc[index[0]][index[1]][index[2]] =
      static_cast<typename AccessorT::ElementType>(linearID);
}

template <typename AccessorT, typename T, int dims>
void set_subscript(AccessorT &acc, cl::sycl::id<1> index,
                   cl::sycl::vec<T, dims> vec) {
  acc[index[0]] = vec;
}

template <typename AccessorT, typename T, int dims>
void set_subscript(AccessorT &acc, cl::sycl::id<2> index,
                   cl::sycl::vec<T, dims> vec) {
  acc[index[0]][index[1]] = vec;
}

template <typename AccessorT, typename T, int dims>
void set_subscript(AccessorT &acc, cl::sycl::id<3> index,
                   cl::sycl::vec<T, dims> vec) {
  acc[index[0]][index[1]][index[2]] = vec;
}

/** Errror Types
 * When functors are executed on a parallel for the output type is an accessor,
 * but for functors executed on the host the output type is just an integer
 * pointer.
 */
typedef cl::sycl::accessor<int, 1, cl::sycl::access::mode::write,
                           cl::sycl::access::target::global_buffer>
    errorOutputKernel_t;

typedef int *errorOutputHost_t;

/*******************************
 *  Kernel for accessor read tests
 */
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target, typename ErrorOutputT>
class accessor_reads {
  cl::sycl::accessor<T, dim, mode, target> accessorA_;
  cl::sycl::accessor<T, dim, mode, target> accessorB_;
  ErrorOutputT errorAccessor_;
  cl::sycl::range<dim> r_;

 public:
  accessor_reads(cl::sycl::accessor<T, dim, mode, target> accessorA,
                 cl::sycl::accessor<T, dim, mode, target> accessorB,
                 ErrorOutputT errorAccessor, cl::sycl::range<dim> r)
      : accessorA_(accessorA),
        accessorB_(accessorB),
        errorAccessor_(errorAccessor),
        r_(r) {}

  void operator()(cl::sycl::id<dim> i) {
    size_t linearID = compute_linear_id(i, r_);
    T elem;

    /** check [id<1>] access syntax
    */
    elem =
        get_linear_subscript<decltype(accessorA_), dim>(accessorA_, linearID);
    if (check_element_valid(elem, linearID)) {
      errorAccessor_[0] = 1;
    }

    /** check [int] access syntax
    */
    elem = get_subscript(accessorB_, i);
    if (check_element_valid(elem, linearID)) {
      errorAccessor_[0] = 1;
    }
  };
};  // accessor_reads

/*******************************
 *  Kernel for accessor write tests
 */
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class accessor_writes {
  cl::sycl::accessor<T, dim, mode, target> accessorA_;
  cl::sycl::accessor<T, dim, mode, target> accessorB_;
  cl::sycl::range<dim> r_;

 public:
  accessor_writes(cl::sycl::accessor<T, dim, mode, target> accessorA,
                  cl::sycl::accessor<T, dim, mode, target> accessorB,
                  cl::sycl::range<dim> r)
      : accessorA_(accessorA), accessorB_(accessorB), r_(r) {}

  void operator()(cl::sycl::id<dim> i) {
    size_t linearID = compute_linear_id(i, r_);

    /** check [id<1>] access syntax
    */
    accessorA_[i] = static_cast<T>(linearID);

    /** check [int] access syntax
    */
    set_subscript(accessorB_, i, linearID);
  };
};  // accessor_writes

/*******************************
 *  Kernel for image accessor write tests
 */
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class image_accessor_writes_kernel {
  cl::sycl::accessor<T, dim, mode, target> accessorA_;
  cl::sycl::accessor<T, dim, mode, target> accessorB_;
  cl::sycl::range<dim> r_;

 public:
  image_accessor_writes_kernel(
      cl::sycl::accessor<T, dim, mode, target> accessorA,
      cl::sycl::accessor<T, dim, mode, target> accessorB,
      cl::sycl::range<dim> r)
      : accessorA_(accessorA), accessorB_(accessorB), r_(r) {}

  void operator()(cl::sycl::id<dim> i) {
    size_t linearID = compute_linear_id(i, r_);
    typename T::element_type elem0 =
        static_cast<typename T::element_type>(linearID);
    typename T::element_type elemN = static_cast<typename T::element_type>(0);

    /** check [id<1>] access syntax
    */
    accessorA_[i] = T(elem0, elemN, elemN, elemN);

    /** check [int] access syntax
    */
    set_subscript(accessorB_, i, T(elem0, elemN, elemN, elemN));
  }
};  // image_accessor_writes

/*******************************
 *  Kernel for accessor read tests
 */
template <typename T, int dim, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target, typename ErrorOutputT>
class image_accessor_reads_kernel {
  cl::sycl::accessor<T, dim, mode, target> accessorA_;
  cl::sycl::accessor<T, dim, mode, target> accessorB_;
  cl::sycl::accessor<T, dim, mode, target> accessorC_;
  cl::sycl::accessor<T, dim, mode, target> accessorD_;
  cl::sycl::sampler sampler_;

  ErrorOutputT errorAccessor_;
  cl::sycl::range<dim> r_;

 public:
  image_accessor_reads_kernel(
      cl::sycl::accessor<T, dim, mode, target> accessorA,
      cl::sycl::accessor<T, dim, mode, target> accessorB,
      cl::sycl::accessor<T, dim, mode, target> accessorC,
      cl::sycl::accessor<T, dim, mode, target> accessorD,
      cl::sycl::sampler sampler, ErrorOutputT errorAccessor,
      cl::sycl::range<dim> r)
      : accessorA_(accessorA),
        accessorB_(accessorB),
        accessorC_(accessorC),
        accessorD_(accessorD),
        sampler_(sampler),
        errorAccessor_(errorAccessor),
        r_(r) {}

  void operator()(cl::sycl::id<dim> i) {
    size_t linearID = compute_linear_id(i, r_);
    T elem;

    /** check [id<1>] access syntax
    */
    elem = accessorA_[i];
    if (check_element_valid(elem, linearID)) {
      errorAccessor_[0] = 1;
    }

    /** check [id<1>](sampler) access syntax
    */
    elem = accessorB_[i](sampler);
    if (check_element_valid(elem, linearID)) {
      errorAccessor_[0] = 1;
    }

    /** check [int] access syntax
    */
    elem = accessorC_[i[0]];
    if (check_element_valid(elem, linearID)) {
      errorAccessor_[2] = 1;
    }

    /** check [int](sampler) access syntax
    */
    elem = accessorD_[id[0]](sampler);
    if (check_element_valid(elem, linearID)) {
      errorAccessor_[3] = 1;
    }
  }
};  // accessor_reads

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class buffer_accessor_other_apis {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    T data[size];
    memset(data, 0xFF, sizeof(data));
    cl::sycl::buffer<T, dims> buffer(data, range);

    queue.submit([&](cl::sycl::handler &handler) {
      auto accessor =
          make_accessor<T, dims, size, mode, target>()(buffer, handler, range);

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

      /** check get() method
      */
      auto bufferID = accessor.get();

      if (typeid(bufferID) != typeid(cl_mem)) {
        FAIL(log, "get() does not return cl_mem");
      }

      /** check get_pointer() method
      */
      auto accessorPointer = accessor.get_pointer();

      if (typeid(accessorPointer) !=
          typeid(typename explicit_pointer<T, target>::type)) {
        FAIL(log, "get() does not return explicit_ptr<T>");
      }
    });
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode>
class buffer_accessor_other_apis<T, dims, size, mode,
                                 cl::sycl::access::target::host_buffer> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    T data[size];
    memset(data, 0xFF, sizeof(data));
    cl::sycl::buffer<T, dims> buffer(data, range);

    cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
        accessor(buffer);

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

    /** check get() method
    */
    auto bufferID = accessor.get();

    if (typeid(bufferID) != typeid(cl_mem)) {
      FAIL(log, "get() does not return cl_mem");
    }

    /** check get_pointer() method
    */
    auto accessorPointer = accessor.get_pointer();

    if (typeid(accessorPointer) != typeid(T *)) {
      FAIL(log, "get() does not return T*");
    }
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class image_accessor_other_apis {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    char data[size];
    memset(data, 0, sizeof(data));
    cl::sycl::image<dims> image(
        data, cl::sycl::image_format::channel_order::RGBA,
        cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);

    queue.submit([&](cl::sycl::handler &handler) {
      cl::sycl::accessor<T, dims, mode, target> accessor(image, handler);

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

      /** check get() method
      */
      auto imageID = accessor.get();

      if (typeid(imageID) != typeid(cl_mem)) {
        FAIL(log, "get() does not return cl_mem");
      }
    });
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode>
class image_accessor_other_apis<T, dims, size, mode,
                                cl::sycl::access::target::host_image> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    char data[size];
    memset(data, 0, sizeof(data));
    cl::sycl::image<dims> image(
        data, cl::sycl::image_format::channel_order::RGBA,
        cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);

    cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
        accessor(image);

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

    /** check get() method
    */
    auto imageID = accessor.get();

    if (typeid(imageID) != typeid(cl_mem)) {
      FAIL(log, "get() does not return cl_image");
    }
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class buffer_accessor_writes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    T dataA[size];
    T dataB[size];
    memset(dataA, 0xFF, sizeof(dataA));
    memset(dataB, 0xFF, sizeof(dataB));

    {
      cl::sycl::buffer<T, dims> bufferA(dataA, range);
      cl::sycl::buffer<T, dims> bufferB(dataB, range);

      queue.submit([&](cl::sycl::handler &handler) {
        auto accessorA = make_accessor<T, dims, size, mode, target>()(
            bufferA, handler, range);
        auto accessorB = make_accessor<T, dims, size, mode, target>()(
            bufferB, handler, range);

        handler.parallel_for(range,
                             accessor_writes<T, dims, size, mode, target>(
                                 accessorA, accessorB, range));
      });
    }

    if (!check_linear_index(dataA, size)) {
      FAIL(log, "operator[id<N>] did not assign to the correct index");
    }

    if (!check_linear_index(dataB, size)) {
      FAIL(log, "operator[int][int][int] did not assign to the correct index");
    }
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode>
class buffer_accessor_writes<T, dims, size, mode,
                             cl::sycl::access::target::host_buffer> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    T dataA[size];
    T dataB[size];
    memset(dataA, 0xFF, sizeof(dataA));
    memset(dataB, 0xFF, sizeof(dataB));

    {
      cl::sycl::buffer<T, dims> bufferA(dataA, range);
      cl::sycl::buffer<T, dims> bufferB(dataB, range);

      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
          accessorA(bufferA);
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
          accessorB(bufferB);

      auto idList = create_id_list<dims>(range);
      for (cl::sycl::id<dims> id : idList) {
        accessor_writes<T, dims, size, mode,
                        cl::sycl::access::target::host_buffer>(
            accessorA, accessorB, range)(id);
      }
    }

    if (!check_linear_index(dataA, size)) {
      FAIL(log, "operator[id<N>] did not assign to the correct index");
    }

    if (!check_linear_index(dataB, size)) {
      FAIL(log, "operator[int][int][int] did not assign to the correct index");
    }
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class buffer_accessor_reads {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    T dataA[size];
    T dataB[size];
    for (int i = 0; i < size; i++) {
      dataA[i] = static_cast<char>(i);
      dataB[i] = static_cast<char>(i);
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

        handler.parallel_for(
            range,
            accessor_reads<T, dims, size, mode, target, errorOutputKernel_t>(
                accessorA, accessorB, errorAccessor, range));
      });
    }

    if (errors[0] != 0) {
      FAIL(log, "operator[id<N>] did not assign to the correct index");
    }
    if (errors[1] != 0) {
      FAIL(log, "operator[int][int][int] did not assign to the correct index");
    }
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode>
class buffer_accessor_reads<T, dims, size, mode,
                            cl::sycl::access::target::host_buffer> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    T dataA[size];
    T dataB[size];
    for (int i = 0; i < size; i++) {
      dataA[i] = static_cast<char>(i);
      dataB[i] = static_cast<char>(i);
    }
    int errors[2] = {0};

    {
      cl::sycl::buffer<T, dims> bufferA(dataA, range);
      cl::sycl::buffer<T, dims> bufferB(dataB, range);

      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
          accessorA(bufferA);
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_buffer>
          accessorB(bufferB);

      auto idList = create_id_list<dims>(range);
      for (cl::sycl::id<dims> id : idList) {
        accessor_reads<T, dims, size, mode,
                       cl::sycl::access::target::host_buffer,
                       errorOutputHost_t>(accessorA, accessorB, errors,
                                          range)(id);
      }
    }

    if (errors[0] != 0) {
      FAIL(log, "operator[id<N>] did not assign to the correct index");
    }
    if (errors[1] != 0) {
      FAIL(log, "operator[int][int][int] did not assign to the correct index");
    }
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class image_accessor_writes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    char dataA[size];
    char dataB[size];
    memset(dataA, 0, sizeof(dataA));
    memset(dataB, 0, sizeof(dataB));

    {
      cl::sycl::image<dims> imageA(
          dataA, cl::sycl::image_format::channel_order::RGBA,
          cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);
      cl::sycl::image<dims> imageB(
          dataB, cl::sycl::image_format::channel_order::RGBA,
          cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);

      queue.submit([&](cl::sycl::handler &handler) {
        cl::sycl::accessor<T, dims, mode, target> accessorA(imageA, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorB(imageB, handler);

        handler.parallel_for(
            range, image_accessor_writes_kernel<T, dims, size, mode, target>(
                       accessorA, accessorB, range));
      });
    }

    if (!check_linear_index(dataA, size)) {
      FAIL(log, "operator[id<N>] did not assign to the correct index");
    }

    if (!check_linear_index(dataB, size)) {
      FAIL(log, "operator[int][int][int] did not assign to the correct index");
    }
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode>
class image_accessor_writes<T, dims, size, mode,
                            cl::sycl::access::target::host_image> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    char dataA[size];
    char dataB[size];
    memset(dataA, 0, sizeof(dataA));
    memset(dataB, 0, sizeof(dataB));

    {
      cl::sycl::image<dims> imageA(
          dataA, cl::sycl::image_format::channel_order::RGBA,
          cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);
      cl::sycl::image<dims> imageB(
          dataB, cl::sycl::image_format::channel_order::RGBA,
          cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);

      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
          accessorA(imageA);
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
          accessorB(imageB);

      auto idList = create_id_list<dims>(range);
      for (cl::sycl::id<dims> id : idList) {
        image_accessor_writes_kernel<T, dims, size, mode,
                                     cl::sycl::access::target::host_image>(
            accessorA, accessorB, range)(id);
      }
    }

    if (!check_linear_index(dataA, size)) {
      FAIL(log, "operator[id] did not assign to the correct index");
    }

    if (!check_linear_index(dataB, size)) {
      FAIL(log, "operator[int][int][int] did not assign to the correct index");
    }
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class image_accessor_reads {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    char dataA[size];
    char dataB[size];
    char dataC[size];
    char dataD[size];

    init_linear_index(dataA, size);
    init_linear_index(dataB, size);
    init_linear_index(dataC, size);
    init_linear_index(dataD, size);

    int errors[4] = {0};

    {
      cl::sycl::image<dims> imageA(
          dataA, cl::sycl::image_format::channel_order::RGBA,
          cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);
      cl::sycl::image<dims> imageB(
          dataB, cl::sycl::image_format::channel_order::RGBA,
          cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);
      cl::sycl::image<dims> imageC(
          dataA, cl::sycl::image_format::channel_order::RGBA,
          cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);
      cl::sycl::image<dims> imageD(
          dataB, cl::sycl::image_format::channel_order::RGBA,
          cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);
      cl::sycl::buffer<int, 1> errorBuffer(errors, cl::sycl::range<1>(4));

      queue.submit([&](cl::sycl::handler &handler) {
        cl::sycl::accessor<T, dims, mode, target> accessorA(imageA, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorB(imageB, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorC(imageC, handler);
        cl::sycl::accessor<T, dims, mode, target> accessorD(imageD, handler);
        cl::sycl::accessor<int, 1, cl::sycl::access::mode::write,
                           cl::sycl::access::target::global_buffer>
            errorAccessor(errorBuffer, handler);
        cl::sycl::sampler sampler(false,
                                  cl::sycl::sampler_addressing_mode::none,
                                  cl::sycl::sampler_filter_mode::nearest);

        handler.parallel_for(
            range, image_accessor_reads_kernel<T, dims, size, mode, target,
                                               errorOutputKernel_t>(
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
      FAIL(log, "operator[int][int][int] did not assign to the correct index");
    }
    if (errors[3] != 0) {
      FAIL(log,
           "operator[int][int][int](sampler) did not assign to the correct "
           "index");
    }
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode>
class image_accessor_reads<T, dims, size, mode,
                           cl::sycl::access::target::host_image> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    char dataA[size];
    char dataB[size];
    char dataC[size];
    char dataD[size];

    init_linear_index(dataA, size);
    init_linear_index(dataB, size);
    init_linear_index(dataC, size);
    init_linear_index(dataD, size);

    int errors[4] = {0};

    {
      cl::sycl::image<dims> imageA(
          dataA, cl::sycl::image_format::channel_order::RGBA,
          cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);
      cl::sycl::image<dims> imageB(
          dataB, cl::sycl::image_format::channel_order::RGBA,
          cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);
      cl::sycl::image<dims> imageC(
          dataA, cl::sycl::image_format::channel_order::RGBA,
          cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);
      cl::sycl::image<dims> imageD(
          dataB, cl::sycl::image_format::channel_order::RGBA,
          cl::sycl::image_format::channel_type::UNSIGNED_INT8, range);

      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
          accessorA(imageA);
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
          accessorB(imageB);
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
          accessorC(imageC);
      cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::host_image>
          accessorD(imageD);

      cl::sycl::sampler sampler(false, cl::sycl::sampler_addressing_mode::none,
                                cl::sycl::sampler_filter_mode::nearest);

      auto idList = create_id_list<dims>(range);
      for (cl::sycl::id<dims> id : idList) {
        image_accessor_reads_kernel<T, dims, size, mode,
                                    cl::sycl::access::target::host_image,
                                    errorOutputHost_t>(
            accessorA, accessorB, accessorC, accessorD, sampler, errors,
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
      FAIL(log, "operator[int][int][int] did not assign to the correct index");
    }
    if (errors[3] != 0) {
      FAIL(log,
           "operator[int][int][int](sampler) did not assign to the correct "
           "index");
    }
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class buffer_accessor_subscripts;

template <typename T, int dims, int size, cl::sycl::access::target target>
class buffer_accessor_subscripts<T, dims, size, cl::sycl::access::mode::read,
                                 target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor subscript operators for read
    */
    buffer_accessor_reads<T, dims, size, cl::sycl::access::mode::read, target>
        readTests;
    readTests(log, queue, range);
  }
};

template <typename T, int dims, int size, cl::sycl::access::target target>
class buffer_accessor_subscripts<T, dims, size, cl::sycl::access::mode::write,
                                 target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor subscript operators for write
    */
    buffer_accessor_writes<T, dims, size, cl::sycl::access::mode::write, target>
        writeTests;
    writeTests(log, queue, range);
  }
};

template <typename T, int dims, int size, cl::sycl::access::target target>
class buffer_accessor_subscripts<T, dims, size,
                                 cl::sycl::access::mode::read_write, target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor subscript operators for read
    */
    buffer_accessor_reads<T, dims, size, cl::sycl::access::mode::read_write,
                          target>
        readTests;
    readTests(log, queue, range);

    /** check buffer accessor subscript operators for write
    */
    buffer_accessor_writes<T, dims, size, cl::sycl::access::mode::read_write,
                           target>
        writeTests;
    writeTests(log, queue, range);
  }
};

template <typename T, int dims, int size, cl::sycl::access::target target>
class buffer_accessor_subscripts<
    T, dims, size, cl::sycl::access::mode::discard_write, target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor subscript operators for write
    */
    buffer_accessor_writes<T, dims, size, cl::sycl::access::mode::discard_write,
                           target>
        writeTests;
    writeTests(log, queue, range);
  }
};

template <typename T, int dims, int size, cl::sycl::access::target target>
class buffer_accessor_subscripts<
    T, dims, size, cl::sycl::access::mode::discard_read_write, target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor subscript operators for read
    */
    buffer_accessor_reads<T, dims, size,
                          cl::sycl::access::mode::discard_read_write, target>
        readTests;
    readTests(log, queue, range);

    /** check buffer accessor subscript operators for write
    */
    buffer_accessor_writes<T, dims, size,
                           cl::sycl::access::mode::discard_read_write, target>
        writeTests;
    writeTests(log, queue, range);
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class buffer_accessor_apis {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check image accessor subscript operators
    */
    buffer_accessor_subscripts<T, dims, size, mode, target> subscriptTests;
    subscriptTests(log, queue, range);

    /** check image accessor other apis
    */
    buffer_accessor_other_apis<T, dims, size, mode, target> otherAPITests;
    otherAPITests(log, queue, range);
  }
};

template <typename T, int dims, int size, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class image_accessor_apis;

template <typename T, int dims, int size, cl::sycl::access::target target>
class image_accessor_apis<T, dims, size, cl::sycl::access::mode::read, target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check image accessor subscript operators for read
    */
    image_accessor_reads<T, dims, size, cl::sycl::access::mode::read, target>
        readTests;
    readTests(log, queue, range);

    /** check buffer accessor other apis
    */
    image_accessor_other_apis<T, dims, size, cl::sycl::access::mode::read,
                              target>
        otherAPITests;
    otherAPITests(log, queue, range);
  }
};

template <typename T, int dims, int size, cl::sycl::access::target target>
class image_accessor_apis<T, dims, size, cl::sycl::access::mode::write,
                          target> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check image accessor subscript operators for write
    */
    image_accessor_writes<T, dims, size, cl::sycl::access::mode::write, target>
        writeTests;
    writeTests(log, queue, range);

    /** check buffer accessor other apis
    */
    image_accessor_other_apis<T, dims, size, cl::sycl::access::mode::write,
                              target>
        otherAPITests;
    otherAPITests(log, queue, range);
  }
};

template <typename T, int dims, int size, cl::sycl::access::target target>
class buffer_accessor_modes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor api for read
    */
    buffer_accessor_apis<T, dims, size, cl::sycl::access::mode::read, target>
        readTests;
    readTests(log, queue, range);

    /** check buffer accessor api for write
    */
    buffer_accessor_apis<T, dims, size, cl::sycl::access::mode::write, target>
        writeTests;
    writeTests(log, queue, range);

    /** check buffer accessor api for read_write
    */
    buffer_accessor_apis<T, dims, size, cl::sycl::access::mode::read_write,
                         target>
        readWriteTests;
    readWriteTests(log, queue, range);

    /** check buffer accessor api for disccard_write
    */
    buffer_accessor_apis<T, dims, size, cl::sycl::access::mode::discard_write,
                         target>
        discardWriteTests;
    discardWriteTests(log, queue, range);

    /** check buffer accessor api for discard_read_write
    */
    buffer_accessor_apis<T, dims, size,
                         cl::sycl::access::mode::discard_read_write, target>
        discardReadWriteTests;
    discardReadWriteTests(log, queue, range);
  }
};

template <typename T, int dims, int size>
class buffer_accessor_modes<T, dims, size,
                            cl::sycl::access::target::host_buffer> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor api for read
    */
    buffer_accessor_apis<T, dims, size, cl::sycl::access::mode::read,
                         cl::sycl::access::target::host_buffer>
        readTests;
    readTests(log, queue, range);

    /** check buffer accessor api for write
    */
    buffer_accessor_apis<T, dims, size, cl::sycl::access::mode::write,
                         cl::sycl::access::target::host_buffer>
        writeTests;
    writeTests(log, queue, range);

    /** check buffer accessor api for read_write
    */
    buffer_accessor_apis<T, dims, size, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::host_buffer>
        readWriteTests;
    readWriteTests(log, queue, range);
  }
};

template <typename T, int dims, int size>
class buffer_accessor_modes<T, dims, size,
                            cl::sycl::access::target::constant_buffer> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor api for read
    */
    buffer_accessor_apis<T, dims, size, cl::sycl::access::mode::read,
                         cl::sycl::access::target::constant_buffer>
        readTests;
    readTests(log, queue, range);
  }
};

template <typename T, int dims, int size>
class buffer_accessor_modes<T, dims, size, cl::sycl::access::target::local> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor api for read_write
    */
    buffer_accessor_apis<T, dims, size, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
        readTests;
    readTests(log, queue, range);
  }
};

template <typename T, int dims, int size, cl::sycl::access::target target>
class image_accessor_modes {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check image accessor api for read
    */
    image_accessor_apis<T, dims, size, cl::sycl::access::mode::read, target>
        readTests;
    readTests(log, queue, range);

    /** check image accessor api for write
    */
    image_accessor_apis<T, dims, size, cl::sycl::access::mode::write, target>
        writeTests;
    writeTests(log, queue, range);
  }
};

template <typename T, int dims, int size>
class buffer_accessor_targets {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check buffer accessor api for global_buffer
    */
    buffer_accessor_modes<T, dims, size,
                          cl::sycl::access::target::global_buffer>
        globalBufferTests;
    globalBufferTests(log, queue, range);

    /** check buffer accessor api for constant_buffer
    */
    buffer_accessor_modes<T, dims, size,
                          cl::sycl::access::target::constant_buffer>
        constantTests;
    constantTests(log, queue, range);

    /** check buffer accessor api for host_buffer
    */
    buffer_accessor_modes<T, dims, size, cl::sycl::access::target::host_buffer>
        hostBufferTests;
    hostBufferTests(log, queue, range);

    /** check buffer accessor api for local
    */
    buffer_accessor_modes<T, dims, size, cl::sycl::access::target::local>
        localTests;
    localTests(log, queue, range);
  }
};

template <typename T, int dims, int size>
class image_accessor_targets {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue,
                  cl::sycl::range<dims> range) {
    /** check image accessor api for image
    */
    image_accessor_modes<T, dims, size, cl::sycl::access::target::image>
        imageTests;
    imageTests(log, queue, range);

    /** check image accessor api for host_image
    */
    image_accessor_modes<T, dims, size, cl::sycl::access::target::host_image>
        hostImageTests;
    hostImageTests(log, queue, range);
  }
};

template <typename T>
class buffer_accessor_dims {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue) {
    const int size = 32;

    /** check buffer accessor api for 1 dimension
    */
    cl::sycl::range<1> range1d(size);
    buffer_accessor_targets<T, 1, size> acc1d;
    acc1d(log, queue, range1d);

    /** check buffer accessor api for 2 dimension
    */
    cl::sycl::range<2> range2d(size, size);
    buffer_accessor_targets<T, 2, (size * size)> acc2d;
    acc2d(log, queue, range2d);

    /** check buffer accessor api for 3 dimension
    */
    cl::sycl::range<3> range3d(size, size, size);
    buffer_accessor_targets<T, 3, (size * size * size)> acc3d;
    acc3d(log, queue, range3d);
  }
};

template <typename T>
class image_accessor_dims {
 public:
  void operator()(util::logger &log, cl::sycl::queue &queue) {
    const int size = 32;

    /** check image accessor api for 1 dimension
    */
    cl::sycl::range<1> range1d(size);
    image_accessor_targets<T, 1, size> acc1d;
    acc1d(log, queue, range1d);

    /** check image accessor api for 2 dimension
    */
    cl::sycl::range<2> range2d(size, size);
    image_accessor_targets<T, 2, (size * size)> acc2d;
    acc2d(log, queue, range2d);

    /** check image accessor api for 3 dimension
    */
    cl::sycl::range<3> range3d(size, size, size);
    image_accessor_targets<T, 3, (size * size * size)> acc3d;
    acc3d(log, queue, range3d);
  }
};

struct user_struct {
  float a;
  int b;
  char c;

  // The type of the element returned by
  // the indexing operator
  typedef int element_type;

  user_struct() : a(0), b(0), c(0){};

  user_struct(int val) : a(0), b(val), c(0) {}

  element_type operator[](size_t index) const { return b; }
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
      cts_selector selector;
      cl::sycl::queue queue(selector);

      /** check buffer accessor api for int
      */
      buffer_accessor_dims<int> intTests;
      intTests(log, queue);

      /** check buffer accessor api for float
      */
      buffer_accessor_dims<float> floatTests;
      floatTests(log, queue);

      /** check buffer accessor api for double
      */
      buffer_accessor_dims<double> doubleTests;
      intTests(log, queue);

      /** check buffer accessor api for char
      */
      buffer_accessor_dims<char> charTests;
      charTests(log, queue);

      /** check buffer accessor api for vec
      */
      buffer_accessor_dims<cl::sycl::int2> vecTests;
      vecTests(log, queue);

      /** check buffer accessor api for user_struct
      */
      buffer_accessor_dims<user_struct> userStructTests;
      userStructTests(log, queue);

      /** check image accessor api for int4
      */
      image_accessor_dims<cl::sycl::int4> int4Tests;
      int4Tests(log, queue);

      /** check image accessor api for uint4
      */
      image_accessor_dims<cl::sycl::uint4> uint4Tests;
      uint4Tests(log, queue);

      /** check image accessor api for float4
      */
      image_accessor_dims<cl::sycl::float4> float4Tests;
      float4Tests(log, queue);

      queue.wait_and_throw();
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "a sycl exception was caught");
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace accessor_api__ */
