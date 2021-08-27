/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef SYCL_CONFORMANCE_SUITE_MATH_VECTOR_H
#define SYCL_CONFORMANCE_SUITE_MATH_VECTOR_H

#include "../tests/common/sycl.h"

template <typename T, int dim>
T getElement(const sycl::vec<T, dim> f, int ix);

template <typename T, int dim>
void setElement(const sycl::vec<T, dim> &f, int ix, T value);

#define CASE_GET_ELEMENT(NUM, COMPONENT) \
  case NUM:                              \
    return f.s##COMPONENT();

#define CASE_SET_ELEMENT(NUM, COMPONENT, VALUE) \
  case NUM:                                     \
    f.s##COMPONENT() = VALUE;                   \
    break;

template <typename T, int dim>
struct getComponent {
  T &operator()(sycl::vec<T, dim> &f, int number) = delete;
};

template <typename T>
struct getComponent<T, 1> {
  static const unsigned dim = 1;
  T operator()(sycl::vec<T, dim> &f, int number) {
    switch (number) {
      CASE_GET_ELEMENT(0, 0);
      default:
        return T(0);
    }
  }
};

template <typename T>
struct getComponent<T, 2> {
  static const unsigned dim = 2;
  T operator()(sycl::vec<T, dim> &f, int number) {
    switch (number) {
      CASE_GET_ELEMENT(0, 0);
      CASE_GET_ELEMENT(1, 1);
      default:
        return T(0);
    }
  }
};

template <typename T>
struct getComponent<T, 3> {
  static const unsigned dim = 3;
  T operator()(sycl::vec<T, dim> &f, int number) {
    switch (number) {
      CASE_GET_ELEMENT(0, 0);
      CASE_GET_ELEMENT(1, 1);
      CASE_GET_ELEMENT(2, 2);
      default:
        return T(0);
    }
  }
};

template <typename T>
struct getComponent<T, 4> {
  static const unsigned dim = 4;
  T operator()(sycl::vec<T, dim> &f, int number) const {
    switch (number) {
      CASE_GET_ELEMENT(0, 0)
      CASE_GET_ELEMENT(1, 1);
      CASE_GET_ELEMENT(2, 2);
      CASE_GET_ELEMENT(3, 3);
      default:
        return T(0);
    }
  }
};

template <typename T>
struct getComponent<T, 8> {
  static const unsigned dim = 8;
  T operator()(sycl::vec<T, dim> &f, int number) const {
    switch (number) {
      CASE_GET_ELEMENT(0, 0)
      CASE_GET_ELEMENT(1, 1);
      CASE_GET_ELEMENT(2, 2);
      CASE_GET_ELEMENT(3, 3);
      CASE_GET_ELEMENT(4, 4);
      CASE_GET_ELEMENT(5, 5);
      CASE_GET_ELEMENT(6, 6);
      CASE_GET_ELEMENT(7, 7);
      default:
        return T(0);
    }
  }
};

template <typename T>
struct getComponent<T, 16> {
  static const unsigned dim = 16;
  T operator()(sycl::vec<T, dim> &f, int number) const {
    switch (number) {
      CASE_GET_ELEMENT(0, 0)
      CASE_GET_ELEMENT(1, 1);
      CASE_GET_ELEMENT(2, 2);
      CASE_GET_ELEMENT(3, 3);
      CASE_GET_ELEMENT(4, 4);
      CASE_GET_ELEMENT(5, 5);
      CASE_GET_ELEMENT(6, 6);
      CASE_GET_ELEMENT(7, 7);
      CASE_GET_ELEMENT(8, 8);
      CASE_GET_ELEMENT(9, 9);
      CASE_GET_ELEMENT(10, A);
      CASE_GET_ELEMENT(11, B);
      CASE_GET_ELEMENT(12, C);
      CASE_GET_ELEMENT(13, D);
      CASE_GET_ELEMENT(14, E);
      CASE_GET_ELEMENT(15, F);
      default:
        return T(0);
    }
  }
};

template <typename T, int dim>
struct setComponent {
  T &operator()(sycl::vec<T, dim> &f, int number) const = delete;
};

template <typename T>
struct setComponent<T, 1> {
  static const unsigned dim = 1;
  void operator()(sycl::vec<T, dim> &f, int number, T value) const {
    switch (number) {
      CASE_SET_ELEMENT(0, 0, value)
      default:
        break;
    }
  }
};

template <typename T>
struct setComponent<T, 2> {
  static const unsigned dim = 2;
  void operator()(sycl::vec<T, dim> &f, int number, T value) const {
    switch (number) {
      CASE_SET_ELEMENT(0, 0, value)
      CASE_SET_ELEMENT(1, 1, value);
      default:
        break;
    }
  }
};

template <typename T>
struct setComponent<T, 3> {
  static const unsigned dim = 3;
  void operator()(sycl::vec<T, dim> &f, int number, T value) const {
    switch (number) {
      CASE_SET_ELEMENT(0, 0, value)
      CASE_SET_ELEMENT(1, 1, value);
      CASE_SET_ELEMENT(2, 2, value);
      default:
        break;
    }
  }
};

template <typename T>
struct setComponent<T, 4> {
  static const unsigned dim = 4;
  void operator()(sycl::vec<T, dim> &f, int number, T value) const {
    switch (number) {
      CASE_SET_ELEMENT(0, 0, value)
      CASE_SET_ELEMENT(1, 1, value);
      CASE_SET_ELEMENT(2, 2, value);
      CASE_SET_ELEMENT(3, 3, value);
      default:
        break;
    }
  }
};

template <typename T>
struct setComponent<T, 8> {
  static const unsigned dim = 8;
  void operator()(sycl::vec<T, dim> &f, int number, T value) const {
    switch (number) {
      CASE_SET_ELEMENT(0, 0, value)
      CASE_SET_ELEMENT(1, 1, value);
      CASE_SET_ELEMENT(2, 2, value);
      CASE_SET_ELEMENT(3, 3, value);
      CASE_SET_ELEMENT(4, 4, value);
      CASE_SET_ELEMENT(5, 5, value);
      CASE_SET_ELEMENT(6, 6, value);
      CASE_SET_ELEMENT(7, 7, value);
      default:
        break;
    }
  }
};

template <typename T>
struct setComponent<T, 16> {
  static const unsigned dim = 16;
  void operator()(sycl::vec<T, dim> &f, int number, T value) const {
    switch (number) {
      CASE_SET_ELEMENT(0, 0, value)
      CASE_SET_ELEMENT(1, 1, value);
      CASE_SET_ELEMENT(2, 2, value);
      CASE_SET_ELEMENT(3, 3, value);
      CASE_SET_ELEMENT(4, 4, value);
      CASE_SET_ELEMENT(5, 5, value);
      CASE_SET_ELEMENT(6, 6, value);
      CASE_SET_ELEMENT(7, 7, value);
      CASE_SET_ELEMENT(8, 8, value);
      CASE_SET_ELEMENT(9, 9, value);
      CASE_SET_ELEMENT(10, A, value);
      CASE_SET_ELEMENT(11, B, value);
      CASE_SET_ELEMENT(12, C, value);
      CASE_SET_ELEMENT(13, D, value);
      CASE_SET_ELEMENT(14, E, value);
      CASE_SET_ELEMENT(15, F, value);
      default:
        break;
    }
  }
};

#undef CASE_GET_ELEMENT
#undef CASE_SET_ELEMENT

template <typename T, int dim>
T getElement(sycl::vec<T, dim> f, int ix) {
  return getComponent<T, dim>()(f, ix);
}

template <typename T, int dim>
void setElement(sycl::vec<T, dim> &f, int ix, T value) {
  setComponent<T, dim>()(f, ix, value);
}

#endif  // SYCL_CONFORMANCE_SUITE_MATH_VECTOR_H
