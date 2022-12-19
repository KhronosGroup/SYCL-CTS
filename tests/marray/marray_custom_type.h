#ifndef SYCL_CTS_TEST_MARRAY_MARRAY_CUSTOM_TYPE_H
#define SYCL_CTS_TEST_MARRAY_MARRAY_CUSTOM_TYPE_H

#include <cstdint>

/// Custom type that meets the NumericType requirement.
struct custom_int {
  custom_int() = default;
  constexpr custom_int(int i) : x{i} {};
  operator int() { return x; }
  custom_int(const custom_int& other) = default;
  custom_int(custom_int&& other) noexcept = default;

  custom_int& operator=(const custom_int& other) = default;

  custom_int& operator=(custom_int&& other) noexcept = default;

  custom_int& operator=(const int& val) {
    *this = custom_int(val);
    return *this;
  }

  custom_int& operator+=(const custom_int& val) {
    x += val.x;
    return *this;
  }
  custom_int& operator-=(const custom_int& val) {
    x -= val.x;
    return *this;
  }
  custom_int& operator*=(const custom_int& val) {
    x *= val.x;
    return *this;
  }
  custom_int& operator/=(const custom_int& val) {
    x /= val.x;
    return *this;
  }
  custom_int& operator%=(const custom_int& val) {
    x %= val.x;
    return *this;
  }
  custom_int& operator&=(const custom_int& val) {
    x &= val.x;
    return *this;
  }
  custom_int& operator|=(const custom_int& val) {
    x |= val.x;
    return *this;
  }
  custom_int& operator^=(const custom_int& val) {
    x ^= val.x;
    return *this;
  }
  custom_int& operator<<=(const custom_int& val) {
    x <<= val.x;
    return *this;
  }
  custom_int& operator>>=(const custom_int& val) {
    x >>= val.x;
    return *this;
  }

  custom_int operator+() const { return custom_int(+x); }

  custom_int operator-() const { return custom_int(-x); }

  custom_int operator~() const { return custom_int(~x); }

  bool operator!() const { return !x; }

  custom_int& operator++() {
    x++;
    return *this;
  }

  custom_int operator++(int) {
    custom_int tmp(*this);
    operator++();
    return tmp;
  }

  custom_int& operator--() {
    x--;
    return *this;
  }

  custom_int operator--(int) {
    custom_int tmp(*this);
    operator--();
    return tmp;
  }

  int x;
};

static_assert(std::is_default_constructible_v<custom_int>);
static_assert(std::is_copy_constructible_v<custom_int>);
static_assert(std::is_copy_assignable_v<custom_int>);
static_assert(std::is_destructible_v<custom_int>);

inline custom_int operator+(const custom_int& lhs, const custom_int& rhs) {
  return {lhs.x + rhs.x};
}
inline custom_int operator-(const custom_int& lhs, const custom_int& rhs) {
  return {lhs.x - rhs.x};
}
inline custom_int operator*(const custom_int& lhs, const custom_int& rhs) {
  return {lhs.x * rhs.x};
}
inline custom_int operator/(const custom_int& lhs, const custom_int& rhs) {
  return {lhs.x / rhs.x};
}
inline custom_int operator%(const custom_int& lhs, const custom_int& rhs) {
  return {lhs.x % rhs.x};
}
inline custom_int operator&(const custom_int& lhs, const custom_int& rhs) {
  return {lhs.x & rhs.x};
}
inline custom_int operator|(const custom_int& lhs, const custom_int& rhs) {
  return {lhs.x | rhs.x};
}
inline custom_int operator^(const custom_int& lhs, const custom_int& rhs) {
  return {lhs.x ^ rhs.x};
}
inline custom_int operator<<(const custom_int& lhs, const custom_int& rhs) {
  return {lhs.x << rhs.x};
}
inline custom_int operator>>(const custom_int& lhs, const custom_int& rhs) {
  return {lhs.x >> rhs.x};
}
inline bool operator&&(const custom_int& lhs, const custom_int& rhs) {
  return (!!lhs) && (!!rhs);
}
inline bool operator||(const custom_int& lhs, const custom_int& rhs) {
  return (!!lhs) || (!!rhs);
}

inline bool operator==(const custom_int& lhs, const custom_int& rhs) {
  return lhs.x == rhs.x;
}
inline bool operator!=(const custom_int& lhs, const custom_int& rhs) {
  return lhs.x != rhs.x;
}
inline bool operator<(const custom_int& lhs, const custom_int& rhs) {
  return lhs.x < rhs.x;
}
inline bool operator<=(const custom_int& lhs, const custom_int& rhs) {
  return lhs.x <= rhs.x;
}
inline bool operator>(const custom_int& lhs, const custom_int& rhs) {
  return lhs.x > rhs.x;
}
inline bool operator>=(const custom_int& lhs, const custom_int& rhs) {
  return lhs.x >= rhs.x;
}

#endif  // SYCL_CTS_TEST_MARRAY_MARRAY_CUSTOM_TYPE_H
