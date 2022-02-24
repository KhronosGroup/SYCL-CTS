/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_EXCEPTIONS_H
#define __SYCLCTS_TESTS_COMMON_EXCEPTIONS_H

#include <catch2/catch_tostring.hpp>
#include <catch2/catch_translate_exception.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <sycl/sycl.hpp>

#include "../../util/logger.h"
#include "../common/conversion.h"
#include "../common/enums.h"
#include "../common/macros.h"

#include <sstream>
#include <string>
#include <utility>

namespace sycl_cts {
namespace detail {

// TODO: Remove when all three implementations support the sycl::exception API
// entirely
namespace support {

#if defined(__COMPUTECPP__)
constexpr bool exception_code = false;
constexpr bool exception_category = false;
constexpr bool sycl_errc_for = false;
constexpr bool sycl_error_category_for = false;
constexpr bool sycl_make_error_code = false;
#elif defined(__HIPSYCL__)
constexpr bool exception_code = true;
constexpr bool exception_category = false;
constexpr bool sycl_errc_for = false;
constexpr bool sycl_error_category_for = false;
constexpr bool sycl_make_error_code = false;
#elif defined(__DPCPP__)
constexpr bool exception_code = true;
constexpr bool exception_category = true;
constexpr bool sycl_errc_for = true;
constexpr bool sycl_error_category_for = false;
constexpr bool sycl_make_error_code = ;
#else
constexpr bool exception_code = true;
constexpr bool exception_category = true;
constexpr bool sycl_errc_for = true;
constexpr bool sycl_error_category_for = true;
constexpr bool sycl_make_error_code = true;
#endif

}  // namespace support
}  // namespace detail

/**
 * Helper function to get details for SYCL exception
 * Can be used, for example, for INFO(... << stringify_sycl_exception(ex))
 * expressions
 *
 * TODO: Move function body to the common/exceptions.cpp
 */
inline std::string stringify_sycl_exception(const sycl::exception& e) {
  std::ostringstream out;
  out << "SYCL exception\n";

  // Define helpers to format exception details
  auto append_str = [&out](const char* description, std::string&& value) {
    // Uses r-value reference directly
    out << "with " << description << ": '" << value << "'\n";
  };
  auto append_cstr = [&append_str](const char* description, const char* value) {
    if (!value) {
      value = "nullptr";
    }
    append_str(description, {value});
  };

  // Collect exception details, considering possible implementation gaps
  if constexpr (!detail::support::exception_category) {
    append_cstr("category", "not supported by implementation");
  } else {
    // Using reference to avoid object slicing
    const auto& category = e.category();
    append_cstr("category name", category.name());
  }

  if constexpr (!detail::support::exception_code) {
    append_cstr("code", "not supported by implementation");
  } else {
    // Using reference to avoid object slicing
    const auto& code = e.code();
    append_str("code value", std::to_string(code.value()));
    append_str("code message", code.message());
  }

  append_cstr("what", e.what());

  return out.str();
}

/**
 *  Matchers' implementation details
 */
namespace detail {

/**
 *  Matcher for std::error_category value, with no error code comparison
 */
class matcher_exception_category : public Catch::Matchers::MatcherGenericBase {
 public:
  matcher_exception_category(const std::error_category& category)
      : m_category(category) {}

  static constexpr bool is_supported_by_implementation() {
    return detail::support::exception_category;
  }

  bool match(const sycl::exception& other) const {
    if constexpr (!is_supported_by_implementation()) {
      // There should be no compilation failures, but every CHECK_THROWS_MATCHES
      // should fail with this matcher
      return false;
    } else {
      return other.category() == m_category;
    }
  }

  std::string describe() const override {
    return std::string("has category: '") + m_category.name() + "'";
  }

 private:
  const std::error_category& m_category;
};

/**
 *  Matcher for sycl::errc error codes
 *  C++ provides semantic match for std::error_code by operator==
 */
struct matcher_exception_code_semantic_match
    : public Catch::Matchers::MatcherGenericBase {
  matcher_exception_code_semantic_match(const sycl::errc& code_value)
      : m_code_value(code_value) {}

  static constexpr bool is_supported_by_implementation() {
    return detail::support::exception_code;
  }

  bool match(const sycl::exception& other) const {
    if constexpr (!is_supported_by_implementation()) {
      // There should be no compilation failures, but every CHECK_THROWS_MATCHES
      // should fail with this matcher
      return false;
    } else {
      // Semantic match
      return other.code() == m_code_value;
    }
  }

  std::string describe() const override {
    std::ostringstream result;
    result << "has code semantically matching to ";
    result << to_integral(m_code_value);
    if constexpr (detail::support::sycl_make_error_code) {
      result << " (" << sycl::make_error_code(m_code_value).message() << ")";
    }
    return result.str();
  }

 private:
  const sycl::errc& m_code_value;
};

/**
 *  Matcher for backend-specific category and error codes
 */
template <sycl::backend Backend, typename CodeT>
class matcher_equals_exception_for
    : public Catch::Matchers::MatcherGenericBase {
 public:
  matcher_equals_exception_for(const CodeT& code_value)
      : m_code_value(code_value) {}

  static constexpr bool is_supported_by_implementation() {
    return detail::support::exception_code &&
           detail::support::exception_category &&
           detail::support::sycl_error_category_for;
  }

  bool match(const sycl::exception& other) const {
    if constexpr (!is_supported_by_implementation()) {
      // There should be no compilation failures, but every CHECK_THROWS_MATCHES
      // should fail with this matcher
      return false;
    } else {
      // Error code is validated by value, with no semantic match
      const auto& expected_category = sycl::error_category_for<Backend>();
      return (other.category() == expected_category) &&
             (other.code().value() == m_code_value);
    }
  }

  std::string describe() const override {
    std::ostringstream result;
    result << "has code value " << to_integral(m_code_value);

    if constexpr (!detail::support::sycl_error_category_for) {
      result << " for backend-specific category (not supported)";
    } else {
      const auto& expected_category = sycl::error_category_for<Backend>();
      result << " for backend-specific category '" << expected_category().name();
      result << "'";
    }
    return result.str();
  }

 private:
  const CodeT& m_code_value;
};
}  // namespace detail

/**
 *  Provides matcher for std::error_category value, with no error code
 *  comparison
 *
 *  Usage example:
 *     CHECK_THROWS_MATCHES(action, sycl::exception,
 *         has_exception_category(sycl::sycl_category()) &&
 *         matches_exception_code(sycl::errc::invalid));
 */
inline auto has_exception_category(const std::error_category& category) {
  return detail::matcher_exception_category(category);
}

/**
 *  Provides matcher for sycl::errc error codes with semantic match
 *
 *  Usage example:
 *     CHECK_THROWS_MATCHES(action, sycl::exception,
 *         matches_exception_code(sycl::errc::feature_not_supported) ||
 *         matches_exception_code(sycl::errc::invalid));
 */
inline auto matches_exception_code(const sycl::errc& code) {
  return detail::matcher_exception_code_semantic_match(code);
}

/**
 *  Provides matcher for backend-specific category and error codes
 *
 *  Usage example:
 *    CHECK_THROWS_MATCHES(action(), sycl::exception,
 *        equals_exception_for<sycl::backend::opencl>(CL_INVALID_PROGRAM));
 */
template <sycl::backend Backend, typename CodeT>
inline auto equals_exception_for(const CodeT& code) {
  if constexpr (detail::support::sycl_errc_for) {
    static_assert(std::is_same_v<CodeT, sycl::errc_for<Backend>> ||
                  std::is_same_v<CodeT, int>);
  }
  return detail::matcher_equals_exception_for<Backend, CodeT>(code);
}

/**
 * Helper function to log details for SYCL exception for legacy tests
 *
 * Deprecated, use Catch2 macroses for new tests instead
 */
inline void log_exception(sycl_cts::util::logger&, const sycl::exception& e) {
  // Print multi-line message in a single Catch2 warning
  WARN(stringify_sycl_exception(e));
}

/**
 * Stringification of SYCL exceptions for Catch2 tests
 * Provides more details than simple what() message
 *
 * Note that neither this macro, not StringMaker specialization would provide
 * support for expressions like `INFO("Exception: " << exception)`
 *
 * TODO: Move out to the common/exceptions.cpp
 */
CATCH_TRANSLATE_EXCEPTION(const sycl::exception& e) {
  return stringify_sycl_exception(e);
}
}  //  namespace sycl_cts

/**
 * Stringification of SYCL exceptions for Catch2 tests
 * Required for custom matchers output
 */
namespace Catch {
template <>
struct StringMaker<sycl::exception> {
  static std::string convert(const sycl::exception& e) {
    return sycl_cts::stringify_sycl_exception(e);
  }
};
}  // namespace Catch

#endif  // __SYCLCTS_TESTS_COMMON_EXCEPTIONS_H
