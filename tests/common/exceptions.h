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
#include <system_error>
#include <utility>

namespace sycl_cts {

// TODO: Remove when all three implementations support the sycl::exception API
// entirely
#if defined(__COMPUTECPP__)

#define SYCL_CTS_SUPPORT_HAS_EXCEPTION_CODE 0
#define SYCL_CTS_SUPPORT_HAS_EXCEPTION_CATEGORY 0
#define SYCL_CTS_SUPPORT_HAS_ERRC_FOR 0
#define SYCL_CTS_SUPPORT_HAS_ERROR_CATEGORY_FOR 0
#define SYCL_CTS_SUPPORT_HAS_MAKE_ERROR_CODE 0

#elif defined(__HIPSYCL__)

#define SYCL_CTS_SUPPORT_HAS_EXCEPTION_CODE 1
#define SYCL_CTS_SUPPORT_HAS_EXCEPTION_CATEGORY 0
#define SYCL_CTS_SUPPORT_HAS_ERRC_FOR 0
#define SYCL_CTS_SUPPORT_HAS_ERROR_CATEGORY_FOR 0
#define SYCL_CTS_SUPPORT_HAS_MAKE_ERROR_CODE 0

#elif defined(__DPCPP__)

#define SYCL_CTS_SUPPORT_HAS_EXCEPTION_CODE 1
#define SYCL_CTS_SUPPORT_HAS_EXCEPTION_CATEGORY 1
#define SYCL_CTS_SUPPORT_HAS_ERRC_FOR 1
#define SYCL_CTS_SUPPORT_HAS_ERROR_CATEGORY_FOR 0
#define SYCL_CTS_SUPPORT_HAS_MAKE_ERROR_CODE 0

#else

#define SYCL_CTS_SUPPORT_HAS_EXCEPTION_CODE 1
#define SYCL_CTS_SUPPORT_HAS_EXCEPTION_CATEGORY 1
#define SYCL_CTS_SUPPORT_HAS_ERRC_FOR 1
#define SYCL_CTS_SUPPORT_HAS_ERROR_CATEGORY_FOR 1
#define SYCL_CTS_SUPPORT_HAS_MAKE_ERROR_CODE 1

#endif

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
  auto append_str = [&out](const char* description, const std::string& value) {
    out << "with " << description << ": '" << value << "'\n";
  };
  auto append_cstr = [&append_str](const char* description, const char* value) {
    if (!value) {
      value = "nullptr";
    }
    append_str(description, {value});
  };

  // Collect exception details, considering possible implementation gaps
#if SYCL_CTS_SUPPORT_HAS_EXCEPTION_CATEGORY == 0
  append_cstr("category", "not supported by implementation");
#else
  // Using reference to avoid object slicing
  const auto& category = e.category();
  append_cstr("category name", category.name());
#endif

#if SYCL_CTS_SUPPORT_HAS_EXCEPTION_CODE == 0
  append_cstr("code", "not supported by implementation");
#else
  // Using reference to avoid object slicing
  const auto& code = e.code();

#if SYCL_CTS_SUPPORT_HAS_ERRC_ENUM == 0
  append_cstr("code", "errc enum not supported by implementation");
#else
  using CodeStringMakerT = Catch::StringMaker<sycl::errc>;
  const auto& errc_value = static_cast<sycl::errc>(code.value());

  append_str("code", CodeStringMakerT::convert(errc_value));
#endif //  SYCL_CTS_SUPPORT_HAS_ERRC_ENUM

  append_str("code raw value", std::to_string(code.value()));
  append_str("code message", code.message());
#endif //  SYCL_CTS_SUPPORT_HAS_EXCEPTION_CODE

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

  bool match(const sycl::exception& other) const {
#if SYCL_CTS_SUPPORT_HAS_EXCEPTION_CATEGORY == 0
    // There should be no compilation failures, but every CHECK_THROWS_MATCHES
    // should fail with this matcher
    return false;
#else
    return other.category() == m_category;
#endif
  }

  std::string describe() const override {
    return std::string("has category: '") + m_category.name() + "'";
  }

 private:
  const std::error_category& m_category;
};

#if SYCL_CTS_SUPPORT_HAS_ERRC_ENUM == 1
/**
 *  Matcher for sycl::errc error codes
 *  C++ provides semantic match for std::error_code by operator==, still SYCL
 *  doesn't support std::error_condition, so we have a strict equality check
 *  here
 */
struct matcher_equals_exception
    : public Catch::Matchers::MatcherGenericBase {
  matcher_equals_exception(const sycl::errc& code_value)
      : m_code_value(code_value) {}

  bool match(const sycl::exception& other) const {
#if SYCL_CTS_SUPPORT_HAS_EXCEPTION_CODE == 0
    // There should be no compilation failures, but every CHECK_THROWS_MATCHES
    // should fail with this matcher
    return false;
#else
    // Compare two std::error_code instances
    return std::is_error_code_enum_v<sycl::errc> &&
           (other.code() == m_code_value);
#endif //  SYCL_CTS_SUPPORT_HAS_EXCEPTION_CODE
  }

  std::string describe() const override {
    using CodeStringMakerT = Catch::StringMaker<sycl::errc>;
    std::ostringstream result;
    result << "has code value ";
    result << to_integral(m_code_value);
    result << " (" << CodeStringMakerT::convert(m_code_value) << ")";
    result << " for sycl_category";
    return result.str();
  }

 private:
  const sycl::errc& m_code_value;
};
#endif //  SYCL_CTS_SUPPORT_HAS_ERRC_ENUM

/**
 *  Matcher for backend-specific category and error codes
 */
template <sycl::backend Backend, typename CodeT>
class matcher_equals_exception_for
    : public Catch::Matchers::MatcherGenericBase {
 public:
  matcher_equals_exception_for(const CodeT& code_value)
      : m_code_value(code_value) {}

  bool match(const sycl::exception& other) const {

#if (SYCL_CTS_SUPPORT_HAS_EXCEPTION_CODE == 0) || \
    (SYCL_CTS_SUPPORT_HAS_EXCEPTION_CATEGORY == 0) || \
    (SYCL_CTS_SUPPORT_HAS_ERROR_CATEGORY_FOR == 0)
    // There should be no compilation failures, but every CHECK_THROWS_MATCHES
    // should fail with this matcher
    return false;
#else
    // Error code is validated by value, with no semantic match
    const auto& expected_category = sycl::error_category_for<Backend>();
    return (other.category() == expected_category) &&
           (other.code().value() == m_code_value);
#endif
  }

  std::string describe() const override {
    std::ostringstream result;
    result << "has code value " << to_integral(m_code_value);

#if (SYCL_CTS_SUPPORT_HAS_ERROR_CATEGORY_FOR == 0)
    result << " for backend-specific category (not supported)";
#else
    const auto& expected_category = sycl::error_category_for<Backend>();
    result << " for backend-specific category '" << expected_category().name();
    result << "'";
#endif
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
 *         has_exception_category(sycl::sycl_category()));
 */
inline auto has_exception_category(const std::error_category& category) {
  return detail::matcher_exception_category(category);
}

#if SYCL_CTS_SUPPORT_HAS_ERRC_ENUM == 1
/**
 *  Provides matcher for sycl::errc error codes with sycl_category check
 *
 *  Usage example:
 *     CHECK_THROWS_MATCHES(action, sycl::exception,
 *         equals_exception(sycl::errc::feature_not_supported) ||
 *         equals_exception(sycl::errc::invalid));
 */
inline auto equals_exception(const sycl::errc& code) {
  return detail::matcher_equals_exception(code);
}
#endif //  SYCL_CTS_SUPPORT_HAS_ERRC_ENUM == 1

/**
 *  Provides matcher for backend-specific category and error codes
 *
 *  Usage example:
 *    CHECK_THROWS_MATCHES(action(), sycl::exception,
 *        equals_exception_for<sycl::backend::opencl>(CL_INVALID_PROGRAM));
 */
template <sycl::backend Backend, typename CodeT>
inline auto equals_exception_for(const CodeT& code) {
#if SYCL_CTS_SUPPORT_HAS_ERRC_FOR == 1
  static_assert(std::is_same_v<CodeT, sycl::errc_for<Backend>> ||
                std::is_same_v<CodeT, int>);
#endif
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
