/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides implementation for SYCL exceptions stringification
//
*******************************************************************************/

#include <catch2/catch_translate_exception.hpp>

#include "sycl_exceptions.h"

#include <sstream>

/**
 * Stringification of SYCL exceptions for Catch2 tests
 * Provides more details than simple what() message
 *
 * Note that neither this macro, not StringMaker specialization would provide
 * support for expressions like `INFO("Exception: " << exception)`
 */
CATCH_TRANSLATE_EXCEPTION(const sycl::exception& e) {
  return ::sycl_cts::util::stringify_sycl_exception(e);
}

namespace sycl_cts::util {

/**
 * Helper function to get details for SYCL exception
 */
std::string stringify_sycl_exception(const sycl::exception& e) {
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

  using CodeStringMakerT = Catch::StringMaker<sycl::errc>;
  const auto& errc_value = static_cast<sycl::errc>(code.value());

  append_str("code", CodeStringMakerT::convert(errc_value));

  append_str("code raw value", std::to_string(code.value()));
  append_str("code message", code.message());
#endif //  SYCL_CTS_SUPPORT_HAS_EXCEPTION_CODE

  append_cstr("what", e.what());

  return out.str();
}

}  //  namespace sycl_cts::util
