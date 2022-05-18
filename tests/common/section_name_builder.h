/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Builder for a section name with the fluent interface
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_SECTION_NAME_BUILDER_H
#define __SYCLCTS_TESTS_COMMON_SECTION_NAME_BUILDER_H

#include <catch2/catch_test_macros.hpp>

namespace {
/**
 * @brief Builder for a section name with the fluent interface
 * @details Be aware that Catch2 doesn't support nested sections with the same
 *          name, see https://github.com/catchorg/Catch2/issues/816 for details.
 *          So if you see
 *              Assertion `m_parent' failed.
 *          that's probably the case.
 */
class section_name {
  std::string m_description;
  std::ostringstream m_parameters;

 public:
  section_name(const section_name& other)
      : m_description(other.m_description),
        m_parameters(other.m_parameters.str()) {}

  // Avoid implicit move constructor removal
  section_name(section_name&& other) = default;

  section_name(const std::string& description) : m_description(description) {}

  template <typename T>
  section_name& with(const std::string& name, T&& value) {
    m_parameters << ' ' << name << ": "
                 << Catch::StringMaker<T>::convert(std::forward<T>(value))
                 << ',';
    return *this;
  }

  std::string create() const {
    std::string result(m_description);

    const auto parameters = m_parameters.str();
    if (!parameters.empty()) {
      // remove last comma and re-use first space from parameters
      result += " with" + parameters + "\b \b";
    }
    return result;
  }
};
}  // namespace

#endif  // __SYCLCTS_TESTS_COMMON_SECTION_NAME_BUILDER_H
