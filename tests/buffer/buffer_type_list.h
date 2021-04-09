/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common type list for buffer type coverage
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_BUFFER_TYPE_LIST_H
#define __SYCLCTS_TESTS_BUFFER_TYPE_LIST_H

#include "../common/type_coverage.h"
#include "../common/type_list.h"

struct nested_struct {
  using nested_user_struct = user_struct;
};

namespace get_buffer_types {
namespace different_namespace {
using diff_namespace_struct = user_struct;
}
static const auto vector_types =
    named_type_pack<char, unsigned char, short, unsigned short, int,
                    unsigned int, long, unsigned long, float>(
        {"char", "unsigned char", "short", "unsigned short", "int",
         "unsigned int", "long", "unsigned long", "float"});
static const auto scalar_types =
    named_type_pack<std::size_t, user_struct,
                    different_namespace::diff_namespace_struct,
                    nested_struct::nested_user_struct>(
        {"std::size_t", "user_struct",
         "different_namespace::diff_namespace_struct",
         "nested_struct::nested_user_struct"});
} // namespace get_buffer_types
#endif // __SYCLCTS_TESTS_BUFFER_TYPE_LIST_H
