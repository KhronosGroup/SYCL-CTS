/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide common functions for type coverage
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_TYPE_COVERAGE_H
#define __SYCLCTS_TESTS_COMMON_TYPE_COVERAGE_H

#include <string>
#include <utility>

#include <sycl/sycl.hpp>

#include "../../util/type_traits.h"

/**
 * @brief Retrieve type name; by default just forward the given one
 */
template <typename T>
struct type_name_string {
  static std::string get(std::string dataType) { return dataType; }
};

/**
 * @brief Specialization of type name retrievement for sycl::vec class
 * @param T Type of the data stored in vector
 * @param nElements Number of elements stored in vector
 */
template <typename T, size_t nElements>
struct type_name_string<sycl::vec<T, nElements>> {
  static std::string get(const std::string &dataType) {
    return "sycl::vec<" + dataType + "," + std::to_string(nElements) + ">";
  }
};

/**
 * @brief Specialization of type name retrievement for cl::sycl::marray class
 * @param T Type of the data stored in marray
 * @param nElements Number of elements stored in marray
 */
template <typename T, size_t nElements>
struct type_name_string<cl::sycl::marray<T, nElements>> {
  static std::string get(const std::string &dataType) {
    return "cl::sycl::marray<" + dataType + "," + std::to_string(nElements) +
           ">";
  }
};

/**
 * @brief Type pack to store types
 */
template <typename... T>
struct type_pack {};

/**
 * @brief Generic type pack with no specific type names provided
 */
template <typename... Types>
struct unnamed_type_pack {
  static_assert(sizeof...(Types) > 0, "Empty pack is not supported");

  // Syntax sugar to align usage with the named_type_pack
  static auto inline generate() { return unnamed_type_pack<Types...>{}; }
};

/**
 * @brief Generic type pack with specific type names provided
 */
template <typename... Types>
class named_type_pack {
  template <typename... nameListT>
  named_type_pack(nameListT &&...nameList)
      : names{std::forward<nameListT>(nameList)...} {}
  static_assert(sizeof...(Types) > 0, "Empty pack is not supported");

  template <typename T>
  static inline auto generate_name() {
    if constexpr (has_static_member::to_string<T>::value) {
      const auto result = T::to_string();
      static_assert(std::is_same_v<decltype(result), const std::string>,
                    "Unexpected return type for the T::to_string() method");
      return result;
    } else {
      constexpr auto always_false = !std::is_same_v<T, T>;
      static_assert(always_false,
                    "There is no static method T::to_string() for this type");
    }
  }

 public:
  // We need a specific names to differentiate types on logic level, with no
  // dependency on actual type implementation and typeid
  const std::string names[sizeof...(Types)];

  // Factory function to properly generate the type pack
  //
  // There are two possible use-cases for generation:
  // - either each type has a corresponding name provided,
  // - or each type have a static T::to_string() method available
  //
  // For example:
  //   struct var_decl {
  //     static std::string to_string() { return "variable declaration"; }
  //   };
  //   struct rval_in_expr {
  //     static std::string to_string() { return "rvalue in an expression"; }
  //   };
  //   const auto types =
  //      named_type_pack<char, signed char>::generate("char", "signed char");
  //   const auto contexts =
  //      named_type_pack<var_decl, rval_in_expr>::generate();
  //
  template <typename... nameListT>
  static auto generate(nameListT &&...nameList) {
    if constexpr (sizeof...(nameListT) == 0) {
      // No names provided explicitly, try to generate them
      return named_type_pack<Types...>(generate_name<Types>()...);
    } else {
      // Make requirement explicit to have more clear error message
      static_assert(sizeof...(Types) == sizeof...(nameListT));
      return named_type_pack<Types...>(std::forward<nameListT>(nameList)...);
    }
  }
};

/**
 * @brief Generic value pack to use for any type of compile-time lists
 */
template <typename T, T... values>
struct value_pack {
  // Factory function to generate the corresponding type pack with no names
  // stored
  //
  // Might be useful to store plain integral values or enumeration values.
  // For example:
  //   const auto bytes = value_pack<int, 1, 2, 8>::generate_unnamed();
  //
  static inline auto generate_unnamed() {
    return unnamed_type_pack<std::integral_constant<T, values>...>::generate();
  }

  // Factory function to generate the type pack with stringified values stored
  // within.
  // For example:
  //   enum class {read, write};
  //   template <mode ... values>
  //   using modes = value_pack<mode, values...>;
  //   const auto modes = modes<mode::read, mode::write>::generate_named();
  static inline auto generate_named() {
    return named_type_pack<std::integral_constant<T, values>...>::generate(
        Catch::StringMaker<T>::convert(values)...);
  }

  // Factory function to generate the type pack with names given for each value
  //
  // For example:
  //   enum class ctx : int {
  //     var_decl = 0,
  //     rval_in_expr
  //   };
  //   const auto contexts =
  //     value_pack<ctx, ctx::var_decl, ctx::rval_in_expr>::generate_named(
  //         "variable declaration", "rvalue in an expression");
  //
  template <typename... argsT>
  static inline auto generate_named(argsT &&...args) {
    return named_type_pack<std::integral_constant<T, values>...>::generate(
        std::forward<argsT>(args)...);
  }
};

/**
 * @brief Shortcut for type packs with integers. No overhead as alias doesn't
 * declare a new type. Mostly use for the dimensions.
 */
template <int... values>
using integer_pack = value_pack<int, values...>;

namespace sfinae {
namespace details {
template <typename T>
struct is_type_pack_t : std::false_type {};

template <typename... Types>
struct is_type_pack_t<named_type_pack<Types...>> : std::true_type {};

template <typename... Types>
struct is_type_pack_t<unnamed_type_pack<Types...>> : std::true_type {};
}  // namespace details

template <typename T>
using is_not_a_type_pack =
    std::enable_if_t<!details::is_type_pack_t<T>::value, bool>;

}  // namespace sfinae

/**
 * @brief Generic function to run specific action for every combination of each
 * of the types given by appropriate type pack instances. Virtually any
 * combination of named and unnamed type packs is supported. Supports different
 * types of compile-time value lists via value pack.
 * @tparam Action Functor template for action to run
 * @tparam ActionArgsT Parameter pack to use for functor template instantiation
 * @tparam HeadT The type of the first non-pack argument during the recursion
 * @tparam ArgsT Parameter pack with types of arguments for functor
 * @param head The first non-pack argument to pass into the functor
 * @param args The rest of the arguments to pass into the functor
 */
template <template <typename...> class Action, typename... ActionArgsT,
          typename HeadT, typename... ArgsT,
          sfinae::is_not_a_type_pack<HeadT> = true>
inline void for_all_combinations(HeadT &&head, ArgsT &&...args) {
  // The first non-pack argument passed into the for_all_combinations stops the
  // recursion
  Action<ActionArgsT...>{}(std::forward<HeadT>(head),
                           std::forward<ArgsT>(args)...);
}

/**
 * @brief Overload to handle the iteration over the types within the named type
 * pack
 */
template <template <typename...> class Action, typename... ActionArgsT,
          typename... HeadTypes, typename... ArgsT>
inline void for_all_combinations(const named_type_pack<HeadTypes...> &head,
                                 ArgsT &&...args) {
  // Run the next level of recursion for each type from the head named_type_pack
  // instance. Each recursion level unfolds the first argument passed and adds a
  // type name as the last argument.
  size_t type_name_index = 0;

  ((for_all_combinations<Action, ActionArgsT..., HeadTypes>(
        std::forward<ArgsT>(args)..., head.names[type_name_index]),
    ++type_name_index),
   ...);
  // The unary right fold expression is used for parameter pack expansion.
  // Every expression with comma operator is strictly sequenced, so we can
  // increment safely. And of course the fold expression would not be optimized
  // out due to side-effects.
  // Additional pair of brackets is required because of precedence of increment
  // operator relative to the comma operator.
  //
  // Note that there is actually no difference in left or right fold expression
  // for the comma operator, as it would give the same order of actions
  // execution and the same order of the type name index increment: both the
  // "(expr0, (exr1, expr2))" and "((expr0, expr1), expr2)" would give the same
  //  result as simple "expr0, expr1, expr2"
  assert((type_name_index == sizeof...(HeadTypes)) && "Pack expansion failed");
}

/**
 * @brief Overload to handle the iteration over the types within the unnamed
 * type pack
 */
template <template <typename...> class Action, typename... ActionArgsT,
          typename... HeadTypes, typename... ArgsT>
inline void for_all_combinations(const unnamed_type_pack<HeadTypes...> &head,
                                 ArgsT &&...args) {
  // Using fold expression to iterate over all types within type pack

  size_t typeNameIndex = 0;

  ((for_all_combinations<Action, ActionArgsT..., HeadTypes>(
        std::forward<ArgsT>(args)...),
    ++typeNameIndex),
   ...);
  // Ensure there is no silent miss for coverage
  assert((typeNameIndex == sizeof...(HeadTypes)) && "Pack expansion failed");
}

/**
 * @brief Overload to handle cases where no runtime arguments provided with
 * unnamed type packs
 */
template <template <typename...> class Action, typename... ArgsT>
inline void for_all_combinations() {
  Action<ArgsT...>{}();
}

/**
 * @brief Run action for each of types given by type_pack instance
 * @tparam action Functor template for action to run
 * @tparam actionArgsT Parameter pack to use for functor template instantiation
 * @tparam types Deduced from type_pack parameter pack for list of types to use
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param args Arguments to forward into the call
 */
template <template <typename, typename...> class action,
          typename... actionArgsT, typename... types, typename... argsT>
inline void for_all_types(const type_pack<types...> &, argsT &&...args) {
  // run action for each type from types... parameter pack
  // Using fold expression to iterate over all types within type pack

  size_t typeNameIndex = 0;

  ((action<types, actionArgsT...>{}(std::forward<argsT>(args)...),
    ++typeNameIndex),
   ...);

  // Ensure there is no silent miss for coverage
  assert((typeNameIndex == sizeof...(types)) && "Pack expansion failed");
}

/**
 * @brief Run action for each of types given by named_type_pack instance
 * @tparam action Functor template for action to run
 * @tparam actionArgsT Parameter pack to use for functor template instantiation
 * @tparam types Deduced from type_pack parameter pack for list of types to use
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param typeList Named type pack instance with type names stored
 * @param args Arguments to forward into the call
 */
template <template <typename, typename...> class action,
          typename... actionArgsT, typename... types, typename... argsT>
inline void for_all_types(const named_type_pack<types...> &typeList,
                          argsT &&...args) {
  // run action for each type from types... parameter pack
  // Using fold expression to iterate over all types within type pack

  size_t typeNameIndex = 0;

  ((action<types, actionArgsT...>{}(std::forward<argsT>(args)...,
                                    typeList.names[typeNameIndex]),
    ++typeNameIndex),
   ...);

  // Ensure there is no silent miss for coverage
  assert((typeNameIndex == sizeof...(types)) && "Pack expansion failed");
}

/**
 * @brief Run action for type and for all vectors of this type
 * @tparam action Functor template for action to run
 * @tparam T Type to instantiate functor template with
 * @tparam actionArgsT Parameter pack to use for functor template instantiation
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param args Arguments to forward into the call
 */
template <template <typename, typename...> class action, typename T,
          typename... actionArgsT, typename... argsT>
void for_type_and_vectors(argsT &&...args) {
  static const auto types = type_pack<
      T, typename sycl::template vec<T, 1>, typename sycl::template vec<T, 2>,
      typename sycl::template vec<T, 3>, typename sycl::template vec<T, 4>,
      typename sycl::template vec<T, 8>, typename sycl::template vec<T, 16>>{};
  // Use type_pack without names here for lazy log message construction
  for_all_types<action, actionArgsT...>(types, std::forward<argsT>(args)...);
}

/**
 * @brief Run action for each of types and vectors of types given by
 *        named_type_pack instance
 * @tparam action Functor template for action to run
 * @tparam actionArgsT Parameter pack to use for functor template instantiation
 * @tparam types Deduced from type_pack parameter pack for list of types to use
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param typeList Named type pack instance with underlying type names stored
 * @param args Arguments to forward into the call
 */
template <template <typename, typename...> class action,
          typename... actionArgsT, typename... types, typename... argsT>
void for_all_types_and_vectors(const named_type_pack<types...> &typeList,
                               argsT &&...args) {
  // run action for each type from types... parameter pack
  // Using fold expression to iterate over all types within type pack

  size_t typeNameIndex = 0;

  ((for_type_and_vectors<action, types, actionArgsT...>(
        std::forward<argsT>(args)..., typeList.names[typeNameIndex]),
    ++typeNameIndex),
   ...);

  // Ensure there is no silent miss for coverage
  assert((typeNameIndex == sizeof...(types)) && "Pack expansion failed");
}

/**
 * @brief Run action for type, vectors and marrays of this type
 * @tparam action Functor template for action to run
 * @tparam T Type to instantiate functor template with
 * @tparam actionArgsT Parameter pack to use for functor template instantiation
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param args Arguments to forward into the call
 */
template <template <typename, typename...> class action, typename T,
          typename... actionArgsT, typename... argsT>
void for_type_vectors_marray(argsT &&...args) {
  if constexpr (std::is_same<T, bool>::value) {
    for_all_types<action, actionArgsT...>(
        type_pack<T, typename sycl::template marray<T, 2>,
                  typename sycl::template marray<T, 5>,
                  typename sycl::template marray<T, 10>>{},
        std::forward<argsT>(args)...);
  } else {
    for_all_types<action, actionArgsT...>(
        type_pack<T, typename sycl::template vec<T, 1>,
                  typename sycl::template vec<T, 2>,
                  typename sycl::template vec<T, 3>,
                  typename sycl::template vec<T, 4>,
                  typename sycl::template vec<T, 8>,
                  typename sycl::template vec<T, 16>,
                  typename sycl::template marray<T, 2>,
                  typename sycl::template marray<T, 5>,
                  typename sycl::template marray<T, 10>>{},
        std::forward<argsT>(args)...);
  }
}

/**
 * @brief Run action for each of types, vectors and marrays of types given by
 *        named_type_pack instance
 * @tparam action Functor template for action to run
 * @tparam actionArgsT Parameter pack to use for functor template instantiation
 * @tparam types Deduced from type_pack parameter pack for list of types to use
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param typeList Named type pack instance with underlying type names stored
 * @param args Arguments to forward into the call
 */
template <template <typename, typename...> class action,
          typename... actionArgsT, typename... types, typename... argsT>
void for_all_types_vectors_marray(const named_type_pack<types...> &typeList,
                                  argsT &&...args) {
  // run action for each type from types... parameter pack
  // Using fold expression to iterate over all types within type pack

  size_t typeNameIndex = 0;

  ((for_type_vectors_marray<action, types, actionArgsT...>(
        std::forward<argsT>(args)..., typeList.names[typeNameIndex]),
    ++typeNameIndex),
   ...);

  // Ensure there is no silent miss for coverage
  assert((typeNameIndex == sizeof...(types)) && "Pack expansion failed");
}

/**
 * @brief Run action for type and marrays of this type
 * @tparam action Functor template for action to run
 * @tparam T Type to instantiate functor template with
 * @tparam actionArgsT Parameter pack to use for functor template instantiation
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param args Arguments to forward into the call
 */
template <template <typename, typename...> class action, typename T,
          typename... actionArgsT, typename... argsT>
void for_type_and_marrays(argsT &&...args) {
  for_all_types<action, actionArgsT...>(
      type_pack<T, typename sycl::template marray<T, 2>,
                typename sycl::template marray<T, 5>,
                typename sycl::template marray<T, 10>>{},
      std::forward<argsT>(args)...);
}

/**
 * @brief Run action for each of types and marrays of types given by
 *        named_type_pack instance
 * @tparam action Functor template for action to run
 * @tparam actionArgsT Parameter pack to use for functor template instantiation
 * @tparam types Deduced from type_pack parameter pack for list of types to use
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param typeList Named type pack instance with underlying type names stored
 * @param args Arguments to forward into the call
 */
template <template <typename, typename...> class action,
          typename... actionArgsT, typename... types, typename... argsT>
void for_all_types_and_marrays(const named_type_pack<types...> &typeList,
                               argsT &&...args) {
  // run action for each type from types... parameter pack
  // Using fold expression to iterate over all types within type pack

  size_t typeNameIndex = 0;

  ((for_type_and_marrays<action, types, actionArgsT...>(
        std::forward<argsT>(args)..., typeList.names[typeNameIndex]),
    ++typeNameIndex),
   ...);
}
#endif  // __SYCLCTS_TESTS_COMMON_TYPE_COVERAGE_H
