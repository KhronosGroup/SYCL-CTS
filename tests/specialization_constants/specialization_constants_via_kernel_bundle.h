/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common checks for specialization constants usage via kernel_bundle
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SPEC_CONST_VIA_KERNEL_BUNDLE_H
#define __SYCLCTS_TESTS_SPEC_CONST_VIA_KERNEL_BUNDLE_H

#include "../common/common.h"
#include "../common/once_per_unit.h"
#include "../common/type_coverage.h"
#include "specialization_constants_common.h"

#include <stdexcept>

namespace specialization_constants_via_kernel_bundle {

/** @brief Defines a group of tests from the point of kernel_bundle modification
 *         between specialization constant's value setup and usage
 */
enum class case_group : int {
  same = 0,
  compile,
  link,
  build,
  join,
  compile_join,
  build_join,
  read_from_kernel
};

/** @brief Defines a case to run for every type
 *  @tparam groupId Defines group of tests
 *  @tparam setN Number of times to set the specialization constants' value,
 *               from 0 incl.
 *  @tparam getN Number of times to get the specialization constants' value
 */
template <sycl::bundle_state bndlState, case_group groupId, int setN, int getN>
struct case_id {
  static constexpr auto state = bndlState;
  static constexpr auto group = groupId;
  static constexpr auto n_set = setN;
  static constexpr auto n_get = getN;

  static std::string get_description() {
    std::string result{"case of using "};
    if constexpr (groupId == case_group::same) {
      result += "same";
    }
    if constexpr (groupId == case_group::compile) {
      result += "compiled";
    }
    if constexpr (groupId == case_group::link) {
      result += "linked";
    }
    if constexpr (groupId == case_group::build) {
      result += "built";
    }
    if constexpr (groupId == case_group::join) {
      result += "joined";
    }
    if constexpr (groupId == case_group::compile_join) {
      result += "compiled and joined";
    }
    if constexpr (groupId == case_group::build_join) {
      result += "built and joined";
    }
    if constexpr (groupId == case_group::read_from_kernel) {
      result += "kernel to read values from";
    }
    result += " kernel bundle with bundle_state::";

    if constexpr (bndlState == sycl::bundle_state::input) {
      result += "input";
    }
    if constexpr (bndlState == sycl::bundle_state::object) {
      result += "object";
    }
    if constexpr (bndlState == sycl::bundle_state::executable) {
      result += "executable";
    }
    result += " state, " + std::to_string(setN) + " write(s) and ";
    result += std::to_string(getN) + " read(s)";
    return result;
  }
};

/** @brief Defines values to use with specialization constants
 *  @tparam T Type of the underlying data to work with
 */
template <typename T>
struct values {
  static constexpr int initial = get_spec_const::default_val;
  static constexpr int empty = initial / 2;

  /** @brief Provides a value to use as the reference one on the N-th set
   *         operation for the same specialization constant
   *  @param setIndex Zero-based index of set operation for the same
   *         specialization constant
   */
  static constexpr int reference(int setIndex = 0) {
    return initial + 1 + setIndex;
  }
};

/** @brief Specialization for boolean underlying type; required to distinguish
 *         default and expected values
 */
template <>
struct values<bool> {
  static constexpr int initial = get_spec_const::default_val;
  static constexpr int empty = initial;

  static constexpr int reference(int setIndex = 0) {
    // Ensure that for every consecuent set operation we use the different value
    if (setIndex % 2 == 1) return initial;

    return static_cast<int>(!initial);
  }
};

/** @brief Specialization for container of boolean elements
 */
template <typename sizeT, template <typename, sizeT> class containerT,
          sizeT size>
struct values<containerT<bool, size>> : values<bool> {};

}  // namespace specialization_constants_via_kernel_bundle

/** @brief Specialization constants used for checks
 *  @tparam T Type of the underlying data
 *  @tparam id Specialization of the case id
 */
template <typename T, typename id>
constexpr sycl::specialization_id<T> spec_const_by_kernel_bundle(
    get_spec_const::get_init_value_helper<T>(
        specialization_constants_via_kernel_bundle::values<T>::initial));

namespace specialization_constants_via_kernel_bundle {

template <typename T, typename id>
struct kernel {};

/** @brief Exception to be thrown if implementation does not provide a kernel
 *         bundle for the required kernel
 *  @details SYCL 2020, rev.3, p.4.11.7 states that since the implementation may
 *           not represent all kernels in bundle_state::input or
 *           bundle_state::object
 */
class skip_check : public std::runtime_error {
 public:
  skip_check(const char* desc) : std::runtime_error(desc) {}
};

/** @brief Provides all operations on kernel_bundle creation and modification
 *  @tparam T Type of the underlying data for the specialization constant
 *  @tparam id Specialization of the case id
 */
template <typename T, typename id>
class bundle_factory {
  const sycl::context context;
  const sycl::device device;

 public:
  bundle_factory(const sycl::queue& queue)
      : context(queue.get_context()), device(queue.get_device()) {}

  /** @brief Factory method for kernel_bundle's used for checks
   */
  auto create() {
    if constexpr (id::group != case_group::read_from_kernel) {
      // Literally any bundle will be OK
      return sycl::get_kernel_bundle<id::state>(context, {device});
    } else {
      // We should use only the one containing the kernel required
      auto kernelId = sycl::get_kernel_id<kernel<T, id>>();
      auto bundle =
          sycl::get_kernel_bundle<id::state>(context, {device}, {kernelId});
      if (!bundle.has_kernel(kernelId)) {
        throw skip_check{
            "Implementation does not provide a kernel bundle for the required "
            "kernel"};
      }
      return bundle;
    }
  }

  /** @brief Factory method for empty kernel_bundle creation
   *  @tparam state State of the kernel_bundle to create
   */
  template <sycl::bundle_state state>
  auto create_empty() {
    auto reject_all = [&](const sycl::device_image<state>&) { return false; };
    return sycl::get_kernel_bundle<state>(context, {device}, reject_all);
  }

  /** @brief Factory method to create a modified kernel_bundle according to the
   *         requirements of an actual test case
   *  @tparam group Defines the logic for kernel_bundle modification
   *  @tparam state State of the input kernel_bundle instance
   *  @retval Different types of kernel_bundle
   */
  template <case_group group = id::group, sycl::bundle_state state>
  auto mutate(const sycl::kernel_bundle<state>& bundle) {
    if constexpr (group == case_group::same) {
      // Using common-by-reference semantics for kernel_bundle
      return bundle;

    } else if constexpr (group == case_group::compile) {
      return sycl::compile(bundle);

    } else if constexpr (group == case_group::link) {
      return sycl::link(sycl::compile(bundle));

    } else if constexpr (group == case_group::build) {
      return sycl::build(bundle);

    } else if constexpr (group == case_group::join) {
      auto otherBundle = create_empty<state>();
      return sycl::join<state>({bundle, otherBundle});

    } else if constexpr (group == case_group::compile_join) {
      auto compiled = mutate<case_group::compile>(bundle);
      auto joined = mutate<case_group::join>(compiled);
      return joined;

    } else if constexpr (group == case_group::build_join) {
      auto built = mutate<case_group::build>(bundle);
      auto joined = mutate<case_group::join>(built);
      return joined;

    } else if constexpr (group == case_group::read_from_kernel) {
      return mutate<case_group::build>(bundle);

    } else {
      static_assert(group != group, "Unknown test group");
    }
  }

  /** @brief Checks if online compiler is required for the given kernel_bundle
   *         initial state and group of tests
   */
  static constexpr bool requires_online_compiler() {
    return id::state == sycl::bundle_state::input;
  }

  /** @brief Checks if online linker is required for the given kernel_bundle
   *         initial state and group of tests
   */
  static constexpr bool requires_online_linker() {
    if constexpr (id::state == sycl::bundle_state::object) {
      return true;
    } else if constexpr (id::state == sycl::bundle_state::executable) {
      return false;
    } else {
      return (id::group != case_group::same) &&
             (id::group != case_group::join) &&
             (id::group != case_group::compile) &&
             (id::group != case_group::compile_join) &&
             (id::group != case_group::read_from_kernel);
    }
  }

  /** @brief Checks if all preconditions are met to run specific test group
   *         with the specific initial state of the kernel_bundle
   */
  bool check_preconditions(sycl_cts::util::logger& log) {
    if constexpr (requires_online_compiler()) {
      if (!device.has(sycl::aspect::online_compiler)) {
        once_per_unit::log(log, "Device does not support online compilation");
        return false;
      }
    }
    if constexpr (requires_online_linker()) {
      if (!device.has(sycl::aspect::online_linker)) {
        once_per_unit::log(log, "Device does not support online linkage");
        return false;
      }
    }
    return true;
  }
};

/** @brief Set specialization constant's value the required number of times
 *  @tparam T Underlying data type, required to specify an actual instance of
 *            the specialization constant
 *  @tparam id Case id type, required to specify an actual instance of
 *             the specialization constant
 */
template <typename T, typename id>
void set_value(sycl::kernel_bundle<id::state>& bundle) {
  using namespace get_spec_const;

  if constexpr (id::n_set > 0) {
    static_assert(id::state == sycl::bundle_state::input,
                  "kernel_bundle's state should be bundle_state::input to call"
                  " set_specialization_constant");
    for (int i = 0; i < id::n_set; ++i) {
      // Prepare value to store
      T value { get_init_value_helper<T>(0) };
      fill_init_values(value, values<T>::reference(i));

      bundle.template set_specialization_constant<
          spec_const_by_kernel_bundle<T, id>>(value);
    }
  }
}

/** @brief Read specialization constant's value the required number of times and
 *         verify the latest read retrieves an expected value
 *  @tparam T Underlying data type, required to specify an actual instance of
 *            the specialization constant
 *  @tparam id Case id type, required to specify an actual instance of
 *             the specialization constant
 *  @tparam specStorageT Actual type used to access the specialization constant.
 *                       It is either kernel_bundle or kernel_handler.
 */
template <typename T, typename id, typename specStorageT>
bool check_value(specStorageT&& storage) {
  using namespace get_spec_const;

  // Prepare to read value
  T value { get_init_value_helper<T>(0) };
  fill_init_values(value, values<T>::empty);

  // Prepare to compare values
  T expected { get_init_value_helper<T>(0) };
  if constexpr (id::n_set == 0) {
    fill_init_values(expected, values<T>::initial);
  } else {
    const int effectiveSetIndex = id::n_set - 1;
    fill_init_values(expected, values<T>::reference(effectiveSetIndex));
  }

  for (int i = 0; i < id::n_get; ++i) {
    value = storage.template get_specialization_constant<
        spec_const_by_kernel_bundle<T, id>>();
  }
  return check_equal_values(value, expected);
}

/** @brief Run an actual check for the type and case id given
 */
template <typename T, typename id>
void run_check(sycl_cts::util::logger& log, const std::string& typeName) {
  try {
    bool pass = false;

    auto queue = sycl_cts::util::get_cts_object::queue();
    bundle_factory<T, id> bundle_factory(queue);

    if (!bundle_factory.check_preconditions(log)) return;

    {
      sycl::buffer<bool, 1> passBuffer(&pass, sycl::range<1>(1));

      queue.submit([&](sycl::handler& cgh) {
        auto srcBundle = bundle_factory.create();

        // Set value if needed
        set_value<T, id>(srcBundle);

        auto bundle = bundle_factory.template mutate<id::group>(srcBundle);

        if constexpr (id::group != case_group::read_from_kernel) {
          static_cast<void>(passBuffer);

          pass = check_value<T, id>(bundle);
          // No kernel submitted
        } else {
          cgh.use_kernel_bundle(bundle);

          auto passAccessor =
              passBuffer.template get_access<sycl::access_mode::write>(cgh);
          cgh.single_task<kernel<T, id>>([=](sycl::kernel_handler h) {
            passAccessor[0] = check_value<T, id>(h);
          });
        }
      });
    }
    if (!pass) {
      std::string message{"Check failed for "};
      message += id::get_description();
      message += " for type " + type_name_string<T>::get(typeName);
      FAIL(log, message);
    }
  } catch (const skip_check& reason) {
    std::string message{"Check skipped for "};
    message += id::get_description();
    message += " for type " + type_name_string<T>::get(typeName);
    message += ". ";
    message += reason.what();
    log.note(message);
    // Pass to the next check
  } catch (...) {
    std::string message{"Exception fired for "};
    message += id::get_description();
    message += " for type " + type_name_string<T>::get(typeName);
    log.note(message);
    // Skip all further checks
    throw;
  }
}

/** @brief Run all checks for the type given
 */
template <typename T, sycl::bundle_state state, typename... argsT>
void run_all_checks(argsT&&... args) {
  // Cases for all kernel_bundle states
  {
    /** Read a spec constant from a kernel_bundle without writing its value.
     */
    constexpr auto group = case_group::same;
    run_check<T, case_id<state, group, 0, 1>>(std::forward<argsT>(args)...);
  }
  {
    /** Read a spec constant from a joined kernel_bundle without writing its
     *  value.
     */
    constexpr auto group = case_group::join;
    run_check<T, case_id<state, group, 0, 1>>(std::forward<argsT>(args)...);
  }

  if constexpr (state == sycl::bundle_state::input) {
    {
      /** Set the value in a kernel_bundle and then read it from the same
       *  bundle.
       *  Set the value in a kernel_bundle twice and then read it from the same
       *  bundle.
       */
      constexpr auto group = case_group::same;
      run_check<T, case_id<state, group, 1, 1>>(std::forward<argsT>(args)...);
      run_check<T, case_id<state, group, 2, 1>>(std::forward<argsT>(args)...);
    }
    {
      /** Read a spec constant from a compiled kernel_bundle without writing its
       *  value.
       *  Set the value in a kernel_bundle and read it from the compiled bundle.
       */
      constexpr auto group = case_group::compile;
      run_check<T, case_id<state, group, 0, 1>>(std::forward<argsT>(args)...);
      run_check<T, case_id<state, group, 1, 1>>(std::forward<argsT>(args)...);
      run_check<T, case_id<state, group, 2, 1>>(std::forward<argsT>(args)...);
    }
    {
      /** Read a spec constant from a linked kernel_bundle without writing its
       *  value.
       *  Set the value in a kernel_bundle and read it from the linked bundle.
       *  Set the value in a kernel_bundle twice and read it from the linked
       *  bundle.
       */
      constexpr auto group = case_group::link;
      run_check<T, case_id<state, group, 0, 1>>(std::forward<argsT>(args)...);
      run_check<T, case_id<state, group, 1, 1>>(std::forward<argsT>(args)...);
      run_check<T, case_id<state, group, 2, 1>>(std::forward<argsT>(args)...);
    }
    {
      /** Read a spec constant from a built kernel_bundle without writing its
       *  value.
       *  Set the value in a kernel_bundle and read it from the built bundle.
       *  Set the value in a kernel_bundle twice and read it from the built
       *  bundle.
       */
      constexpr auto group = case_group::build;
      run_check<T, case_id<state, group, 0, 1>>(std::forward<argsT>(args)...);
      run_check<T, case_id<state, group, 1, 1>>(std::forward<argsT>(args)...);
      run_check<T, case_id<state, group, 2, 1>>(std::forward<argsT>(args)...);
    }
    {
      /** Set the value in a kernel_bundle and read it from the joined bundle.
       *  Set the value in a kernel_bundle twice and read it from the joined
       *  bundle.
       */
      constexpr auto group = case_group::join;
      run_check<T, case_id<state, group, 1, 1>>(std::forward<argsT>(args)...);
      run_check<T, case_id<state, group, 2, 1>>(std::forward<argsT>(args)...);
    }
    {
      /** Set the value in a kernel_bundle, compile and read it from the joined
       *  bundle.
       *  Set the value in a kernel_bundle twice, compile and read it from the
       *  joined bundle
       */
      constexpr auto group = case_group::compile_join;
      run_check<T, case_id<state, group, 1, 1>>(std::forward<argsT>(args)...);
      run_check<T, case_id<state, group, 2, 1>>(std::forward<argsT>(args)...);
    }
    {
      /** Set the value in a kernel_bundle, build and read it from the joined
       *  bundle.
       *  Set the value in a kernel_bundle twice, build and read it from the
       *  joined bundle.
       */
      constexpr auto group = case_group::build_join;
      run_check<T, case_id<state, group, 1, 1>>(std::forward<argsT>(args)...);
      run_check<T, case_id<state, group, 2, 1>>(std::forward<argsT>(args)...);
    }
    {
      /** Read a spec constant from a kernel without writing its value.
       *  Set the value in a kernel_bundle and read it from a kernel.
       *  Set the value in a kernel_bundle twice and read it from a kernel.
       *  Set the value in a kernel_bundle and read it twice from a kernel.
       */
      constexpr auto group = case_group::read_from_kernel;
      run_check<T, case_id<state, group, 0, 1>>(std::forward<argsT>(args)...);
      run_check<T, case_id<state, group, 1, 1>>(std::forward<argsT>(args)...);
      run_check<T, case_id<state, group, 2, 1>>(std::forward<argsT>(args)...);
      run_check<T, case_id<state, group, 1, 2>>(std::forward<argsT>(args)...);
    }
  }
}

template <typename T>
struct check_all {
  template <typename... argsT>
  void operator()(argsT&&... args) {
    run_all_checks<T, sycl::bundle_state::input>(std::forward<argsT>(args)...);
    run_all_checks<T, sycl::bundle_state::object>(std::forward<argsT>(args)...);
    run_all_checks<T, sycl::bundle_state::executable>(
        std::forward<argsT>(args)...);
  }
};

}  // namespace specialization_constants_via_kernel_bundle
#endif  // __SYCLCTS_TESTS_SPEC_CONST_VIA_KERNEL_BUNDLE_H
