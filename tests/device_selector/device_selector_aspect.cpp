/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022-2023 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "../common/type_coverage.h"

#include <random>
#include <sstream>

namespace device_selector_aspect {
using namespace sycl_cts;

/** Return a named value pack of all defined aspects. */
static auto get_aspect_pack() {
// FIXME: remove when https://github.com/intel/llvm/issues/8324 is fixed
#ifdef SYCL_CTS_COMPILING_WITH_DPCPP
  WARN(
      "DPCPP does not define sycl::aspect::emulated."
      "Skipping test cases for this aspect.");
#endif
  static const auto types =
      value_pack<sycl::aspect, sycl::aspect::cpu, sycl::aspect::gpu,
                 sycl::aspect::accelerator, sycl::aspect::custom,
                 sycl::aspect::host_debuggable, sycl::aspect::fp16,
                 sycl::aspect::fp64, sycl::aspect::atomic64,
                 sycl::aspect::image, sycl::aspect::online_compiler,
                 sycl::aspect::online_linker, sycl::aspect::queue_profiling,
                 sycl::aspect::usm_device_allocations,
                 sycl::aspect::usm_host_allocations,
                 sycl::aspect::usm_atomic_host_allocations,
                 sycl::aspect::usm_shared_allocations,
                 sycl::aspect::usm_atomic_shared_allocations,
                 sycl::aspect::usm_system_allocations
#ifndef SYCL_CTS_COMPILING_WITH_DPCPP
                 ,
                 sycl::aspect::emulated
#endif
                 >::generate_named();
  return types;
}

/**
 Check whether all specified constructors for \p aspect_selector are
 available. */
void check_no_aspects() {
  const auto selector_vector = sycl::aspect_selector({});
  const auto selector_vector_deny = sycl::aspect_selector({}, {});
  const auto selector_args = sycl::aspect_selector();
  const auto selector_params = sycl::aspect_selector<>();
}

/**
 Given a list of aspects and a list of forbidden aspects, find out if a
 conforming device exists. */
bool device_exists(const std::vector<sycl::aspect>& aspect_list,
                   const std::vector<sycl::aspect>& deny_list) {
  const auto devices = sycl::device::get_devices();
  return std::any_of(
      devices.cbegin(), devices.cend(), [&](const sycl::device& dev) {
        const auto dev_has_aspect = [&](const sycl::aspect& aspect) {
          return dev.has(aspect);
        };
        return std::all_of(aspect_list.cbegin(), aspect_list.cend(),
                           dev_has_aspect) &&
               std::none_of(deny_list.cbegin(), deny_list.cend(),
                            dev_has_aspect);
      });
}

/**
 Given a list of aspects, find out if the selector finds a conforming
 device. */
template <typename Selector>
void test_selector_accept(const Selector& selector,
                          const std::vector<sycl::aspect>& accept_list) {
  sycl::device device{selector};
  for (const auto& aspect : accept_list) {
    CHECK(device.has(aspect));
  }

  sycl::queue queue{selector};
  for (const auto& aspect : accept_list) {
    CHECK(queue.get_device().has(aspect));
  }

  sycl::platform platform{selector};
  for (const auto& aspect : accept_list) {
    check_return_type<bool>(platform.has(aspect),
                            "sycl::platform::has(sycl::aspect)");
  }
}

/**
 Given a list of forbidden aspects, find out if the selector finds a
 conforming device. */
template <typename Selector>
void test_selector_deny(const Selector& selector,
                        const std::vector<sycl::aspect>& deny_list) {
  sycl::device device{selector};
  for (const auto& aspect : deny_list) {
    CHECK((!device.has(aspect)));
  }

  sycl::queue queue{selector};
  for (const auto& aspect : deny_list) {
    CHECK((!queue.get_device().has(aspect)));
  }

  sycl::platform platform{selector};
  // If all devices in the platform have an aspect, the platform itself has the
  // aspect. Hence, selected platform should not have any denied aspect.
  for (const auto& aspect : deny_list) {
    CHECK((!platform.has(aspect)));
  }
}

/**
 Given a selector that selects a device not available in the system,
 check if the error behavior is correct. */
template <typename Selector>
void check_selector_exception(const Selector& s) {
  INFO(
      "device with requested aspects does not exist, checking if error is "
      "correct");

  try {
    sycl::device device{s};
    FAIL("selected a device when none are available");
  } catch (const sycl::exception& e) {
    // ComputeCpp cannot make comparison
#ifndef SYCL_CTS_COMPILING_WITH_COMPUTECPP
    CHECK(sycl::errc::runtime == e.code());
#endif  // SYCL_CTS_COMPILING_WITH_COMPUTECPP
  }
}

/**
 Tests whether a given selector conforms to a given list of required
 aspects. */
template <typename Selector>
void test_selector(const Selector& selector,
                   const std::vector<sycl::aspect>& accept_list) {
  if (device_exists(accept_list, {})) {
    test_selector_accept(selector, accept_list);
  } else {
    check_selector_exception(selector);
  }
}

/**
 Tests whether a given selector conforms to a given list of required
 aspects and a list of denied aspects. */
template <typename Selector>
void test_selector(const Selector& selector,
                   const std::vector<sycl::aspect>& accept_list,
                   const std::vector<sycl::aspect>& deny_list) {
  if (device_exists(accept_list, deny_list)) {
    test_selector_accept(selector, accept_list);
    test_selector_deny(selector, deny_list);
  } else {
    check_selector_exception(selector);
  }
}

/**
 Checks aspect selector's constructors: using an accept list
 (and no deny list), using an accept list and a deny list, using accepted
 variadic function arguments, and using accepted variadic template
 parameters. */
template <sycl::aspect... Aspects>
void check_aspect_selector(const std::vector<sycl::aspect>& deny_list) {
  const std::vector<sycl::aspect> accept_list{Aspects...};
  test_selector(sycl::aspect_selector(accept_list), accept_list);
  if (!deny_list.empty()) {
    test_selector(sycl::aspect_selector(accept_list, deny_list), accept_list,
                  deny_list);
  }
  test_selector(sycl::aspect_selector(Aspects...), accept_list);
  test_selector(sycl::aspect_selector<Aspects...>(), accept_list);
}

/**
 Functor that checks a selector with a single aspect and an empty list of
 denied aspects. */
template <typename AspectT>
class check_for_single_aspect {
  static constexpr sycl::aspect aspect = AspectT::value;

 public:
  void operator()(const std::string& aspect_name) {
    INFO("for aspect " << aspect_name);
    check_aspect_selector<aspect>({});
  }
};

/**
 Functor that checks a selector with two aspects and an empty list of
 denied aspects. */
template <typename Aspect1T, typename Aspect2T>
class check_for_two_aspects {
  static constexpr sycl::aspect aspect1 = Aspect1T::value;
  static constexpr sycl::aspect aspect2 = Aspect2T::value;

 public:
  void operator()(const std::string& aspect1_name,
                  const std::string& aspect2_name) {
    INFO("for aspects " << aspect1_name << " and " << aspect2_name);
    check_aspect_selector<aspect1, aspect2>({});
  }
};

/**
 Returns a randomly-sized list of random denied aspects that are not part of
 the requested aspects \p selected_aspect_list. */
template <typename Rng>
std::vector<sycl::aspect> generate_denied_list(
    const std::vector<sycl::aspect>& aspect_list,
    const std::vector<sycl::aspect>& selected_aspect_list, Rng& rng) {
  std::vector<sycl::aspect> denied_list;
  for (const auto& aspect : aspect_list) {
    bool is_selected =
        std::find(selected_aspect_list.begin(), selected_aspect_list.end(),
                  aspect) != selected_aspect_list.end();
    // if the aspect is already part of the selected aspects, it cannot be
    // part of the denied aspects
    if (is_selected) {
      continue;
    }
    // else, select the aspect with a probability of
    // 1 over (#aspects - #selected aspects), which has an expected value of 1
    unsigned int rng_range = Rng::max() - Rng::min();
    unsigned int non_selected_aspect_count =
        aspect_list.size() - selected_aspect_list.size();
    if (rng() - Rng::min() < rng_range / non_selected_aspect_count) {
      denied_list.push_back(aspect);
    }
  }
  return denied_list;
}

/**
 Functor that checks a selector with multiple aspects and optionally
 a list of denied aspects. */
template <typename... AspectsT>
class check_for_multiple_aspects {
  static constexpr std::size_t aspect_count = sizeof...(AspectsT);

 public:
  void operator()(
      const std::vector<sycl::aspect>& deny_list,
      const std::array<std::string, aspect_count>& accept_aspect_names) {
    std::ostringstream os;
    os << "for aspects (" << aspect_count << "):\n";
    for (const auto& aspect_name : accept_aspect_names) {
      os << aspect_name << "\n";
    }
    os << "for denied aspects (" << deny_list.size() << "):\n";
    for (const auto& aspect : deny_list) {
      os << Catch::StringMaker<sycl::aspect>::convert(aspect) << "\n";
    }
    INFO(os.str());

    check_aspect_selector<AspectsT::value...>(deny_list);
  }
};

template <typename... SelectedAspectsT>
constexpr void check_subset() {
  // create a value pack using the selected aspects
  const auto aspect_pack =
      value_pack<sycl::aspect, SelectedAspectsT::value...>::generate_named();
  check_for_multiple_aspects<SelectedAspectsT...>{}({}, aspect_pack.names);
}

template <typename... AspectsT, std::size_t... Indices>
constexpr void create_subset(std::tuple<AspectsT...> aspect_tuple,
                             std::index_sequence<Indices...>) {
  // expand sequence to index into the aspect tuple
  check_subset<
      typename std::tuple_element<Indices, decltype(aspect_tuple)>::type...>();
}

template <std::size_t SmallestSubset, typename... AspectsT,
          std::size_t... Sizes>
constexpr void create_subsets(std::tuple<AspectsT...> aspect_tuple,
                              std::index_sequence<Sizes...>) {
  // expand sizes pack to create one sequence for each desired subset size
  (create_subset(aspect_tuple,
                 std::make_index_sequence<SmallestSubset + Sizes>{}),
   ...);
}

/** Checks subsets of size \p SmallestSubset, \p SmallestSubset + 1, etc. */
template <std::size_t SmallestSubset, typename... AspectsT>
constexpr void check_for_subset(const named_type_pack<AspectsT...>&) {
  static_assert(sizeof...(AspectsT) >= SmallestSubset);
  constexpr std::size_t subset_count = sizeof...(AspectsT) + 1 - SmallestSubset;
  // tuple of aspects to pass to function
  auto aspect_tuple = std::tuple<AspectsT...>{};
  // one index for each subset size
  auto subset_sizes = std::make_index_sequence<subset_count>{};
  create_subsets<SmallestSubset>(aspect_tuple, subset_sizes);
}

/**
 Get next state value for a linear congruential engine with the same
 parameters as std::minstd_rand. */
constexpr unsigned int get_next_state(unsigned int state) {
  return (48271 * state) % 2147483647;
}

/** \p Seed is unique for this random set. */
template <unsigned int Seed, typename... SelectedAspectsT>
void check_random_set(const std::vector<sycl::aspect>& aspect_list) {
  // create a value pack and vector for the selected aspects
  const auto selected_aspect_pack =
      value_pack<sycl::aspect, SelectedAspectsT::value...>::generate_named();
  std::vector<sycl::aspect> selected_aspect_list = {SelectedAspectsT::value...};

  // generate a vector of denied aspects using a runtime random number
  // generator, choose *some* different seed
  std::minstd_rand rng_runtime(Seed + 1);
  const std::vector<sycl::aspect> deny_list =
      generate_denied_list(aspect_list, selected_aspect_list, rng_runtime);

  check_for_multiple_aspects<SelectedAspectsT...>{}(deny_list,
                                                    selected_aspect_pack.names);
}

/** \p Seed is unique for this random set. */
template <unsigned int Seed, typename... AspectsT, unsigned int... Seeds>
void fill_random_set(std::tuple<AspectsT...> aspect_tuple,
                     std::integer_sequence<unsigned int, Seeds...>) {
  std::vector<sycl::aspect> aspect_list = {AspectsT::value...};
  constexpr unsigned int aspect_count = sizeof...(AspectsT);
  // The maximum number of seeds used before this set, plus one for offset.
  // Due to the simple nature of the LCG, the + 1 is required as otherwise
  // the same sequence is produced as % aspect_count is applied on the index.
  constexpr unsigned int seed_offset = Seed * (aspect_count + 1);
  // Expand seeds pack to:
  // 1. Obtain a unique seed among all random sets.
  // 2. Use the seed to generate a random index into the aspect pack.
  // 3. Create a new aspect pack of selected aspects.
  check_random_set<Seed, typename std::tuple_element<
                             get_next_state(seed_offset + Seeds) % aspect_count,
                             decltype(aspect_tuple)>::type...>(aspect_list);
}

/** \p Seed is unique for this random set. */
template <unsigned int Seed, typename... AspectsT>
constexpr void create_random_set(std::tuple<AspectsT...> aspect_tuple) {
  // obtain random non-zero length, no longer than number of aspects
  constexpr unsigned int size =
      1 + get_next_state(Seed) % (sizeof...(AspectsT) - 1);
  // one index for each random aspect in this set
  auto aspect_seeds = std::make_integer_sequence<unsigned int, size>{};
  fill_random_set<Seed>(aspect_tuple, aspect_seeds);
}

template <typename... AspectsT, unsigned int... Seeds>
constexpr void create_random_sets(
    std::tuple<AspectsT...> aspect_tuple,
    std::integer_sequence<unsigned int, Seeds...>) {
  // expand seeds pack to create a random set with each seed
  (create_random_set<Seeds>(aspect_tuple), ...);
}

/**
 Checks \p Count randomly-sized sets of random aspects with a randomly-sized
 list of random denied aspects. */
template <unsigned int Count, typename... AspectsT>
constexpr void check_for_random_set(const named_type_pack<AspectsT...>&) {
  // tuple of aspects to pass to function
  auto aspect_tuple = std::tuple<AspectsT...>{};
  // one index for each random set, which forms the seed for the random number
  // generator used to generate the size and elements of the set
  auto set_sizes = std::make_integer_sequence<unsigned int, Count>{};
  create_random_sets(aspect_tuple, set_sizes);
}

template <typename AspectT>
class kernel_aspect;

#ifndef SYCL_CTS_COMPILING_WITH_DPCPP
/**
 Functor that checks any_device_has and all_devices_have functionality. */
template <typename AspectT>
class check_any_device_has_all_devices_have {
  static constexpr sycl::aspect aspect = AspectT::value;

 public:
  void operator()(const std::string& aspect_name) {
    using k_name = kernel_aspect<AspectT>;

    sycl::queue queue = util::get_cts_object::queue();
    queue.submit(
        [&](sycl::handler& cgh) { cgh.single_task<k_name>([=]() {}); });

    auto exec_device_has_aspect = [=](auto d) {
      return d.has(aspect) && sycl::is_compatible<k_name>(d);
    };
    auto devices = sycl::device::get_devices();
    bool any_device =
        any_of(devices.begin(), devices.end(), exec_device_has_aspect);
    bool all_devices =
        all_of(devices.begin(), devices.end(), exec_device_has_aspect);
    if (any_device) {
      INFO(
          "Check that if some device has aspect A, "
          "any_device_has_v<A> is true.");
      CHECK(std::is_base_of_v<std::true_type, sycl::any_device_has<aspect>>);
      CHECK(any_device_has_v < aspect >>);
    }

    if (!all_devices) {
      INFO(
          "Check that if some device does not have aspect A, "
          "all_devices_have_v<A> is false.");
      CHECK(std::is_base_of_v<std::false_type, sycl::all_devices_have<aspect>>);
      CHECK_FALSE(all_devices_have_v < aspect >>);
    }
  }
};
#endif

TEST_CASE("aspect", "[device_selector]") {
#if SYCL_CTS_COMPILING_WITH_COMPUTECPP
  WARN("ComputeCPP cannot compare exception code. Workaround is in place.");
#endif

  // check whether all constructors compile when no aspects are specified
  check_no_aspects();

  // obtain a named value pack of all defined aspects
  const auto aspect_pack = get_aspect_pack();

  // every single aspect
  for_all_combinations<check_for_single_aspect>(aspect_pack);

  // every possible combination of two aspects
  for_all_combinations<check_for_two_aspects>(aspect_pack, aspect_pack);

  // a subset of three aspects, four aspects, five aspects, etc.
  check_for_subset<3>(aspect_pack);

  // randomly-sized list of random aspects (greater than two), in addition to a
  // randomly-generated list of forbidden aspects
  constexpr unsigned int random_aspects_count = 100;
  check_for_random_set<random_aspects_count>(aspect_pack);
}

// FIXME: re-enable when sycl::any_device_has, sycl::all_devices_have are
// implemented
DISABLED_FOR_TEST_CASE(DPCPP)
("Check any_device_has and all_devices_have", "[device_selector]")({
  const auto aspect_pack = get_aspect_pack();

  for_all_combinations<check_any_device_has_all_devices_have>(aspect_pack);
});

}  // namespace device_selector_aspect
