/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2024 The Khronos Group Inc.
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

#include "../../group_functions/group_functions_common.h"
#include "non_uniform_group_common.h"

constexpr size_t init = 8;
constexpr size_t test_size = 8;

template <bool with_init, typename OpT, typename IteratorT>
size_t get_reduce_reference(IteratorT first, IteratorT end) {
  // Cast `init` to size_t so that guards are introduced in verification
  if constexpr (with_init)
    return std::accumulate(first, end, size_t(init), OpT());
  else
    return std::accumulate(first + 1, end, size_t(*first), OpT());
}

template <bool with_init, typename OpT, typename InputT, typename OutputT>
void result_verifier(const std::vector<InputT>& v_input,
                     const std::vector<OutputT>& v_output,
                     const std::vector<uint32_t>& sg_ids,
                     const std::vector<uint32_t>& nug_ids) {
  std::map<std::pair<uint32_t, uint32_t>, OutputT> reference_results;
  auto op = OpT();

  for (size_t i = 0; i < sg_ids.size(); ++i) {
    uint32_t sg_id = sg_ids[i];
    // Max values indicate items not participating.
    if (sg_id == std::numeric_limits<uint32_t>::max()) continue;

    uint32_t nug_id = nug_ids[i];
    auto key = std::make_pair(sg_id, nug_id);
    InputT input = v_input[i];

    auto iter = reference_results.find(key);
    if (iter == reference_results.end()) {
      // First may need to apply init value.
      OutputT value = with_init ? op(InputT(init), input) : OutputT(input);
      reference_results.emplace(std::make_pair(key, value));
    } else {
      iter->second = op(iter->second, input);
    }
  }

  bool res = false;
  for (size_t i = 0; i < sg_ids.size(); ++i) {
    uint32_t sg_id = sg_ids[i];
    // Max values indicate items not participating.
    if (sg_id == std::numeric_limits<uint32_t>::max()) continue;

    uint32_t nug_id = nug_ids[i];
    auto key = std::make_pair(sg_id, nug_id);

    OutputT expected = reference_results[key];
    OutputT actual = v_output[i];

    if (expected > util::exact_max<OutputT>) continue;

    INFO("Verifying reduction result of element with sub-group ID " +
         std::to_string(sg_id) + " and non-uniform group ID " +
         std::to_string(nug_id));
    CHECK(expected == actual);
  }
}

template <typename GroupT, typename T, typename OpT>
class joint_reduce_group_kernel;

/**
 * @brief Provides test for joint reduce by group
 * @tparam GroupT Non-uniform group type to use for testing
 * @tparam T Type for reduced values
 * @tparam OpT Type for binary operator
 */
template <typename GroupT, typename T, typename OpT>
void joint_reduce_group(sycl::queue& queue, const std::string& op_name) {
  const std::string group_name = NonUniformGroupHelper<GroupT>::get_name();

  INFO("Testing joint_reduce_group for " + group_name + " and " + op_name);
  if (!NonUniformGroupHelper<GroupT>::is_supported(queue.get_device())) {
    SKIP("Device does not support " + group_name);
  }

  const std::string test_name =
      "std::iterator_traits<Ptr>::value_type joint_reduce(GroupT g, Ptr first, "
      "Ptr last, BinaryOperation binary_op)";

  sycl::range<1> work_group_range =
      sycl_cts::util::work_group_range<1>(queue, test_size);
  size_t work_group_size = work_group_range.size();

  const size_t sizes[3] = {4, work_group_size / 2, 2 * work_group_size};

  for (size_t test_case = 0;
       test_case < NonUniformGroupHelper<GroupT>::num_test_cases; ++test_case) {
    const std::string test_case_name =
        NonUniformGroupHelper<GroupT>::get_test_case_name(test_case);
    INFO("Running test case (" + std::to_string(test_case) + ") with " +
         test_case_name);

    for (size_t size : sizes) {
      std::vector<T> v(size);
      std::iota(v.begin(), v.end(), 1);

      // array to return results
      std::vector<T> res(work_group_size);
      // participation markers
      std::vector<unsigned char> participating(work_group_size, 0);
      {
        sycl::buffer<T, 1> v_sycl(v.data(), sycl::range<1>(size));
        sycl::buffer<T, 1> res_sycl(res.data(),
                                    sycl::range<1>(work_group_size));
        sycl::buffer<unsigned char, 1> participating_sycl(
            participating.data(), sycl::range<1>(work_group_size));

        queue.submit([&](sycl::handler& cgh) {
          auto v_acc =
              v_sycl.template get_access<sycl::access::mode::read_write>(cgh);
          auto res_acc =
              res_sycl.template get_access<sycl::access::mode::read_write>(cgh);
          auto participating_acc =
              participating_sycl.get_access<sycl::access::mode::read_write>(
                  cgh);

          sycl::nd_range<1> executionRange(work_group_range, work_group_range);

          cgh.parallel_for<joint_reduce_group_kernel<GroupT, T, OpT>>(
              executionRange, [=](sycl::nd_item<1> item) {
                size_t index = item.get_global_linear_id();
                sycl::sub_group sub_group = item.get_sub_group();

                // If this item is not participating in the group, leave early.
                if (!NonUniformGroupHelper<GroupT>::should_participate(
                        sub_group, test_case))
                  return;

                GroupT non_uniform_group =
                    NonUniformGroupHelper<GroupT>::create(sub_group, test_case);
                participating_acc[index] = 1;

                T* v_begin = v_acc.get_pointer();
                T* v_end = v_begin + v_acc.size();

                ASSERT_RETURN_TYPE(
                    T,
                    sycl::joint_reduce(non_uniform_group, v_begin, v_end,
                                       OpT()),
                    "Return type of joint_reduce(GroupT g, Ptr first, Ptr "
                    "last, BinaryOperation binary_op) is wrong\n");

                res_acc[index] = sycl::joint_reduce(non_uniform_group, v_begin,
                                                    v_end, OpT());
              });
        });
      }

      const auto expected =
          get_reduce_reference<false, OpT>(v.begin(), v.end());

      if (expected <= util::exact_max<T>) {
        for (size_t i = 0; i < work_group_size; ++i) {
          if (!participating[i]) continue;

          std::string work_group =
              sycl_cts::util::work_group_print(work_group_range);
          CAPTURE(group_name, work_group, size, i);
          INFO("Verifying value of "
               << test_name << " with " << op_name
               << " operation and Ptr = " << type_name<T>() << "*");
          CHECK(res[i] == expected);
        }
      }
    }
  }
}

template <typename GroupT, typename T, typename OperatorT>
class invoke_joint_reduce_group {
 public:
  void operator()(sycl::queue& queue, const std::string& op_name) {
    if constexpr (type_traits::group_algorithms::is_legal_operator_v<
                      T, OperatorT>) {
      joint_reduce_group<GroupT, T, OperatorT>(queue, op_name);
    }
  }
};

template <typename GroupT, typename T, typename U, typename OpT>
class init_joint_reduce_group_kernel;

/**
 * @brief Provides test for joint reduce by group with init
 * @tparam GroupT Non-uniform group type to use for testing
 * @tparam T Type for init and result values
 * @tparam U Type for reduced values
 * @tparam OpT Type for binary operator
 */
template <typename GroupT, typename T, typename U, typename OpT>
void init_joint_reduce_group(sycl::queue& queue, const std::string& op_name) {
  const std::string group_name = NonUniformGroupHelper<GroupT>::get_name();

  INFO("Testing joint_reduce_group with init for " + group_name + " and " +
       op_name);
  if (!NonUniformGroupHelper<GroupT>::is_supported(queue.get_device())) {
    SKIP("Device does not support " + group_name);
  }

  const std::string test_name =
      "T joint_reduce(GroupT g, Ptr first, Ptr last, T init, "
      "BinaryOperation binary_op)";

  sycl::range<1> work_group_range =
      sycl_cts::util::work_group_range<1>(queue, test_size);
  size_t work_group_size = work_group_range.size();

  const size_t sizes[3] = {4, work_group_size / 2, 2 * work_group_size};

  for (size_t test_case = 0;
       test_case < NonUniformGroupHelper<GroupT>::num_test_cases; ++test_case) {
    const std::string test_case_name =
        NonUniformGroupHelper<GroupT>::get_test_case_name(test_case);
    INFO("Running test case (" + std::to_string(test_case) + ") with " +
         test_case_name);

    for (size_t size : sizes) {
      std::vector<U> v(size);
      std::iota(v.begin(), v.end(), 1);

      // array to return results
      std::vector<T> res(work_group_size);
      // participation markers
      std::vector<unsigned char> participating(work_group_size, 0);
      {
        sycl::buffer<U, 1> v_sycl(v.data(), sycl::range<1>(size));
        sycl::buffer<T, 1> res_sycl(res.data(),
                                    sycl::range<1>(work_group_size));
        sycl::buffer<unsigned char, 1> participating_sycl(
            participating.data(), sycl::range<1>(work_group_size));

        queue.submit([&](sycl::handler& cgh) {
          auto v_acc =
              v_sycl.template get_access<sycl::access::mode::read_write>(cgh);
          auto res_acc =
              res_sycl.template get_access<sycl::access::mode::read_write>(cgh);
          auto participating_acc =
              participating_sycl.get_access<sycl::access::mode::read_write>(
                  cgh);

          sycl::nd_range<1> executionRange(work_group_range, work_group_range);

          cgh.parallel_for<init_joint_reduce_group_kernel<GroupT, T, U, OpT>>(
              executionRange, [=](sycl::nd_item<1> item) {
                size_t index = item.get_global_linear_id();
                sycl::sub_group sub_group = item.get_sub_group();

                // If this item is not participating in the group, leave early.
                if (!NonUniformGroupHelper<GroupT>::should_participate(
                        sub_group, test_case))
                  return;

                GroupT non_uniform_group =
                    NonUniformGroupHelper<GroupT>::create(sub_group, test_case);
                participating_acc[index] = 1;

                U* v_begin = v_acc.get_pointer();
                U* v_end = v_begin + v_acc.size();

                ASSERT_RETURN_TYPE(
                    T,
                    sycl::joint_reduce(non_uniform_group, v_begin, v_end,
                                       T(init), OpT()),
                    "Return type of joint_reduce(GroupT g, Ptr first, Ptr "
                    "last, T init, BinaryOperation binary_op) is wrong\n");

                res_acc[index] = sycl::joint_reduce(non_uniform_group, v_begin,
                                                    v_end, T(init), OpT());
              });
        });
      }
      const auto expected = get_reduce_reference<true, OpT>(v.begin(), v.end());

      if (expected <= util::exact_max<T>) {
        for (size_t i = 0; i < work_group_size; ++i) {
          if (!participating[i]) continue;

          std::string work_group =
              sycl_cts::util::work_group_print(work_group_range);
          CAPTURE(group_name, work_group, size, i);
          INFO("Verifying value of "
               << test_name << " with " << op_name
               << " operation and Ptr = " << type_name<T>() << "*");
          CHECK(res[i] == expected);
        }
      }
    }
  }
}

template <typename GroupT, typename RetT, typename ReducedT, typename OperatorT>
class invoke_init_joint_reduce_group {
 public:
  void operator()(sycl::queue& queue, const std::string& op_name) {
    if constexpr (type_traits::group_algorithms::is_legal_operator_v<
                      RetT, OperatorT>) {
      init_joint_reduce_group<GroupT, RetT, ReducedT, OperatorT>(queue,
                                                                 op_name);
    }
  }
};

template <typename GroupT, typename T, typename OpT>
class reduce_over_group_kernel;

/**
 * @brief Provides test for reduce over group values
 * @tparam GroupT Non-uniform group type to use for testing
 * @tparam T Type for reduced values
 * @tparam OpT Type for binary operator
 */
template <typename GroupT, typename T, typename OpT>
void reduce_over_group(sycl::queue& queue, const std::string& op_name) {
  const std::string group_name = NonUniformGroupHelper<GroupT>::get_name();

  INFO("Testing reduce_over_group for " + group_name + " and " + op_name);
  if (!NonUniformGroupHelper<GroupT>::is_supported(queue.get_device())) {
    SKIP("Device does not support " + group_name);
  }

  const std::string test_name =
      "T reduce_over_group(GroupT g, T x, BinaryOperation binary_op)";

  sycl::range<1> work_group_range =
      sycl_cts::util::work_group_range<1>(queue, test_size);
  size_t work_group_size = work_group_range.size();

  for (size_t test_case = 0;
       test_case < NonUniformGroupHelper<GroupT>::num_test_cases; ++test_case) {
    const std::string test_case_name =
        NonUniformGroupHelper<GroupT>::get_test_case_name(test_case);
    INFO("Running test case (" + std::to_string(test_case) + ") with " +
         test_case_name);

    bool res = false;
    // array to input data
    std::vector<T> v(work_group_size);
    std::iota(v.begin(), v.end(), 1);
    // array to reduce results
    std::vector<T> nug_output(work_group_size, 0);
    // Sub-group ID and non-uniform group ID (Max int means the item is not
    // participating in a reduction)
    std::vector<uint32_t> sg_id(work_group_size,
                                std::numeric_limits<uint32_t>::max());
    std::vector<uint32_t> nug_id(work_group_size,
                                 std::numeric_limits<uint32_t>::max());

    {
      sycl::buffer<T, 1> v_sycl(v.data(), sycl::range<1>(work_group_size));
      sycl::buffer<T, 1> nug_output_sycl(nug_output.data(),
                                         sycl::range<1>(work_group_size));
      sycl::buffer<uint32_t, 1> sg_id_sycl(sg_id.data(),
                                           sycl::range<1>(work_group_size));
      sycl::buffer<uint32_t, 1> nug_id_sycl(nug_id.data(),
                                            sycl::range<1>(work_group_size));

      queue.submit([&](sycl::handler& cgh) {
        auto v_acc = v_sycl.template get_access<sycl::access::mode::read>(cgh);
        auto nug_output_acc =
            nug_output_sycl.template get_access<sycl::access::mode::read_write>(
                cgh);
        auto sg_id_acc =
            sg_id_sycl.template get_access<sycl::access::mode::read_write>(cgh);
        auto nug_id_acc =
            nug_id_sycl.template get_access<sycl::access::mode::read_write>(
                cgh);
        sycl::nd_range<1> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<reduce_over_group_kernel<GroupT, T, OpT>>(
            executionRange, [=](sycl::nd_item<1> item) {
              size_t index = item.get_global_linear_id();

              sycl::sub_group sub_group = item.get_sub_group();

              // If this item is not participating in the group, leave early.
              if (!NonUniformGroupHelper<GroupT>::should_participate(sub_group,
                                                                     test_case))
                return;

              GroupT non_uniform_group =
                  NonUniformGroupHelper<GroupT>::create(sub_group, test_case);

              sg_id_acc[index] = sub_group.get_group_linear_id();
              nug_id_acc[index] = non_uniform_group.get_group_linear_id();

              ASSERT_RETURN_TYPE(T,
                                 sycl::reduce_over_group(non_uniform_group,
                                                         v_acc[index], OpT()),
                                 "Return type of reduce_over_group(GroupT g, "
                                 "T x, BinaryOperation binary_op) is wrong\n");
              nug_output_acc[index] = sycl::reduce_over_group(
                  non_uniform_group, v_acc[index], OpT());
            });
      });
    }

    // Verify return value for reduce_over_group on GroupT
    {
      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(group_name, work_group);
      result_verifier<false, OpT>(v, nug_output, sg_id, nug_id);
    }
  }
}

template <typename GroupT, typename T, typename OperatorT>
class invoke_reduce_over_group {
 public:
  void operator()(sycl::queue& queue, const std::string& op_name) {
    if constexpr (type_traits::group_algorithms::is_legal_operator_v<
                      T, OperatorT>) {
      reduce_over_group<GroupT, T, OperatorT>(queue, op_name);
    }
  }
};

template <typename GroupT, typename T, typename U, typename OpT>
class init_reduce_over_group_kernel;

/**
 * @brief Provides test for reduce over group values with init
 * @tparam GroupT Non-uniform group type to use for testing
 * @tparam T Type for init and result values
 * @tparam U Type for group values
 * @tparam OpT Type for binary operator
 */
template <typename GroupT, typename T, typename U, typename OpT>
void init_reduce_over_group(sycl::queue& queue, const std::string& op_name) {
  const std::string group_name = NonUniformGroupHelper<GroupT>::get_name();

  INFO("Testing reduce_over_group with init for " + group_name + " and " +
       op_name);
  if (!NonUniformGroupHelper<GroupT>::is_supported(queue.get_device())) {
    SKIP("Device does not support " + group_name);
  }

  const std::string test_name =
      "T reduce_over_group(GroupT g, V x, T init, BinaryOperation binary_op)";

  sycl::range<1> work_group_range =
      sycl_cts::util::work_group_range<1>(queue, test_size);
  size_t work_group_size = work_group_range.size();

  for (size_t test_case = 0;
       test_case < NonUniformGroupHelper<GroupT>::num_test_cases; ++test_case) {
    const std::string test_case_name =
        NonUniformGroupHelper<GroupT>::get_test_case_name(test_case);
    INFO("Running test case (" + std::to_string(test_case) + ") with " +
         test_case_name);

    bool res = false;
    // array to input data
    std::vector<U> v(work_group_size);
    std::iota(v.begin(), v.end(), 1);
    // array to reduce results
    std::vector<T> nug_output(work_group_size, 0);
    // Sub-group ID and non-uniform group ID (Max int means the item is not
    // participating in a reduction)
    std::vector<uint32_t> sg_id(work_group_size,
                                std::numeric_limits<uint32_t>::max());
    std::vector<uint32_t> nug_id(work_group_size,
                                 std::numeric_limits<uint32_t>::max());

    {
      sycl::buffer<U, 1> v_sycl(v.data(), sycl::range<1>(work_group_size));
      sycl::buffer<T, 1> nug_output_sycl(nug_output.data(),
                                         sycl::range<1>(work_group_size));
      sycl::buffer<uint32_t, 1> sg_id_sycl(sg_id.data(),
                                           sycl::range<1>(work_group_size));
      sycl::buffer<uint32_t, 1> nug_id_sycl(nug_id.data(),
                                            sycl::range<1>(work_group_size));

      queue.submit([&](sycl::handler& cgh) {
        auto v_acc = v_sycl.template get_access<sycl::access::mode::read>(cgh);
        auto nug_output_acc =
            nug_output_sycl.template get_access<sycl::access::mode::read_write>(
                cgh);
        auto sg_id_acc =
            sg_id_sycl.template get_access<sycl::access::mode::read_write>(cgh);
        auto nug_id_acc =
            nug_id_sycl.template get_access<sycl::access::mode::read_write>(
                cgh);
        sycl::nd_range<1> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<init_reduce_over_group_kernel<GroupT, T, U, OpT>>(
            executionRange, [=](sycl::nd_item<1> item) {
              size_t index = item.get_global_linear_id();

              sycl::sub_group sub_group = item.get_sub_group();

              // If this item is not participating in the group, leave early.
              if (!NonUniformGroupHelper<GroupT>::should_participate(sub_group,
                                                                     test_case))
                return;

              GroupT non_uniform_group =
                  NonUniformGroupHelper<GroupT>::create(sub_group, test_case);

              sg_id_acc[index] = sub_group.get_group_linear_id();
              nug_id_acc[index] = non_uniform_group.get_group_linear_id();

              ASSERT_RETURN_TYPE(
                  T,
                  sycl::reduce_over_group(non_uniform_group, v_acc[index],
                                          T(init), OpT()),
                  "Return type of reduce_over_group(GroupT g, V x, T init, "
                  "BinaryOperation binary_op) is wrong\n");
              nug_output_acc[index] = sycl::reduce_over_group(
                  non_uniform_group, v_acc[index], T(init), OpT());
            });
      });
    }

    // Verify return value for reduce_over_group on GroupT
    {
      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(group_name, work_group);
      result_verifier<true, OpT>(v, nug_output, sg_id, nug_id);
    }
  }
}

template <typename GroupT, typename RetT, typename ReducedT, typename OperatorT>
class invoke_init_reduce_over_group {
 public:
  void operator()(sycl::queue& queue, const std::string& op_name) {
    if constexpr (type_traits::group_algorithms::is_legal_operator_v<
                      RetT, OperatorT>) {
      init_reduce_over_group<GroupT, RetT, ReducedT, OperatorT>(queue, op_name);
    }
  }
};
