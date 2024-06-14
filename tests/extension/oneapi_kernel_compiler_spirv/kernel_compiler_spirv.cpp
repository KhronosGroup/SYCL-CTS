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

#include "../../common/common.h"

namespace kernel_compiler_spirv::tests {

template <typename... Ts>
std::vector<std::byte> createByteVector(Ts&&... args) noexcept {
  return {std::byte(std::forward<Ts>(args))...};
}

static const std::vector<std::byte> kernels = createByteVector(
#include "kernels.inc"
);
static const std::vector<std::byte> kernels_fp16 = createByteVector(
#include "kernels_fp16.inc"
);
static const std::vector<std::byte> kernels_fp64 = createByteVector(
#include "kernels_fp64.inc"
);

#ifdef SYCL_EXT_ONEAPI_AUTO_LOCAL_RANGE

sycl::kernel_bundle<sycl::bundle_state::executable> loadKernelsFromVector(
    sycl::queue& q, const std::vector<std::byte>& kernels) {
  namespace syclex = sycl::ext::oneapi::experimental;

  // Create a kernel bundle from the binary SPIR-V.
  sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
      syclex::create_kernel_bundle_from_source(
          q.get_context(), syclex::source_language::spirv, kernels);

  // Build the SPIR-V module for our device.
  sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
      syclex::build(kb_src);
  return kb_exe;
}

void testSimpleKernel(sycl::queue& q, const sycl::kernel& kernel,
                      int multiplier, int added) {
  const auto num_args = kernel.get_info<sycl::info::kernel::num_args>();
  REQUIRE(num_args == 2);

  constexpr int N = 4;
  std::array<int, N> input_array{0, 1, 2, 3};

  sycl::buffer<int> input_buffer{input_array.data(), sycl::range<1>(N)};
  sycl::buffer<int> output_buffer{sycl::range<1>(N)};

  q.submit([&](sycl::handler& cgh) {
    cgh.set_args(sycl::accessor{input_buffer, cgh, sycl::read_only},
                 sycl::accessor{output_buffer, cgh, sycl::write_only});
    cgh.parallel_for(sycl::range<1>{N}, kernel);
  });

  sycl::host_accessor out{output_buffer};
  for (int i = 0; i < N; i++) {
    CHECK(out[i] == ((i * multiplier) + added));
  }
}

/*
For each type T, the matching SPIR-V kernel takes four parameters:

  1. [in]     a:   T
  2. [in]     b:   OpTypePointer(CrossWorkgroup) to T
  3. [in/out] tmp: OpTypePointer(Workgroup) to T
  4. [out]    c:   OpTypePointer(CrossWorkgroup) to T

The kernel computes the following expressions:

  1. *tmp = (a * a);
  2. *c = (*tmp) + ((*b) * (*b));

This test case sets the four parameters, runs the kernel, and asserts that
output c has the expected value.
*/
template <typename T>
void testParam(sycl::queue& q, const sycl::kernel& kernel) {
  const auto num_args = kernel.get_info<sycl::info::kernel::num_args>();
  REQUIRE(num_args == 4);

  // Kernel computes sum of squared inputs.
  const T a = 2;
  const T b = 5;
  const T expected = (a * a) + (b * b);

  sycl::buffer a_buffer(&a, sycl::range<1>(1));

  T* const b_ptr = sycl::malloc_shared<T>(1, q);
  b_ptr[0] = b;

  T output{};
  sycl::buffer output_buffer(&output, sycl::range<1>(1));

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<T, 1> local(1, cgh);
    // Pass T for scalar parameter.
    cgh.set_arg(0, a);
    // Pass USM pointer for OpTypePointer(CrossWorkgroup) parameter.
    cgh.set_arg(1, b_ptr);
    // Pass sycl::accessor for OpTypePointer(CrossWorkgroup) parameter.
    cgh.set_arg(2, sycl::accessor{output_buffer, cgh, sycl::write_only});
    // Pass sycl::local_accessor for OpTypePointer(Workgroup) parameter.
    cgh.set_arg(3, local);
    cgh.parallel_for(sycl::range<1>{1}, kernel);
  });

  sycl::host_accessor out{output_buffer};
  CHECK(out[0] == expected);
  sycl::free(b_ptr, q);
}

void testStruct(sycl::queue& q, const sycl::kernel& kernel) {
  const auto num_args = kernel.get_info<sycl::info::kernel::num_args>();
  REQUIRE(num_args == 2);

  // This definition must match the one used in the kernel.
  struct S {
    std::int32_t i;
    cl_float f;
    std::int32_t* p;
    struct Inner {
      std::int32_t i;
      float f;
      std::int32_t* p;
    } inner;
  };

  // Any constants can be used to initialize this input.
  std::int32_t* const in_p0 = sycl::malloc_shared<std::int32_t>(1, q);
  std::int32_t* const in_p1 = sycl::malloc_shared<std::int32_t>(1, q);
  *in_p0 = 3;
  *in_p1 = 6;
  S input{1, 2.0f, in_p0, S::Inner{4, 5.0f, in_p1}};

  std::int32_t* const out_p0 = sycl::malloc_shared<std::int32_t>(1, q);
  std::int32_t* const out_p1 = sycl::malloc_shared<std::int32_t>(1, q);
  *out_p0 = 0;
  *out_p1 = 0;
  S* output = sycl::malloc_shared<S>(1, q);
  *output = S{0, 0, out_p0, S::Inner{0, 0, out_p1}};

  q.submit([&](sycl::handler& cgh) {
     cgh.set_args(input, output);
     cgh.parallel_for(sycl::range<1>{1}, kernel);
   }).wait();

  // For each scalar struct member, output == (2 * input). For pointer members,
  // *output == (2 * (*input)).
  CHECK(output->i == input.i * 2);
  CHECK(output->f == input.f * 2);
  CHECK(*output->p == (*input.p) * 2);
  CHECK(output->inner.i == input.inner.i * 2);
  CHECK(output->inner.f == input.inner.f * 2);
  CHECK(*output->inner.p == (*input.inner.p) * 2);

  sycl::free(output, q);
  sycl::free(in_p0, q);
  sycl::free(in_p1, q);
  sycl::free(out_p0, q);
  sycl::free(out_p1, q);
}

void testKernels() {
  const auto getKernel =
      [](sycl::kernel_bundle<sycl::bundle_state::executable>& bundle,
         const std::string& name) {
        return bundle.ext_oneapi_get_kernel(name);
      };

  sycl::queue q;
  auto bundle = loadKernelsFromVector(q, kernels);

  // Test kernel retrieval functions.
  {
    CHECK(bundle.ext_oneapi_has_kernel("my_kernel"));
    CHECK(bundle.ext_oneapi_has_kernel("OpTypeInt8"));
    CHECK(bundle.ext_oneapi_has_kernel("OpTypeInt16"));
    CHECK(bundle.ext_oneapi_has_kernel("OpTypeInt32"));
    CHECK(bundle.ext_oneapi_has_kernel("OpTypeInt64"));
    CHECK(bundle.ext_oneapi_has_kernel("OpTypeInt9") == false);
    CHECK(bundle.ext_oneapi_has_kernel("") == false);

    CHECK_NOTHROW(bundle.ext_oneapi_get_kernel("my_kernel"));
    CHECK_NOTHROW(bundle.ext_oneapi_get_kernel("OpTypeInt8"));
    CHECK_NOTHROW(bundle.ext_oneapi_get_kernel("OpTypeInt16"));
    CHECK_NOTHROW(bundle.ext_oneapi_get_kernel("OpTypeInt32"));
    CHECK_NOTHROW(bundle.ext_oneapi_get_kernel("OpTypeInt64"));
    CHECK_THROWS_AS(bundle.ext_oneapi_get_kernel("OpTypeInt9"),
                    sycl::exception);
    CHECK_THROWS_AS(bundle.ext_oneapi_get_kernel(""), sycl::exception);
  }

  // Test simple kernel.
  testSimpleKernel(q, getKernel(bundle, "my_kernel"), 2, 100);

  // Test parameters.
  testParam<std::int8_t>(q, getKernel(bundle, "OpTypeInt8"));
  testParam<std::int16_t>(q, getKernel(bundle, "OpTypeInt16"));
  testParam<std::int32_t>(q, getKernel(bundle, "OpTypeInt32"));
  testParam<std::int64_t>(q, getKernel(bundle, "OpTypeInt64"));
  testParam<float>(q, getKernel(bundle, "OpTypeFloat32"));

  // Test OpTypeFloat16 parameters.
  if (q.get_device().has(sycl::aspect::fp16)) {
    auto fp16_bundle = loadKernelsFromVector(q, kernels_fp16);
    testParam<sycl::half>(q, getKernel(fp16_bundle, "OpTypeFloat16"));
  }

  // Test OpTypeFloat64 parameters.
  if (q.get_device().has(sycl::aspect::fp64)) {
    auto fp64_bundle = loadKernelsFromVector(q, kernels_fp64);
    testParam<double>(q, getKernel(fp64_bundle, "OpTypeFloat64"));
  }

  // Test OpTypeStruct parameters.
  testStruct(q, getKernel(bundle, "OpTypeStruct"));
}

#endif

TEST_CASE("Test case for \"Kernel Compiler SPIR-V\" extension",
          "[oneapi_kernel_compiler_spirv]") {
#ifndef SYCL_EXT_ONEAPI_KERNEL_COMPILER_SPIRV
  SKIP("SYCL_EXT_ONEAPI_KERNEL_COMPILER_SPIRV is not defined");
#else
  testKernels();
#endif
}

}  // namespace kernel_compiler_spirv::tests
