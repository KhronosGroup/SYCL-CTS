#!/usr/bin/env python3
# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#
#   Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
#   Copyright (c) 2022 The Khronos Group Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ************************************************************************

import sys
import argparse
from string import Template
sys.path.append('../common/')
from common_python_vec import (Data, make_fp_or_byte_explicit, make_func_call,
                               wrap_with_test_func, write_source_file,
                               wrap_with_extension_checks, get_types,
                               remove_namespaces_whitespaces, cast_to_bool)

TEST_NAME = 'LOAD_STORE'

global_multi_ptr_load_store_test_template = Template(
    """
    {
        ${type} inputData${type_as_str}${size}[${size}] = {${in_order_vals}};
        ${type} outputData${type_as_str}${size}[${size}] = {${val}};
        ${type} swizzleInputData${type_as_str}${size}[${size}] = {${reverse_order_vals}};
        ${type} swizzleOutputData${type_as_str}${size}[${size}] = {${val}};
        {
          sycl::buffer<${type}, 1> inBuffer${type_as_str}${size}(inputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> outBuffer${type_as_str}${size}(outputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleInBuffer${type_as_str}${size}(swizzleInputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleOutBuffer${type_as_str}${size}(swizzleOutputData${type_as_str}${size}, sycl::range<1>(${size}));

          testQueue.submit([&](sycl::handler &cgh) {
            auto inPtr${type_as_str}${size} = inBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto outPtr${type_as_str}${size} = outBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            auto swizzleInPtr${type_as_str}${size} = swizzleInBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto swizzleOutPtr${type_as_str}${size} = swizzleOutBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            cgh.single_task<class ${kernelName}>([=]() {
              auto testVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});

              auto multiPtrIn${type_as_str}${size} = inPtr${type_as_str}${size}.get_multi_ptr<${decorated}>();
              sycl::global_ptr<const ${type}, ${decorated}> constMultiPtrIn${type_as_str}${size} = multiPtrIn${type_as_str}${size};
              auto multiPtrOut${type_as_str}${size} = outPtr${type_as_str}${size}.get_multi_ptr<${decorated}>();
              testVec${type_as_str}${size}.load(0, constMultiPtrIn${type_as_str}${size});
              testVec${type_as_str}${size}.store(0, multiPtrOut${type_as_str}${size});

              auto cleanVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});
              sycl::vec<${type}, ${size}> swizzledVec {cleanVec${type_as_str}${size}.template swizzle<${swizVals}>()};

              auto multiPtrInSwizzle${type_as_str}${size} = swizzleInPtr${type_as_str}${size}.get_multi_ptr<${decorated}>();
              sycl::global_ptr<const ${type}, ${decorated}> constMultiPtrInSwizzle${type_as_str}${size} = multiPtrInSwizzle${type_as_str}${size};
              auto multiPtrOutSwizzle${type_as_str}${size} = swizzleOutPtr${type_as_str}${size}.get_multi_ptr<${decorated}>();
              swizzledVec.load(0, constMultiPtrInSwizzle${type_as_str}${size});
              swizzledVec.store(0, multiPtrOutSwizzle${type_as_str}${size});
            });
          });

        }
        check_array_equality<${type}, ${size}>(log, inputData${type_as_str}${size}, outputData${type_as_str}${size});
        check_array_equality<${type}, ${size}>(log, swizzleInputData${type_as_str}${size}, swizzleOutputData${type_as_str}${size});

        testQueue.wait_and_throw();
    }
      """)

local_multi_ptr_load_store_test_template = Template(
    """
    {
        ${type} inputData${type_as_str}${size}[${size}] = {${in_order_vals}};
        ${type} outputData${type_as_str}${size}[${size}] = {${val}};
        ${type} swizzleInputData${type_as_str}${size}[${size}] = {${reverse_order_vals}};
        ${type} swizzleOutputData${type_as_str}${size}[${size}] = {${val}};
        {
          sycl::buffer<${type}, 1> inBuffer${type_as_str}${size}(inputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> outBuffer${type_as_str}${size}(outputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleInBuffer${type_as_str}${size}(swizzleInputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleOutBuffer${type_as_str}${size}(swizzleOutputData${type_as_str}${size}, sycl::range<1>(${size}));

          testQueue.submit([&](sycl::handler &cgh) {
            auto inPtr${type_as_str}${size} = inBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto outPtr${type_as_str}${size} = outBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            auto swizzleInPtr${type_as_str}${size} = swizzleInBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto swizzleOutPtr${type_as_str}${size} = swizzleOutBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            sycl::local_accessor<${type}> inLocalPtr${type_as_str}${size}(${size}, cgh);
            sycl::local_accessor<${type}> outLocalPtr${type_as_str}${size}(${size}, cgh);

            sycl::local_accessor<${type}> swizzleInLocalPtr${type_as_str}${size}(${size}, cgh);
            sycl::local_accessor<${type}> swizzleOutLocalPtr${type_as_str}${size}(${size}, cgh);

            cgh.parallel_for<class ${kernelName}>(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)), [=](sycl::nd_item<1>)  {
              for (unsigned i = 0; i < ${size}; ++i) {
                inLocalPtr${type_as_str}${size}[i] = inPtr${type_as_str}${size}[i];
                swizzleInLocalPtr${type_as_str}${size}[i] = swizzleInPtr${type_as_str}${size}[i];
              }

              auto testVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});

              auto rawPtrIn${type_as_str}${size} = inLocalPtr${type_as_str}${size}.get_multi_ptr<${decorated}>();
              sycl::local_ptr<const ${type}, ${decorated}> constRawPtrIn${type_as_str}${size} = rawPtrIn${type_as_str}${size};
              auto rawPtrOut${type_as_str}${size} = outLocalPtr${type_as_str}${size}.get_multi_ptr<${decorated}>();
              testVec${type_as_str}${size}.load(0, constRawPtrIn${type_as_str}${size});
              testVec${type_as_str}${size}.store(0, rawPtrOut${type_as_str}${size});

              auto cleanVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});
              sycl::vec<${type}, ${size}> swizzledVec {cleanVec${type_as_str}${size}.template swizzle<${swizVals}>()};

              auto rawPtrInSwizzle${type_as_str}${size} = swizzleInLocalPtr${type_as_str}${size}.get_multi_ptr<${decorated}>();
              sycl::local_ptr<const ${type}, ${decorated}> constRawPtrInSwizzle${type_as_str}${size} = rawPtrInSwizzle${type_as_str}${size};
              auto rawPtrOutSwizzle${type_as_str}${size} = swizzleOutLocalPtr${type_as_str}${size}.get_multi_ptr<${decorated}>();
              swizzledVec.load(0, constRawPtrInSwizzle${type_as_str}${size});
              swizzledVec.store(0, rawPtrOutSwizzle${type_as_str}${size});

              for (unsigned i = 0; i < ${size}; ++i) {
                outPtr${type_as_str}${size}[i] = outLocalPtr${type_as_str}${size}[i];
                swizzleOutPtr${type_as_str}${size}[i] = swizzleOutLocalPtr${type_as_str}${size}[i];
              }
            });
          });

        }
        check_array_equality<${type}, ${size}>(log, inputData${type_as_str}${size}, outputData${type_as_str}${size});
        check_array_equality<${type}, ${size}>(log, swizzleInputData${type_as_str}${size}, swizzleOutputData${type_as_str}${size});

        testQueue.wait_and_throw();
    }
      """)

private_multi_ptr_load_store_test_template = Template(
    """
    {
        ${type} inputData${type_as_str}${size}[${size}] = {${in_order_vals}};
        ${type} outputData${type_as_str}${size}[${size}] = {${val}};
        ${type} swizzleInputData${type_as_str}${size}[${size}] = {${reverse_order_vals}};
        ${type} swizzleOutputData${type_as_str}${size}[${size}] = {${val}};
        {
          sycl::buffer<${type}, 1> inBuffer${type_as_str}${size}(inputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> outBuffer${type_as_str}${size}(outputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleInBuffer${type_as_str}${size}(swizzleInputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleOutBuffer${type_as_str}${size}(swizzleOutputData${type_as_str}${size}, sycl::range<1>(${size}));

          testQueue.submit([&](sycl::handler &cgh) {
            auto inPtr${type_as_str}${size} = inBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto outPtr${type_as_str}${size} = outBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            auto swizzleInPtr${type_as_str}${size} = swizzleInBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto swizzleOutPtr${type_as_str}${size} = swizzleOutBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            sycl::local_accessor<${type}> inLocalPtr${type_as_str}${size}(${size}, cgh);
            sycl::local_accessor<${type}> outLocalPtr${type_as_str}${size}(${size}, cgh);

            sycl::local_accessor<${type}> swizzleInLocalPtr${type_as_str}${size}(${size}, cgh);
            sycl::local_accessor<${type}> swizzleOutLocalPtr${type_as_str}${size}(${size}, cgh);

            cgh.parallel_for<class ${kernelName}>(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)), [=](sycl::nd_item<1>) {
              ${type} inPrivatePtr${type_as_str}${size}[${size}];
              ${type} outPrivatePtr${type_as_str}${size}[${size}];

              ${type} swizzleInPrivatePtr${type_as_str}${size}[${size}];
              ${type} swizzleOutPrivatePtr${type_as_str}${size}[${size}];

              for (unsigned i = 0; i < ${size}; ++i) {
                inPrivatePtr${type_as_str}${size}[i] = inPtr${type_as_str}${size}[i];
                swizzleInPrivatePtr${type_as_str}${size}[i] = swizzleInPtr${type_as_str}${size}[i];
              }

              auto testVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});

              auto rawPtrIn${type_as_str}${size} = sycl::address_space_cast<sycl::access::address_space::private_space, ${decorated}>(inPrivatePtr${type_as_str}${size});
              sycl::private_ptr<const ${type}, ${decorated}> constRawPtrIn${type_as_str}${size} = rawPtrIn${type_as_str}${size};
              auto rawPtrOut${type_as_str}${size} = sycl::address_space_cast<sycl::access::address_space::private_space, ${decorated}>(outPrivatePtr${type_as_str}${size});
              testVec${type_as_str}${size}.load(0, constRawPtrIn${type_as_str}${size});
              testVec${type_as_str}${size}.store(0, rawPtrOut${type_as_str}${size});

              auto cleanVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});
              sycl::vec<${type}, ${size}> swizzledVec {cleanVec${type_as_str}${size}.template swizzle<${swizVals}>()};

              auto rawPtrInSwizzle${type_as_str}${size} = sycl::address_space_cast<sycl::access::address_space::private_space, ${decorated}>(swizzleInPrivatePtr${type_as_str}${size});
              sycl::private_ptr<const ${type}, ${decorated}> constRawPtrInSwizzle${type_as_str}${size} = rawPtrInSwizzle${type_as_str}${size};
              auto rawPtrOutSwizzle${type_as_str}${size} = sycl::address_space_cast<sycl::access::address_space::private_space, ${decorated}>(swizzleOutPrivatePtr${type_as_str}${size});
              swizzledVec.load(0, constRawPtrInSwizzle${type_as_str}${size});
              swizzledVec.store(0, rawPtrOutSwizzle${type_as_str}${size});

              for (unsigned i = 0; i < ${size}; ++i) {
                outPtr${type_as_str}${size}[i] = outPrivatePtr${type_as_str}${size}[i];
                swizzleOutPtr${type_as_str}${size}[i] = swizzleOutPrivatePtr${type_as_str}${size}[i];
              }
            });
          });

        }
        check_array_equality<${type}, ${size}>(log, inputData${type_as_str}${size}, outputData${type_as_str}${size});
        check_array_equality<${type}, ${size}>(log, swizzleInputData${type_as_str}${size}, swizzleOutputData${type_as_str}${size});

        testQueue.wait_and_throw();
    }
      """)

global_raw_ptr_load_store_test_template = Template(
    """
    {
        ${type} inputData${type_as_str}${size}[${size}] = {${in_order_vals}};
        ${type} outputData${type_as_str}${size}[${size}] = {${val}};
        ${type} swizzleInputData${type_as_str}${size}[${size}] = {${reverse_order_vals}};
        ${type} swizzleOutputData${type_as_str}${size}[${size}] = {${val}};
        {
          sycl::buffer<${type}, 1> inBuffer${type_as_str}${size}(inputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> outBuffer${type_as_str}${size}(outputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleInBuffer${type_as_str}${size}(swizzleInputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleOutBuffer${type_as_str}${size}(swizzleOutputData${type_as_str}${size}, sycl::range<1>(${size}));

          testQueue.submit([&](sycl::handler &cgh) {
            auto inPtr${type_as_str}${size} = inBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto outPtr${type_as_str}${size} = outBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            auto swizzleInPtr${type_as_str}${size} = swizzleInBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto swizzleOutPtr${type_as_str}${size} = swizzleOutBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            cgh.single_task<class ${kernelName}>([=]() {
              auto testVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});

              std::add_pointer_t<${type}> rawPtrIn${type_as_str}${size} = inPtr${type_as_str}${size}.get_multi_ptr<sycl::access::decorated::no>().get_raw();
              const std::add_pointer_t<${type}> constRawPtrIn${type_as_str}${size} = rawPtrIn${type_as_str}${size};
              std::add_pointer_t<${type}> rawPtrOut${type_as_str}${size} = outPtr${type_as_str}${size}.get_multi_ptr<sycl::access::decorated::no>().get_raw();
              testVec${type_as_str}${size}.load(0, constRawPtrIn${type_as_str}${size});
              testVec${type_as_str}${size}.store(0, rawPtrOut${type_as_str}${size});

              auto cleanVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});
              sycl::vec<${type}, ${size}> swizzledVec {cleanVec${type_as_str}${size}.template swizzle<${swizVals}>()};

              std::add_pointer_t<${type}> rawPtrInSwizzle${type_as_str}${size} = swizzleInPtr${type_as_str}${size}.get_multi_ptr<sycl::access::decorated::no>().get_raw();
              const std::add_pointer_t<${type}> constRawPtrInSwizzle${type_as_str}${size} = rawPtrInSwizzle${type_as_str}${size};
              std::add_pointer_t<${type}> rawPtrOutSwizzle${type_as_str}${size} = swizzleOutPtr${type_as_str}${size}.get_multi_ptr<sycl::access::decorated::no>().get_raw();
              swizzledVec.load(0, constRawPtrInSwizzle${type_as_str}${size});
              swizzledVec.store(0, rawPtrOutSwizzle${type_as_str}${size});
            });
          });

        }
        check_array_equality<${type}, ${size}>(log, inputData${type_as_str}${size}, outputData${type_as_str}${size});
        check_array_equality<${type}, ${size}>(log, swizzleInputData${type_as_str}${size}, swizzleOutputData${type_as_str}${size});

        testQueue.wait_and_throw();
    }
      """)

local_raw_ptr_load_store_test_template = Template(
    """
    {
        ${type} inputData${type_as_str}${size}[${size}] = {${in_order_vals}};
        ${type} outputData${type_as_str}${size}[${size}] = {${val}};
        ${type} swizzleInputData${type_as_str}${size}[${size}] = {${reverse_order_vals}};
        ${type} swizzleOutputData${type_as_str}${size}[${size}] = {${val}};
        {
          sycl::buffer<${type}, 1> inBuffer${type_as_str}${size}(inputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> outBuffer${type_as_str}${size}(outputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleInBuffer${type_as_str}${size}(swizzleInputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleOutBuffer${type_as_str}${size}(swizzleOutputData${type_as_str}${size}, sycl::range<1>(${size}));

          testQueue.submit([&](sycl::handler &cgh) {
            auto inPtr${type_as_str}${size} = inBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto outPtr${type_as_str}${size} = outBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            auto swizzleInPtr${type_as_str}${size} = swizzleInBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto swizzleOutPtr${type_as_str}${size} = swizzleOutBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            sycl::local_accessor<${type}> inLocalPtr${type_as_str}${size}(${size}, cgh);
            sycl::local_accessor<${type}> outLocalPtr${type_as_str}${size}(${size}, cgh);

            sycl::local_accessor<${type}> swizzleInLocalPtr${type_as_str}${size}(${size}, cgh);
            sycl::local_accessor<${type}> swizzleOutLocalPtr${type_as_str}${size}(${size}, cgh);

            cgh.parallel_for<class ${kernelName}>(sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)), [=](sycl::nd_item<1>) {
              for (unsigned i = 0; i < ${size}; ++i) {
                inLocalPtr${type_as_str}${size}[i] = inPtr${type_as_str}${size}[i];
                swizzleInLocalPtr${type_as_str}${size}[i] = swizzleInPtr${type_as_str}${size}[i];
              }

              auto testVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});

              std::add_pointer_t<${type}> rawPtrIn${type_as_str}${size} = inLocalPtr${type_as_str}${size}.get_multi_ptr<sycl::access::decorated::no>().get_raw();
              const std::add_pointer_t<${type}> constRawPtrIn${type_as_str}${size} = rawPtrIn${type_as_str}${size};
              std::add_pointer_t<${type}> rawPtrOut${type_as_str}${size} = outLocalPtr${type_as_str}${size}.get_multi_ptr<sycl::access::decorated::no>().get_raw();
              testVec${type_as_str}${size}.load(0, constRawPtrIn${type_as_str}${size});
              testVec${type_as_str}${size}.store(0, rawPtrOut${type_as_str}${size});

              auto cleanVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});
              sycl::vec<${type}, ${size}> swizzledVec {cleanVec${type_as_str}${size}.template swizzle<${swizVals}>()};

              std::add_pointer_t<${type}> rawPtrInSwizzle${type_as_str}${size} = swizzleInLocalPtr${type_as_str}${size}.get_multi_ptr<sycl::access::decorated::no>().get_raw();
              const std::add_pointer_t<${type}> constRawPtrInSwizzle${type_as_str}${size} = rawPtrInSwizzle${type_as_str}${size};
              std::add_pointer_t<${type}> rawPtrOutSwizzle${type_as_str}${size} = swizzleOutLocalPtr${type_as_str}${size}.get_multi_ptr<sycl::access::decorated::no>().get_raw();
              swizzledVec.load(0, constRawPtrInSwizzle${type_as_str}${size});
              swizzledVec.store(0, rawPtrOutSwizzle${type_as_str}${size});

              for (unsigned i = 0; i < ${size}; ++i) {
                outPtr${type_as_str}${size}[i] = outLocalPtr${type_as_str}${size}[i];
                swizzleOutPtr${type_as_str}${size}[i] = swizzleOutLocalPtr${type_as_str}${size}[i];
              }
            });
          });

        }
        check_array_equality<${type}, ${size}>(log, inputData${type_as_str}${size}, outputData${type_as_str}${size});
        check_array_equality<${type}, ${size}>(log, swizzleInputData${type_as_str}${size}, swizzleOutputData${type_as_str}${size});

        testQueue.wait_and_throw();
    }
      """)

private_raw_ptr_load_store_test_template = Template(
    """
    {
        ${type} inputData${type_as_str}${size}[${size}] = {${in_order_vals}};
        ${type} outputData${type_as_str}${size}[${size}] = {${val}};
        ${type} swizzleInputData${type_as_str}${size}[${size}] = {${reverse_order_vals}};
        ${type} swizzleOutputData${type_as_str}${size}[${size}] = {${val}};
        {
          sycl::buffer<${type}, 1> inBuffer${type_as_str}${size}(inputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> outBuffer${type_as_str}${size}(outputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleInBuffer${type_as_str}${size}(swizzleInputData${type_as_str}${size}, sycl::range<1>(${size}));
          sycl::buffer<${type}, 1> swizzleOutBuffer${type_as_str}${size}(swizzleOutputData${type_as_str}${size}, sycl::range<1>(${size}));

          testQueue.submit([&](sycl::handler &cgh) {
            auto inPtr${type_as_str}${size} = inBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto outPtr${type_as_str}${size} = outBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            auto swizzleInPtr${type_as_str}${size} = swizzleInBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);
            auto swizzleOutPtr${type_as_str}${size} = swizzleOutBuffer${type_as_str}${size}.get_access<sycl::access_mode::read_write>(cgh);

            cgh.single_task<class ${kernelName}>([=]() {
              ${type} inPrivatePtr${type_as_str}${size}[${size}];
              ${type} outPrivatePtr${type_as_str}${size}[${size}];

              ${type} swizzleInPrivatePtr${type_as_str}${size}[${size}];
              ${type} swizzleOutPrivatePtr${type_as_str}${size}[${size}];

              for (unsigned i = 0; i < ${size}; ++i) {
                inPrivatePtr${type_as_str}${size}[i] = inPtr${type_as_str}${size}[i];
                swizzleInPrivatePtr${type_as_str}${size}[i] = swizzleInPtr${type_as_str}${size}[i];
              }

              auto testVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});

              std::add_pointer_t<${type}> rawPtrIn${type_as_str}${size} = inPrivatePtr${type_as_str}${size};
              const std::add_pointer_t<${type}> constRawPtrIn${type_as_str}${size} = rawPtrIn${type_as_str}${size};
              std::add_pointer_t<${type}> rawPtrOut${type_as_str}${size} = outPrivatePtr${type_as_str}${size};
              testVec${type_as_str}${size}.load(0, constRawPtrIn${type_as_str}${size});
              testVec${type_as_str}${size}.store(0, rawPtrOut${type_as_str}${size});

              auto cleanVec${type_as_str}${size} = sycl::vec<${type}, ${size}>(${val});
              sycl::vec<${type}, ${size}> swizzledVec {cleanVec${type_as_str}${size}.template swizzle<${swizVals}>()};

              std::add_pointer_t<${type}> rawPtrInSwizzle${type_as_str}${size} = swizzleInPrivatePtr${type_as_str}${size};
              const std::add_pointer_t<${type}> constRawPtrInSwizzle${type_as_str}${size} = rawPtrInSwizzle${type_as_str}${size};
              std::add_pointer_t<${type}> rawPtrOutSwizzle${type_as_str}${size} = swizzleOutPrivatePtr${type_as_str}${size};
              swizzledVec.load(0, constRawPtrInSwizzle${type_as_str}${size});
              swizzledVec.store(0, rawPtrOutSwizzle${type_as_str}${size});

              for (unsigned i = 0; i < ${size}; ++i) {
                outPtr${type_as_str}${size}[i] = outPrivatePtr${type_as_str}${size}[i];
                swizzleOutPtr${type_as_str}${size}[i] = swizzleOutPrivatePtr${type_as_str}${size}[i];
              }
            });
          });

        }
        check_array_equality<${type}, ${size}>(log, inputData${type_as_str}${size}, outputData${type_as_str}${size});
        check_array_equality<${type}, ${size}>(log, swizzleInputData${type_as_str}${size}, swizzleOutputData${type_as_str}${size});

        testQueue.wait_and_throw();
    }
      """)


def gen_kernel_name(type_str, size, decorated):
    return 'KERNEL_load_store_' + remove_namespaces_whitespaces(type_str) + str(size) + decorated

def wrap_with_deprecated(test_string):
    string = '#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS \n'
    string += test_string
    string += '#endif \n'
    return string


def gen_load_store_test(type_str, size):
    no_whitespace_type_str = remove_namespaces_whitespaces(type_str)
    test_string = ''
    for decoration in ['yes', 'no', 'legacy']:
        multi_ptr_test_string = global_multi_ptr_load_store_test_template.substitute(
            type=type_str,
            type_as_str=no_whitespace_type_str,
            size=size,
            val=Data.value_default_dict[type_str],
            in_order_vals=', '.join(
                make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size])),
            reverse_order_vals=', '.join(
                make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size][::-1])),
            kernelName=gen_kernel_name(type_str, size, decoration + '_global'),
            swizVals=', '.join(Data.swizzle_elem_list_dict[size]),
            decorated=('sycl::access::decorated::' + decoration))
        multi_ptr_test_string += local_multi_ptr_load_store_test_template.substitute(
            type=type_str,
            type_as_str=no_whitespace_type_str,
            size=size,
            val=Data.value_default_dict[type_str],
            in_order_vals=', '.join(
                make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size])),
            reverse_order_vals=', '.join(
                make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size][::-1])),
            kernelName=gen_kernel_name(type_str, size, decoration + '_local'),
            swizVals=', '.join(Data.swizzle_elem_list_dict[size]),
            decorated=('sycl::access::decorated::' + decoration))
        multi_ptr_test_string += private_multi_ptr_load_store_test_template.substitute(
            type=type_str,
            type_as_str=no_whitespace_type_str,
            size=size,
            val=Data.value_default_dict[type_str],
            in_order_vals=', '.join(
                make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size])),
            reverse_order_vals=', '.join(
                make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size][::-1])),
            kernelName=gen_kernel_name(type_str, size, decoration + '_private'),
            swizVals=', '.join(Data.swizzle_elem_list_dict[size]),
            decorated=('sycl::access::decorated::' + decoration))
        if decoration == 'legacy':
            multi_ptr_test_string = wrap_with_deprecated(multi_ptr_test_string)
        test_string += multi_ptr_test_string
    test_string += global_raw_ptr_load_store_test_template.substitute(
        type=type_str,
        type_as_str=no_whitespace_type_str,
        size=size,
        val=Data.value_default_dict[type_str],
        in_order_vals=', '.join(
            make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size])),
        reverse_order_vals=', '.join(
            make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size][::-1])),
        kernelName=gen_kernel_name(type_str, size, 'raw_global'),
        swizVals=', '.join(Data.swizzle_elem_list_dict[size]))
    test_string += local_raw_ptr_load_store_test_template.substitute(
        type=type_str,
        type_as_str=no_whitespace_type_str,
        size=size,
        val=Data.value_default_dict[type_str],
        in_order_vals=', '.join(
            make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size])),
        reverse_order_vals=', '.join(
            make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size][::-1])),
        kernelName=gen_kernel_name(type_str, size, 'raw_local'),
        swizVals=', '.join(Data.swizzle_elem_list_dict[size]))
    test_string += private_raw_ptr_load_store_test_template.substitute(
        type=type_str,
        type_as_str=no_whitespace_type_str,
        size=size,
        val=Data.value_default_dict[type_str],
        in_order_vals=', '.join(
            make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size])),
        reverse_order_vals=', '.join(
            make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size][::-1])),
        kernelName=gen_kernel_name(type_str, size, 'raw_private'),
        swizVals=', '.join(Data.swizzle_elem_list_dict[size]))
    return wrap_with_test_func(TEST_NAME, type_str,
                               wrap_with_extension_checks(
                                   type_str, test_string), str(size))


def make_tests(type_str, input_file, output_file):
    if type_str == 'bool':
        Data.vals_list_dict = cast_to_bool(Data.vals_list_dict)

    test_string = ''
    func_calls = ''
    for size in Data.standard_sizes:
        test_string += gen_load_store_test(type_str, size)
        func_calls += make_func_call(TEST_NAME, type_str, str(size))
    write_source_file(test_string, func_calls, TEST_NAME, input_file,
                      output_file, type_str)

def main():
    argparser = argparse.ArgumentParser(
        description='Generates vector swizzles opencl test')
    argparser.add_argument(
        'template',
        metavar='<code template path>',
        help='Path to code template')
    argparser.add_argument(
        '-type',
        dest='ty',
        required=True,
        choices=get_types(),
        help='Type to generate the test for')
    argparser.add_argument(
        '-o',
        required=True,
        dest="output",
        metavar='<out file>',
        help='CTS test output')
    args = argparser.parse_args()

    make_tests(args.ty, args.template, args.output)


if __name__ == '__main__':
    main()
