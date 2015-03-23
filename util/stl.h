/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

// std::vector
#include <vector>

// std::string
#include <string>

// std::mutex
// std::lock_guard
#include <mutex>

// assert()
#include <assert.h>

// std::unique_ptr
#include <memory>

// cout
#include <iostream>
#include <fstream>

// std::sort()
#include <algorithm>

// std::atomic_uint
#include <atomic>

// memcpy
#include <cstring>

namespace sycl_cts {
namespace util {

/** alias to STRING
 */
using STRING = std::string;

/** alias to std::vector
 */
template <class T>
using VECTOR = std::vector<T>;

/** alias to std::mutex
 */
using MUTEX = std::mutex;

/** alias to std::lock_guard
 */
template <class T>
using LOCK_GUARD = std::lock_guard<std::mutex>;

/** alias to std::unique_ptr
 */
template <class T>
using UNIQUE_PTR = std::unique_ptr<T>;

/** alias to std::ifstream
 */
using IFSTREAM = std::ifstream;

/** std atomic
 */
using ATOMIC_INT = std::atomic_int;

}  // util
}  // sycl_cts
