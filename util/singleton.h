/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
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

#ifndef __SYCLCTS_UTIL_SINGLETON_H
#define __SYCLCTS_UTIL_SINGLETON_H

#include "stl.h"

namespace sycl_cts {
namespace util {

/** implement a singleton interface to all derived class.
 *  template argument T must be the derived class.
 */
template <class T>
class singleton {
  // must be a friend of class T to access the constructor
  friend T;

  // singleton instance
  static std::unique_ptr<T> m_instance;

 public:
  /** destructor
   *  ensure that we release the singleton instance
   */
  virtual ~singleton() { release(); }

  /** get singleton instance
   */
  static T &instance() {
    // if the instance has not be created
    if (m_instance.get() == nullptr) {
      m_instance.reset(new T());
    }
    assert(m_instance.get() != nullptr);

    // return the singleton instance
    return *(m_instance.get());
  }

  /** release this instance
   */
  static void release() { m_instance.release(); }
};

/** instance of the singleton
 */
template <class T>
std::unique_ptr<T> singleton<T>::m_instance;

/** easy singleton accessors
 */
template <class T>
static inline T &get() {
  return T::instance();
}

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_SINGLETON_H