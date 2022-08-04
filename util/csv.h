/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
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

#ifndef __SYCLCTS_UTIL_CSV_H
#define __SYCLCTS_UTIL_CSV_H

#include "stl.h"

namespace sycl_cts {
namespace util {

/** category, testname, timeout, regression y/n
 */

/** comma separated values file parser
 */
class csv {
 public:
  /** constructor destructor
   */
  csv();
  ~csv();

  /** load a CSV file from disk
   */
  bool load_file(const std::string &path);

  /** release all stored csv information
   */
  void release();

  /** return the number of csv rows (lines)
   */
  int32_t size();

  /** return the last error message set
   */
  bool get_last_error(std::string &out);

  /** extract a csv value
   */
  bool get_item(int32_t row, int32_t column, std::string &out);

 protected:
  // the last error message set
  std::string m_error;

  // raw items from the csv file
  std::vector<std::string> m_items;

  // indices for the start of a row
  std::vector<int> m_rowIndex;
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_CSV_H