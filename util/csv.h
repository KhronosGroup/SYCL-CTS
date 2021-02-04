/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
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