/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_CMDARG_H
#define __SYCLCTS_UTIL_CMDARG_H

#include "stl.h"
#include "singleton.h"

namespace sycl_cts {
namespace util {

/** command line parser
 */
class cmdarg : public singleton<cmdarg> {
public:
  /** parse a set of given command line arguments
   *  @return, false if there was an error parsing
   *           true if the cmd line was parsed
   */
  bool parse(const int argc, const char **args);

  /** search for a specific key
   */
  bool find_key(const std::string &key) const;

  /** find a value from a given key
   *  @param, key, the key to try and locate
   *  @param, value, string to receive the value that was associated
   *                 with the given key
   */
  bool get_value(const std::string &key, std::string &value) const;

  /** return the last error message given
   *  @param, string to receive the last error message
   */
  bool get_last_error(std::string &out) const;

protected:
  /** a simple key value pair container
   */
  struct pair {
    std::string key;
    std::string value;
  };

  /** the options list
   */
  std::vector<pair> m_pairs;

  /** add a pair to the list
   */
  void push_pair(const pair &opt);

  // the last error message
  std::string m_error;
};

} // namespace util
} // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_CMDARG_H