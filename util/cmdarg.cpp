/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "cmdarg.h"
#include "singleton.h"

namespace sycl_cts {
namespace util {

/** add a pair to the list of pairs
 */
void cmdarg::push_pair(const pair &opt) {
  // the key must not be empty
  assert(!opt.key.empty());

  m_pairs.push_back(opt);
}

/** parse a set of given command line arguments
 *  @return, false if there was an error parsing
 *           true if the cmd line was parsed
 */
bool cmdarg::parse(const int argc, const char **args) {
  // args[0] is speficied for the executable name, so we must skip it
  // thus here we treat args as base 1.

  // early exit if no argument are present
  if (argc <= 1) return true;

  // check the pointer is valid
  assert(args != nullptr);

  pair kvp;

  // loop over all of the given arguments
  for (int i = 1; i < argc; i++) {
    // isolate the argument we are currently looking at
    const char *arg = args[i];

    // '-' marks a key
    if (arg[0] == '-') {
      // if we previously had a key then we must push it
      // with or without a value
      if (!kvp.key.empty()) {
        // push the previous option onto the list
        push_pair(kvp);
        // erase the option
        kvp.key.clear();
        kvp.value.clear();
      }

      // set the new key
      kvp.key = arg;
    } else {
      // check if we have a key for this value
      if (kvp.key.empty()) {
        m_error = (std::string("expecting key before '") + arg) + "'";
        return false;
      }

      // check if we are overwriting a value
      if (!kvp.value.empty()) {
        m_error = (std::string("duplicate argument to '") + kvp.key) + "'";
        return false;
      }

      // set the current value
      kvp.value = arg;
    }
  }

  // push any remaining key values
  push_pair(kvp);

  return true;
}

/** return the last error message set
 */
bool cmdarg::get_last_error(std::string &out) const {
  out = m_error;
  return !m_error.empty();
}

/** return the last error message given
 *  @param, string to receive the last error message
 */
bool cmdarg::find_key(const std::string &key) const {
  // number of pairs that have been parsed
  int32_t nPairs = int32_t(m_pairs.size());

  for (int32_t i = 0; i < nPairs; i++) {
    const pair &kvp = m_pairs.at(size_t(i));

    // check if we can match the key
    if (kvp.key == key) return true;
  }

  // key could not be matched
  return false;
}

/** find a value from a given key
 *  @param, key, the key to try and locate
 *  @param, value, string to receive the value that was associated
 *                 with the given key
 */
bool cmdarg::get_value(const std::string &key, std::string &value) const {
  // number of pairs that have been parsed
  int32_t nPairs = int32_t(m_pairs.size());

  for (int32_t i = 0; i < nPairs; i++) {
    const pair &kvp = m_pairs.at(size_t(i));

    // check if we can match the key
    if (kvp.key == key) {
      // return the value via its reference
      value = kvp.value;
      return true;
    }
  }

  // key could not be matched
  return false;
}

}  // namespace util
}  // namespace sycl_cts
