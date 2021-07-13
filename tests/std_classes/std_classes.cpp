/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME std_classes

namespace std_classes__ {
using namespace sycl_cts;

/** check std::vector
*/
template <class T, class Alloc>
using vectorClass = std::vector<T, Alloc>;

/** check std::string
*/
using stringClass = std::string;

/** check std::function
*/
template <class R, class... Args>
using functionClass = std::function<R(Args...)>;

/** check std::mutex
*/
using mutexClass = std::mutex;

/** check std::unique_ptr
*/
template <class T, class D>
using uniquePtrClass = std::unique_ptr<T, D>;

/** check std::shared_ptr
*/
template <class T>
using sharedPtrClass = std::shared_ptr<T>;

/** check std::weak_ptr
*/
template <class T>
using weakPtrClass = std::weak_ptr<T>;

/** check std::hash
*/
template <class T>
using hashClass = std::hash<T>;

/** check std::exception_ptr
*/
using exceptionPtrClass = std::exception_ptr;

struct custom_deleter {
  void operator()(int *p) const {};
};

/** tests the availability of std classes
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
  */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
  */
  void run(util::logger &log) override {
    /* Try instantiating these classes */
    {
      /** check std::vector
      */
      std::vector<int> vector;

      /** check std::string
      */
      stringClass string;

      /** check std::function
      */
      functionClass<void> function;

      /** check std::mutex
      */
      mutexClass mutex;

      /** check std::unique_ptr
      */
      uniquePtrClass<int, custom_deleter> uniquePtr;

      /** check std::shared_ptr
      */
      sharedPtrClass<int> sharedPtr;

      /** check std::weak_ptr
      */
      weakPtrClass<int> weakPtr;

      /** check std::hash
      */
      hashClass<int> hash;

      /** check std::exception_ptr
      */
      exceptionPtrClass exceptionPtr;
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace std_classes__ */
