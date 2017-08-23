/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME stream_api

namespace stream_api__ {
using namespace sycl_cts;

/**
 * Function that streams a type using the cl::sycl::stream object.
 */
template <typename T>
void stream_type(cl::sycl::stream& os, T var) {
  os << var;
}

/** test cl::sycl::stream interface
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger& log) override {
    try {
      cts_selector testSelector;

      cl::sycl::queue testQueue(testSelector);
      testQueue.submit([&](cl::sycl::handler& cgh) {

        cl::sycl::stream os(2048, 80);

        auto size = os.get_size();

        if (typeid(size) != typeid(size_t)) {
          FAIL(log,
               "cl::sycl::context::get_size() does not "
               "return size_t");
        }

        auto maxStatementSize = os.get_max_statement_size();

        if (typeid(maxStatementSize) != typeid(size_t)) {
          FAIL(log,
               "cl::sycl::context::get_max_statement_size() does not "
               "return size_t");
        }
      });

      testQueue.submit([&](cl::sycl::handler& cgh) {

        cl::sycl::stream os;

        cgh.single_task<class TEST_NAME>([=]() {
          /** check stream operator for basic types
          */
          os << cl::sycl::string_class("hello world!"));
          os << char('c');
          os << unsigned char('c');
          os << int(5);
          os << unsigned int(5);
          os << short(5);
          os << unsigned short(5);
          os << long(5);
          os << unsigned long(5);
          os << float(5.5f);
          os << double(5.5);
          os << size_t(5);
          os << true;

          /** check stream operator for sycl types
          */
          os << id<3>(1, 2, 3);
          os << range<3>(1, 2, 3);

          /** check stream operator for sycl vec types
          */
          cl::sycl::vec<float, 4> f4(1.0f, 2.0f, 3.0f, 4.0f);
          os << f4;
          os << f4.wzyx();

          /** check stream operator for manipulators
          */
          os << cl::sycl::endl;
          os << cl::sycl::precision << float(5.0f);
          os << cl::sycl::scientific << float(5.0f);
          os << cl::sycl::hex << float(5.0f);
          os << cl::sycl::oct << float(5.0f);
          os << cl::sycl::showbase << int(5);
          os << cl::sycl::showpos << int(-5) << int(5);
        });
      });

      testQueue.submit([&](cl::sycl::handler& cgh) {

        cgh.parallel_for<class TEST_NAME>(nd_range<3>(range<3>(16, 8, 4), range<3>(8, 4, 2), [=](nd_item ndItem) {
          /** check stream operator for nd_item
          */
          os << ndItem;
        });
      });

      testQueue.submit([&](cl::sycl::handler& cgh) {

        cgh.parallel_for<class TEST_NAME>(range<3>(16, 8, 4), [=](item it) {
          /** check stream operator for item
          */
          os << it;
        });
      });

      testQueue.submit([&](cl::sycl::handler& cgh) {

        cgh.parallel_for_work_group<class TEST_NAME>(range<3>(16, 8, 4),
                                                     [=](group gp) {
                                                       /** check stream operator
                                                        * for group
                                                       */
                                                       os << gp;
                                                     });
      });
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace stream_api__ */
