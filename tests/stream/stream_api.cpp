/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#define SYCL_SIMPLE_SWIZZLES

#include "../stream/stream_api_common.h"

#define TEST_NAME stream_api_core

namespace TEST_NAMESPACE {

using namespace sycl_cts;

class test_kernel_0;
class test_kernel_1;

template <int dims>
class test_kernel_2;

template <int dims>
class test_kernel_3;

template <int dims>
class test_kernel_4;

/**
 * Function that create a cl::sycl::stream object and streams nd_item.
 */
template <int dims>
void check_nd_item_dims(cl::sycl::range<dims> &range1, cl::sycl::range<dims> &range2) {
  auto testQueue = util::get_cts_object::queue();
  testQueue.submit([&](cl::sycl::handler &cgh) {

    cl::sycl::stream os(2048, 80, cgh);

    cgh.parallel_for<class test_kernel_2<dims>>(
        cl::sycl::nd_range<dims>(range1, range2),
        [=](cl::sycl::nd_item<dims> ndItem) {
          /** check stream operator for nd_item
          */
          check_type(os, ndItem);

          // check stream operator for nd_range
          check_type(os, ndItem.get_nd_range());
        });
  });

  testQueue.wait_and_throw();
}

/**
 * Function that create a cl::sycl::stream object and streams item.
 */
template <int dims>
void check_item_dims(cl::sycl::range<dims> &range){
  auto testQueue = util::get_cts_object::queue();
  testQueue.submit([&](cl::sycl::handler &cgh) {

    cl::sycl::stream os(2048, 80, cgh);

    cgh.parallel_for<class test_kernel_3<dims>>(range,
                                          [=](cl::sycl::item<dims> it) {
                                            /** check stream operator for
                                              * item
                                            */
                                            check_type(os, it);
                                          });
  });

  testQueue.wait_and_throw();
}

/**
 * Function that create a cl::sycl::stream object and streams group and h_item.
 */
template <int dims>
void check_group_h_item_dims(cl::sycl::range<dims> &range1, cl::sycl::range<dims> &range2) {
  auto testQueue = util::get_cts_object::queue();
  testQueue.submit([&](cl::sycl::handler &cgh) {

    cl::sycl::stream os(2048, 80, cgh);

    cgh.parallel_for_work_group<class test_kernel_4<dims>>(range1, range2,
        [=](const cl::sycl::group<dims> gp) {
          /** check stream operator for cl::sycl::group
          */
          check_type(os, gp);

          gp.parallel_for_work_item([&](cl::sycl::h_item<dims> hit) {
            /** check stream operator for cl::sycl::h_item
            */
            check_type(os, hit);
          });
        });
  });

  testQueue.wait_and_throw();
}

/** test cl::sycl::stream interface
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      /** check cl::sycl::stream_manipulator
      */
      check_enum_class_value(cl::sycl::stream_manipulator::dec);
      check_enum_class_value(cl::sycl::stream_manipulator::hex);
      check_enum_class_value(cl::sycl::stream_manipulator::oct);
      check_enum_class_value(cl::sycl::stream_manipulator::noshowbase);
      check_enum_class_value(cl::sycl::stream_manipulator::showbase);
      check_enum_class_value(cl::sycl::stream_manipulator::noshowpos);
      check_enum_class_value(cl::sycl::stream_manipulator::showpos);
      check_enum_class_value(cl::sycl::stream_manipulator::endl);
      check_enum_class_value(cl::sycl::stream_manipulator::fixed);
      check_enum_class_value(cl::sycl::stream_manipulator::scientific);
      check_enum_class_value(cl::sycl::stream_manipulator::hexfloat);
      check_enum_class_value(cl::sycl::stream_manipulator::defaultfloat);
      check_enum_class_value(cl::sycl::stream_manipulator::flush);

      /** Check stream interface
      */
      {
        auto testQueue = util::get_cts_object::queue();
        testQueue.submit([&](cl::sycl::handler &cgh) {

          cl::sycl::stream os(2048, 80, cgh);

          /** check get_size()
          */
          {
            auto size = os.get_size();
            check_return_type<size_t>(log, size,
                                      "cl::sycl::context::get_size()");
          }

          /** check get_max_statement_size()
          */
          {
            auto maxStatementSize = os.get_max_statement_size();
            check_return_type<size_t>(
                log, maxStatementSize,
                "cl::sycl::context::get_max_statement_size()");
          }

          cgh.single_task<class test_kernel_0>([=]() {});
        });

        testQueue.wait_and_throw();
      }

      /** check stream operator for supported types
      */
      {
        auto testQueue = util::get_cts_object::queue();
        testQueue.submit([&](cl::sycl::handler &cgh) {

          cl::sycl::stream os(2048, 80, cgh);

          cgh.single_task<class test_kernel_1>([=]() {

            /** check stream operator for basic types
            */
            check_type(os, "hello world!");
            check_type(os, const_cast<char *>("hello world!"));
            check_all_vec_dims(os, char('c'));
            check_all_vec_dims(os, static_cast<signed char>('c'));
            check_all_vec_dims(os, static_cast<unsigned char>('c'));
            check_all_vec_dims(os, int(5));
            check_all_vec_dims(os, static_cast<unsigned int>(5));
            check_all_vec_dims(os, short(5));
            check_all_vec_dims(os, static_cast<unsigned short>(5));
            check_all_vec_dims(os, long(5));
            check_all_vec_dims(os, static_cast<unsigned long>(5));
            check_all_vec_dims(os, static_cast<long long>(5));
            check_all_vec_dims(os, static_cast<unsigned long long>(5));
            check_all_vec_dims(os, float(5.5f));
            check_type(os, true);
            check_type(os, size_t(5));

            // check stream operator for pointers
            int a = 5;
            int *aPtr = &a;
            check_type(os, aPtr);
            const int * aConstPtr = &a;
            check_type(os, aConstPtr);
            auto multiPtr = cl::sycl::private_ptr<int>(aPtr);
            check_type(os, multiPtr);

            /** check stream operator for cl types
            */
            check_all_vec_dims(os, cl_char('c'));
            check_all_vec_dims(os, static_cast<cl_uchar>('c'));
            check_all_vec_dims(os, cl_int(5));
            check_all_vec_dims(os, static_cast<cl_uint>(5));
            check_all_vec_dims(os, cl_short(5));
            check_all_vec_dims(os, static_cast<cl_ushort>(5));
            check_all_vec_dims(os, cl_long(5));
            check_all_vec_dims(os, static_cast<cl_ulong>(5));
            check_all_vec_dims(os, cl_float(5.5f));
            check_type(os, static_cast<cl_bool>(true));

            /** check stream operator for sycl types
            */
            check_all_vec_dims(os, cl::sycl::byte(72));

            check_type(os, cl::sycl::id<1>(1));
            check_type(os, cl::sycl::id<2>(1, 2));
            check_type(os, cl::sycl::id<3>(1, 2, 3));

            check_type(os, cl::sycl::range<1>(1));
            check_type(os, cl::sycl::range<2>(1, 2));
            check_type(os, cl::sycl::range<3>(1, 2, 3));

            check_type(os, cl::sycl::nd_range<1>(cl::sycl::range<1>(2),
                                        cl::sycl::range<1>(1)));
            check_type(os, cl::sycl::nd_range<2>(cl::sycl::range<2>(2, 4),
                                        cl::sycl::range<2>(1, 2)));
            check_type(os, cl::sycl::nd_range<3>(cl::sycl::range<3>(2, 4, 1),
                                        cl::sycl::range<3>(1, 2, 1)));

            /** check stream operator for manipulators
            */
            os << cl::sycl::endl;
            os << cl::sycl::setprecision(5) << float(5.0f);
            os << cl::sycl::setw(3) << float(5.0f);
            os << cl::sycl::hex << float(5.0f);
            os << cl::sycl::oct << float(5.0f);
            os << cl::sycl::dec << float(5.0f);
            os << cl::sycl::showbase << int(5);
            os << cl::sycl::noshowbase << int(5);
            os << cl::sycl::showpos << int(-5) << int(5);
            os << cl::sycl::noshowpos << int(-5) << int(5);
            os << cl::sycl::fixed << float(5.0f);
            os << cl::sycl::scientific << float(5.0f);
            os << cl::sycl::hexfloat << float(5.0f);
            os << cl::sycl::defaultfloat << float(5.0f);
            os << cl::sycl::flush;
          });
        });

        testQueue.wait_and_throw();
      }

      /** check stream operator for cl::sycl::nd_item
      */
      {
        cl::sycl::range<1> r11(2);
        cl::sycl::range<1> r12(1);
        check_nd_item_dims(r11, r12);

        cl::sycl::range<2> r21(2, 4);
        cl::sycl::range<2> r22(1, 2);
        check_nd_item_dims(r21, r22);

        cl::sycl::range<3> r31(2, 4, 1);
        cl::sycl::range<3> r32(1, 2, 1);
        check_nd_item_dims(r31, r32);

      }

      /** check stream operator for cl::sycl::item
      */
      {
        cl::sycl::range<1> r1(4);
        check_item_dims(r1);

        cl::sycl::range<2> r2(4, 2);
        check_item_dims(r2);

        cl::sycl::range<3> r3(4, 2, 1);
        check_item_dims(r3);
      }

      /** check stream operator for cl::sycl::group and cl::sycl::h_item
      */
      {
        cl::sycl::range<1> r11(4);
        cl::sycl::range<1> r12(1);
        check_group_h_item_dims(r11, r12);

        cl::sycl::range<2> r21(4, 2);
        cl::sycl::range<2> r22(1, 1);
        check_group_h_item_dims(r21, r22);

        cl::sycl::range<3> r31(4, 2, 1);
        cl::sycl::range<3> r32(1, 1, 1);
        check_group_h_item_dims(r31, r32);

      }
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
