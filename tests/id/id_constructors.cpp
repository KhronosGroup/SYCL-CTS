/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/common_by_value.h"

#define TEST_NAME id_constructors

namespace TEST_NAMESPACE {
using namespace sycl_cts;

class id_it1;
class id_it2;
class id_it3;

/** test sycl::id initialization
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
      // check default constructors

      // dim 1
      {
        sycl::id<1> id;
        if ((id[0] != 0) || (id.get(0) != 0)) {
          FAIL(log, "default constructed id failed for dim = 1");
        }
      }

      // dim 2
      {
        sycl::id<2> id;
        if ((id[0] != 0) || (id.get(0) != 0) || (id[1] != 0) ||
            (id.get(1) != 0)) {
          FAIL(log, "default constructed id failed for dim = 2");
        }
      }

      // dim 3
      {
        sycl::id<3> id;
        if ((id[0] != 0) || (id.get(0) != 0) || (id[1] != 0) ||
            (id.get(1) != 0) || (id[2] != 0) || (id.get(2) != 0)) {
          FAIL(log, "default constructed id failed for dim = 3");
        }
      }

      // use across all the dimensions
      size_t sizes[] = {16, 8, 4};

      // construct from a range, explicit dimensions perform deep copy and a
      // move

      // dim 1
      {
        sycl::id<1> id_explicit(sizes[0]);
        if ((id_explicit[0] != sizes[0]) || (id_explicit.get(0) != sizes[0])) {
          FAIL(log, "id with size_t was not constructed correctly for dim = 1");
        }

        sycl::range<1> range(sizes[0]);
        sycl::id<1> id(range);
        if ((id[0] != sizes[0]) || (id.get(0) != sizes[0])) {
          FAIL(log, "id with range was not constructed correctly for dim = 1");
        }

        sycl::id<1> id_deep(id);
        if ((id_deep[0] != sizes[0]) || (id_deep.get(0) != sizes[0])) {
          FAIL(log, "id with id was not constructed correctly for dim = 1");
        }

        sycl::id<1> id_moved_constr(std::move(id));
        if ((id_moved_constr[0] != sizes[0]) ||
            (id_moved_constr.get(0) != sizes[0])) {
          FAIL(log,
               "id with id was not move constructed correctly for dim = 1");
        }

        sycl::id<1> id_move_assign;
        id_move_assign = std::move(id_deep);
        if ((id_move_assign[0] != sizes[0]) ||
            (id_move_assign.get(0) != sizes[0])) {
          FAIL(log, "id with id was not move assigned correctly for dim = 1");
        }

        check_equality_comparable_generic(log, id_explicit, std::string("id"));
      }

      // dim 2
      {
        sycl::id<2> id_explicit(sizes[0], sizes[1]);
        if ((id_explicit[0] != sizes[0]) || (id_explicit.get(0) != sizes[0]) ||
            (id_explicit[1] != sizes[1]) || (id_explicit.get(1) != sizes[1])) {
          FAIL(log, "id with size_t was not constructed correctly for dim = 2");
        }

        sycl::range<2> range(sizes[0], sizes[1]);
        sycl::id<2> id(range);
        if ((id[0] != sizes[0]) || (id.get(0) != sizes[0]) ||
            (id[1] != sizes[1]) || (id.get(1) != sizes[1])) {
          FAIL(log, "id with range was not constructed correctly for dim = 2");
        }

        sycl::id<2> id_deep(id);
        if ((id_deep[0] != sizes[0]) || (id_deep.get(0) != sizes[0]) ||
            (id_deep[1] != sizes[1]) || (id_deep.get(1) != sizes[1])) {
          FAIL(log, "id with id was not constructed correctly for dim = 2");
        }

        sycl::id<2> id_moved_constr(std::move(id));
        if ((id_moved_constr[0] != sizes[0]) ||
            (id_moved_constr.get(0) != sizes[0]) ||
            (id_moved_constr[1] != sizes[1]) ||
            (id_moved_constr.get(1) != sizes[1])) {
          FAIL(log,
               "id with id was not move constructed correctly for dim = 2");
        }

        sycl::id<2> id_move_assign;
        id_move_assign = std::move(id_deep);
        if ((id_move_assign[0] != sizes[0]) ||
            (id_move_assign.get(0) != sizes[0]) ||
            (id_move_assign[1] != sizes[1]) ||
            (id_move_assign.get(1) != sizes[1])) {
          FAIL(log, "id with id was not move assigned correctly for dim = 2");
        }

        check_equality_comparable_generic(log, id_explicit, std::string("id"));
      }

      // dim 3
      {
        sycl::id<3> id_explicit(sizes[0], sizes[1], sizes[2]);
        if ((id_explicit[0] != sizes[0]) || (id_explicit.get(0) != sizes[0]) ||
            (id_explicit[1] != sizes[1]) || (id_explicit.get(1) != sizes[1]) ||
            (id_explicit[2] != sizes[2]) || (id_explicit.get(2) != sizes[2])) {
          FAIL(log, "id with size_t was not constructed correctly for dim = 3");
        }

        sycl::range<3> range(sizes[0], sizes[1], sizes[2]);
        sycl::id<3> id(range);
        if ((id[0] != sizes[0]) || (id.get(0) != sizes[0]) ||
            (id[1] != sizes[1]) || (id.get(1) != sizes[1]) ||
            (id[2] != sizes[2]) || (id.get(2) != sizes[2])) {
          FAIL(log, "id with range was not constructed correctly for dim = 3");
        }

        sycl::id<3> id_deep(id);
        if ((id_deep[0] != sizes[0]) || (id_deep.get(0) != sizes[0]) ||
            (id_deep[1] != sizes[1]) || (id_deep.get(1) != sizes[1]) ||
            (id_deep[2] != sizes[2]) || (id_deep.get(2) != sizes[2])) {
          FAIL(log, "id with id was not constructed correctly for dim = 3");
        }

        sycl::id<3> id_moved_constr(std::move(id));
        if ((id_moved_constr[0] != sizes[0]) ||
            (id_moved_constr.get(0) != sizes[0]) ||
            (id_moved_constr[1] != sizes[1]) ||
            (id_moved_constr.get(1) != sizes[1]) ||
            (id_moved_constr[2] != sizes[2]) ||
            (id_moved_constr.get(2) != sizes[2])) {
          FAIL(log,
               "id with id was not move constructed correctly for dim = 3");
        }

        sycl::id<3> id_move_assign;
        id_move_assign = std::move(id_deep);
        if ((id_move_assign[0] != sizes[0]) ||
            (id_move_assign.get(0) != sizes[0]) ||
            (id_move_assign[1] != sizes[1]) ||
            (id_move_assign.get(1) != sizes[1]) ||
            (id_move_assign[2] != sizes[2]) ||
            (id_move_assign.get(2) != sizes[2])) {
          FAIL(log, "id with id was not move assigned correctly for dim = 3");
        }

        check_equality_comparable_generic(log, id_explicit, std::string("id"));
      }

      // construct from an item

      // dim 1
      {
        auto q = util::get_cts_object::queue();
        bool success = true;
        {
          sycl::buffer<bool, 1> b(&success, sycl::range<1>(1));
          q.submit([&](sycl::handler &cgh) {
            auto hasSucceded =
                b.get_access<sycl::access_mode::read_write,
                             sycl::target::device>(cgh);

            auto my_range = sycl::range<1>(sizes[0]);

            auto my_kernel = [=](sycl::item<1> item) {
              sycl::id<1> id(item);
              if (id.get(0) != item.get_id(0)) {
                hasSucceded[0] = false;
              }
            };
            cgh.parallel_for<class id_it1>(my_range, my_kernel);
          });

          q.wait_and_throw();
        }
        if (!success) {
          FAIL(log, "id with item was not constructed correctly for dim = 1");
        }
      }

      // dim 2
      {
        auto q = util::get_cts_object::queue();
        bool success = true;
        {
          sycl::buffer<bool, 1> b(&success, sycl::range<1>(1));
          q.submit([&](sycl::handler &cgh) {
            auto hasSucceded =
                b.get_access<sycl::access_mode::read_write,
                             sycl::target::device>(cgh);

            auto my_range = sycl::range<2>(sizes[0], sizes[1]);

            auto my_kernel = [=](sycl::item<2> item) {
              sycl::id<2> id(item);
              if ((id.get(0) != item.get_id(0)) ||
                  (id.get(1) != item.get_id(1))) {
                hasSucceded[0] = false;
              }

            };
            cgh.parallel_for<class id_it2>(my_range, my_kernel);
          });

          q.wait_and_throw();
        }
        if (!success) {
          FAIL(log, "id with item was not constructed correctly for dim = 2");
        }
      }

      // dim 3
      {
        auto q = util::get_cts_object::queue();
        bool success = true;
        {
          sycl::buffer<bool, 1> b(&success, sycl::range<1>(1));
          q.submit([&](sycl::handler &cgh) {
            auto hasSucceded =
                b.get_access<sycl::access_mode::read_write,
                             sycl::target::device>(cgh);

            auto my_range = sycl::range<3>(sizes[0], sizes[1], sizes[2]);

            auto my_kernel = [=](sycl::item<3> item) {
              sycl::id<3> id(item);
              if ((id.get(0) != item.get_id(0)) ||
                  (id.get(1) != item.get_id(1)) ||
                  (id.get(2) != item.get_id(2))) {
                hasSucceded[0] = false;
              }

            };
            cgh.parallel_for<class id_it3>(my_range, my_kernel);
          });

          q.wait_and_throw();
        }
        if (!success) {
          FAIL(log, "id with item was not constructed correctly for dim = 3");
        }
      }
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      std::string errorMsg =
          "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace id_constructors__ */
