/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME nd_item_api

namespace test_nd_item__ {
using namespace sycl_cts;

size_t getIndex(sycl::id<1> Id, sycl::range<1> Range) {
  return Id.get(0);
}

size_t getIndex(sycl::id<2> Id, sycl::range<2> Range) {
  return Id.get(1) + Id.get(0) * Range.get(1);
}

size_t getIndex(sycl::id<3> Id, sycl::range<3> Range) {
  return Id.get(2) + Id.get(1) * Range.get(0) + Id.get(0) * Range.get(0) * Range.get(1);
}

template <int dimensions>
class kernel_nd_item {
 protected:
  typedef sycl::accessor<int, dimensions, sycl::access::mode::read,
                             sycl::target::global_buffer>
      t_readAccess;
  typedef sycl::accessor<int, dimensions, sycl::access::mode::write,
                             sycl::target::global_buffer>
      t_writeAccess;

  t_readAccess m_globalID;
  t_readAccess m_localID;
  t_writeAccess m_o;

 public:
  kernel_nd_item(t_readAccess inG_, t_readAccess inL_, t_writeAccess out_)
      : m_globalID(inG_), m_localID(inL_), m_o(out_) {}

  void operator()(sycl::nd_item<dimensions> myitem) const {
    bool failed = false;

    /* test global ID*/
    sycl::id<dimensions> global_id = myitem.get_global_id();
    size_t globals[dimensions];

    for (int i = 0; i < dimensions; ++i) {
      globals[i] = myitem.get_global_id(i);
      if (globals[i] != global_id.get(i)) {
        failed = true;
      }
    }

    sycl::id<dimensions> local_id = myitem.get_local_id();
    size_t locals[dimensions];

    for (int i = 0; i < dimensions; ++i) {
      locals[i] = myitem.get_local_id(i);
      if (locals[i] != local_id.get(i)) {
        failed = true;
      }
    }

    /* test group ID*/
    sycl::group<dimensions> group_id = myitem.get_group();
    size_t groups[dimensions];

    for (int i = 0; i < dimensions; ++i) {
      groups[i] = myitem.get_group(i);
      if (groups[i] != group_id.get_id(i)) {
        failed = true;
      }
    }

    /* test range*/
    sycl::range<dimensions> globalRange = myitem.get_global_range();
    size_t global_ranges[dimensions];

    size_t globalIndex = getIndex(global_id, globalRange);
    if (m_globalID[global_id] != globalIndex) {
      failed = true;
    }
    for (int i = 0; i < dimensions; ++i) {
      global_ranges[i] = myitem.get_global_range(i);
      if (global_ranges[i] != globalRange.get(i)) {
        failed = true;
      }
    }

    sycl::range<dimensions> localRange = myitem.get_local_range();
    size_t local_ranges[dimensions];

    size_t localIndex = getIndex(local_id, localRange);
    if (m_localID[local_id] != localIndex) {
      failed = true;
    }
    for (int i = 0; i < dimensions; ++i) {
      local_ranges[i] = myitem.get_local_range(i);
      if (local_ranges[i] != localRange.get(i)) {
        failed = true;
      }
    }

    for (int i = 0; i < dimensions; ++i) {
      if (group_id.get_id(i) != (global_id.get(i) / localRange.get(i)))
        failed = true;
    }

    /* test number of groups*/
    sycl::id<dimensions> num_groups = myitem.get_group_range();
    size_t nGroups[dimensions];

    for (int i = 0; i < dimensions; ++i) {
      nGroups[i] = myitem.get_group_range(i);
      if (nGroups[i] != num_groups.get(i)) {
        failed = true;
      }
    }

    for (int i = 0; i < dimensions; ++i) {
      size_t ratio = globalRange.get(i) / localRange.get(i);
      if (ratio != num_groups.get(i)) {
        failed = true;
      }
    }

    /* test NDrange and offset*/
    sycl::id<dimensions> offset = myitem.get_offset();
    sycl::nd_range<dimensions> NDRange = myitem.get_nd_range();

    sycl::range<dimensions> ndGlobal = NDRange.get_global_range();
    sycl::range<dimensions> ndLocal = NDRange.get_local_range();
    sycl::id<dimensions> ndOffset = NDRange.get_offset();

    for (int i = 0; i < dimensions; ++i) {
      bool are_same = true;
      are_same &= globalRange.get(i) == ndGlobal.get(i);
      are_same &= localRange.get(i) == ndLocal.get(i);
      are_same &= offset.get(i) == ndOffset.get(i);

      failed = !are_same ? true : failed;
    }

    /* test linear_id */
    size_t glid = myitem.get_global_linear_id();
    size_t llid = myitem.get_local_linear_id();
    size_t grlid = myitem.get_group_linear_id();
    size_t groupIndex = getIndex(group_id.get_id(), myitem.get_group_range());

    if (glid != globalIndex)
      failed = true;
    if (llid != localIndex)
      failed = true;
    if (grlid != groupIndex)
      failed = true;

    /* write back success or failure*/
    m_o[global_id] = failed ? 0 : 1;
  }
};

/* test that kernel returns expected data, i.e. all 1s*/
int check_nd_item(int *buf, const int nWidth, const int nHeight,
                  const int nDepth) {
  int nErrors = 0;
  for (int i = 0; i < nWidth * nHeight * nDepth; i++) {
    nErrors += (buf[i] == 0);
  }
  return nErrors;
}

/* Fill buffers with global and local work item ids*/
void populate(int *globalBuf, int *localBuf, const int *localSize,
              const int *globalSize) {
  for (int k = 0; k < globalSize[2]; k++) {
    for (int j = 0; j < globalSize[1]; j++) {
      for (int i = 0; i < globalSize[0]; i++) {
        const int globIndex =
            (k * globalSize[0] * globalSize[1]) + (j * globalSize[0]) + i;
        globalBuf[globIndex] = globIndex;

        int local_i = i % localSize[0];
        int local_j = j % localSize[1];
        int local_k = k % localSize[2];

        const int locIndex = (local_k * localSize[0] * localSize[1]) +
                             (local_j * localSize[0]) + local_i;
        localBuf[globIndex] = locIndex;
      }
    }
  }
}

void test_item(util::logger &log, sycl::queue &queue) {
  /* set sizes*/
  const int globalSize[3] = {16, 16, 16};
  const int localSize[3] = {4, 4, 4};
  const int nSize = globalSize[0] * globalSize[1] * globalSize[2];

  /* allocate host buffers */
  std::unique_ptr<int> globalIDs(new int[nSize]);
  std::unique_ptr<int> localIDs(new int[nSize]);
  std::unique_ptr<int> dataOut(new int[nSize]);

  /* set host buffers */
  populate(globalIDs.get(), localIDs.get(), localSize, globalSize);
  ::memset(dataOut.get(), 0, nSize * sizeof(int));

  /* create ranges*/
  sycl::range<1> globalRange(globalSize[0]);
  sycl::range<1> localRange(localSize[0]);
  sycl::nd_range<1> dataRange(globalRange, localRange);

  /* test 1 Dimension*/
  {
    /* create ranges*/
    sycl::range<1> globalRange(globalSize[0]);
    sycl::range<1> localRange(localSize[0]);
    sycl::nd_range<1> dataRange(globalRange, localRange);

    sycl::buffer<int, 1> bufGlob(globalIDs.get(), globalRange);
    sycl::buffer<int, 1> bufLoc(localIDs.get(), globalRange);
    sycl::buffer<int, 1> bufOut(dataOut.get(), globalRange);

    queue.submit([&](sycl::handler &cgh) {
      auto accG =
          bufGlob.template get_access<sycl::access::mode::read>(cgh);
      auto accL = bufLoc.template get_access<sycl::access::mode::read>(cgh);
      auto accOut =
          bufOut.template get_access<sycl::access::mode::write>(cgh);

      kernel_nd_item<1> kernel_1d(accG, accL, accOut);
      cgh.parallel_for(dataRange, kernel_1d);

    });
  }
  /* check no errors are returned*/
  if (check_nd_item(dataOut.get(), globalSize[0], 1, 1)) {
    FAIL(log, "item API inconsistency");
    return;
  }

  /* test 2 Dimensions */
  ::memset(dataOut.get(), 0, nSize * sizeof(int));
  {
    /* create ranges*/
    sycl::range<2> globalRange(globalSize[0], globalSize[1]);
    sycl::range<2> localRange(localSize[0], localSize[1]);
    sycl::nd_range<2> dataRange(globalRange, localRange);

    sycl::buffer<int, 2> bufGlob(globalIDs.get(), globalRange);
    sycl::buffer<int, 2> bufLoc(localIDs.get(), globalRange);
    sycl::buffer<int, 2> bufOut(dataOut.get(), globalRange);

    queue.submit([&](sycl::handler &cgh) {
      auto accG =
          bufGlob.template get_access<sycl::access::mode::read>(cgh);
      auto accL = bufLoc.template get_access<sycl::access::mode::read>(cgh);
      auto accOut =
          bufOut.template get_access<sycl::access::mode::write>(cgh);

      kernel_nd_item<2> kernel_2d(accG, accL, accOut);
      cgh.parallel_for(dataRange, kernel_2d);
    });
  }
  /* check no errors are returned */
  if (check_nd_item(dataOut.get(), globalSize[0], globalSize[1], 1)) {
    FAIL(log, "item API inconsistency");
    return;
  }

  /* test 3 Dimensions */
  ::memset(dataOut.get(), 0, nSize * sizeof(int));
  {
    /* create ranges*/
    sycl::range<3> globalRange(globalSize[0], globalSize[1], globalSize[2]);
    sycl::range<3> localRange(localSize[0], localSize[1], localSize[2]);
    sycl::nd_range<3> dataRange(globalRange, localRange);

    sycl::buffer<int, 3> bufGlob(globalIDs.get(), globalRange);
    sycl::buffer<int, 3> bufLoc(localIDs.get(), globalRange);
    sycl::buffer<int, 3> bufOut(dataOut.get(), globalRange);

    queue.submit([&](sycl::handler &cgh) {
      auto accG = bufGlob.get_access<sycl::access::mode::read>(cgh);
      auto accL = bufLoc.get_access<sycl::access::mode::read>(cgh);
      auto accOut = bufOut.get_access<sycl::access::mode::write>(cgh);

      kernel_nd_item<3> kernel_3d(accG, accL, accOut);
      cgh.parallel_for(dataRange, kernel_3d);
    });
  }
  /* check no errors are returned */
  if (check_nd_item(dataOut.get(), globalSize[0], globalSize[1],
                    globalSize[2])) {
    FAIL(log, "item API inconsistency");
  }
}

/** test sycl::nd_item
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
  *  @param info, test_base::info structure as output
  */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
  *  @param log, test transcript logging class
  */
  void run(util::logger &log) override {
    try {
      auto cmd_queue = util::get_cts_object::queue();

      test_item(log, cmd_queue);

      cmd_queue.wait_and_throw();
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      sycl::string_class errorMsg =
          "a SYCL exception was caught: " + sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace test_nd_item__ */
