/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
//  This file is a common utility for the implementation of
//  accessor_constructors.cpp and accessor_api.cpp.
//
**************************************************************************/
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_UTILITY_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_CONSTRUCTORS_UTILITY_H

#include "../common/common.h"

namespace TEST_NAMESPACE {

	using namespace sycl_cts;

	/** Helper function that calculates the amount of elements
	 *  of a range
	 */
	template <size_t dims>
	size_t getElementsCount(const cl::sycl::range<dims> &range);

	/** Specializations of for getElementsCount each supported
	 *  dimensionality
	 */
	template <>
	size_t getElementsCount<1>(const cl::sycl::range<1> &range) {
		return range[0];
	}

	template <>
	size_t getElementsCount<2>(const cl::sycl::range<2> &range) {
		return range[0] * range[1];
	}

	template <>
	size_t getElementsCount<3>(const cl::sycl::range<3> &range) {
		return range[0] * range[1] * range[2];
	}

	/** Helper function that calculates a range from a size so
	 *  that each dimension equals size
	 */
	template <size_t dims>
	cl::sycl::range<dims> getRange(const size_t &size);

	/** Specializations of for getRange each supported
	 *  dimensionality
	 */
	template <>
	cl::sycl::range<1> getRange<1>(const size_t &size) {
		return cl::sycl::range<1>(size);
	}
	template <>
	cl::sycl::range<2> getRange<2>(const size_t &size) {
		return cl::sycl::range<2>(size, size);
	}
	template <>
	cl::sycl::range<3> getRange<3>(const size_t &size) {
		return cl::sycl::range<3>(size, size, size);
	}

	/** Helper function that calculates an id from a size so
	*  that each dimension equals size
	*/
	template <size_t dims>
	cl::sycl::id<dims> getId(const size_t &size);

	/** Specializations of for getId each supported
	*  dimensionality
	*/
	template <>
	cl::sycl::id<1> getId<1>(const size_t &size) {
		return cl::sycl::id<1>(size);
	}
	template <>
	cl::sycl::id<2> getId<2>(const size_t &size) {
		return cl::sycl::id<2>(size, size);
	}
	template <>
	cl::sycl::id<3> getId<3>(const size_t &size) {
		return cl::sycl::id<3>(size, size, size);
	}



}  // namespace accessor_utility__

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_H
