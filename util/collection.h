/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include <vector>

#include "test_base.h"
#include "singleton.h"
#include "csv.h"

namespace sycl_cts
{
namespace util
{

/** this class is a central repository of tests
 */
class collection
    : public singleton<collection>
{
public:

    /** constructor
     */
    collection( );
    
    /** add a test to the collection
     *  @param test, the test to be added
     */
    void add_test( test_base * test );

    /** run all tests in the collection
     */
    void run_all( );

    /** release all registered tests
     */
    void release( );
    
    /** list all tests in the collection
     */
    void list( );

    /** specify the test session parameters
     *  via a csv file
     *  @param csv, the csv file containing the parameters
     */
    void set_test_parameters( const csv & params );

    /** get the total number of tests in this collection
     */
    int get_test_count( ) const;

    /** return a specific test 
     */
    test_base * get_test( int index );

    /** prepare the list of tests for execution
     */
    void prepare( );

protected:
    
    // the test collection itself
    std::vector<test_base*> m_tests;

};

}; // namespace util
}; // namespace sycl_cts