/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include "stl.h"
#include "test_base.h"
#include "singleton.h"

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

    /** test structure
     */
    struct testinfo
    {
        test_base  *m_test;
        bool        m_skip;
        int         m_timeout;
    };

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

    /** load a test filter (csv file)
     *  @param csvPath, the csv file path fir filtering the tests
     */
    bool filter_tests_csv( const STRING & csvPath );

    /** filter tests by name
     */
    bool filter_tests_name( const STRING & testName );

    /** get the total number of tests in this collection
     */
    int get_test_count( ) const;

    /** return a specific test 
     */
    testinfo & get_test( int index );

    /** prepare the list of tests for execution
     */
    void prepare( );

protected:
    
    /** set the skip status of a test by name
     */
    void set_test_skip( const STRING & testName, bool skip );

    // the test collection itself
    VECTOR<testinfo> m_tests;

};

}; // namespace util
}; // namespace sycl_cts