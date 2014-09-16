/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <assert.h>
#include <iostream>
#include <algorithm>

#include "collection.h"
#include "printer.h"

// conformance test suite namespace
namespace sycl_cts
{
namespace util
{
    
/** c style function to add a test to the collection
 *  a c function is used to avoid having to pull in an extra
 *  header for the definition of the collection object.
 */
extern void register_test( test_base * test )
{
    assert( test != nullptr );
    collection::instance().add_test( test );
}

/** constructor
 */
collection::collection( )
    : m_tests( )
{
}

/** add a test to the collection
 *  @param test, the test to be added
 */
void collection::add_test( test_base * test )
{
    assert( test != nullptr );
    // add this test to the collection
    m_tests.push_back( test );
}

/**
 *
 */
void collection::release( )
{
    // iterate over all tests in the collection
    const size_t numTests = m_tests.size( );
    for ( int i = 0; i < numTests; i++ ) {

        // locate a specific test
        test_base * test = m_tests[i];
        assert( test != nullptr );

        // free all tests
        delete test;

    }

    // clear the vector
    m_tests.clear( );
}

/** output a list of all the tests contained by this collection
 */
void collection::list( )
{
    // iterate over all tests in the collection
    const size_t numTests = m_tests.size( );

    // print the number of tests in this executable
    std::cout << std::to_string( numTests ) + " tests in this executable\n";

    // iterate over all of the contained tests
    for ( int i = 0; i < numTests; i++ ) {
        
        // locate a specific test
        test_base * test = m_tests[i];
        assert( test != nullptr );

        test_base::info testInfo;
        test->get_info( testInfo );

        // form one line for each tests
        std::string line;
        line = testInfo.m_name + "\n";

        // output on stdout
        std::cout << line;
    }
}

/** get the total number of tests in this collection
 */
int collection::get_test_count( ) const
{
    size_t size = m_tests.size( );
    return (int) size;
}

/** return a specific test 
 */
test_base * collection::get_test( int index )
{
    // get the number of tests in the collection
    size_t nTests = m_tests.size( );

    // check that index is in range
    assert( index >= 0 && index < nTests );

    test_base * test = m_tests[index];

    // check that test is really valid
    assert( test != nullptr );

    return test;
}

/** specify the test session parameters
 *  via a csv file
 *  @param csv, the csv file containing the parameters
 */
void collection::set_test_parameters( const csv & csvFile )
{
}

/** this function act as a comparator between two tests
 *  for sorting them alphabetically based on their names
 *  with std::sort( )
 */
static bool test_order_func( const test_base *a, const test_base *b )
{
    // get test a info
    assert( a != nullptr );
    test_base::info aInfo;
    a->get_info( aInfo );

    // get test b info
    assert( b != nullptr );
    test_base::info bInfo;
    b->get_info( bInfo );

    // use std::string compare operator
    return aInfo.m_name < bInfo.m_name;
}

/** prepare the list of tests for execution
 *
 */
void collection::prepare( )
{
    // sort the list of tests using our sort function
    std::sort( 
        m_tests.begin( ),
        m_tests.end( ), 
        test_order_func );
}

}; // namespace util
}; // namespace sycl_cts