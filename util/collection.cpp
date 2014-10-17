/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "collection.h"
#include "printer.h"
#include "csv.h"

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
    get<collection>().add_test( test );
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
void collection::add_test( test_base * testobj )
{
    assert( testobj != nullptr );

    // encapsulate in testinfo structure
    testinfo test =
    {
        testobj ,   // test
        false   ,   // skip
        -1          // timeout
    };

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
    for ( int i = 0; i < numTests; i++ )
    {
        // locate a specific test
        testinfo & test = m_tests[i];

        assert( test.m_test != nullptr );

        // free this test
        delete test.m_test;
        test.m_test = nullptr;
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
        testinfo & test = m_tests[i];

        assert( test.m_test != nullptr );

        test_base::info testInfo;
        test.m_test->get_info( testInfo );

        // form one line for each tests
        STRING line;
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
collection::testinfo & collection::get_test( int index )
{
    // get the number of tests in the collection
    size_t nTests = m_tests.size( );

    // check that index is in range
    assert( index >= 0 && index < nTests );

    // grab the test info structure
    testinfo & test = m_tests[index];
    
    // check that test is really valid
    assert( test.m_test != nullptr );

    return test;
}

/** perform a partial string match.
 *  return true if entire string 'b' can be found at the beginning of
 *  string 'a'.
 */
static inline 
bool partial_strcmp( const STRING & a, const STRING & b )
{
    const char * _a = a.c_str( );
    const char * _b = b.c_str( );

    // advance both string in lock step
    for ( ;; _a++, _b++ )
    {
        // if 'b' is totally consumed then we pass
        if ( *_b == '\0' )
            return true;

        // if they are equal advance
        if ( *_a == *_b )
            continue;

        // otherwise 'b' does not prefix 'a'
        return false;
    }
}

/** set the skip status of a test by name
 */
void collection::set_test_skip( const STRING & testName, bool skip )
{
    // TODO: since all tests are in alphabetic order this could be optimized
    //       yet the linear search here is likely not a hot spot
    for ( int i = 0; i < m_tests.size( ); i++ )
    {
        testinfo & info = m_tests[i];
        sycl_cts::util::test_base::info testInfo;
        info.m_test->get_info( testInfo );

        // if 'testName' is prefix of test name
        if ( partial_strcmp( testInfo.m_name, testName ) )
        {
            // we do not want to skip this test
            info.m_skip = false;
        }
    }
}

/** load a test filter (csv file)
 *  @param csvPath, the csv file filtering the tests
 */
bool collection::filter_tests_csv( const STRING & csvPath )
{
    // try to load the csv file
    csv csvFile;
    if (! csvFile.load_file( csvPath ) )
    {
        // unable to load the CSV file
        return false;
    }

    // pre-pass sets all tests to be skipped
    for ( int i = 0; i < m_tests.size( ); i++ )
    {
        testinfo & info = m_tests[i];
        info.m_skip = true;
    }

    // loop over all rows in the CSV file
    for ( int r = 0; r < csvFile.size( ); r++ )
    {
        // first column is test name
        STRING csvName;
        if ( !csvFile.get_item( r, 0, csvName ) )
            continue;

        // check for empty string
        if ( csvName.empty( ) )
            continue;

        // enable a test with this name
        set_test_skip( csvName, false );
    }

    return true;
}

/**
 */
bool collection::filter_tests_name( const STRING & name )
{
    // todo: extend to take a comma separated list of names

    // pre-pass sets all tests to be skipped
    for ( int i = 0; i < m_tests.size( ); i++ )
    {
        testinfo & info = m_tests[i];
        info.m_skip = true;
    }

    // enable a test with this name
    set_test_skip( name, false );

    return true;
}

/** this function act as a comparator between two tests
 *  for sorting them alphabetically based on their names
 *  with std::sort( )
 */
static bool test_order_func( 
    const collection::testinfo & a, 
    const collection::testinfo & b )
{
    // get test a info
    assert( a.m_test != nullptr );

    test_base::info aInfo;
    a.m_test->get_info( aInfo );

    // get test b info
    assert( b.m_test != nullptr );

    test_base::info bInfo;
    b.m_test->get_info( bInfo );

    // use STRING compare operator
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