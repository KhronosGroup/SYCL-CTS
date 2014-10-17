/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "executor.h"
#include "singleton.h"
#include "printer.h"
#include "collection.h"

namespace sycl_cts
{
namespace util
{

/** execute all tests in the collection
 */
void executor::run_all( )
{
    // find the number of tests in the collection
    const int nTests = get<collection>().get_test_count( );

    // iterate over all tests
    for ( int i = 0; i < nTests; i++ ) 
    {

        // locate a specific test
        collection::testinfo & info = get<collection>().get_test( i );
        test_base * test = info.m_test;

        // do not execute any test marked to be skipped
        if ( info.m_skip )
            continue;

        // scope for the logger
        {
            test_base::info testInfo;
            test->get_info( testInfo );

            logger logger;

            // get a new log id for this logging session
            printer::logid logId = get<printer>( ).new_log_id( );

            // write the test info header
            get<printer>( ).write( logId, testInfo );   

            // we must install an exception handler here so that if a test
            // fails to catch a thrown exception it wont crash the entire
            // test suite
            try
            {
                // ask the test to set itself up
                if ( test->setup( logger ) )
                {
                    // ask the test to execute
                    test->run( logger );
                }
                // enforce that each test must give a result
                assert( logger.get_result() != logger::epending );

                // ask the test to clean up after itself
                test->cleanup( );
            }
            catch ( ... )
            {
                logger.fail( "Exception thrown and not caught by test case!", 0 );
            }

            // write the test transcript
            get<printer>( ).write( logId, logger.get_info() );   

            // if we received a fatal error then we must exit
            if ( logger.get_result( ) == logger::efatal )
                break;
        }
    }
}

}; // namespace util
}; // namespace sycl_cts