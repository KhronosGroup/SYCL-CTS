/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "./../../util/test_manager.h"
#include "./../../util/collection.h"

using namespace sycl_cts::util;

/** program exit callback
 */
void exit_handler()
{
    // inform the test manager of program exit
    get<test_manager>().on_exit();
}

/** test suite entry point
 */
int main( int argc, const char **args )
{
    // register an exit handler
    atexit( exit_handler );

    // prepare the test collection for use
    get<collection>().prepare();

    // get a handle to the test manager instance
    test_manager &testManager = get<test_manager>();

    // inform the test manager the cts has launched
    testManager.on_start();

    // parse the command line
    if ( !testManager.parse( argc, args ) )
    {
        return -1;
    }

    // if the test harness will execute
    if ( testManager.will_execute() )
    {
        // run all of the specified tests
        if ( !testManager.run() )
        {
            return -1;
        }
    }

    return 0;
}
