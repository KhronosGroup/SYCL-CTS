/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "./../../util/collection.h"
#include "./../../util/printer.h"
#include "./../../util/cmdarg.h"
#include "./../../util/csv.h"
#include "./../../util/executor.h"
#include "./../../util/selector.h"

/** callback triggered at program exit
 *  this is currently only used for debugging and should
 *  be removed later
 */
void on_exit( )
{
    // temporary
    // wait for [enter] before exiting for debugging the output
    getchar( );
}

/** print usage information for the '--help' option
 */
static void print_usage( )
{
    const char *usage = R"(
SYCL CONFORMANCE TEST SUITE
Usage:
    --help         Show this help message
    --json   -j    Print test results in JSON format
    --text   -t    Print test results in text format 
    --csv    -c    CSV file for specifying tests to run
    --list   -l    List the tests compiled in this executable
    --device       Select a device to target:
            'host'
            'opencl_cpu'
            'opencl_gpu'

)";
    puts( usage );
}

/** parse the command line arguments
 */
static void parse_command_line( int argc, const char ** args )
{
    // get more convenient references to each singleton
    sycl_cts::util::cmdarg & cmdarg = 
        sycl_cts::util::cmdarg::instance( );
    sycl_cts::util::printer & printer = 
        sycl_cts::util::printer::instance( );
    sycl_cts::util::collection & collection = 
        sycl_cts::util::collection::instance( );
    
    // try to parse all of the command line arguments
    if ( !cmdarg.parse( argc, args ) )
    {
        // print an error message
        std::string error;
        if ( cmdarg.get_last_error( error ) )
            std::cout << error;
        // exit the program
        exit( 1 );
    }

    // show the usage information
    if ( cmdarg.find_key( "--help" ) )
    {
        print_usage( );
        exit( 0 );
    }

    // list all of the tests in this binary
    if ( cmdarg.find_key( "--list" ) || cmdarg.find_key( "-l" ) )
    {
        collection.list( );
        exit( 0 );
    }

    // load a csv file used for specifying test parameters
    std::string csvfile;
    if ( cmdarg.get_value( "--csv", csvfile ) || cmdarg.find_key( "-c" ) )
    {
        sycl_cts::util::csv csvFile;

        // try to load the csv file
        if (! csvFile.load( csvfile ) )
        {
            // print an error message
            std::string error;
            if ( csvFile.get_last_error( error ) )
                std::cout << error;
            exit( 1 );
        }

        // forward the csv file on to the collection for filtering
        collection.set_test_parameters( csvFile );
    }

    // set JSON output formatting
    if ( cmdarg.find_key( "--json" ) || cmdarg.find_key( "-j" ) )
        printer.set_format( sycl_cts::util::printer::ejson );

    // set text output formatting
    if ( cmdarg.find_key( "--text" ) || cmdarg.find_key( "-t" ) )
        printer.set_format( sycl_cts::util::printer::etext );

    // set the default sycl cts device
    std::string deviceName;
    if ( cmdarg.get_value( "--device", deviceName ) )
    {
        sycl_cts::util::selector::instance().set_default( deviceName );
    }
}

/** test suite entry point
 */
int main( int argc, const char **args )
{
    // prepare the test collection for use
    sycl_cts::util::collection & collection = 
        sycl_cts::util::collection::instance( );
    collection.prepare( );

    // parse the command line
    parse_command_line( argc, args );
    
    // register a cleanup handler
    atexit( on_exit );
    
    // get more convenient references singletons
    sycl_cts::util::executor & executor = 
        sycl_cts::util::executor::instance( );
    sycl_cts::util::printer & printer = 
        sycl_cts::util::printer::instance( );

    // execute all tests
    executor.run_all( );

    // flush the printer so all output appears on stdout
    printer.finish( );

    return 0;
}
