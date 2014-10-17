/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <stdlib.h>

#include "test_manager.h"
#include "cmdarg.h"
#include "collection.h"
#include "printer.h"
#include "selector.h"
#include "executor.h"

#if defined (_MSC_VER)
extern "C"
extern long __stdcall IsDebuggerPresent( );
#endif

namespace sycl_cts
{
namespace util
{

/**
 */
test_manager::test_manager( )
    : m_willExecute ( false )
    , m_wimpyMode   ( false )
{
}

/**
 */
void test_manager::on_start( )
{
    // prepare the test collection for use
    get<util::collection>( ).prepare( );
}

/**
 */
void test_manager::on_exit( )
{
    // flush the printer so all output appears on stdout
    get<util::printer>( ).finish( );

    // in debug mode, halt before exit
#if defined( _MSC_VER )
    if ( IsDebuggerPresent( ) != 0 )
        getchar( );
#endif
}

/**
 */
bool test_manager::parse( const int argc, const char **args )
{
    // get more convenient references to each singleton
    util::cmdarg     & cmdarg     = get<util::cmdarg>( );
    util::printer    & printer    = get<util::printer>( );
    util::collection & collection = get<util::collection>( );
    util::selector   & selector   = get<util::selector>( );
    
    // try to parse all of the command line arguments
    if ( !cmdarg.parse( argc, args ) )
    {
        // print an error message
        util::STRING error;
        if ( cmdarg.get_last_error( error ) )
            std::cout << error;
        return false;
    }

    // show the usage information
    if ( cmdarg.find_key( "--help" ) )
    {
        print_usage( );
        return true;
    }

    // list all of the tests in this binary
    if ( cmdarg.find_key( "--list" ) || cmdarg.find_key( "-l" ) )
    {
        collection.list( );
        return true;
    }

    // load a csv file used for specifying test parameters
    util::STRING csvfile;
    if ( cmdarg.get_value( "--csv", csvfile ) || cmdarg.find_key( "-c" ) )
    {
        // forward the csv file on to the collection for filtering
        if (! collection.filter_tests_csv( csvfile ) )
        {
            puts( "unable to load csv file" );
            return false;
        }
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
        selector.set_default( deviceName );

    // filter by the given test name
    std::string testName;
    if ( cmdarg.get_value( "--test", testName ) )
        collection.filter_tests_name( testName );

    // check for wimpy mode being enabled
    if ( cmdarg.find_key( "--wimpy" ) || cmdarg.find_key( "-w" ) )
        m_wimpyMode = true;

    // the test suite will try to execute tests
    m_willExecute = true;
    return true;
}

/** 
 */
bool test_manager::run( )
{
    // execute all tests
    get<util::executor>( ).run_all( );
    
    return true;
}

/** 
 */
void test_manager::print_usage( )
{
    const char *usage = R"(
SYCL CONFORMANCE TEST SUITE
Usage:
    --help           Show this help message
    --json   -j      Print test results in JSON format
    --text   -t      Print test results in text format 
    --csv    -c      CSV file for specifying tests to run
    --list   -l      List the tests compiled in this executable
    --wimpy  -w      Run with reduced test complexity (faster)

    --device [name]  Select a device to target:
            'host'
            'opencl_cpu'
            'opencl_gpu'
    --test [name]   Specify a specific test to run by name, eg:
            '--test unary_math_sin'

)";
    puts( usage );
}

/** 
 */
bool test_manager::will_execute( ) const
{
    return m_willExecute;
}

/** 
 */
bool test_manager::wimpy_mode_enabled( ) const
{
    return m_wimpyMode;
}

}; // namespace util
}; // namespace sycl_cts