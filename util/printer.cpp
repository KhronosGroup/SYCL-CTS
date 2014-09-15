/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include <iostream>
#include <assert.h>

#include "printer.h"

namespace sycl_cts
{
namespace util
{

/** constructor
 */
printer::printer( )
    : m_format( printer::etext )
{
}

/** destructor
 */
printer::~printer()
{
    release( );
}

/** set the current output format for the printer
 */
void printer::set_format( printer::eformat fmt )
{
    m_format = fmt;
}

/** output a test log in the currently set format
 *  @param log, the log object to output
 */
void printer::write( 
    const test_base::info & testInfo )
{
    // acquire the output lock
    std::lock_guard<std::mutex> lock( m_outputMutex );

    switch ( m_format )
    {
    case ( ejson ) :
        write_json( testInfo );
        break;

    case ( etext ) :
        write_text( testInfo );
        break;

    default:
        assert( !"should not get here" );
    }

}


/** output a test log in the currently set format
 *  @param log, the log object to output
 */
void printer::write(
    const logger::info & logInfo )
{
    // acquire the output lock
    std::lock_guard<std::mutex> lock( m_outputMutex );

    switch ( m_format )
    {
    case ( ejson ) :
        write_json( logInfo );
        break;

    case ( etext ) :
        write_text( logInfo );
        break;

    default:
        assert( !"should not get here" );
    }

}

/** convert a logger::result enum to a std::string
 *  @param res, an enum value
 */
std::string printer::result_as_string( logger::result res )
{
    switch ( res )
    {
    case ( logger::efail ):
        return "fail";
    case ( logger::efatal ):
        return "fatal";
    case ( logger::epass ):
        return "pass";
    case ( logger::eskip ):
        return "skip";
    case ( logger::epending ):
        return "pending";
    default:
        assert( !"should not get here" );
        return "";
    }
}

/** output a string to stdout
 *  @param str, the string to output
 */
void printer::output( const std::string & str )
{
    // write output to stdout
    std::cout << str;
}

/** output a string to stdout followed by a new line
 *  @param str, the string to output
 */
void printer::outputln( const std::string & str )
{
    // forward on to the main output function
    output( str + "\n" );
}

/** output a key value pair with string value
 *
 */
void printer::output_kvp(
    const std::string & key,
    const std::string & value,
    const bool comma )
{
    output( "\"" + key + "\":\"" + value + "\"" );
    if ( comma )
        output( "," );
}

/** output a key value pair with integer value
 */
void printer::output_kvp(
    const std::string & key,
    const int & value,
    const bool comma )
{
    output( "\"" + key + "\":" + std::to_string( value ) );
    if ( comma )
        output( "," );
}

void printer::write_json(
    const test_base::info & testInfo )
{
    // signal that this is a test header
    output( "{\"type\"=1," );

    // JSON object string
    output_kvp( "name"     , testInfo.m_name     , true  );
    output_kvp( "file"     , testInfo.m_file     , true  );
    output_kvp( "buildTime", testInfo.m_buildTime, true  );
    output_kvp( "buildDate", testInfo.m_buildDate, false );
    
    outputln( "}" );
}

/** output a test log in JSON form
 *  @param log, the log object to output
 *  @param info, info about the test that generated the log
 */
void printer::write_json(
    const logger::info & logInfo )
{
    // signal that these are test results
    output( "{\"type\"=2," );
    {
        // basic test information
        output_kvp( "result", logInfo.m_result, true );
        
        // if logs were made
        const size_t nLogs = logInfo.m_log.size( );
        if ( nLogs > 0 )
        {
            // JSON value array
            output( "\"notes\":[" );
            for ( int i = 0; i < nLogs; i++ )
            {
                // commas prefix all but the first note
                if ( i > 0 )
                    output( "," );
                output( "\"" + logInfo.m_log[i] + "\"" );
            }
            output( "]" );
        }
    }
    outputln( "}" );
}

/** instruct the printer to finish all printing
 *  operations. importantly, this terminates the root JSON object
 */
void printer::finish( )
{
    // make sure stdout was flushed
    fflush( stdout );
}

/** output a test log in a text form
 *  @param log, the log object to output
 *  @param info, info about the test that generated the log
 */
void printer::write_text(
    const test_base::info & testInfo )
{
    // output test name
    outputln( "    test: " + testInfo.m_name );
    
    // output compilation info
    outputln( "compiled: " + testInfo.m_buildDate + " - " + testInfo.m_buildTime );

}

/** output a test log in a text form
 *  @param log, the log object to output
 *  @param info, info about the test that generated the log
 */
void printer::write_text(
    const logger::info & logInfo )
{
    // all verbose log entries
    const size_t nLogs = logInfo.m_log.size( );
    for ( int i = 0; i < nLogs; i++ )
    {
        outputln( "        - " + logInfo.m_log[i] );
    }

    // output test result
    outputln( "  result: " + result_as_string( logInfo.m_result ) );

    // blank line between tests
    outputln( "" );
}

}; // namespace util
}; // namespace sycl_cts
