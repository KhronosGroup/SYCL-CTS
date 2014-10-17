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
    , m_nextLogId( 0 )
{
}

/** destructor
 */
printer::~printer()
{
    release( );
}

/**
 * 
 */
printer::logid printer::new_log_id( )
{
    // acquire the log id issue mutex
    LOCK_GUARD<MUTEX> lock( m_logIdMutex );

    return m_nextLogId++;
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
    printer::logid id,
    const test_base::info & testInfo )
{
    // acquire the output lock
    LOCK_GUARD<MUTEX> lock( m_outputMutex );

    switch ( m_format )
    {
    case ( ejson ) :
        write_json( id, testInfo );
        break;

    case ( etext ) :
        write_text( id, testInfo );
        break;

    default:
        assert( !"should not get here" );
    }

}

/** output a test log in the currently set format
 *  @param log, the log object to output
 */
void printer::write(
    printer::logid id,
    const logger::info & logInfo )
{
    // acquire the output lock
    LOCK_GUARD<MUTEX> lock( m_outputMutex );

    switch ( m_format )
    {
    case ( ejson ) :
        write_json( id, logInfo );
        break;

    case ( etext ) :
        write_text( id, logInfo );
        break;

    default:
        assert( !"should not get here" );
    }

}

/** convert a logger::result enum to a STRING
 *  @param res, an enum value
 */
STRING printer::result_as_string( logger::result res )
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
void printer::output( const STRING & str )
{
    // write output to stdout
    std::cout << str;
}

/** output a string to stdout followed by a new line
 *  @param str, the string to output
 */
void printer::outputln( const STRING & str )
{
    // forward on to the main output function
    output( str + "\n" );
}

/** output a key value pair with string value
 *
 */
void printer::output_kvp(
    const STRING & key,
    const STRING & value,
    const bool comma )
{
    output( "\"" + key + "\":\"" + value + "\"" );
    if ( comma )
        output( "," );
}

/** output a key value pair with integer value
 */
void printer::output_kvp(
    const STRING & key,
    const int & value,
    const bool comma )
{
    output( "\"" + key + "\":" + std::to_string( value ) );
    if ( comma )
        output( "," );
}

void printer::write_json(
    printer::logid id,
    const test_base::info & testInfo )
{
    // signal that this is a test header
    output( "{\"type\":1," );

    // output the log id to bind header to footer
    output_kvp( "id", id, true );

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
    printer::logid id,
    const logger::info & logInfo )
{
    // signal that these are test results
    output( "{\"type\":2," );
    
    // output the log id to bind header to footer
    output_kvp( "id", id, true );

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
        output( "]," );
    }
    
    // output line number
    output_kvp( "line", logInfo.m_line, true );

    // basic test information
    output_kvp( "result", logInfo.m_result, false );

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
    printer::logid id,
    const test_base::info & testInfo )
{
    outputln( "#" + std::to_string( id ) );

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
    printer::logid id,
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

    // display line number for non passes
    if ( logInfo.m_result != logger::result::epass )
        outputln( "    line: " + std::to_string( logInfo.m_line ) );

    // blank line between tests
    outputln( "" );
}

}; // namespace util
}; // namespace sycl_cts
