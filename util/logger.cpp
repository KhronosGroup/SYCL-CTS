/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#ifdef _MSC_VER
# define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdarg.h>
#include <assert.h>
#include <cstdio>

#include "logger.h"

namespace sycl_cts
{
namespace util
{

/** add one entry into the test log
 *  @param str, string entry to add
 */
void logger::add_to_log( const STRING & str )
{
    if ( ( str.empty( ) ) || ( str.length( ) == 0 ) )
        return;
    m_info.m_log.push_back( str );
}

/** constructor
 */
logger::logger( )
{
    // record a pass initially until it is overridden
    // by a failing test
    m_info.m_result = result::epass;

    m_info.m_time = 0;

    m_info.m_line = 0;
}

/** notify a test has failed
 *  @param reason, optional descriptive string for fail
 *  @param line, test line number that threw the error
 */
void logger::fail( const STRING & str, const int line )
{
    add_to_log( str );
    m_info.m_result = logger::efail;
    m_info.m_line = line;
}

/** notify a test has passed
 *  @param reason, optional descriptive string for pass
 */
void logger::pass( const STRING & str )
{
    add_to_log( str );
    m_info.m_result = logger::epass;
}

/** notify a test has been skipped
 *  @param reason, optional descriptive string for skip
 */
void logger::skip( const STRING & str )
{
    add_to_log( str );
    m_info.m_result = logger::eskip;
}

/** report fatal error and abort program
 *  @param reason, optional descriptive string for fatal error
 */
void logger::fatal( const STRING & str )
{
    add_to_log( str );
    m_info.m_result = logger::efatal;
}

/** output verbose information
 *  @param string
 */
void logger::note( const STRING & str )
{
    // push this into the log list
    add_to_log( str );
}

/** output verbose information
 *  @param variable argument list, printf syntax
 */
void logger::note( const char *fmt, ... )
{
    assert( fmt != nullptr );

    va_list args;
    va_start( args, fmt );
    
    // temporary buffer of 1kb
    char buffer[1024];

    // use string formatting to print into buffer
    if ( vsnprintf( buffer, sizeof( buffer ), fmt, args ) <= 0 )
    {
        // error
        assert( !"sprintf failed" );
        return;
    }
    // enforce terminal character
    buffer[sizeof(buffer)-1] = '\0';
    // cast to a string object
    STRING newLogItem( buffer );
    // push this into the log list
    add_to_log( newLogItem );

    va_end( args );
}

/** return the the internal state structure
 */
const logger::info & logger::get_info( ) const
{
    return m_info;
}

/** destructor
 *  will terminate log output appropriately
 */
logger::~logger( )
{
}

/** has a test result been emitted
    *  @return, false = result not yet specified
    */
logger::result logger::get_result( ) const
{
    return m_info.m_result;
}

}; // namespace util
}; // namespace sycl_cts
