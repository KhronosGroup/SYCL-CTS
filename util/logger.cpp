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
void logger::add_to_log( const std::string & str )
{
    if ( ( str.empty( ) ) || ( str.length( ) == 0 ) )
        return;
    m_info.m_log.push_back( str );
}

/** constructor
 */
logger::logger( )
{
}

/** notify a test has failed
 *  @param reason, optional descriptive string for fail
 */
void logger::fail( const std::string & str )
{
    add_to_log( str );
    m_info.m_result = logger::efail;
}

/** notify a test has passed
 *  @param reason, optional descriptive string for pass
 */
void logger::pass( const std::string & str )
{
    add_to_log( str );
    m_info.m_result = logger::epass;
}

/** notify a test has been skipped
 *  @param reason, optional descriptive string for skip
 */
void logger::skip( const std::string & str )
{
    add_to_log( str );
    m_info.m_result = logger::eskip;
}

/** report fatal error and abort program
 *  @param reason, optional descriptive string for fatal error
 */
void logger::fatal( const std::string & str )
{
    add_to_log( str );
    m_info.m_result = logger::efatal;
}

/** output verbose information
 *  @param string
 */
void logger::note( const std::string & str )
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
    std::string newLogItem( buffer );
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
