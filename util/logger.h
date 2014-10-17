/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include "stl.h"

namespace sycl_cts
{
namespace util
{

/** the logger class records all output during testing
 *  and so forms a transcript of an executed test
 */
class logger
{
public:

    /** test result values
     */
    enum result
    {
        epending = 0,
        epass    ,
        efail    ,
        eskip    ,
        efatal   ,
        etimeout ,
    };

    /** test information structure
     */
    struct info
    {
        VECTOR<STRING> m_log;
        result         m_result;
        int            m_time;

        // line number associated with result
        int            m_line;
    };

    /** signal the beginning of a test
     */
    logger( );

    /** destructor
     */
    ~logger( );
    
    /** notify a test has failed
     *  @param reason, optional descriptive string for fail
     *  @param line, line number in test that reports the failure
     */
    void fail( const STRING & reason, const int line );

    /** notify a test has passed
     *  @param reason, optional descriptive string for pass
     */
    void pass( const STRING & reason = STRING() );

    /** notify a test has been skipped
     *  @param reason, optional descriptive string for skip
     */
    void skip( const STRING & reason = STRING() );
    
    /** report fatal error and abort program
     *  @param reason, optional descriptive string for fatal error
     */
    void fatal( const STRING & reason = STRING() );
    
    /** output verbose information
     *  @param string
     */
    void note( const STRING & str );

    /** output verbose information
     *  @param variable argument list, printf syntax
     */
    void note( const char *fmt, ... );

    /** return the test result as result enum
     */
    result get_result( ) const;

    /** return the the internal state structure
     *  containing the test results
     */
    const info & get_info( ) const;

protected:
    
    // add a string to the log
    void add_to_log( const STRING & str );

    // the internal state structure
    info m_info;

    // disable copy constructors
    logger( const logger & );
    
}; // class logger

}; // namespace util
}; // namespace sycl_cts