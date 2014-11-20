/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include "stl.h"
#include "test_base.h"

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
        epass,
        efail,
        eskip,
        efatal,
        etimeout,
    };

    /** constructor
     */
    logger();

    /** destructor
     */
    ~logger();

    /* emit a test preamble
     */
    void preamble( const struct test_base::info &testInfo );

    /** notify a test has failed
     *  @param reason, optional descriptive string for fail
     *  @param line, line number in test that reports the failure
     */
    void fail( const STRING &reason, const int line );

    /** notify a test has been skipped
     *  @param reason, optional descriptive string for skip
     */
    void skip( const STRING &reason = STRING() );

    /** report fatal error and abort program
     *  @param reason, optional descriptive string for fatal error
     */
    void fatal( const STRING &reason = STRING() );

    /** output verbose information
     *  @param string
     */
    void note( const STRING &str );

    /** output verbose information
     *  @param variable argument list, printf syntax
     */
    void note( const char *fmt, ... );

    /** beginning of a test
     */
    void test_start();

    /** end of a test
     */
    void test_end();

    /** send a progress update
     *
     *  sent as 'items' done of 'total'
     */
    void progress( int item, int total );

    /** return the test result as result enum
     */
    result get_result() const;

protected:
    // unique log identifier
    int32_t m_logId;

    // test result
    result m_result;

    // disable copy constructors
    logger( const logger & );

};  // class logger

};  // namespace util
};  // namespace sycl_cts
