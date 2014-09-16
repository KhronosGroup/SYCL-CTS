/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include "logger.h"

// conformance test suite namespace
namespace sycl_cts
{
namespace util
{

/** Base class for all SYCL tests
 */
class test_base
{
public:

    /** encapsulate information about a test
     */
    struct info
    {
        std::string m_name;
        std::string m_file;
        std::string m_buildDate;
        std::string m_buildTime;
    };

    /** virtual destructor
     */
    virtual ~test_base( )
    {
        /* call cleanup to ensure internals are released */
        cleanup( );
    }

    /** return information about this test
     *  @param info, test_base::info structure as output
     */
    virtual void get_info( test_base::info & out ) const = 0;

    /** called before this test is executed
     *  @param log for emitting test notes and results
     */
    virtual bool setup( logger & log )
    {
        // stub
        return true;
    }

    /** execute this test
     *  @param log for emitting test notes and results
     */
    virtual void run( logger & log ) = 0;

    /** called after this test has executed
     *  @param log for emitting test notes and results
     */
    virtual void cleanup( )
    {
        // empty
    }

}; // class test_base

}; // namespace util
}; // namespace sycl_cts