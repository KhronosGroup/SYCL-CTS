/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include "stl.h"
#include "singleton.h"
#include "logger.h"
#include "test_base.h"

namespace sycl_cts
{
namespace util
{

/** printer class
 *  this class handles the output from the logger class
 */
class printer
    : public singleton<printer>
{
public:

    enum eformat
    {
        ejson = 0,
        etext
    };
    
    enum epacket
    {
        // details for a test which is about to run
        header     = 0,
        // transcript of a test that has run
        transcript ,
    };

    /**
     */
    typedef int logid;

    /** ask the printer to generate a new log id so that
     *  log headers and footers can be matched up
     */
    logid new_log_id( );

    /** constructor
     */
    printer( );

    /** destructor
     */
    virtual ~printer( );

    /** set the output format
     */
    void set_format( eformat fmt );
    
    /** output a tests info header
     */
    void write( 
        printer::logid id,
        const test_base::info & testInfo );
    
    /** output a test log
     */
    void write(
        printer::logid id,
        const logger::info & logInfo );

    /** instruct the printer to finish all printing
     *  operations. importantly, this terminates the root JSON object
     */
    void finish( );

protected:

    /** Convert a logger::result enum to a string
     */
    STRING result_as_string( logger::result res );

    /** output a string to stdout
     */
    void output( const STRING & str );

    /** output a string to stdout followed by a new line
     */
    void outputln( const STRING & str );

    /** output a key value pair in JSON style
     *  @param key, key to emit
     *  @param value, value to emit
     *  @param comma, true=postfix comma
     */
    void output_kvp(
        const STRING & key,
        const STRING & value,
        const bool comma );

    /** output a key value pair in JSON style
     *  @param key, key to emit
     *  @param value, value to emit
     *  @param comma, true=postfix comma
     */
    void output_kvp(
        const STRING & key,
        const int & value,
        const bool comma );

    /** announce the start of a test
     *  @param id, identifier to match up test headers and transcripts
     *  @param testInfo, details of the test about the be executed
     */
    void write_json( 
        printer::logid id,
        const test_base::info & testInfo );

    /** output a test log in JSON form
     *  @param id, identifier to match up test headers and transcripts
     *  @param logInfo, a test execution transcript
     */
    void write_json(
        printer::logid id,
        const logger::info & logInfo );

    /** announce the start of a test
     *  @param id, identifier to match up test headers and transcripts
     *  @param testInfo, details of the test about the be executed
     */
    void write_text( 
        printer::logid id,
        const test_base::info & testInfo );
    
    /** output the results of a test
     *  @param id, identifier to match up test headers and transcripts
     *  @param logInfo, a test execution transcript
     */
    void write_text(
        printer::logid id,
        const logger::info & logInfo );

    // mutex for write operations so that two tests
    // wont be written to stdout at the same time
    MUTEX m_outputMutex;

    // the output format
    eformat m_format;
    
    // mutex for issuing log ids
    MUTEX m_logIdMutex;

    // next log id to be issued from new_log_id()
    volatile logid m_nextLogId;

};

}; // namespace util
}; // namespace sycl_cts