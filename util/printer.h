/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

#include <mutex>

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
        const test_base::info & testInfo );
    
    /** output a test log
     */
    void write(
        const logger::info & logInfo );

    /** instruct the printer to finish all printing
     *  operations. importantly, this terminates the root JSON object
     */
    void finish( );

protected:

    /** Convert a logger::result enum to a string
     */
    std::string result_as_string( logger::result res );

    /** output a string to stdout
     */
    void output( const std::string & str );

    /** output a string to stdout followed by a new line
     */
    void outputln( const std::string & str );

    /** output a key value pair in JSON style
     *  @param key, key to emit
     *  @param value, value to emit
     *  @param comma, true=postfix comma
     */
    void output_kvp(
        const std::string & key,
        const std::string & value,
        const bool comma );

    /** output a key value pair in JSON style
     *  @param key, key to emit
     *  @param value, value to emit
     *  @param comma, true=postfix comma
     */
    void output_kvp(
        const std::string & key,
        const int & value,
        const bool comma );

    /** announce the start of a test
     *  
     */
    void write_json( 
        const test_base::info & testInfo );

    /** output a test log in JSON form
     *  
     */
    void write_json(
        const logger::info & logInfo );

    /** announce the start of a test
     *  
     */
    void write_text( 
        const test_base::info & testInfo );
    
    /** output the results of a test
     *  @param log, the log object to output
     */
    void write_text(
        const logger::info & logInfo );

    // mutex for write operations so that two tests
    // wont be written to stdout at the same time
    std::mutex m_outputMutex;

    // the output format
    eformat m_format;
    
};

}; // namespace util
}; // namespace sycl_cts