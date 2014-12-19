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
#include "test_base.h"

namespace sycl_cts
{
namespace util
{

/** printer class
 *  this class handles the output from the logger class
 */
class printer : public singleton<printer>
{
public:
    enum eformat
    {
        ejson = 0,
        etext
    };

    enum epacket
    {
        /* test attributes */
        name = 0,
        file,
        line,
        date,
        progress,
        note,
        result,

        /* test execution */
        test_start,
        test_end,

        /* test listing */
        list_test_name,
        list_test_count,
    };

    /** a string output channel
     */
    class channel
    {
    public:
        virtual ~channel(){}

        /* output string over channel */
        virtual void write( const STRING &msg ) = 0;

        /* output string with newline */
        virtual void writeln( const STRING &msg ) = 0;

        /* flush the output channel */
        virtual void flush() = 0;
    };

    /** formats a packet of information
     */
    class formatter
    {
    public:
        virtual ~formatter()
        {
        }

        /* print a packet */
        virtual void write( channel &out, int32_t id, epacket packet, const STRING &data ) = 0;

        /* print a packet */
        virtual void write( channel &out, int32_t id, epacket packet, int data ) = 0;
    };

    /** ask the printer to generate a new log id so that
     *  log headers and footers can be matched up
     */
    int32_t new_log_id();

    /** destructor
     */
    printer();

    /** destructor
     */
    virtual ~printer();

    /** set the output format
     */
    void set_format( eformat fmt );

    /** redirect the printer to write to a file
     */
    bool set_file_channel( const char *m_path );

    /** write a packet to the printer
     */
    void write( int32_t id, epacket packet, STRING data );

    /** write a packet to the printer
     */
    void write( int32_t id, epacket packet, int data );

    /** instruct the printer to finish all printing
     *  operations. importantly, this terminates the root JSON object
     */
    virtual void finish();

    /** global stdout printing functions
     */
    void print( const char *fstr, ... );
    void print( const STRING &str );

protected:
    // next log id to be issued from new_log_id()
    ATOMIC_INT m_nextLogId;

    // the packet formatter to use
    formatter *m_formatter;

    // the output channel to use
    channel *m_channel;
};

}  // namespace util
}  // namespace sycl_cts
