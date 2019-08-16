#!/usr/bin/env python3
################################################################################
##
##  SYCL 1.2.1 Conformance Test Suite
##
##  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
##
################################################################################

import subprocess
import json
import time
import argparse
import tempfile
import os

__author__ = "Codeplay"

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# global variables
#
g_binary_path   = None
g_csv_path      = None
g_packets       = {}
g_list_tests    = False
g_junit_path    = None
g_platform      = "intel"
g_device        = "opencl_cpu"

g_types = \
    [
        "name"           ,
        "file"           ,
        "line"           ,
        "date"           ,
        "progress"       ,
        "note"           ,
        "result"         ,
        "test_start"     ,
        "test_end"       ,
        "list_test_name" ,
        "list_test_count",
    ]

g_results = \
    [
        "pending" ,
        "pass"    ,
        "fail"    ,
        "skip"    ,
        "fatal"   ,
        "timeout" ,
    ]

g_xml_escapes = \
    [
        # note '&' MUST be handled first!
        ( '&' , '&amp;' ),
        ( '>' , '&gt;'  ),
        ( '<' , '&lt;'  ),
        ( '\'', '&apos;'),
        ( '\"', '&quot;'),
    ]

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# exit with an error message
#
def error_exit( string ):
    print(string)
    exit( )

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# process with xml escape codes
#
def xml_escape( msg ):
    for e in g_xml_escapes:
        msg = msg.replace( e[0], e[1] )
    return msg

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
def write_junit_test_case( id ):

    l_name = find_packet_data( id, 'name' )[0]
    l_result = get_test_result( id )

    l_xml = "  <testcase name='" + l_name + "' time='0'"

    # test pass
    if l_result == 'pass':
        return l_xml + "/>\n"

    # test fail or skip
    if l_result == 'fail' or l_result == 'skip':
        l_xml += ">\n"

        l_note_list = find_packet_data( id, 'note' )
        l_stacktrace = ""
        for x in l_note_list:
            l_stacktrace += '. ' + x + '\n'

        l_stacktrace = xml_escape( l_stacktrace )
        if l_result == 'fail':

            l_file = find_packet_data( id, 'file' )
            if len( l_file ) > 0:
                l_stacktrace += "@file: " + l_file[0] + "\n"
            l_line = find_packet_data( id, 'line' )
            if len( l_line ) > 0:
                l_stacktrace += "@line: " + l_line[0] + "\n"

            l_xml += "    <failure message='error'>\n" + l_stacktrace \
                  +  "    </failure>\n"

        if l_result == 'skip':
            l_xml += "    <skipped>\n" + l_stacktrace \
                  +  "    </skipped>\n"

        return l_xml + "  </testcase>\n"

    # fatal or unknown result
    return l_xml + ">\n" \
                 + "    <failure message='fatal'>\nresult was " + l_result + "\n" \
                 + "    </failure>\n" \
                 + "  </testcase>\n"

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# produce a junit xml file
#
def write_junit_summary( ):
    global g_packets
    global g_junit_path
    global g_xml_escapes

    print('exporting JUNIT XML to: \'' + g_junit_path + '\'')

    try:
        l_xml  = "<?xml version='1.0' ?>\n"
        l_xml += "<testsuite" \
              +  " name='SYCL_CTS'" \
              +  ">\n"

        # itterate over all the tests
        for id in g_packets:

            l_xml += write_junit_test_case( id )

        l_xml += "</testsuite>\n"

        with open( g_junit_path, "w") as f:
            f.write( l_xml )

    except Exception as e:
        print("Exception thrown while writing junit output")
        print(str(e))
        pass

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
def find_packet_data( id, type ):

    l_list = []

    try:
        # find list of packet with this id
        l_packet = g_packets[ id ]
        # itterate over those packets
        for x in l_packet:
            # if packet is of correct type
            if x[ 'type' ] == type:
                # return the data field
                l_list.append( x[ 'data' ] )

    except Exception as e:
        pass

    return l_list

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# lookup the test result for a given test ID
#
def get_test_result( id ):

    try:
        #
        if not id in g_packets:
            return None

        # get list of packets with this id
        l_list = g_packets[id]

        for x in l_list:
            # check for result packet
            if x[ 'type' ] == 'result':
                l_data = x[ 'data' ]
                return g_results[ int( l_data ) ]

    except Exception as e:
        pass

    return None

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# print a final test summary
#
def print_summary( ):
    global g_packets

    l_total   = 0
    l_passes  = 0
    l_fails   = 0
    l_skipped = 0
    l_failing_tests = []

    try:
        # iterate over all the test ids
        for id in g_packets:

            # get the test result for this id
            l_result = get_test_result( id )

            if l_result == 'pass':
                l_passes += 1

            if l_result == 'fail':
                l_fails += 1
                l_failing_tests.append( find_packet_data( id, 'name' )[0] )

            if l_result == 'skip':
                l_skipped += 1

            if not (l_result is None):
                l_total += 1

        print(" " + str( l_total ) + ' tests ran in total')

        print('  - passed : ' + str( l_passes ))
        print('  - failed : ' + str( l_fails ))
        for x in l_failing_tests:
            print('    + ' + str( x ))
        print('  - skipped: ' + str( l_skipped ))

        if l_total == 0:
            l_percent = 0.0
        else:
            l_percent = (100 * l_passes) / l_total
        print('  = ' + str( l_percent ) + '% pass rate')

    except Exception as e:
        print('Exception thrown: ' + e.message)
        pass

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
def dispatch_packet( id, type, data ):

    if ( type is "test_start" ):
        print("--- " + find_packet_data( id, 'name' )[0])
        pass

    if ( type is "test_end" ):

        l_notes = find_packet_data( id, 'note' )
        l_result = g_results[ int( find_packet_data( id, 'result' )[0] ) ]

        # print all of the notes
        for x in l_notes:
            print("  . " + str( x ))

        # print extended test info for non passes
        if not l_result == 'pass':

            l_file = find_packet_data( id, 'file' )
            l_date = find_packet_data( id, 'date' )
            l_line = find_packet_data( id, 'line' )
            if l_file: print("  - " + l_file[0])
            if l_date: print("  - " + l_date[0])
            if l_line: print("  - line " + l_line[0])

        # print the result
        print("  - " + l_result)
        print(" ")

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# process a fully formed JSON object
#
def process_json( object ):
    global g_packets
    global g_types

    try:

        # packet must have an ID field
        if not ('id' in object):
            return
        l_id = int( object[ 'id' ] )

        # packet must have type field
        if not ('type' in object):
            return
        l_type = g_types[ int( object[ 'type' ]) ]
        object['type'] = l_type

        # packet must have data field
        if not ('data' in object):
            return
        l_data = object[ 'data' ]
        # convert to string
        if type( l_data ) is int:
            l_data = str( l_data )
            object['data'] = l_data

        # add this packet to the packet store
        if not (l_id in g_packets):
            g_packets[l_id] = []
        g_packets[ l_id ].append( object )

        # dispatch the packet we just received
        dispatch_packet( l_id, l_type, l_data )

    except Exception as e:
        pass

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# use brace matching to isolate json packets
#
def find_packet( in_buffer ):

    l_start = in_buffer.find( '{' )
    l_end   =-1
    l_scope = 1

    if l_start > -1:
        for x in range( l_start+1, len( in_buffer ) ):
            if in_buffer[x] == '{':
                l_scope += 1
            if in_buffer[x] == '}':
                l_scope -= 1
            if l_scope is 0:
                l_end = x+1
                break

    return l_start, l_end

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# dispatch a line of JSON
#
def dispatch_buffer( in_buffer ):

    # try to extract a single packet
    (l_start,l_end) = find_packet( in_buffer )
    if l_start is -1 or l_end is -1:
        return in_buffer
    l_packet  = in_buffer[ l_start : l_end ]
    # work out the remainder of the packet
    in_buffer = in_buffer[ l_end : ]

    # convert any file path '\' to unix '/'
    l_packet = l_packet.replace( "\\", "/")

    # convert json string
    try:
        l_json = json.loads( l_packet )

    except Exception as e:
        error_exit( "received malformed json input: " + l_packet )

    # pass the json object on for processing
    process_json( l_json )

    return in_buffer

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# generate a valid temporary file name
#
def get_temp_filename( ):
    (l_fd, l_name) = tempfile.mkstemp()
    os.close( l_fd )
    return l_name

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# launch cts test binary
#
def launch_cts_binary( list ):
    global g_binary_path
    global g_csv_path
    global g_platform
    global g_device

    if not g_binary_path:
        error_exit( "path to cts binary unknown" )

    # construct argument
    l_args = " --json"
    if g_csv_path:
        l_args = l_args + " --csv " + g_csv_path

    if g_platform:
        l_args += " --platform " + g_platform

    if g_device:
        l_args += " --device " + g_device

    # add list flag if listing
    if list:
        l_args += " --list"

    # generate a temporary file name
    intermediate_file = os.path.abspath( "log.txt" )
    # touch the file to make sure it exists
    output_fd = open( intermediate_file, "w" )
    output_fd.close()

    # generate command line for launching test binary
    l_command = g_binary_path + l_args + " --file " + intermediate_file

    # execute the cts test suite executable
    with open( get_temp_filename(), "w" ) as scrap_io:
        l_exe = subprocess.Popen( l_command, stdout=scrap_io, stderr=scrap_io, shell=True )

    # read the intermediate file
    with open( intermediate_file, "r" ) as l_stdOutLogR:

        l_buffer = ""

        # iterate over all lines
        while True:

            # read a line of input
            l_line = l_stdOutLogR.readline()
            # strip new line characters
            l_line = l_line.rstrip('\r\n')
            # append to the buffer
            l_buffer += l_line

            # while we are dispatching packets
            l_old_len = 0
            while l_old_len != len( l_buffer ):
                l_old_len = len( l_buffer )
                l_buffer = dispatch_buffer( l_buffer )

            # check if we are at the eof
            if l_line == '':
                # check for program exit
                l_exe.poll()
                if not l_exe.returncode is None:
                    # done reading
                    break

    # close the file handle
    if l_stdOutLogR:
        l_stdOutLogR.close()

    # make sure the exe has terminates
    l_exe.wait()
    scrap_io.close()

    return True

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# list the tests contained in a test executable
#
def list_tests( ):

    if not ( -1 in g_packets ):
        return
    l_list = g_packets[ -1 ]
    l_count = find_packet_data( -1, 'list_test_count' )

    print(l_count[0] + " tests in executable")

    for x in l_list:
        if x['type'] == 'list_test_name':
            print("  . " + x['data'])

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# parse command line arguments
#
def parse_args( ):
    global g_binary_path
    global g_csv_path
    global g_junit_path
    global g_list_tests
    global g_platform
    global g_device

    devices = ['host', 'opencl_cpu', 'opencl_gpu', 'accelerator']
    platforms = ['host', 'amd', 'arm', 'intel', 'nvidia']

    parser = argparse.ArgumentParser( description="Khronos SYCL CTS" )

    parser.add_argument( "-b", "--binpath", help="specify path to the cts executable file" )

    parser.add_argument( "--csvpath", help="specify path to csv file for filtering tests" )
    parser.add_argument( "--list", help="list all tests in a test binary", action="store_true" )
    parser.add_argument( "-j", "--junit", help="specify output path for a junit xml file" )
    parser.add_argument( "-p", "--platform", choices=platforms, help="The platform to run on " )
    parser.add_argument( "-d", "--device", choices=devices, help="The device to run on " )

    args = parser.parse_args()

    if 'binpath' in args:
        g_binary_path = args.binpath

    if not g_binary_path:
        error_exit( "'--binpath' option must be specified. try '--help'" )

    if 'csvpath' in args:
        g_csv_path = args.csvpath

    if 'list' in args:
        g_list_tests = args.list

    if 'junit' in args:
        g_junit_path = args.junit

    if 'platform' in args:
        g_platform = args.platform

    if 'device' in args:
        g_device = args.device

    return True

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# entry point
#
def main( ):
    global g_list_tests

    # parse command line arguments
    if not parse_args( ):
        return

    # launch test suite
    if not launch_cts_binary( g_list_tests ):
        return

    # are we in test list mode
    if g_list_tests:
        # list all of the tests
        list_tests( )
        return

    # print a summary of the test
    print_summary( )

    if g_junit_path:
        # print a junit summary
        write_junit_summary()

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
if __name__ == '__main__':
    main( )
