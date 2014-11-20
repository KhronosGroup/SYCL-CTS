#!/usr/bin/python
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#  SYCL Conformance Test Suite
#
#  Copyright: (c) 2014 by Codeplay Software LTD. All Rights Reserved.
#

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
        "timeout"
    ]

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# exit with an error message
#
def error_exit( string ):
    print string
    exit( )

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
        # itterate over all the test ids
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

            l_total += 1

        print " " + str( l_total ) + ' tests ran in total'

        print '  - passed : ' + str( l_passes )
        print '  - failed : ' + str( l_fails )
        for x in l_failing_tests:
            print '    + ' + str( x )
        print '  - skipped: ' + str( l_skipped )

        l_percent = (100 * l_passes) / l_total
        print '  = ' + str( l_percent ) + '% pass rate'

    except Exception as e:
        pass

    return


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
def dispatch_packet( id, type, data ):

    if ( type is "test_start" ):
        print "--- " + find_packet_data( id, 'name' )[0]
        pass

    if ( type is "test_end" ):

        l_notes = find_packet_data( id, 'note' )
        l_result = g_results[ int( find_packet_data( id, 'result' )[0] ) ]

        # print all of the notes
        for x in l_notes:
            print "  . " + str( x )

        # print extended test info for non passes
        if not l_result == 'pass':

            l_file = find_packet_data( id, 'file' )
            l_date = find_packet_data( id, 'date' )
            l_line = find_packet_data( id, 'line' )
            if l_file: print "  - " + l_file[0]
            if l_date: print "  - " + l_date[0]
            if l_line: print "  - line " + l_line[0]

        # print the result
        print "  - " + l_result
        print " "

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
# dispatch a line of JSON
#
def dispatch_line( line ):

    # trim any garbage from the start of the packet
    l_ix = line.find( '{' )
    if l_ix > -1:
        line = line[ l_ix : ]

    # convert any file path '\' to unix '/'
    line = line.replace( "\\", "/")

    # simple check for json input
    if line[0] == '{':

        # strip new line characters
        line = line.rstrip('\r\n')

        # convert json string
        try:
            l_json = json.loads( line )

        except Exception as e:
            error_exit( "received malformed json input" )

        # pass the json object on for processing
        process_json( l_json )

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# launch cts test binary
#
def launch_cts_binary( list ):
    global g_binary_path
    global g_csv_path

    if not g_binary_path:
        error_exit( "path to cts binary unknown" )

    # construct argument
    l_args = " --json"
    if g_csv_path:
        l_args = l_args + " --csv " + g_csv_path

    # add list flag if listing
    if list:
        l_args += " --list"

    # generate a temporary file name
    (output_fd, temp_name) = tempfile.mkstemp()
    os.close( output_fd )

    # create a temporary
    with open( temp_name, "w" ) as l_stdOutLogW:

        # generate command
        l_command = g_binary_path + l_args

        # execute the cts test suite
        l_exe = subprocess.Popen( l_command, stdout=l_stdOutLogW, stderr=None, shell=True )

    # read the newly created temporary file
    with open( temp_name, "r" ) as l_stdOutLogR:

        # iterate over all lines
        while True:

            # read a line of input
            l_line = l_stdOutLogR.readline()

            # if we are at the eof
            if l_line == "":
                # check for program exit
                l_exe.poll()
                if not l_exe.returncode is None:
                    # done reading
                    break

            # pass this line to the dispatcher
            else:
                dispatch_line( l_line )

    # close the file handles
    if l_stdOutLogW:
        l_stdOutLogW.close()
    if l_stdOutLogR:
        l_stdOutLogR.close()

    # make sure the exe has terminates
    l_exe.wait()

    # delete the temporary file
    os.remove( temp_name )

    return True

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# list the tests contained in a test executable
#
def list_tests( ):

    if not ( -1 in g_packets ):
        return
    l_list = g_packets[ -1 ]

    l_count = find_packet_data( -1, 'list_test_count' )

    print l_count[0] + " tests in executable"

    for x in l_list:
        if x['type'] == 'list_test_name':
            print "  . " + x['data']

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# parse command line arguments
#
def parse_args( ):
    global g_binary_path
    global g_csv_path
    global g_list_tests

    parser = argparse.ArgumentParser( description="Khronos SYCL CTS" )

    parser.add_argument( "-b", "--binpath", help="path to the cts executable file" )

    parser.add_argument( "--csvpath", help="path to csv file for filtering tests" )
    parser.add_argument( "--list", help="list all tests in a test binary", action="store_true" )

    args = parser.parse_args()

    if 'binpath' in args:
        g_binary_path = args.binpath

    if not g_binary_path:
        error_exit( "'--binpath' option must be specified. try '--help'" )

    if 'csvpath' in args:
        g_csv_path = args.csvpath

    if 'list' in args:
        g_list_tests = args.list

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

    else:
        # print a summary of the test
        print_summary( )

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
if __name__ == '__main__':
    main( )
