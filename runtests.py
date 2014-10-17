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
g_results       = []
g_headers       = {}
g_list_tests    = False

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# exit with an error message
#
def error_exit( string ):
    print string
    exit( )

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# display a summary of all test executed
#
def print_summary( ):
    global g_results
    global g_headers

    l_results = [0,0,0,0]
    l_total = 0

    l_fails = []

    try:
        # loop threw each packet
        for packet in g_results:

            # increment total test executed
            l_total += 1

            if 'result' in packet:
                # find the test result
                l_result = int( packet[ 'result' ])

                if 'id' in packet:
                    l_id = packet[ 'id' ]
                    if l_id in g_headers:
                        l_header = g_headers[ l_id ]

                # check for a fail
                if l_result is 2 and l_header:
                    l_fails.append( l_header )

                # increment this result type
                l_results[ l_result ] += 1

            else:
                print "Test result missing result field"

    except Exception as e:
        print "Error constructing summary"

    # print the summary
    print str( l_total ) + ' tests ran in total'
    print ' - passed : ' + str( l_results[1] )
    print ' - failed : ' + str( l_results[2] )
    for x in l_fails:
        if 'name' in x:
            print '   + ' + x[ 'name' ]
    print ' - skipped: ' + str( l_results[3] )

    # calculate conformance percentage
    l_score = 0
    if l_total > 0:
        l_score = ((l_results[1] * 100) / l_total)
    print ' = ' + str( l_score ) + '% conformance'

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# store a received json packet
#
def store_result( object ):
    global g_results

    # g_packets must be a valid list
    if not g_results:
        g_results = []

    try:
        g_results.append( object )

    except Exception as e:
        print "error storing JSON object"

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# store a test header packet
#
def store_header( object ):
    global g_headers

    if not g_headers:
        g_headers = {}

    # add this item to the header dictionary
    if 'id' in object:
        l_id = object['id']
        g_headers[ l_id ] = object

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# print a test header
#
def print_test_header( object ):

    # print out the test name
    if 'name' in object:
        print '## ' + object[ 'name' ]

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# print test body
#
def print_test_body( object ):
    global g_headers

    # check header has been received
    if 'id' in object:
        l_id = object['id']
        if l_id in g_headers:
            l_header = g_headers[l_id]
        else:
            print ' !  error:  test body received before header'

    # if there are any notes
    if 'notes' in object:
        for x in object[ 'notes' ]:
            print " ?   note: " + x

    # print out the result
    if 'result' in object:
        l_result = object[ 'result' ]
        l_res_to_string = [ 'pending', 'pass', 'fail', 'skip' ]
        print ' + result: ' +  l_res_to_string[ l_result ]

        # print line number of failure
        if (l_result is 2) and ('line' in object):

            # if we could match this body with its header
            if l_header:

                # print the test source path
                if 'file' in l_header:
                    print ' !   file: ' + l_header[ 'file' ]

                # print the test build time
                if ('buildDate' in l_header) and ('buildTime' in l_header):
                    print ' !  built: ' + l_header[ 'buildDate' ] + ", " + l_header[ 'buildTime' ]

            # print the line number of the fail
            print " !   line: " + str( object[ 'line' ] )

    print ""

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# process a fully formed JSON object
#
def process_json( object ):

    if 'type' in object:

        l_type = object[ 'type' ]

        # packet is a header
        if l_type is 1:
            store_header( object )
            print_test_header( object )

        # packet is a test transcript
        if l_type is 2:
            store_result( object )
            print_test_body( object )

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
def launch_cts_binary( ):
    global g_binary_path
    global g_csv_path

    if not g_binary_path:
        error_exit( "path to cts binary unknown" )

    # construct argument
    l_args = " --json"
    if g_csv_path:
        l_args = l_args + " --csv " + g_csv_path

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
    global g_binary_path

    # construct argument
    l_args = " --json --list"



    print "listing tests..."

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

    # are we in test list mode
    if g_list_tests:

        # list all of the tests
        list_tests( )

    else:
        # launch test suite
        if not launch_cts_binary( ):
            return

        # print a summary of the test
        print_summary( )

    return

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#
if __name__ == '__main__':
    main( )
