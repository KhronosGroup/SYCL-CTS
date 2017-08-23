#!/usr/bin/python
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#
#  SYCL Conformance Test Suite
#
#  Copyright: (c) 2017 by Codeplay Software LTD. All Rights Reserved.
#


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# run clang format over all files
#
import os
import subprocess

__author__ = "Codeplay"

gClangFormat = 'clang-format'

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# run clang format for the given file
#
def visit_file( afile ):
    # construct the command line argument
    l_cmd = gClangFormat + ' -style=file -i ' + afile
    print l_cmd
    # run clang format
    subprocess.call( l_cmd, shell=True )

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# collect all files in the current directory
#
def walk_dir( a_dirs, a_suffixs ):
    l_files = []
    # for all given directorys
    for l_dir in a_dirs:
        # walk every directory
        for ( l_dir_path, l_dir_names, l_file_names ) in os.walk( l_dir ):
            # convert path specifier
            l_dir_path = l_dir_path.replace( '\\', '/' )
            # itterate over each file
            for l_file_name in l_file_names:
                # check if file has suffix
                if l_file_name.endswith( a_suffixs ):
                    # add to list
                    l_files.append( l_dir_path + '/' + l_file_name )
    # return list of matches
    return l_files

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# program entry point
#
def main( ):

    # collect all files with a given suffix
    l_file_list = []
    l_file_list.extend( walk_dir( ['util', 'tests'], ( '.cpp', '.h', '.template' ) ))

    # itterate over all matches
    for l_name in l_file_list:
        try:
            # try to hash the file
            visit_file( l_name )

        except Exception as e:
            pass
    return

# call the entry point
main( )
