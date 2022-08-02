/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_PRINTER_H
#define __SYCLCTS_UTIL_PRINTER_H

#include "stl.h"
#include "singleton.h"
#include "test_base.h"

namespace sycl_cts {
namespace util {

/** printer class
 *  this class handles the output from the logger class
 */
class printer : public singleton<printer> {
 public:
  enum eformat { ejson = 0, etext };

  enum epacket {
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
  class channel {
   public:
    virtual ~channel() {}

    /* output string over channel */
    virtual void write(const std::string &msg) = 0;

    /* output string with newline */
    virtual void writeln(const std::string &msg) = 0;

    /* flush the output channel */
    virtual void flush() = 0;
  };

  /** formats a packet of information
   */
  class formatter {
   public:
    virtual ~formatter() {}

    /* print a packet */
    virtual void write(channel &out, int32_t id, epacket packet,
                       const std::string &data) = 0;

    /* print a packet */
    virtual void write(channel &out, int32_t id, epacket packet, int data) = 0;
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
  void set_format(eformat fmt);

  /** redirect the printer to write to a file
   */
  bool set_file_channel(const char *m_path);

  /** write a packet to the printer
   */
  void write(int32_t id, epacket packet, std::string data);

  /** write a packet to the printer
   */
  void write(int32_t id, epacket packet, int data);

  /** instruct the printer to finish all printing
   *  operations. importantly, this terminates the root JSON object
   */
  virtual void finish();

  /** global stdout printing functions
   */
  void print(const char *fstr, ...);
  void print(const std::string &str);

 protected:
  // next log id to be issued from new_log_id()
  std::atomic_int m_nextLogId;

  // the packet formatter to use
  formatter *m_formatter;

  // the output channel to use
  channel *m_channel;
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_PRINTER_H