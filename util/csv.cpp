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

#include "csv.h"

namespace sycl_cts {
namespace util {

/** constructor
 */
csv::csv() : m_error(), m_items(), m_rowIndex() {}

/** destructor
 */
csv::~csv() { release(); }

/** load a CSV file from disk
 */
bool csv::load_file(const std::string &path) {
  // load the raw file
  std::ifstream stream(path, std::ios::in | std::ios::binary);
  if (!stream.is_open()) {
    m_error = "unable to open file";
    return false;
  }

  // start parsing a new row
  m_rowIndex.push_back(0);

  // make a buffer to pull together each entry
  int32_t index = 0;
  char buffer[64];

  // parse all characters in the input file
  char ch = '\0';
  for (;;) {
    // read a character from the stream
    stream.read(&ch, sizeof(ch));
    if (!stream.good()) break;

    switch (ch) {
      // skip over these characters
      case (' '):
      case ('\r'):
        continue;

      // new line start of new row
      case ('\n'): {
        // null terminate the buffer string
        buffer[index] = '\0';
        // add string to items list
        m_items.push_back(std::string(buffer));
        index = 0;

        // index of next item marks start of new row
        int32_t items = int32_t(m_items.size());
        m_rowIndex.push_back(items);
      } break;

      // marks next entry in a column
      case (','): {
        // null terminate the buffer string
        buffer[index] = '\0';
        // add string to items list
        m_items.push_back(std::string(buffer));
        index = 0;
      } break;

      // add new character to the buffer
      default:
        // check for buffer overflow
        if (index >= int32_t(sizeof(buffer) - 1u)) {
          m_error = "value exceeds buffer length";
          return false;
        } else
          buffer[index++] = ch;

    };  // switch
  };    // while

  // push any remaining item
  if (index > 0) {
    // null terminate the buffer string
    buffer[index] = '\0';
    // add string to items list
    m_items.push_back(std::string(buffer));
  }

  return true;
}

/**
 */
void csv::release() {
  m_items.clear();
  m_rowIndex.clear();
}

/**
 */
bool csv::get_item(int32_t row, int32_t column, std::string &out) {
  out = std::string("");
  const int32_t nRowIndices = int32_t(m_rowIndex.size());
  const int32_t nItems = int32_t(m_items.size());

  // test if row is valid
  if (row < 0 || row >= nRowIndices) {
    m_error = "row index out of bounds";
    return false;
  }

  // find the location of the requested element
  int32_t index = m_rowIndex.at(size_t(row)) + column;

  // find the index of the end of this row
  int32_t limit = nItems - 1;
  if ((row + 1) < nRowIndices) limit = m_rowIndex.at(size_t(row + 1)) - 1;

  // test the index doesn't pass end of the row
  if (index < 0 || index > limit) {
    m_error = "column index out of bounds";
    return false;
  }

  // output the item asked for
  assert(index < nItems && index >= 0);
  out = m_items.at(size_t(index));

  // success
  return true;
}

/** return the number of csv rows (line)
 */
int32_t csv::size() { return int32_t(m_rowIndex.size()); }

/** return the last error message set by a csv object
 */
bool csv::get_last_error(std::string &out) {
  out = m_error;
  return !m_error.empty();
}

}  // namespace util
}  // namespace sycl_cts
