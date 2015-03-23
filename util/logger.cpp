/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdarg.h>
#include <assert.h>
#include <cstdio>

#include "logger.h"
#include "printer.h"

namespace sycl_cts {
namespace util {

/** constructor
 */
logger::logger() : m_result(result::epass) {
  /* get a new unique log identifier */
  m_logId = get<printer>().new_log_id();
}

/** test preamble
 */
void logger::preamble(const test_base::info &info) {
  get<printer>().write(m_logId, printer::epacket::name, info.m_name);
  get<printer>().write(m_logId, printer::epacket::file, info.m_file);
}

/** notify a test has failed
 *  @param reason, optional descriptive string for fail
 *  @param line, test line number that threw the error
 */
void logger::fail(const STRING &str, const int line) {
  m_result = logger::efail;
  get<printer>().write(m_logId, printer::epacket::line, line);
  get<printer>().write(m_logId, printer::epacket::note, str);
}

/** notify a test has been skipped
 *  @param reason, optional descriptive string for skip
 */
void logger::skip(const STRING &str) {
  m_result = logger::eskip;
  get<printer>().write(m_logId, printer::epacket::note, str);
}

/** report fatal error and abort program
 *  @param reason, optional descriptive string for fatal error
 */
void logger::fatal(const STRING &str) {
  m_result = logger::efatal;
  get<printer>().write(m_logId, printer::epacket::note, str);
}

/** output verbose information
 *  @param string
 */
void logger::note(const STRING &str) {
  // output via the printer
  get<printer>().write(m_logId, printer::epacket::note, str);
}

/** beginning of a test
 */
void logger::test_start() {
  get<printer>().write(m_logId, printer::epacket::test_start, 0);
}

/** tests end
 */
void logger::test_end() {
  get<printer>().write(m_logId, printer::epacket::result, m_result);
  get<printer>().write(m_logId, printer::epacket::test_end, 0);
}

/** output verbose information
 *  @param variable argument list, printf syntax
 */
void logger::note(const char *fmt, ...) {
  assert(fmt != nullptr);

  char buffer[1024];

  va_list args;
  va_start(args, fmt);
  if (vsnprintf(buffer, sizeof(buffer), fmt, args) <= 0)
    assert(!"vsnprintf failed");
  va_end(args);

  // enforce terminal character
  buffer[sizeof(buffer) - 1] = '\0';

  // output via the printer
  get<printer>().write(m_logId, printer::epacket::note, STRING(buffer));
}

/** send a progress report
 */
void logger::progress(int item, int total) {
  int percent = (total > 0) ? ((item * 100) / total) : 0;
  get<printer>().write(m_logId, printer::epacket::progress, percent);
}

/** return true if the log has been marked as fail
    */
bool logger::has_failed() {
  return (m_result == efail) || (m_result == efatal);
}

/** destructor
 *  will terminate log output appropriately
 */
logger::~logger() {}

/** has a test result been emitted
 *  @return, false = result not yet specified
 */
logger::result logger::get_result() const { return m_result; }

}  // namespace util
}  // namespace sycl_cts
