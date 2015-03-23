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
#include <iostream>
#include <assert.h>
#include <cstdio>

#include "printer.h"
#include "logger.h"

namespace sycl_cts {
namespace util {

/** standard output channel
 */
class stdout_channel : public printer::channel {
  MUTEX m_outputMutex;

 public:
  virtual void write(const STRING &msg) {
    if (!msg.empty()) {
      LOCK_GUARD<MUTEX> lock(m_outputMutex);
      std::cout << msg;
    }
  }

  virtual void writeln(const STRING &msg) {
    if (!msg.empty()) {
      LOCK_GUARD<MUTEX> lock(m_outputMutex);
      std::cout << msg << std::endl;
    }
  }

  virtual void flush() { fflush(stdout); }
};

/** file output channel
 */
class file_channel : public printer::channel {
  MUTEX m_outputMutex;
  FILE *m_file;

 public:
  file_channel() : m_outputMutex(), m_file(nullptr) {}

  ~file_channel() { close(); }

  bool open(const char *path) {
    LOCK_GUARD<MUTEX> lock(m_outputMutex);
    if (m_file != nullptr) fclose(m_file);
    m_file = nullptr;
    m_file = fopen(path, "wb");
    return m_file != nullptr;
  }

  void close() {
    if (m_file != nullptr) {
      LOCK_GUARD<MUTEX> lock(m_outputMutex);
      fflush(m_file);
      fclose(m_file);
      m_file = nullptr;
    }
  }

  STRING sanitize(const STRING &in) {
    STRING t = STRING(in);
    for (uint32_t i = 0; i < t.length(); i++) {
      char &ch = t[i];
      if (ch < 0x20 || ch >= 0x7f) ch = '.';
    }
    return t;
  }

  virtual void write(const STRING &msg) {
    STRING t = sanitize(msg);
    if (t.empty() || m_file == nullptr)
      return;
    else {
      LOCK_GUARD<MUTEX> lock(m_outputMutex);
      fputs(t.c_str(), m_file);
      fflush(m_file);
    }
  }

  virtual void writeln(const STRING &msg) {
    STRING t = sanitize(msg);
    if (t.empty() || m_file == nullptr)
      return;
    else {
      LOCK_GUARD<MUTEX> lock(m_outputMutex);
      fputs(t.c_str(), m_file);
      fputs("\n", m_file);
      fflush(m_file);
    }
  }

  virtual void flush() {
    if (m_file != nullptr) fflush(m_file);
  }
};

/** JSON printer
 */
class json_formatter : public printer::formatter {
 public:
  virtual void write(printer::channel &out, int32_t id, printer::epacket packet,
                     const STRING &data) {
    STRING strId = std::to_string(id);
    STRING strPacket = std::to_string(packet);
    out.writeln("{\"id\":" + strId + ",\"type\":" + strPacket + ",\"data\":\"" +
                data + "\"}");
  }

  virtual void write(printer::channel &out, int32_t id, printer::epacket packet,
                     int data) {
    STRING strId = std::to_string(id);
    STRING strPacket = std::to_string(packet);
    STRING strData = std::to_string(data);
    out.writeln("{\"id\":" + strId + ",\"type\":" + strPacket + ",\"data\":\"" +
                strData + "\"}");
  }
};

/** human readable text printer
 */
class text_formatter : public printer::formatter {
 public:
  virtual void write(printer::channel &out, int32_t, printer::epacket packet,
                     const STRING &data) {
    switch (packet) {
      default:
        // ignore packets we dont know
        return;
      case (printer::name):
        out.write("--- ");
        break;
      case (printer::line):
        out.write("  . line: ");
        break;
      case (printer::note):
      case (printer::list_test_name):
        out.write("  . ");
        break;
      case (printer::list_test_count):
        out.writeln(data + " tests in executable");
        return;
    }
    out.writeln(data);
  }

  virtual void write(printer::channel &out, int32_t id, printer::epacket packet,
                     int data) {
    switch (packet) {
      case (printer::result): {
        switch (data) {
          case (logger::epass):
            out.writeln("  - pass\n");
            break;
          case (logger::efail):
            out.writeln("  - fail\n");
            break;
          case (logger::eskip):
            out.writeln("  - skip\n");
            break;
          case (logger::efatal):
            out.writeln("  - fatal\n");
            break;
        }
      }
        return;
      case (printer::progress):
        out.write("\r  . progress " + std::to_string(data) + "%");
        if (data == 100) out.write("\n");
        return;
      default:
        // stringify and pass to string handler
        write(out, id, packet, std::to_string(data));
    }
  }
};

/** local static variables
 */
namespace {
stdout_channel gStdoutChannel;
file_channel gFileChannel;
json_formatter gJsonFormat;
text_formatter gTextFormat;
};

/** constructor
 */
printer::printer()
    : m_nextLogId(), m_formatter(&gTextFormat), m_channel(&gStdoutChannel) {}

/** destructor
 */
printer::~printer() { release(); }

/** generate a new unique identifier
 */
int32_t printer::new_log_id() {
  int32_t newId = m_nextLogId.fetch_add(1);
  return newId;
}

/** set the current output format for the printer
 */
void printer::set_format(printer::eformat fmt) {
  switch (fmt) {
    case (printer::eformat::ejson):
      m_formatter = &gJsonFormat;
      break;
    case (printer::eformat::etext):
      m_formatter = &gTextFormat;
      break;
    default:
      assert(!"Unknown printer format");
  }
}

/** redirect the printer to write to a file
 */
bool printer::set_file_channel(const char *path) {
  if (!gFileChannel.open(path)) return false;
  m_channel = &gFileChannel;
  return true;
}

/** write a packet using the set formatter and channel
 */
void printer::write(int32_t id, epacket packet, STRING data) {
  if (m_formatter) m_formatter->write(*m_channel, id, packet, data);
}

/** write a packet using the set formatter and channel
 */
void printer::write(int32_t id, epacket packet, int data) {
  if (m_formatter) m_formatter->write(*m_channel, id, packet, data);
}

/** global printf
 */
void printer::print(const char *fmt, ...) {
  assert(fmt != nullptr);

  char buffer[1024];

  va_list args;
  va_start(args, fmt);
  if (vsnprintf(buffer, sizeof(buffer), fmt, args) <= 0)
    assert(!"vsnprintf() failed");
  va_end(args);

  // enforce terminal character
  buffer[sizeof(buffer) - 1] = '\0';

  // output string via channel
  if (m_channel) m_channel->write(STRING(buffer));
}

/** global print
 */
void printer::print(const STRING &str) {
  if (m_channel) m_channel->write(str);
}

/** finish all writing operations
 */
void printer::finish() {
  if (m_channel) m_channel->flush();
}

}  // namespace util
}  // namespace sycl_cts
