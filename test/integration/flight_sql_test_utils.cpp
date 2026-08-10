/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flight_sql_test_utils.hpp"

#include <boost/asio/buffer.hpp>
#include <boost/asio/connect_pipe.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/readable_pipe.hpp>
#include <boost/asio/writable_pipe.hpp>
#include <boost/asio/write.hpp>
#include <boost/process/v2.hpp>

#include <dirent.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <format>

namespace gqe_test {

int find_free_port()
{
  scoped_socket sock{socket(AF_INET, SOCK_STREAM, 0)};
  if (sock.fd < 0) { return -1; }
  struct sockaddr_in addr = {};
  addr.sin_family         = AF_INET;
  addr.sin_addr.s_addr    = htonl(INADDR_LOOPBACK);
  addr.sin_port           = 0;
  bind(sock.fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
  socklen_t len = sizeof(addr);
  getsockname(sock.fd, reinterpret_cast<struct sockaddr*>(&addr), &len);
  return ntohs(addr.sin_port);
}

bool wait_for_port(int port, int timeout_seconds)
{
  for (int attempt = 0; attempt < timeout_seconds * 2; ++attempt) {
    usleep(500'000);
    scoped_socket sock{socket(AF_INET, SOCK_STREAM, 0)};
    struct sockaddr_in addr = {};
    addr.sin_family         = AF_INET;
    addr.sin_addr.s_addr    = htonl(INADDR_LOOPBACK);
    addr.sin_port           = htons(port);
    if (connect(sock.fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0) {
      return true;
    }
  }
  return false;
}

std::pair<int, std::string> run_command_with_output(std::string_view cmd)
{
  namespace bp   = boost::process::v2;
  namespace asio = boost::asio;

  asio::io_context ctx;
  asio::readable_pipe out{ctx};

  // Run through the shell so command strings with pipes/redirects still work, and merge
  // stderr into stdout as the previous popen(cmd + " 2>&1", "r") did.
  auto const args = std::to_array<std::string>({"-c", std::format("{} 2>&1", cmd)});
  bp::process proc(ctx, "/bin/sh", args, bp::process_stdio{.out = out});

  boost::system::error_code ec;
  std::string output;
  asio::read(out, asio::dynamic_buffer(output), ec);  // reads until EOF
  return {proc.wait(), output};
}

std::vector<pid_t> find_child_pids(pid_t parent)
{
  std::vector<pid_t> children;
  DIR* dir = opendir("/proc");
  if (!dir) return children;
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type != DT_DIR) continue;
    char* end;
    long pid = strtol(entry->d_name, &end, 10);
    if (*end != '\0' || pid <= 0) continue;
    auto stat_path = std::format("/proc/{}/stat", pid);
    FILE* f        = fopen(stat_path.c_str(), "r");
    if (!f) continue;
    int child_pid, ppid;
    char comm[256], state;
    if (fscanf(f, "%d %255s %c %d", &child_pid, comm, &state, &ppid) == 4) {
      if (ppid == parent) { children.push_back(static_cast<pid_t>(child_pid)); }
    }
    fclose(f);
  }
  closedir(dir);
  return children;
}

std::pair<int, std::string> run_sql(std::string_view client_bin, int port, std::string_view sql)
{
  namespace bp   = boost::process::v2;
  namespace asio = boost::asio;

  // Pipe the SQL to the client's stdin directly, without a shell. Interpolating it into
  // `echo "..."` would let the shell strip the double quotes around quoted identifiers
  // (e.g. CREATE TABLE "MixedCase"), silently turning the test into an unquoted-identifier
  // test. Piping preserves the statement verbatim, special characters and all.
  asio::io_context ctx;
  asio::writable_pipe in{ctx};
  asio::readable_pipe out{ctx};
  asio::writable_pipe sink{ctx};

  boost::system::error_code ec;
  // Give the child's stdout and stderr the write end of a single pipe (equivalent to 2>&1)
  // so the returned string is the combined stream and CLI errors reach the assertions.
  asio::connect_pipe(out, sink, ec);

  auto const args = std::to_array<std::string>(
    {"--server-url", std::format("http://127.0.0.1:{}", port), "--sql-file", "-"});
  bp::process proc(
    ctx, std::string(client_bin), args, bp::process_stdio{.in = in, .out = sink, .err = sink});

  // Close the parent's copy of the write end so the read below sees EOF once the child exits.
  sink.close();

  // The client reads all of stdin before producing output, so writing the whole statement
  // and then reading the response cannot deadlock.
  asio::write(in, asio::buffer(sql.data(), sql.size()), ec);
  in.close();

  std::string output;
  asio::read(out, asio::dynamic_buffer(output), ec);  // reads until EOF

  int exit_code = proc.wait();
  return {exit_code, output};
}

std::string_view to_string(InputFormat mode)
{
  switch (mode) {
    case InputFormat::Sql: return "sql";
    case InputFormat::Physical: return "physical";
  }
  return "";  // unreachable; switch is exhaustive
}

std::pair<int, std::string> run_tpch_query(std::string_view run_script,
                                           InputFormat mode,
                                           int port,
                                           std::string_view queries_dir,
                                           int query_num,
                                           std::optional<std::string_view> ref_dir)
{
  auto cmd =
    std::format("{} --mode {} --server-url http://127.0.0.1:{}", run_script, to_string(mode), port);
  if (ref_dir.has_value()) { cmd += std::format(" --validate {}", *ref_dir); }
  cmd += std::format(" {} {}", queries_dir, query_num);
  return run_command_with_output(cmd);
}

int detect_gpu_count()
{
  auto const* visible = std::getenv("CUDA_VISIBLE_DEVICES");
  auto cmd =
    (visible && visible[0] != '\0')
      ? std::format("nvidia-smi --query-gpu=name --format=csv,noheader --id={} | wc -l", visible)
      : std::string("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l");
  auto [rc, output] = run_command_with_output(cmd);
  return (rc == 0) ? std::atoi(output.c_str()) : 0;
}

}  // namespace gqe_test
