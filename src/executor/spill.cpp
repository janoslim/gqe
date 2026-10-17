/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, flowcsd study. SPDX-License-Identifier: Apache-2.0
 *
 * flowcsd SSD spill mode: resident and spilled partitions, their GPU <-> SSD I/O, and the spill
 * tasks (see include/gqe/executor/spill.hpp for the task graph and the settings).
 *
 * I/O path: spill bytes move between GPU memory and the data SSDs only through cuFile (GDS) and
 * registered GPU staging buffers, never through host memory:
 *   write: an input task's rows of its written partitions -> the columns the join reads -> narrow
 *          integer columns -> one cudf::contiguous_split (one packed slice per partition) -> write
 *          staging buffer of the slice's drive -> cuFileWrite on a write thread
 *   read : cuFileRead on a read thread -> its read staging buffer -> partition buffer (query pool)
 *          -> cudf::unpack -> restore narrowed columns -> cudf::concatenate -> placeholders for the
 *          columns the join does not read
 * - Packing: one contiguous_split per input task and side packs all of the task's written
 *   partitions with one setup and one synchronization (v3 packed every slice with its own
 *   cudf::chunked_pack, about 0.3 ms each whatever the slice size). Destaging a resident partition
 *   (eviction) packs its resident slices as one slice.
 * - Files: one append-only file per (operator, side, drive). A write staging buffer collects
 *   consecutive slices of one file and becomes one write request at the file's end when it is full,
 *   when stage C starts reading that side, or when its drive needs a buffer and has none free; a
 *   slice that does not fit continues in the next buffer, contiguous in the file. Every slice
 *   starts 4 KiB-aligned (each one's end is padded), and the slice index keeps its true size.
 * - Narrowing (lossless): per input task, each integer column, and each DATE/TIMESTAMP column
 *   through its integer count of days (seconds, ...), whose (largest - smallest value) fits a
 *   narrower unsigned type is stored as (value - smallest) in that type; the slice index keeps the
 *   column's type, the subtracted value and the stored type, and the read adds the value back.
 *   Floating-point, string, decimal and other columns are written as they are.
 * - Placement: a slice goes to the drive expected to finish its writes first (bytes queued, in
 *   flight or staged on the drive, divided by the drive's write rate in this query), so a drive
 *   whose writes slow down gets less data; ties go to the drive holding fewer bytes of the operator
 *   side, which balances the read-back.
 * - Threads: per drive, write threads and read threads with a FIFO queue each, so stage-C reads
 *   never wait behind another operator's queued writes. Writes run behind the compute thread: it
 *   waits only while every write staging buffer of a drive is queued or in flight. Stage C reads
 *   the next two spilled partitions of a side ahead, on both drives at once.
 * - Staging buffers are registered with cuFileBufRegister: GDS writes from unregistered GPU memory
 *   were about 3x slower on the WD SN770 (0.42 vs 1.43 GiB/s; gdsio 1 MiB, 4 threads; study run
 *   20261017T073632Z-spill-gds-gate). libcudf's data_sink cannot write from a caller-registered
 *   buffer (its kvikio is private), so the spill files use cuFile directly.
 * - Request size and queue depth follow gdsio on both drives (study run
 *   20261017T090657Z-gqe-spill-io-sizing): 4 MiB reads with 4 in flight reached 3.27 (SN770) and
 *   3.32 (Netac) GiB/s, 16 MiB requests were not faster, and writes did best with about 2 in flight.
 */

#include <gqe/executor/spill.hpp>
#include <gqe/executor/spill_filter.hpp>
#include <gqe/executor/spill_narrowing.hpp>

#include <gqe/executor/eval.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

namespace gqe {

namespace {

// O_DIRECT and GDS need 4 KiB-aligned file offsets and sizes. Every slice starts at a multiple of
// this in its file and is padded to one; the slice index keeps the true packed size.
constexpr std::size_t file_alignment_bytes = 4096;
// Write staging, per drive: a buffer collects consecutive slices of one (operator side, drive) and
// becomes one write request. Two writes in flight per drive did best in gdsio; the other buffers
// let the compute thread pack on while they are written. All staging buffers live outside the
// query pool, so the pool (the memory budget) stays the size of the baseline mode's.
constexpr std::size_t write_buffer_bytes = std::size_t{8} << 20;
constexpr int write_buffers_per_drive    = 8;
constexpr int write_threads_per_drive    = 2;
// Reads: requests of at most 4 MiB, 4 in flight per drive; each read thread has its own staging
// buffer of that size.
constexpr std::size_t read_buffer_bytes = std::size_t{4} << 20;
constexpr int read_threads_per_drive    = 4;
// Spill files grow in preallocated steps of this size, so writes land inside the file size on
// allocated blocks: ext4 takes the inode lock exclusively for size-extending direct writes, which
// would serialize the writes in flight on one file.
constexpr std::size_t file_preallocation_bytes = std::size_t{1} << 30;
// One spilled side of a partition may hold at most 1/8 of the query pool once read back (its
// unnarrowed size with the placeholders): reading it takes about twice that (file buffer and
// restored columns, then the concatenated table), the next partitions are read ahead, and the join
// then holds both sides, its hash table and its output. Partitions are never split again: a larger
// one fails the query with "spill partition too large" as soon as its written slices exceed this.
constexpr std::size_t partition_limit_divisor = 8;
// Default resident budget: 1/4 of the query pool (spill_resident_budget_bytes in spill.hpp).
constexpr std::size_t resident_budget_divisor = 4;
// Stage C keeps up to this many spilled partitions of a side read ahead, while their buffers
// together stay within the partition limit.
constexpr std::size_t read_ahead_partitions = 2;

int64_t monotonic_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

std::size_t round_up_to_alignment(std::size_t bytes)
{
  return (bytes + file_alignment_bytes - 1) / file_alignment_bytes * file_alignment_bytes;
}

// An environment variable as an integer; nullopt when it is unset or empty.
std::optional<int64_t> read_int_env(char const* name)
{
  char const* value = std::getenv(name);
  if (value == nullptr || *value == '\0') { return std::nullopt; }
  return std::stoll(value);
}

std::vector<std::string> split_directories(std::string const& directory_list)
{
  std::vector<std::string> directories;
  std::stringstream list_stream(directory_list);
  std::string directory;
  while (std::getline(list_stream, directory, ':')) {
    if (!directory.empty()) { directories.push_back(directory); }
  }
  return directories;
}

// GQE_SPILL_OPERATOR_PARTITIONS as signature -> P. Entries are ';'-separated; a signature contains
// '=' itself (keys=<n>), so an entry splits at its last '='.
std::map<std::string, int32_t> parse_operator_partitions(std::string const& entry_list)
{
  std::map<std::string, int32_t> partitions;
  std::stringstream list_stream(entry_list);
  std::string entry;
  while (std::getline(list_stream, entry, ';')) {
    if (entry.empty()) { continue; }
    auto const separator = entry.rfind('=');
    if (separator == std::string::npos) {
      throw std::runtime_error("spill: GQE_SPILL_OPERATOR_PARTITIONS entry without '=': " + entry);
    }
    auto const count = std::stoi(entry.substr(separator + 1));
    if (count <= 0) {
      throw std::runtime_error("spill: GQE_SPILL_OPERATOR_PARTITIONS entry with P <= 0: " + entry);
    }
    partitions[entry.substr(0, separator)] = count;
  }
  return partitions;
}

/**
 * @brief Wall time during which at least one request of one kind was in flight.
 */
class busy_clock {
 public:
  void start()
  {
    std::lock_guard lock(_mutex);
    if (_in_flight++ == 0) { _busy_since_ns = monotonic_ns(); }
  }

  void stop()
  {
    std::lock_guard lock(_mutex);
    if (--_in_flight == 0) { _busy_ns += monotonic_ns() - _busy_since_ns; }
  }

  // Busy time so far, a busy interval still running included (placement reads it mid-query).
  int64_t busy_ns()
  {
    std::lock_guard lock(_mutex);
    if (_in_flight == 0) { return _busy_ns; }
    return _busy_ns + monotonic_ns() - _busy_since_ns;
  }

  void reset()
  {
    std::lock_guard lock(_mutex);
    _busy_ns = 0;
  }

 private:
  std::mutex _mutex;
  int _in_flight         = 0;
  int64_t _busy_since_ns = 0;
  int64_t _busy_ns       = 0;
};

// Spill I/O of one drive (spill directory) during the running query.
struct drive_counters {
  std::atomic<int64_t> write_bytes{0};
  std::atomic<int64_t> write_requests{0};
  std::atomic<int64_t> read_bytes{0};
  std::atomic<int64_t> read_requests{0};
  busy_clock writing;
  busy_clock reading;
};

// Counters of the running query for the `flowcsd_spill` log lines (one query runs at a time,
// node_manager.hpp `_execute_mutex`). Times are of the compute thread unless named busy.
struct spill_counters {
  std::atomic<int64_t> hybrid_joins{0};
  std::atomic<int64_t> hybrid_aggregations{0};
  std::atomic<int64_t> largest_partitions{0};     // largest P of the query's operators
  std::atomic<int64_t> spilled_operators{0};      // operators that wrote at least one slice
  std::atomic<int64_t> files{0};
  std::atomic<int64_t> resident_budget_bytes{0};  // budget of the query's operators
  std::atomic<int64_t> resident_bytes{0};         // held bytes kept in GPU memory, never written
  std::atomic<int64_t> partition_ns{0};       // evaluating keys and hash-partitioning
  std::atomic<int64_t> pack_ns{0};            // packing written slices, copying them to staging
  std::atomic<int64_t> narrow_write_ns{0};    // narrowing before packing
  std::atomic<int64_t> filter_ns{0};          // probe-side filters: building and applying
  std::atomic<int64_t> join_phase_ns{0};      // stage-C partitions with spill data
  std::atomic<int64_t> narrow_restore_ns{0};  // restoring narrowed columns after unpacking
  std::atomic<int64_t> exposed_write_ns{0};   // waiting for staging buffers or writes
  std::atomic<int64_t> exposed_read_ns{0};    // waiting for partition reads
  std::atomic<int64_t> narrowed_write_bytes{0};    // packed bytes of the written slices, no padding
  std::atomic<int64_t> unnarrowed_write_bytes{0};  // the same slices packed without narrowing
  busy_clock writing;  // any spill write in flight on any drive
  busy_clock reading;  // any spill read in flight on any drive
  busy_clock io;       // any spill write or read in flight on any drive
  std::vector<std::unique_ptr<drive_counters>> drives;  // one per spill directory
};

spill_settings const& settings() { return get_spill_settings(); }

spill_counters& counters()
{
  // The counters hold atomics and mutexes, so they are built in place; one drive per directory.
  static spill_counters instance;
  static std::once_flag drives_created;
  std::call_once(drives_created, [] {
    for (std::size_t drive = 0; drive < settings().directories.size(); drive++) {
      instance.drives.push_back(std::make_unique<drive_counters>());
    }
  });
  return instance;
}

// Serializes the spill-set bookkeeping and the write staging of the compute thread (the I/O
// threads never take it).
std::mutex& spill_mutex()
{
  static std::mutex instance;
  return instance;
}

// Resident held partitions of all operators of the running query, and the probe-side filters of
// its joins, share one budget. Guarded by spill_mutex(). `open_sets` are the operators whose k can
// still drop: their stage B (joins) or stage C (aggregations) has not started.
struct resident_registry {
  int64_t live_bytes = 0;  // held bytes in GPU memory, until the operator frees them
  std::vector<spill_set*> open_sets;
};

resident_registry& registry()
{
  static resident_registry instance;
  return instance;
}

void check_cufile(CUfileError_t status, std::string const& what)
{
  if (status.err == CU_FILE_SUCCESS) { return; }
  throw std::runtime_error("spill: " + what + " failed: " + cufileop_status_error(status.err));
}

// cuFileRead/cuFileWrite return -1 with errno for system errors and -(CUfileOpError) for cuFile
// errors (cufile.h:403,427).
std::string cufile_io_error(ssize_t result)
{
  if (result == -1) { return std::strerror(errno); }
  if (result < 0) { return cufileop_status_error(static_cast<CUfileOpError>(-result)); }
  return "short transfer of " + std::to_string(result) + " bytes";
}

// kvikio (inside libcudf) opens the cuFile driver on its own for the Parquet reads. Opening it
// here as well is harmless; it stays open until the process exits, as kvikio's does.
void open_cufile_driver()
{
  static std::once_flag driver_opened;
  std::call_once(driver_opened, [] {
    auto const status = cuFileDriverOpen();
    if (status.err != CU_FILE_DRIVER_ALREADY_OPEN) { check_cufile(status, "cuFileDriverOpen"); }
  });
}

/**
 * @brief Completion of a group of I/O requests: the writes of one operator side, or the reads of
 * one partition.
 */
class io_batch {
 public:
  void add()
  {
    std::lock_guard lock(_mutex);
    _pending++;
  }

  void finish(std::string const& error)
  {
    std::lock_guard lock(_mutex);
    if (!error.empty() && _error.empty()) { _error = error; }
    if (--_pending == 0) { _done.notify_all(); }
  }

  // Wait for every request and return the waiting time (ns); throws the first request's error.
  int64_t wait()
  {
    auto const start_ns = monotonic_ns();
    std::unique_lock lock(_mutex);
    _done.wait(lock, [this] { return _pending == 0; });
    if (!_error.empty()) { throw std::runtime_error(_error); }
    return monotonic_ns() - start_ns;
  }

  // Wait without reporting errors; for clean-up after a failure.
  void wait_quietly()
  {
    std::unique_lock lock(_mutex);
    _done.wait(lock, [this] { return _pending == 0; });
  }

 private:
  std::mutex _mutex;
  std::condition_variable _done;
  int _pending = 0;
  std::string _error;
};

// One write staging buffer written to its file.
struct write_request {
  CUfileHandle_t handle   = nullptr;
  std::size_t file_offset = 0;
  std::size_t bytes       = 0;
  int buffer              = -1;  // write staging buffer, numbered over all drives
  std::shared_ptr<io_batch> batch;
};

// One read into `destination` through the read thread's staging buffer.
struct read_request {
  CUfileHandle_t handle   = nullptr;
  std::size_t file_offset = 0;
  std::size_t bytes       = 0;
  uint8_t* destination    = nullptr;
  std::shared_ptr<io_batch> batch;
};

}  // namespace

/**
 * @brief One append-only spill file: the written slices of one operator side on one drive.
 *
 * A write staging buffer collects the file's next bytes and becomes one write request at the
 * file's end when it is full or flushed. Only the compute thread changes a file (holding
 * spill_mutex()); the I/O threads only use its cuFile handle.
 */
struct spill_file {
  int drive = 0;
  std::string path;
  int fd                      = -1;
  CUfileHandle_t handle       = nullptr;
  std::size_t written_bytes   = 0;   // bytes of the write requests submitted so far
  std::size_t allocated_bytes = 0;   // bytes preallocated with fallocate
  int open_buffer             = -1;  // write staging buffer holding the next request; -1: none
  std::size_t open_bytes      = 0;   // bytes in it, a multiple of 4 KiB between slices
  int64_t write_requests      = 0;
  std::shared_ptr<io_batch> writes;  // the side's submitted writes
};

/**
 * @brief Registered staging buffers and background I/O threads of the running query.
 *
 * Staging region, registered once: [write buffers, drive by drive | read buffers, one per read
 * thread, drive by drive].
 */
class spill_io {
 public:
  spill_io()
    : _device(rmm::get_current_cuda_device()),
      _num_drives(static_cast<int>(settings().directories.size())),
      _queued_write_bytes(std::make_unique<std::atomic<int64_t>[]>(_num_drives)),
      _open_bytes(_num_drives, 0),
      _open_files(_num_drives),
      _free_write_buffers(_num_drives),
      _write_queues(_num_drives),
      _read_queues(_num_drives)
  {
    open_cufile_driver();
    auto const write_buffers = static_cast<std::size_t>(_num_drives * write_buffers_per_drive);
    auto const read_buffers  = static_cast<std::size_t>(_num_drives * read_threads_per_drive);
    _read_region_offset      = write_buffers * write_buffer_bytes;
    auto const staging_bytes = _read_region_offset + read_buffers * read_buffer_bytes;
    std::size_t free_device_bytes  = 0;
    std::size_t total_device_bytes = 0;
    GQE_CUDA_TRY(cudaMemGetInfo(&free_device_bytes, &total_device_bytes));
    GQE_LOG_INFO(
      "flowcsd_spill buffers: {} write staging buffers of {} bytes ({} per drive) and {} read "
      "staging buffers of {} bytes ({} per drive), registered outside the query pool; free device "
      "memory before: {} bytes",
      write_buffers,
      write_buffer_bytes,
      write_buffers_per_drive,
      read_buffers,
      read_buffer_bytes,
      read_threads_per_drive,
      free_device_bytes);
    GQE_CUDA_TRY(cudaMalloc(&_staging, staging_bytes));
    check_cufile(cuFileBufRegister(_staging, staging_bytes, 0), "cuFileBufRegister");
    for (int drive = 0; drive < _num_drives; drive++) {
      for (int buffer = 0; buffer < write_buffers_per_drive; buffer++) {
        _free_write_buffers[drive].push_back(drive * write_buffers_per_drive + buffer);
      }
      for (int thread = 0; thread < write_threads_per_drive; thread++) {
        _threads.emplace_back([this, drive] { run_writes(drive); });
      }
      for (int thread = 0; thread < read_threads_per_drive; thread++) {
        auto const read_buffer_offset =
          _read_region_offset +
          static_cast<std::size_t>(drive * read_threads_per_drive + thread) * read_buffer_bytes;
        _threads.emplace_back(
          [this, drive, read_buffer_offset] { run_reads(drive, read_buffer_offset); });
      }
    }
  }

  ~spill_io()
  {
    {
      std::lock_guard lock(_queue_mutex);
      _stopping = true;
    }
    _queue_ready.notify_all();
    for (auto& thread : _threads) {
      thread.join();
    }
    cuFileBufDeregister(_staging);
    cudaFree(_staging);
  }

  spill_io(spill_io const&)            = delete;
  spill_io& operator=(spill_io const&) = delete;

  [[nodiscard]] int num_drives() const noexcept { return _num_drives; }

  /**
   * @brief The drive expected to finish writing a slice of `bytes` first: (bytes queued or in
   * flight on the drive + bytes in its open staging buffers + the slice) / the drive's write rate in
   * this query (equal rates until every drive has finished a write). Ties go to the drive holding
   * fewer `side_bytes` (bytes of the operator side already there). Callers hold spill_mutex().
   */
  [[nodiscard]] int choose_drive(std::size_t bytes, std::vector<std::size_t> const& side_bytes) const
  {
    auto const& drives      = counters().drives;
    bool const rates_known  = std::all_of(
      drives.begin(), drives.end(), [](auto const& drive) { return drive->write_requests > 0; });
    int best                = 0;
    double best_finish      = std::numeric_limits<double>::infinity();
    for (int drive = 0; drive < _num_drives; drive++) {
      double bytes_per_ns = 1;
      if (rates_known) {
        auto const busy_ns = std::max<int64_t>(drives[drive]->writing.busy_ns(), 1);
        bytes_per_ns = static_cast<double>(drives[drive]->write_bytes.load()) /
                       static_cast<double>(busy_ns);
      }
      auto const ahead_bytes =
        static_cast<double>(_queued_write_bytes[drive].load() + _open_bytes[drive] + bytes);
      double const finish = ahead_bytes / bytes_per_ns;
      bool const earlier  = finish < best_finish;
      bool const tie_with_fewer_side_bytes =
        finish == best_finish && side_bytes[drive] < side_bytes[best];
      if (earlier || tie_with_fewer_side_bytes) {
        best        = drive;
        best_finish = finish;
      }
    }
    return best;
  }

  /**
   * @brief Append `bytes` of device memory to the end of `file` through its write staging buffer
   * and pad the end to 4 KiB; returns the file offset where they start. Waiting for a free staging
   * buffer is exposed write time, added to `waited_ns` too. Callers hold spill_mutex().
   */
  std::size_t append(spill_file& file,
                     uint8_t const* data,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     int64_t& waited_ns)
  {
    auto const file_offset = file.written_bytes + file.open_bytes;
    while (bytes > 0) {
      if (file.open_buffer < 0) { open_write_buffer(file, stream, waited_ns); }
      auto const copy_bytes = std::min(bytes, write_buffer_bytes - file.open_bytes);
      GQE_CUDA_TRY(cudaMemcpyAsync(write_buffer(file.open_buffer) + file.open_bytes,
                                   data,
                                   copy_bytes,
                                   cudaMemcpyDeviceToDevice,
                                   stream.value()));
      file.open_bytes += copy_bytes;
      _open_bytes[file.drive] += copy_bytes;
      data += copy_bytes;
      bytes -= copy_bytes;
      if (file.open_bytes == write_buffer_bytes) { submit(file, stream); }
    }
    // The next slice starts 4 KiB-aligned; the padding bytes are never read back.
    if (file.open_buffer >= 0) {
      auto const padded_bytes = round_up_to_alignment(file.open_bytes);
      _open_bytes[file.drive] += padded_bytes - file.open_bytes;
      file.open_bytes = padded_bytes;
      if (file.open_bytes == write_buffer_bytes) { submit(file, stream); }
    }
    return file_offset;
  }

  /**
   * @brief Queue the file's partially filled staging buffer, if any, as one write. Callers hold
   * spill_mutex().
   */
  void flush(spill_file& file, rmm::cuda_stream_view stream)
  {
    if (file.open_buffer >= 0) { submit(file, stream); }
  }

  /**
   * @brief Give the file's open staging buffer back unwritten (clean-up: the file is about to be
   * deleted). Callers hold spill_mutex().
   */
  void discard(spill_file& file)
  {
    if (file.open_buffer < 0) { return; }
    forget_open_file(file);
    _open_bytes[file.drive] -= file.open_bytes;
    release_write_buffer(file.drive, file.open_buffer);
    file.open_buffer = -1;
    file.open_bytes  = 0;
  }

  /**
   * @brief Queue reads of [file_offset, file_offset + bytes) of `file` into `destination`, at most
   * one read staging buffer per request.
   */
  void queue_read(spill_file const& file,
                  std::size_t file_offset,
                  std::size_t bytes,
                  uint8_t* destination,
                  std::shared_ptr<io_batch> const& reads)
  {
    {
      std::lock_guard lock(_queue_mutex);
      for (std::size_t done_bytes = 0; done_bytes < bytes; done_bytes += read_buffer_bytes) {
        auto const request_bytes = std::min(read_buffer_bytes, bytes - done_bytes);
        reads->add();
        _read_queues[file.drive].push_back(read_request{
          file.handle, file_offset + done_bytes, request_bytes, destination + done_bytes, reads});
      }
    }
    _queue_ready.notify_all();
  }

 private:
  [[nodiscard]] uint8_t* write_buffer(int buffer) const noexcept
  {
    return static_cast<uint8_t*>(_staging) + static_cast<std::size_t>(buffer) * write_buffer_bytes;
  }

  void open_write_buffer(spill_file& file, rmm::cuda_stream_view stream, int64_t& waited_ns)
  {
    file.open_buffer = take_write_buffer(file.drive, stream, waited_ns);
    file.open_bytes  = 0;
    _open_files[file.drive].push_back(&file);
  }

  // A free write staging buffer of `drive`. When every one is queued, in flight or open, the
  // drive's open buffers (of other operator sides) are queued first so the drive keeps writing,
  // then the compute thread waits for a write to finish.
  int take_write_buffer(int drive, rmm::cuda_stream_view stream, int64_t& waited_ns)
  {
    {
      std::lock_guard lock(_buffer_mutex);
      if (!_free_write_buffers[drive].empty()) { return pop_free_write_buffer(drive); }
    }
    auto const open_files = _open_files[drive];  // submit() removes them from the list
    for (auto* open_file : open_files) {
      submit(*open_file, stream);
    }
    utility::nvtx_scoped_range wait_range("spill_wait_write");
    auto const start_ns = monotonic_ns();
    std::unique_lock lock(_buffer_mutex);
    _buffer_free.wait(lock, [this, drive] { return !_free_write_buffers[drive].empty(); });
    int const buffer   = pop_free_write_buffer(drive);
    auto const wait_ns = monotonic_ns() - start_ns;
    waited_ns += wait_ns;
    counters().exposed_write_ns += wait_ns;
    return buffer;
  }

  // Callers hold _buffer_mutex.
  int pop_free_write_buffer(int drive)
  {
    int const buffer = _free_write_buffers[drive].back();
    _free_write_buffers[drive].pop_back();
    return buffer;
  }

  void release_write_buffer(int drive, int buffer)
  {
    {
      std::lock_guard lock(_buffer_mutex);
      _free_write_buffers[drive].push_back(buffer);
    }
    _buffer_free.notify_all();
  }

  void forget_open_file(spill_file& file)
  {
    auto& open_files = _open_files[file.drive];
    open_files.erase(std::remove(open_files.begin(), open_files.end(), &file), open_files.end());
  }

  // Queue the file's open staging buffer as one write at the file's end.
  void submit(spill_file& file, rmm::cuda_stream_view stream)
  {
    // cuFile reads the staging buffer on a write thread: the copies into it must be complete.
    stream.synchronize();
    auto const request_bytes = file.open_bytes;
    preallocate(file, file.written_bytes + request_bytes);
    forget_open_file(file);
    _open_bytes[file.drive] -= request_bytes;
    _queued_write_bytes[file.drive] += static_cast<int64_t>(request_bytes);
    file.writes->add();
    {
      std::lock_guard lock(_queue_mutex);
      _write_queues[file.drive].push_back(
        write_request{file.handle, file.written_bytes, request_bytes, file.open_buffer, file.writes});
    }
    _queue_ready.notify_all();
    file.written_bytes += request_bytes;
    file.write_requests++;
    file.open_buffer = -1;
    file.open_bytes  = 0;
  }

  // Preallocate the file past `end_bytes` (unwritten extents; the file size grows with them).
  static void preallocate(spill_file& file, std::size_t end_bytes)
  {
    if (end_bytes <= file.allocated_bytes) { return; }
    auto const steps =
      (end_bytes - file.allocated_bytes + file_preallocation_bytes - 1) / file_preallocation_bytes;
    auto const grow_bytes = steps * file_preallocation_bytes;
    if (::fallocate(file.fd,
                    0,
                    static_cast<off_t>(file.allocated_bytes),
                    static_cast<off_t>(grow_bytes)) != 0) {
      throw std::runtime_error("spill: fallocate " + file.path + ": " + std::strerror(errno));
    }
    file.allocated_bytes += grow_bytes;
  }

  // One write thread: run the drive's writes in order until the query's spill I/O goes away.
  void run_writes(int drive)
  {
    pthread_setname_np(pthread_self(), "gqe_spill");
    GQE_CUDA_TRY(cudaSetDevice(_device.value()));
    while (true) {
      write_request request;
      {
        std::unique_lock lock(_queue_mutex);
        _queue_ready.wait(
          lock, [this, drive] { return _stopping || !_write_queues[drive].empty(); });
        if (_write_queues[drive].empty()) { break; }  // stopping and nothing left
        request = std::move(_write_queues[drive].front());
        _write_queues[drive].pop_front();
      }
      std::string error;
      try {
        write_one(drive, request);
      } catch (std::exception const& exception) {
        error = exception.what();
      }
      _queued_write_bytes[drive] -= static_cast<int64_t>(request.bytes);
      release_write_buffer(drive, request.buffer);
      request.batch->finish(error);
    }
  }

  // One read thread: run the drive's reads in order, each through this thread's staging buffer.
  void run_reads(int drive, std::size_t read_buffer_offset)
  {
    pthread_setname_np(pthread_self(), "gqe_spill");
    GQE_CUDA_TRY(cudaSetDevice(_device.value()));
    cudaStream_t copy_stream;
    GQE_CUDA_TRY(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    while (true) {
      read_request request;
      {
        std::unique_lock lock(_queue_mutex);
        _queue_ready.wait(lock, [this, drive] { return _stopping || !_read_queues[drive].empty(); });
        if (_read_queues[drive].empty()) { break; }  // stopping and nothing left
        request = std::move(_read_queues[drive].front());
        _read_queues[drive].pop_front();
      }
      std::string error;
      try {
        read_one(drive, request, read_buffer_offset, copy_stream);
      } catch (std::exception const& exception) {
        error = exception.what();
      }
      request.batch->finish(error);
    }
    cudaStreamDestroy(copy_stream);
  }

  void write_one(int drive, write_request const& request)
  {
    utility::nvtx_scoped_range write_range("spill_write");
    auto& drive_stats = *counters().drives[drive];
    drive_stats.writing.start();
    counters().writing.start();
    counters().io.start();
    auto const result =
      cuFileWrite(request.handle,
                  _staging,
                  request.bytes,
                  static_cast<off_t>(request.file_offset),
                  static_cast<off_t>(static_cast<std::size_t>(request.buffer) * write_buffer_bytes));
    counters().io.stop();
    counters().writing.stop();
    drive_stats.writing.stop();
    if (result != static_cast<ssize_t>(request.bytes)) {
      throw std::runtime_error("spill: cuFileWrite: " + cufile_io_error(result));
    }
    drive_stats.write_bytes += static_cast<int64_t>(request.bytes);
    drive_stats.write_requests++;
  }

  void read_one(int drive,
                read_request const& request,
                std::size_t read_buffer_offset,
                cudaStream_t copy_stream)
  {
    utility::nvtx_scoped_range read_range("spill_read");
    auto& drive_stats = *counters().drives[drive];
    drive_stats.reading.start();
    counters().reading.start();
    counters().io.start();
    auto const result = cuFileRead(request.handle,
                                   _staging,
                                   request.bytes,
                                   static_cast<off_t>(request.file_offset),
                                   static_cast<off_t>(read_buffer_offset));
    if (result == static_cast<ssize_t>(request.bytes)) {
      GQE_CUDA_TRY(cudaMemcpyAsync(request.destination,
                                   static_cast<uint8_t*>(_staging) + read_buffer_offset,
                                   request.bytes,
                                   cudaMemcpyDeviceToDevice,
                                   copy_stream));
      GQE_CUDA_TRY(cudaStreamSynchronize(copy_stream));
    }
    counters().io.stop();
    counters().reading.stop();
    drive_stats.reading.stop();
    if (result != static_cast<ssize_t>(request.bytes)) {
      throw std::runtime_error("spill: cuFileRead: " + cufile_io_error(result));
    }
    drive_stats.read_bytes += static_cast<int64_t>(request.bytes);
    drive_stats.read_requests++;
  }

  rmm::cuda_device_id _device;
  int _num_drives;
  void* _staging                  = nullptr;
  std::size_t _read_region_offset = 0;  // the read buffers start here in the staging region
  std::unique_ptr<std::atomic<int64_t>[]> _queued_write_bytes;  // per drive: queued or in flight
  // Compute thread only (spill_mutex()): per drive, the bytes in open staging buffers and the
  // files those buffers belong to.
  std::vector<std::size_t> _open_bytes;
  std::vector<std::vector<spill_file*>> _open_files;

  std::mutex _buffer_mutex;
  std::condition_variable _buffer_free;
  std::vector<std::vector<int>> _free_write_buffers;  // per drive

  std::mutex _queue_mutex;
  std::condition_variable _queue_ready;
  std::vector<std::deque<write_request>> _write_queues;  // per drive
  std::vector<std::deque<read_request>> _read_queues;    // per drive
  bool _stopping = false;
  std::vector<std::thread> _threads;
};

std::shared_ptr<spill_io> acquire_spill_io()
{
  static std::mutex creation_mutex;
  static std::weak_ptr<spill_io> current;
  std::lock_guard lock(creation_mutex);
  if (auto existing = current.lock()) { return existing; }
  auto created = std::make_shared<spill_io>();
  current      = created;
  return created;
}

namespace {

// An input hash-partitioned on its keys: the partitioned copy, a view of its input columns, where
// each partition starts, and the device bytes of the input columns.
struct partitioned_input {
  std::unique_ptr<cudf::table> table;
  cudf::table_view input_columns;
  std::vector<cudf::size_type> offsets;  // partition p = rows [offsets[p], offsets[p + 1])
  std::size_t input_bytes = 0;
};

// The rows of partitions first .. last-1 (contiguous in a partitioned input).
cudf::table_view partition_rows(partitioned_input const& partitioned, int32_t first, int32_t last)
{
  return cudf::slice(partitioned.input_columns,
                     {partitioned.offsets[first], partitioned.offsets[last]})
    .front();
}

// Device bytes the rows of partitions first .. last-1 take, from the input's bytes per row.
int64_t estimated_bytes(partitioned_input const& partitioned, int32_t first, int32_t last)
{
  auto const total_rows = partitioned.offsets.back();
  if (total_rows == 0) { return 0; }
  auto const rows = partitioned.offsets[last] - partitioned.offsets[first];
  return static_cast<int64_t>(static_cast<double>(partitioned.input_bytes) * rows / total_rows);
}

// Rows as written to SSD, their integer columns narrowed (spill_narrowing.hpp); timed as write-side
// narrowing, including the subtraction kernels, which the packing would otherwise wait for.
narrowed_rows narrow_for_write(cudf::table_view rows)
{
  utility::nvtx_scoped_range narrow_range("spill_narrow_write");
  auto const start_ns = monotonic_ns();
  auto const stream   = cudf::get_default_stream();
  auto narrowed       = narrow_rows(rows, stream);
  stream.synchronize();
  counters().narrow_write_ns += monotonic_ns() - start_ns;
  return narrowed;
}

// Whether a column the join never reads can be left out of written slices: its type has a
// placeholder (placeholder_column).
bool has_placeholder(cudf::data_type type)
{
  return type.id() == cudf::type_id::STRING || cudf::is_fixed_width(type);
}

// The column read back in place of a column the join never reads: `rows` values, allocated but not
// initialized (fixed width), or `rows` empty strings.
std::unique_ptr<cudf::column> placeholder_column(cudf::data_type type,
                                                 cudf::size_type rows,
                                                 rmm::cuda_stream_view stream)
{
  if (type.id() == cudf::type_id::STRING) {
    return cudf::make_column_from_scalar(cudf::string_scalar("", true, stream), rows, stream);
  }
  return cudf::make_fixed_width_column(type, rows, cudf::mask_state::UNALLOCATED, stream);
}

// Device bytes of placeholder_column: the values of a fixed-width column, the offsets of a strings
// column.
std::size_t placeholder_bytes(cudf::data_type type, cudf::size_type rows)
{
  auto const num_rows = static_cast<std::size_t>(rows);
  if (type.id() == cudf::type_id::STRING) { return (num_rows + 1) * sizeof(cudf::size_type); }
  return num_rows * cudf::size_of(type);
}

}  // namespace

spill_settings const& get_spill_settings()
{
  static spill_settings const parsed_settings = [] {
    spill_settings parsed;
    parsed.partitions = static_cast<int32_t>(read_int_env("GQE_SPILL_PARTITIONS").value_or(0));
    if (char const* entries = std::getenv("GQE_SPILL_OPERATOR_PARTITIONS")) {
      parsed.operator_partitions = parse_operator_partitions(entries);
    }
    if (auto const budget = read_int_env("GQE_SPILL_RESIDENT_BUDGET_BYTES")) {
      parsed.resident_budget_bytes = static_cast<std::size_t>(*budget);
    }
    if (char const* directories = std::getenv("GQE_SPILL_DIRS")) {
      parsed.directories = split_directories(directories);
    }
    if (parsed.partitions > 0 && parsed.directories.empty()) {
      throw std::runtime_error("spill: GQE_SPILL_PARTITIONS > 0 needs GQE_SPILL_DIRS");
    }
    return parsed;
  }();
  return parsed_settings;
}

operator_partitions spill_partitions_for(std::string const& signature)
{
  auto const& table = get_spill_settings().operator_partitions;
  auto const entry  = table.find(signature);
  if (entry != table.end()) { return {entry->second, true}; }
  return {get_spill_settings().partitions, false};
}

std::size_t spill_resident_budget_bytes(std::size_t pool_bytes)
{
  return get_spill_settings().resident_budget_bytes.value_or(pool_bytes / resident_budget_divisor);
}

void count_hybrid_operator(hybrid_operator kind)
{
  if (kind == hybrid_operator::join) {
    counters().hybrid_joins++;
  } else {
    counters().hybrid_aggregations++;
  }
}

void reset_spill_counters()
{
  auto& current = counters();
  for (auto* counter : {&current.hybrid_joins,
                        &current.hybrid_aggregations,
                        &current.largest_partitions,
                        &current.spilled_operators,
                        &current.files,
                        &current.resident_budget_bytes,
                        &current.resident_bytes,
                        &current.partition_ns,
                        &current.pack_ns,
                        &current.narrow_write_ns,
                        &current.filter_ns,
                        &current.join_phase_ns,
                        &current.narrow_restore_ns,
                        &current.exposed_write_ns,
                        &current.exposed_read_ns,
                        &current.narrowed_write_bytes,
                        &current.unnarrowed_write_bytes}) {
    counter->store(0);
  }
  current.writing.reset();
  current.reading.reset();
  current.io.reset();
  for (auto& drive : current.drives) {
    for (auto* counter :
         {&drive->write_bytes, &drive->write_requests, &drive->read_bytes, &drive->read_requests}) {
      counter->store(0);
    }
    drive->writing.reset();
    drive->reading.reset();
  }
}

void log_spill_counters(int rank, utility::uuid const& query_id)
{
  auto& current          = counters();
  int64_t write_bytes    = 0;
  int64_t write_requests = 0;
  int64_t read_bytes     = 0;
  int64_t read_requests  = 0;
  // Per drive: bytes and the time that drive had a request in flight, so bytes / busy time is the
  // rate the drive delivered to the spill; requests are cuFileWrite / cuFileRead calls.
  for (std::size_t drive = 0; drive < current.drives.size(); drive++) {
    auto& drive_stats = *current.drives[drive];
    GQE_LOG_INFO(
      "flowcsd_spill_drive task_manager rank={} query={} drive={} directory={} write_bytes={} "
      "write_busy_ns={} write_requests={} read_bytes={} read_busy_ns={} read_requests={}",
      rank,
      query_id,
      drive,
      settings().directories[drive],
      drive_stats.write_bytes.load(),
      drive_stats.writing.busy_ns(),
      drive_stats.write_requests.load(),
      drive_stats.read_bytes.load(),
      drive_stats.reading.busy_ns(),
      drive_stats.read_requests.load());
    write_bytes += drive_stats.write_bytes.load();
    write_requests += drive_stats.write_requests.load();
    read_bytes += drive_stats.read_bytes.load();
    read_requests += drive_stats.read_requests.load();
  }
  // write_bytes are the bytes written (slices padded to 4 KiB); narrowed_write_bytes the packed
  // slices without padding, and unnarrowed_write_bytes the same slices packed without narrowing.
  // io_busy_ns: wall time with any spill write or read in flight (writes and reads overlap).
  GQE_LOG_INFO(
    "flowcsd_spill task_manager rank={} query={} partitions={} resident_budget_bytes={} "
    "hybrid_joins={} hybrid_aggregations={} spilled_operators={} files={} resident_bytes={} "
    "write_bytes={} write_busy_ns={} write_requests={} read_bytes={} read_busy_ns={} "
    "read_requests={} io_busy_ns={} partition_ns={} pack_ns={} narrow_write_ns={} filter_ns={} "
    "join_phase_ns={} narrow_restore_ns={} exposed_write_ns={} exposed_read_ns={} "
    "narrowed_write_bytes={} unnarrowed_write_bytes={}",
    rank,
    query_id,
    current.largest_partitions.load(),
    current.resident_budget_bytes.load(),
    current.hybrid_joins.load(),
    current.hybrid_aggregations.load(),
    current.spilled_operators.load(),
    current.files.load(),
    current.resident_bytes.load(),
    write_bytes,
    current.writing.busy_ns(),
    write_requests,
    read_bytes,
    current.reading.busy_ns(),
    read_requests,
    current.io.busy_ns(),
    current.partition_ns.load(),
    current.pack_ns.load(),
    current.narrow_write_ns.load(),
    current.filter_ns.load(),
    current.join_phase_ns.load(),
    current.narrow_restore_ns.load(),
    current.exposed_write_ns.load(),
    current.exposed_read_ns.load(),
    current.narrowed_write_bytes.load(),
    current.unnarrowed_write_bytes.load());
}

class spill_set {
 public:
  explicit spill_set(spill_operator_plan plan)
    : _operator_id(next_operator_id()),
      _plan(std::move(plan)),
      _io(acquire_spill_io()),
      _sides(_plan.num_sides),
      _resident_partitions(_plan.held_side >= 0 ? _plan.partitions : 0),
      _projected_resident_partitions(_resident_partitions),
      _stage_c_start_ns(_plan.partitions, 0)
  {
    if (_plan.probe_filter_build_tasks > 0) {
      _probe_filter = std::make_unique<probe_filter>(_plan.probe_filter_build_tasks);
    }
    for (std::size_t side = 0; side < _sides.size(); side++) {
      auto& state = _sides[side];
      state.partitions.resize(_plan.partitions);
      state.stage_c_rows.resize(_plan.partitions, 0);
      state.drive_bytes.resize(_io->num_drives(), 0);
      state.dead_column_candidates = _plan.dead_columns[side];
      // Sized once: the I/O bookkeeping keeps pointers to the files.
      state.files.resize(_io->num_drives());
      for (int drive = 0; drive < _io->num_drives(); drive++) {
        state.files[drive].drive  = drive;
        state.files[drive].writes = state.writes;
      }
    }
    counters().resident_budget_bytes = static_cast<int64_t>(_plan.resident_budget_bytes);
    std::lock_guard lock(spill_mutex());
    auto& largest_partitions = counters().largest_partitions;
    largest_partitions       = std::max<int64_t>(largest_partitions, _plan.partitions);
    if (_plan.held_side >= 0) { registry().open_sets.push_back(this); }
  }

  ~spill_set()
  {
    // A failed query can leave I/O in flight: let it finish before the files and buffers go.
    for (auto& side : _sides) {
      side.writes->wait_quietly();
      side.read_ahead.clear();  // each pending read waits for its requests
    }
    {
      std::lock_guard lock(spill_mutex());
      for (auto& side : _sides) {
        for (auto& file : side.files) {
          _io->discard(file);
        }
      }
    }
    bool const wrote_slices = std::any_of(
      _sides.begin(), _sides.end(), [](auto const& side) { return side.slices > 0; });
    if (wrote_slices) { counters().spilled_operators++; }
    log_partition_sizes();
    for (auto& side : _sides) {
      close_and_delete(side);
    }
    std::lock_guard lock(spill_mutex());
    freeze_locked();
    release_probe_filter_locked();
    registry().live_bytes -= _live_resident_bytes;
  }

  spill_set(spill_set const&)            = delete;
  spill_set& operator=(spill_set const&) = delete;

  [[nodiscard]] int32_t partitions() const noexcept { return _plan.partitions; }

  /**
   * @brief Remember the schema of one input of `side` (for partitions that stay empty), which of
   * its columns written slices store, and check that both sides of a join hash keys of the same
   * types (no types: no check).
   */
  void record_input(int32_t side, cudf::table_view input, std::vector<cudf::data_type> key_types)
  {
    std::lock_guard lock(spill_mutex());
    auto& state = _sides[side];
    if (state.empty_input == nullptr) {
      state.empty_input = cudf::empty_like(input);
      choose_stored_columns(state, input);
    }
    if (state.key_types.empty()) { state.key_types = std::move(key_types); }
    for (auto const& other : _sides) {
      // Different key types hash to different partitions, so the join would silently miss rows.
      GQE_EXPECTS(other.key_types.empty() || state.key_types.empty() ||
                    other.key_types == state.key_types,
                  "spill: the join sides have different key types");
    }
  }

  /**
   * @brief Stage A: add the keys of one build input to the join's probe-side filter, if it has one.
   * The filter's GPU memory counts in the query's resident budget, like resident partitions: it is
   * held until stage B ends.
   */
  void add_build_keys(int32_t side, std::vector<cudf::column_view> const& keys)
  {
    std::lock_guard lock(spill_mutex());
    if (_probe_filter == nullptr || side != _plan.held_side) { return; }
    utility::nvtx_scoped_range filter_range("spill_probe_filter_build");
    auto const start_ns = monotonic_ns();
    auto const stream   = cudf::get_default_stream();
    auto const bloom_filters_before = _probe_filter->bloom_filters();
    bool const stopped_before       = _probe_filter->stopped();
    _probe_filter->add_build_keys(keys, stream);
    // Its kernels' time belongs to the filter, not to the partitioning that runs next.
    stream.synchronize();
    auto const build_ns = monotonic_ns() - start_ns;
    _probe_filter_build_ns += build_ns;
    counters().filter_ns += build_ns;
    auto const filter_bytes = static_cast<int64_t>(_probe_filter->device_bytes());
    registry().live_bytes += filter_bytes - _probe_filter_live_bytes;
    _probe_filter_live_bytes = filter_bytes;
    _probe_filter_peak_bytes = std::max(_probe_filter_peak_bytes, filter_bytes);
    log_probe_filter_change(bloom_filters_before, stopped_before);
    evict_over_budget();
  }

  /**
   * @brief Stage B: the probe rows that may find a build key (the filter dropped the others), or
   * nullptr when every row may (no copy). Counts the probe rows before and after the filter.
   */
  std::unique_ptr<cudf::table> filter_probe_rows(cudf::table_view rows,
                                                 std::vector<cudf::column_view> const& keys)
  {
    std::lock_guard lock(spill_mutex());
    _probe_rows_before_filter += rows.num_rows();
    std::unique_ptr<cudf::table> kept_rows;
    if (_probe_filter != nullptr) {
      utility::nvtx_scoped_range filter_range("spill_probe_filter_apply");
      auto const start_ns = monotonic_ns();
      auto const stream   = cudf::get_default_stream();
      kept_rows           = _probe_filter->filter_probe_rows(rows, keys, stream);
      stream.synchronize();
      auto const apply_ns = monotonic_ns() - start_ns;
      _probe_filter_apply_ns += apply_ns;
      counters().filter_ns += apply_ns;
      _probe_filter_applied = _probe_filter->filter_kind() != probe_filter::kind::none;
    }
    _probe_rows_after_filter += kept_rows != nullptr ? kept_rows->num_rows() : rows.num_rows();
    return kept_rows;
  }

  /**
   * @brief Stage B with the whole build resident (k = P): the probe rows pass unsplit and
   * unfiltered, so the filter built in stage A is not needed.
   */
  void pass_probe_rows(int64_t rows)
  {
    std::lock_guard lock(spill_mutex());
    _probe_rows_before_filter += rows;
    _probe_rows_after_filter += rows;
    release_probe_filter_locked();
  }

  /**
   * @brief Stage A: place one input task's partitions. On the held side, the first task with rows
   * projects k; room is made in the budget for the task's partitions 0 .. k-1 (which may lower k)
   * and they are copied into GPU memory. Every other partition of the task is written in one batch.
   */
  void place_task(int32_t side, partitioned_input const& partitioned)
  {
    std::lock_guard lock(spill_mutex());
    int32_t first_written = 0;
    if (side == _plan.held_side) {
      project_resident_partitions(partitioned);
      make_room_for_task(partitioned);
      first_written = _resident_partitions;
      keep_resident_slices(side, partitioned, first_written);
      // The copies' real bytes may differ from the estimate: eviction stays the safety net.
      evict_over_budget();
    }
    write_partitions(side, partitioned, first_written);
  }

  /**
   * @brief Stage B: write one probe task's rows of the spilled partitions k .. P-1 in one batch.
   */
  void write_task(int32_t side, partitioned_input const& partitioned, int32_t first_partition)
  {
    std::lock_guard lock(spill_mutex());
    write_partitions(side, partitioned, first_partition);
  }

  /**
   * @brief k: partitions 0 .. k-1 of the held side are resident. Asking fixes k (stage B starts).
   */
  [[nodiscard]] int32_t resident_partitions()
  {
    std::lock_guard lock(spill_mutex());
    freeze_locked();
    return _resident_partitions;
  }

  /**
   * @brief Every resident slice of `side` as one table, freeing the slices (stage B's build).
   */
  std::unique_ptr<cudf::table> take_resident(int32_t side)
  {
    std::lock_guard lock(spill_mutex());
    return take_resident_locked(_sides[side]);
  }

  /**
   * @brief The operator's resident data has been freed: its bytes leave the query's budget.
   */
  void release_resident()
  {
    std::lock_guard lock(spill_mutex());
    registry().live_bytes -= _live_resident_bytes;
    _live_resident_bytes = 0;
  }

  /**
   * @brief Stage C, before partition `partition_index`'s pipeline: whether it runs, i.e. the
   * partition has spill data on some side or (partition 0) resident data. k is fixed from now on,
   * and the probe-side filter can go: every probe split of stage B is done.
   */
  bool start_stage_c_partition(int32_t partition_index)
  {
    std::lock_guard lock(spill_mutex());
    freeze_locked();
    release_probe_filter_locked();
    bool resident_data = false;
    if (partition_index == 0) {
      for (auto const& side : _sides) {
        for (auto const& partition : side.partitions) {
          resident_data = resident_data || !partition.resident_slices.empty();
        }
      }
    }
    if (!partition_has_spill_data(partition_index) && !resident_data) { return false; }
    _stage_c_partitions++;
    return true;
  }

  /**
   * @brief Stage C: partition `partition_index` of `side` as one table.
   *
   * A spilled partition comes back from SSD (the next spilled partitions of the side are read
   * ahead). Partition 0 also brings every resident partition of the side (an aggregation's; a
   * join's were taken in stage B). The first read of a side first writes out its partially filled
   * staging buffers and waits for its writes.
   */
  std::unique_ptr<cudf::table> read(int32_t side, int32_t partition_index)
  {
    std::lock_guard lock(spill_mutex());
    auto& state = _sides[side];
    if (!state.writes_flushed) { flush_writes(state); }
    // The partition's stage-C time starts at its first read; only partitions with spill data count.
    if (_stage_c_start_ns[partition_index] == 0 && partition_has_spill_data(partition_index)) {
      _stage_c_start_ns[partition_index] = monotonic_ns();
    }
    std::unique_ptr<cudf::table> partition_table;
    if (!state.partitions[partition_index].slices.empty()) {
      partition_table = read_spilled(state, partition_index);
    } else {
      read_ahead(state, partition_index + 1);
      if (partition_index == 0) {
        partition_table = take_resident_locked(state);
      } else {
        partition_table = concatenate_slices(state, {});
      }
    }
    state.stage_c_rows[partition_index] += partition_table->num_rows();
    return partition_table;
  }

  /**
   * @brief Stage C: partition `partition_index` has been joined (or aggregated) into `output_rows`
   * rows. Adds its stage-C time (partitions with spill data) and, for a join, its spilled probe rows
   * and their matches.
   */
  void finish_partition(int32_t partition_index, int64_t output_rows, spill_match_rule match_rule)
  {
    std::lock_guard lock(spill_mutex());
    auto const start_ns = _stage_c_start_ns[partition_index];
    if (start_ns > 0) { counters().join_phase_ns += monotonic_ns() - start_ns; }
    if (_plan.probe_side < 0) { return; }
    auto const probe_rows = _sides[_plan.probe_side].stage_c_rows[partition_index];
    if (probe_rows == 0) { return; }  // nothing spilled: a resident partition was joined in stage B
    _spilled_probe_rows += probe_rows;
    switch (match_rule) {
      case spill_match_rule::output_rows: _matched_probe_rows += output_rows; break;
      case spill_match_rule::probe_rows_minus_output:
        _matched_probe_rows += probe_rows - output_rows;
        break;
      case spill_match_rule::unknown: _match_known = false; break;
    }
  }

 private:
  // One packed slice in its side's file of one drive: one input task's rows of one partition, or
  // one destaged resident partition.
  struct slice_entry {
    int drive;
    std::size_t file_offset;       // 4 KiB-aligned
    std::size_t packed_bytes;      // cudf pack size; the file holds it padded to 4 KiB
    std::size_t restored_bytes;    // its size once read back: unnarrowed + dead-column placeholders
    std::size_t unnarrowed_bytes;  // packed size of its stored columns without narrowing
    cudf::size_type rows;
    std::vector<uint8_t> metadata;                    // cudf pack metadata (host)
    std::shared_ptr<narrowing_plan const> narrowing;  // its narrowed columns; null: none
  };

  struct partition_state {
    std::vector<slice_entry> slices;  // written slices
    std::vector<std::unique_ptr<cudf::table>> resident_slices;  // resident partitions only
    std::size_t stored_bytes   = 0;   // padded bytes of its slices (its read buffer)
    std::size_t restored_bytes = 0;   // its size once read back
    bool read_done             = false;
  };

  // Reads of one spilled partition into one buffer (slices laid out as start_read says).
  struct pending_read {
    int32_t partition = -1;
    std::unique_ptr<rmm::device_buffer> data;
    std::vector<std::size_t> slice_positions;  // where each slice starts in `data`
    std::shared_ptr<io_batch> reads;

    pending_read()                          = default;
    pending_read(pending_read&&)            = default;
    pending_read& operator=(pending_read&&) = delete;  // would free `data` without waiting
    // The read threads copy into `data`: it may go only after they are done (also on failure).
    ~pending_read()
    {
      if (reads != nullptr) { reads->wait_quietly(); }
    }
  };

  struct side_state {
    std::vector<partition_state> partitions;
    std::vector<spill_file> files;         // one per drive, opened by its first slice
    std::vector<std::size_t> drive_bytes;  // per drive: padded bytes of this side's slices there
    std::shared_ptr<io_batch> writes = std::make_shared<io_batch>();
    bool writes_flushed              = false;  // stage C queued the staging buffers, writes done
    int32_t unread_partitions        = 0;      // spilled partitions not read back yet
    int64_t slices                   = 0;
    std::unique_ptr<cudf::table> empty_input;  // schema of this side's inputs
    std::vector<cudf::data_type> key_types;
    std::vector<cudf::size_type> dead_column_candidates;  // from the builder
    std::vector<cudf::size_type> live_columns;  // stored in written slices
    std::vector<cudf::size_type> dead_columns;  // left out, placeholders on read
    std::deque<pending_read> read_ahead;        // spilled partitions being read ahead, in order
    std::vector<int64_t> stage_c_rows;          // per partition: rows returned in stage C
    std::size_t narrowed_bytes   = 0;           // packed bytes of the written slices, no padding
    std::size_t unnarrowed_bytes = 0;           // the same slices packed without narrowing
    std::size_t restored_bytes   = 0;           // the written slices once read back
    std::map<cudf::size_type, std::set<std::size_t>> stored_widths;  // narrowed column -> bytes
  };

  static int64_t next_operator_id()
  {
    static std::atomic<int64_t> operator_count{0};
    return operator_count++;
  }

  // Written slices store every column except the dead ones that have a placeholder type.
  static void choose_stored_columns(side_state& state, cudf::table_view input)
  {
    std::set<cudf::size_type> const candidates(state.dead_column_candidates.begin(),
                                               state.dead_column_candidates.end());
    for (cudf::size_type column = 0; column < input.num_columns(); column++) {
      bool const dead = candidates.count(column) > 0 && has_placeholder(input.column(column).type());
      if (dead) {
        state.dead_columns.push_back(column);
      } else {
        state.live_columns.push_back(column);
      }
    }
  }

  // k is fixed from now on: the operator leaves the set whose partitions can still be written out.
  // Callers hold spill_mutex().
  void freeze_locked()
  {
    auto& open_sets = registry().open_sets;
    open_sets.erase(std::remove(open_sets.begin(), open_sets.end(), this), open_sets.end());
  }

  // At the held side's first input task with rows: that task's bytes times the number of held input
  // tasks is the expected held side, and k is the share of the partitions that fits the budget left
  // by everything held in it now (other operators' resident partitions, probe-side filters); a small
  // held side keeps all P (small_held_side). k only drops. Callers hold spill_mutex().
  void project_resident_partitions(partitioned_input const& partitioned)
  {
    if (_projected || partitioned.offsets.back() == 0) { return; }
    _projected                 = true;
    auto const projected_bytes = static_cast<double>(partitioned.input_bytes) *
                                 static_cast<double>(_plan.held_input_tasks);
    _projected_held_bytes = static_cast<int64_t>(projected_bytes);
    if (small_held_side(0)) {
      _projected_resident_partitions = _plan.partitions;
      return;
    }
    auto const held_elsewhere_bytes = registry().live_bytes - _live_resident_bytes;
    auto const available_bytes =
      std::max<int64_t>(0, static_cast<int64_t>(_plan.resident_budget_bytes) - held_elsewhere_bytes);
    auto projected = _plan.partitions;
    if (projected_bytes > 0) {
      auto const fitting = std::floor(static_cast<double>(_plan.partitions) *
                                      static_cast<double>(available_bytes) / projected_bytes);
      projected = static_cast<int32_t>(std::min(static_cast<double>(_plan.partitions), fitting));
    }
    _projected_resident_partitions = projected;
    _resident_partitions           = std::min(_resident_partitions, projected);
  }

  // Whether the held side is smaller than one write staging buffer, as far as the projection and
  // the held bytes so far (resident and written, plus `incoming_bytes`) know it. Such an operator
  // keeps every partition and is never evicted: freeing less than one write request cannot matter
  // against the budget, but spilling a small build sends its whole probe side to SSD on very few
  // keys (v3 q10 SF1000: a 25-row nation build evicted to k = 1 left 0.81-2.43 GB partitions on a
  // 25-value key, which no P fixes). It lets at most one buffer's bytes per operator past the
  // budget. A budget of 0 (every partition spills) turns it off. Callers hold spill_mutex().
  [[nodiscard]] bool small_held_side(int64_t incoming_bytes) const
  {
    if (_plan.held_side < 0 || _plan.resident_budget_bytes == 0) { return false; }
    auto const known_bytes =
      _resident_bytes + static_cast<int64_t>(_sides[_plan.held_side].restored_bytes) + incoming_bytes;
    return std::max(_projected_held_bytes, known_bytes) < static_cast<int64_t>(write_buffer_bytes);
  }

  // The open operator (its k can still drop) holding the most resident bytes, or null. Operators
  // with a small held side are never chosen (small_held_side).
  static spill_set* largest_open_holder()
  {
    spill_set* holder = nullptr;
    for (auto* candidate : registry().open_sets) {
      if (candidate->_resident_partitions == 0 || candidate->_live_resident_bytes == 0) { continue; }
      if (candidate->small_held_side(0)) { continue; }
      if (holder == nullptr || candidate->_live_resident_bytes > holder->_live_resident_bytes) {
        holder = candidate;
      }
    }
    return holder;
  }

  // Write out the highest resident partition; k drops by one. Callers hold spill_mutex().
  void destage_highest_partition()
  {
    _resident_partitions--;
    destage_partition(_plan.held_side, _resident_partitions);
  }

  // While the query's held bytes (resident partitions and probe-side filters) exceed the budget,
  // the open operator holding the most resident bytes writes out its highest resident partition.
  // Operators past stage A keep theirs: their k is fixed. Callers hold spill_mutex().
  void evict_over_budget()
  {
    auto const budget_bytes = static_cast<int64_t>(_plan.resident_budget_bytes);
    while (registry().live_bytes > budget_bytes) {
      auto* holder = largest_open_holder();
      if (holder == nullptr) { return; }
      holder->destage_highest_partition();
    }
  }

  // Before one input task's resident partitions are copied: make room in the budget for its rows of
  // partitions 0 .. k-1 (estimated from its bytes per row). Open operators write out resident
  // partitions as in evict_over_budget; while none holds any, this operator keeps fewer of the
  // task's partitions. A small held side keeps all of them past the budget (small_held_side).
  // Callers hold spill_mutex().
  void make_room_for_task(partitioned_input const& partitioned)
  {
    auto const budget_bytes = static_cast<int64_t>(_plan.resident_budget_bytes);
    while (_resident_partitions > 0) {
      auto const incoming_bytes = estimated_bytes(partitioned, 0, _resident_partitions);
      if (registry().live_bytes + incoming_bytes <= budget_bytes) { return; }
      if (small_held_side(incoming_bytes)) { return; }
      auto* holder = largest_open_holder();
      if (holder == nullptr) {
        _resident_partitions--;
        continue;
      }
      holder->destage_highest_partition();
    }
  }

  // Copy one input task's rows of partitions 0 .. resident_partitions-1 into GPU memory of their
  // own (the partitioned input is freed when its task ends) and count them in the budget. Callers
  // hold spill_mutex().
  void keep_resident_slices(int32_t side,
                            partitioned_input const& partitioned,
                            int32_t resident_partitions)
  {
    for (int32_t partition = 0; partition < resident_partitions; partition++) {
      auto const slice = partition_rows(partitioned, partition, partition + 1);
      if (slice.num_rows() == 0) { continue; }
      auto copy             = std::make_unique<cudf::table>(slice);
      auto const copy_bytes = static_cast<int64_t>(copy->alloc_size());
      _resident_bytes += copy_bytes;
      _live_resident_bytes += copy_bytes;
      registry().live_bytes += copy_bytes;
      counters().resident_bytes += copy_bytes;
      _sides[side].partitions[partition].resident_slices.push_back(std::move(copy));
    }
  }

  // Pack one input task's rows of partitions first .. P-1 (their stored columns, narrowed once)
  // with one contiguous_split, one packed slice per partition, and append each non-empty slice to
  // a file. Callers hold spill_mutex().
  void write_partitions(int32_t side, partitioned_input const& partitioned, int32_t first_partition)
  {
    auto const num_partitions = _plan.partitions;
    if (first_partition >= num_partitions) { return; }
    auto const rows = partition_rows(partitioned, first_partition, num_partitions);
    if (rows.num_rows() == 0) { return; }
    auto& state         = _sides[side];
    auto const narrowed = narrow_for_write(rows.select(state.live_columns));

    utility::nvtx_scoped_range pack_range("spill_pack");
    auto const start_ns    = monotonic_ns();
    auto const first_row   = partitioned.offsets[first_partition];
    std::vector<cudf::size_type> splits;
    for (auto partition = first_partition + 1; partition < num_partitions; partition++) {
      splits.push_back(partitioned.offsets[partition] - first_row);
    }
    auto packed       = cudf::contiguous_split(narrowed.view, splits, cudf::get_default_stream());
    int64_t waited_ns = 0;
    for (auto partition = first_partition; partition < num_partitions; partition++) {
      auto const slice_rows = partitioned.offsets[partition + 1] - partitioned.offsets[partition];
      if (slice_rows == 0) { continue; }
      write_slice(side,
                  partition,
                  packed[static_cast<std::size_t>(partition - first_partition)].data,
                  slice_rows,
                  narrowed.plan,
                  waited_ns);
    }
    // The packed buffers go when `packed` does, after the staging copies on the same stream.
    counters().pack_ns += monotonic_ns() - start_ns - waited_ns;
  }

  // Append one packed slice to its side's file on the drive chosen for it and index it. Fails with
  // "spill partition too large" as soon as the partition would exceed its limit once read back.
  // Callers hold spill_mutex().
  void write_slice(int32_t side,
                   int32_t partition_index,
                   cudf::packed_columns& packed,
                   cudf::size_type rows,
                   std::shared_ptr<narrowing_plan const> const& narrowing,
                   int64_t& waited_ns)
  {
    auto& state                 = _sides[side];
    auto& partition             = state.partitions[partition_index];
    auto const packed_bytes     = packed.gpu_data->size();
    auto const unnarrowed_bytes = unnarrowed_packed_bytes(packed_bytes, rows, narrowing.get());
    auto const restored_bytes   = unnarrowed_bytes + dead_column_bytes(state, rows);
    check_partition_size(side, partition_index, partition.restored_bytes + restored_bytes);

    auto const stored_bytes = round_up_to_alignment(packed_bytes);
    int const drive         = _io->choose_drive(stored_bytes, state.drive_bytes);
    auto& file              = file_for_write(side, drive);
    auto const file_offset  = _io->append(file,
                                         static_cast<uint8_t const*>(packed.gpu_data->data()),
                                         packed_bytes,
                                         cudf::get_default_stream(),
                                         waited_ns);
    if (partition.slices.empty()) { state.unread_partitions++; }
    partition.slices.push_back(slice_entry{drive,
                                           file_offset,
                                           packed_bytes,
                                           restored_bytes,
                                           unnarrowed_bytes,
                                           rows,
                                           std::move(*packed.metadata),
                                           narrowing});
    partition.stored_bytes += stored_bytes;
    partition.restored_bytes += restored_bytes;
    state.drive_bytes[drive] += stored_bytes;
    state.slices++;
    // Bytes with and without narrowing, and the widths the narrowed columns were stored in.
    state.narrowed_bytes += packed_bytes;
    state.unnarrowed_bytes += unnarrowed_bytes;
    state.restored_bytes += restored_bytes;
    counters().narrowed_write_bytes += static_cast<int64_t>(packed_bytes);
    counters().unnarrowed_write_bytes += static_cast<int64_t>(unnarrowed_bytes);
    if (narrowing != nullptr) {
      for (std::size_t stored = 0; stored < narrowing->size(); stored++) {
        auto const& entry = (*narrowing)[stored];
        if (!entry.narrowed()) { continue; }
        state.stored_widths[state.live_columns[stored]].insert(cudf::size_of(entry.stored_type));
      }
    }
  }

  // Bytes of the placeholders a slice of `rows` rows gets for the side's dead columns.
  static std::size_t dead_column_bytes(side_state const& state, cudf::size_type rows)
  {
    std::size_t bytes = 0;
    for (auto const column : state.dead_columns) {
      bytes += placeholder_bytes(state.empty_input->view().column(column).type(), rows);
    }
    return bytes;
  }

  void check_partition_size(int32_t side, int32_t partition_index, std::size_t restored_bytes) const
  {
    auto const limit_bytes = _plan.pool_bytes / partition_limit_divisor;
    if (restored_bytes <= limit_bytes) { return; }
    throw std::runtime_error("spill partition too large: partition " +
                             std::to_string(partition_index) + " of operator " + _plan.signature +
                             " (side " + std::to_string(side) + ") holds " +
                             std::to_string(restored_bytes) +
                             " bytes once read back, more than 1/" +
                             std::to_string(partition_limit_divisor) + " of the query pool (" +
                             std::to_string(limit_bytes) +
                             " bytes); raise its partitions in spill_partitions.csv");
  }

  // Write one resident partition to SSD as one slice (its resident slices concatenated, narrowed
  // once, packed once) and free it; the partition is spilled from now on. Callers hold
  // spill_mutex().
  void destage_partition(int32_t side, int32_t partition_index)
  {
    auto& state  = _sides[side];
    auto& slices = state.partitions[partition_index].resident_slices;
    if (slices.empty()) { return; }
    auto const stream = cudf::get_default_stream();
    int64_t slice_bytes = 0;
    for (auto const& slice : slices) {
      slice_bytes += static_cast<int64_t>(slice->alloc_size());
    }
    // The stored columns of all slices as one table. A single slice is used as it is (with all its
    // columns); several are concatenated with only the stored ones.
    std::unique_ptr<cudf::table> rows_table;
    cudf::table_view rows;
    {
      utility::nvtx_scoped_range pack_range("spill_pack");
      auto const start_ns = monotonic_ns();
      if (slices.size() == 1) {
        rows_table = std::move(slices.front());
        rows       = rows_table->view().select(state.live_columns);
      } else {
        std::vector<cudf::table_view> stored_slices;
        for (auto const& slice : slices) {
          stored_slices.push_back(slice->view().select(state.live_columns));
        }
        rows_table = cudf::concatenate(stored_slices, stream);
        rows       = rows_table->view();
      }
      counters().pack_ns += monotonic_ns() - start_ns;
    }
    // The slices leave the budget now; the copy goes when it has been written.
    slices.clear();
    _resident_bytes -= slice_bytes;
    _live_resident_bytes -= slice_bytes;
    registry().live_bytes -= slice_bytes;
    counters().resident_bytes -= slice_bytes;

    auto const narrowed = narrow_for_write(rows);
    utility::nvtx_scoped_range pack_range("spill_pack");
    auto const start_ns = monotonic_ns();
    auto packed         = cudf::pack(narrowed.view, stream);
    int64_t waited_ns   = 0;
    write_slice(side, partition_index, packed, rows.num_rows(), narrowed.plan, waited_ns);
    counters().pack_ns += monotonic_ns() - start_ns - waited_ns;
  }

  spill_file& file_for_write(int32_t side, int drive)
  {
    auto& file = _sides[side].files[drive];
    if (file.fd >= 0) { return file; }
    file.path = settings().directories[drive] + "/gqe" + std::to_string(getpid()) + "_op" +
                std::to_string(_operator_id) + "_side" + std::to_string(side) + "_drive" +
                std::to_string(drive) + ".spill";
    file.fd = ::open(file.path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_DIRECT, 0644);
    if (file.fd < 0) {
      throw std::runtime_error("spill: cannot create " + file.path + ": " + std::strerror(errno));
    }
    CUfileDescr_t descriptor{};
    descriptor.handle.fd = file.fd;
    descriptor.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    check_cufile(cuFileHandleRegister(&file.handle, &descriptor),
                 "cuFileHandleRegister " + file.path);
    counters().files++;
    return file;
  }

  [[nodiscard]] bool partition_has_spill_data(int32_t partition_index) const
  {
    return std::any_of(_sides.begin(), _sides.end(), [partition_index](auto const& side) {
      return !side.partitions[partition_index].slices.empty();
    });
  }

  // Resident slices as one table (the side's schema with no rows when there are none).
  static std::unique_ptr<cudf::table> concatenate_slices(
    side_state& state, std::vector<std::unique_ptr<cudf::table>> slices)
  {
    if (slices.empty()) {
      GQE_EXPECTS(state.empty_input != nullptr, "spill: no input schema for an empty partition");
      return std::make_unique<cudf::table>(state.empty_input->view());
    }
    if (slices.size() == 1) { return std::move(slices.front()); }
    utility::nvtx_scoped_range concatenate_range("spill_concatenate");
    std::vector<cudf::table_view> views;
    views.reserve(slices.size());
    for (auto const& slice : slices) {
      views.push_back(slice->view());
    }
    return cudf::concatenate(views, cudf::get_default_stream());
  }

  // Every resident slice of a side as one table; k is fixed from now on. Callers hold
  // spill_mutex().
  std::unique_ptr<cudf::table> take_resident_locked(side_state& state)
  {
    freeze_locked();
    std::vector<std::unique_ptr<cudf::table>> slices;
    for (auto& partition : state.partitions) {
      std::move(partition.resident_slices.begin(),
                partition.resident_slices.end(),
                std::back_inserter(slices));
      partition.resident_slices.clear();
    }
    return concatenate_slices(state, std::move(slices));
  }

  // Stage C starts reading this side: queue its partially filled staging buffers and wait for all
  // of its writes (exposed write time; usually they finished long before). Callers hold
  // spill_mutex().
  void flush_writes(side_state& state)
  {
    auto const stream = cudf::get_default_stream();
    for (auto& file : state.files) {
      _io->flush(file, stream);
    }
    utility::nvtx_scoped_range wait_range("spill_wait_write");
    counters().exposed_write_ns += state.writes->wait();
    state.writes_flushed = true;
  }

  // A spilled partition: take its read-ahead or read it now, read the next ones ahead, then unpack
  // its slices into one table (restoring the narrowed columns, adding the placeholders). The side's
  // files are deleted after its last spilled partition. Callers hold spill_mutex().
  std::unique_ptr<cudf::table> read_spilled(side_state& state, int32_t partition_index)
  {
    auto pending = take_read_ahead(state, partition_index);
    read_ahead(state, partition_index + 1);
    {
      utility::nvtx_scoped_range wait_range("spill_wait_read");
      counters().exposed_read_ns += pending.reads->wait();
    }

    utility::nvtx_scoped_range concatenate_range("spill_concatenate");
    auto& partition = state.partitions[partition_index];
    std::vector<cudf::table_view> stored_slices;
    stored_slices.reserve(partition.slices.size());
    auto const* data = static_cast<uint8_t const*>(pending.data->data());
    for (std::size_t slice = 0; slice < partition.slices.size(); slice++) {
      stored_slices.push_back(cudf::unpack(partition.slices[slice].metadata.data(),
                                           data + pending.slice_positions[slice]));
    }
    auto partition_table = assemble_partition(state, partition, stored_slices);
    partition.read_done  = true;
    state.unread_partitions--;
    if (state.unread_partitions == 0) { close_and_delete(state); }
    // `pending.data` is released here, after the partition was assembled on the same stream.
    return partition_table;
  }

  // The read-ahead of `partition_index`, or its reads queued now. Read-aheads of earlier partitions
  // (partitions read out of order) are dropped once their reads finish.
  pending_read take_read_ahead(side_state& state, int32_t partition_index)
  {
    while (!state.read_ahead.empty() && state.read_ahead.front().partition < partition_index) {
      state.read_ahead.pop_front();
    }
    if (!state.read_ahead.empty() && state.read_ahead.front().partition == partition_index) {
      pending_read pending(std::move(state.read_ahead.front()));
      state.read_ahead.pop_front();
      return pending;
    }
    return start_read(state, partition_index);
  }

  // Keep the next spilled partitions of the side (from `from_partition` on) read ahead: up to
  // read_ahead_partitions of them, while their buffers together stay within the partition limit
  // (the first one always).
  void read_ahead(side_state& state, int32_t from_partition)
  {
    auto const num_partitions = static_cast<int32_t>(state.partitions.size());
    auto const limit_bytes    = _plan.pool_bytes / partition_limit_divisor;
    std::size_t buffered_bytes = 0;
    int32_t next               = from_partition;
    for (auto const& pending : state.read_ahead) {
      buffered_bytes += pending.data->size();
      next = std::max(next, pending.partition + 1);
    }
    while (state.read_ahead.size() < read_ahead_partitions) {
      while (next < num_partitions &&
             (state.partitions[next].slices.empty() || state.partitions[next].read_done)) {
        next++;
      }
      if (next >= num_partitions) { return; }
      auto const next_bytes = state.partitions[next].stored_bytes;
      if (!state.read_ahead.empty() && buffered_bytes + next_bytes > limit_bytes) { return; }
      state.read_ahead.push_back(start_read(state, next));
      buffered_bytes += next_bytes;
      next++;
    }
  }

  // Queue the reads of a spilled partition into one new buffer from the query pool. The slices are
  // laid out in (drive, file offset) order, each at a 4 KiB-aligned position, so slices that lie
  // back to back in a file are read with one run of requests; both drives' reads are queued at
  // once. Its writes finished when stage C flushed the side.
  pending_read start_read(side_state& state, int32_t partition_index)
  {
    auto const& slices = state.partitions[partition_index].slices;
    std::vector<std::size_t> order(slices.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&slices](std::size_t left, std::size_t right) {
      return std::tie(slices[left].drive, slices[left].file_offset) <
             std::tie(slices[right].drive, slices[right].file_offset);
    });
    pending_read pending;
    pending.partition = partition_index;
    pending.reads     = std::make_shared<io_batch>();
    pending.slice_positions.resize(slices.size());
    std::size_t total_bytes = 0;
    for (auto const slice : order) {
      pending.slice_positions[slice] = total_bytes;
      total_bytes += round_up_to_alignment(slices[slice].packed_bytes);
    }
    auto const stream = cudf::get_default_stream();
    pending.data      = std::make_unique<rmm::device_buffer>(total_bytes, stream);
    // The read threads copy into the buffer on their own streams.
    stream.synchronize();
    auto* data = static_cast<uint8_t*>(pending.data->data());
    // One run per stretch of slices that follow each other in one file (and so in the buffer).
    std::size_t run_begin = 0;
    for (std::size_t next = 1; next <= order.size(); next++) {
      if (next < order.size() && follows(slices[order[next - 1]], slices[order[next]])) { continue; }
      auto const& first_slice = slices[order[run_begin]];
      auto const& last_slice  = slices[order[next - 1]];
      auto const run_bytes    = last_slice.file_offset +
                             round_up_to_alignment(last_slice.packed_bytes) -
                             first_slice.file_offset;
      _io->queue_read(state.files[first_slice.drive],
                      first_slice.file_offset,
                      run_bytes,
                      data + pending.slice_positions[order[run_begin]],
                      pending.reads);
      run_begin = next;
    }
    return pending;
  }

  // Whether `next` starts in the same file right where `previous` (padded) ends.
  static bool follows(slice_entry const& previous, slice_entry const& next)
  {
    return next.drive == previous.drive &&
           next.file_offset == previous.file_offset + round_up_to_alignment(previous.packed_bytes);
  }

  // A spilled partition's unpacked slices as one table of the side's columns: the stored columns
  // restored or concatenated, and a placeholder at every dead column's position.
  static std::unique_ptr<cudf::table> assemble_partition(
    side_state const& state,
    partition_state const& partition,
    std::vector<cudf::table_view> const& stored_slices)
  {
    auto const stream  = cudf::get_default_stream();
    auto stored_columns = assemble_stored_columns(partition.slices, stored_slices, stream);
    if (state.dead_columns.empty()) {
      return std::make_unique<cudf::table>(std::move(stored_columns));
    }
    cudf::size_type rows = 0;
    for (auto const& slice : partition.slices) {
      rows += slice.rows;
    }
    std::vector<std::unique_ptr<cudf::column>> columns(state.live_columns.size() +
                                                       state.dead_columns.size());
    for (std::size_t stored = 0; stored < state.live_columns.size(); stored++) {
      columns[state.live_columns[stored]] = std::move(stored_columns[stored]);
    }
    auto const schema = state.empty_input->view();
    for (auto const column : state.dead_columns) {
      columns[column] = placeholder_column(schema.column(column).type(), rows, stream);
    }
    return std::make_unique<cudf::table>(std::move(columns));
  }

  // The stored columns of a partition's slices: a column narrowed in any slice is restored in one
  // pass over all slices (restore_column, timed as restore-side narrowing), the others are
  // concatenated.
  static std::vector<std::unique_ptr<cudf::column>> assemble_stored_columns(
    std::vector<slice_entry> const& slices,
    std::vector<cudf::table_view> const& stored_slices,
    rmm::cuda_stream_view stream)
  {
    bool const narrowed_any = std::any_of(
      slices.begin(), slices.end(), [](auto const& slice) { return slice.narrowing != nullptr; });
    if (!narrowed_any) { return cudf::concatenate(stored_slices, stream)->release(); }

    std::vector<std::unique_ptr<cudf::column>> columns;
    for (cudf::size_type column = 0; column < stored_slices.front().num_columns(); column++) {
      std::vector<cudf::column_view> pieces;
      std::vector<column_narrowing> narrowing;  // per piece; the unnarrowed ones stored as they are
      bool column_narrowed = false;
      for (std::size_t slice = 0; slice < slices.size(); slice++) {
        auto const& piece = pieces.emplace_back(stored_slices[slice].column(column));
        if (slices[slice].narrowing == nullptr) {
          narrowing.push_back(column_narrowing{piece.type(), piece.type(), 0});
          continue;
        }
        narrowing.push_back((*slices[slice].narrowing)[column]);
        column_narrowed = column_narrowed || narrowing.back().narrowed();
      }
      if (!column_narrowed) {
        columns.push_back(cudf::concatenate(pieces, stream));
        continue;
      }
      utility::nvtx_scoped_range restore_range("spill_narrow_restore");
      auto const restore_start_ns = monotonic_ns();
      columns.push_back(restore_column(pieces, narrowing, stream));
      stream.synchronize();  // the restore kernel's time is narrowing time
      counters().narrow_restore_ns += monotonic_ns() - restore_start_ns;
    }
    return columns;
  }

  // One line per side: what this operator spilled and kept resident (the held side), its k
  // (projected and final), its partitions' sizes on SSD and once read back, its write requests,
  // the stage-C pipelines that ran and the rows they read, (joins) how many spilled probe rows
  // matched (-1 when the join's output does not tell, spill_match_rule) and what the probe-side
  // filter did (the same on both sides of a join), how much narrowing saved, and the columns left
  // out of written slices.
  void log_partition_sizes() const
  {
    auto const matched_probe_rows =
      (_plan.probe_side >= 0 && _match_known) ? _matched_probe_rows : int64_t{-1};
    // "none": no filter, or one built in stage A that was not needed (k = P).
    std::string filter_kind = "none";
    if (_probe_filter_applied) {
      filter_kind =
        _probe_filter->filter_kind() == probe_filter::kind::bitmap ? "bitmap" : "bloom";
    }
    auto const filter_false_positive_rate =
      _probe_filter_applied ? _probe_filter->false_positive_rate() : 0.0;
    for (std::size_t side = 0; side < _sides.size(); side++) {
      auto const& state                    = _sides[side];
      std::size_t total_bytes              = 0;
      std::size_t largest_bytes            = 0;
      std::size_t smallest_bytes           = std::numeric_limits<std::size_t>::max();
      std::size_t largest_restored_bytes   = 0;
      for (auto const& partition : state.partitions) {
        if (partition.slices.empty()) { continue; }  // resident or empty
        total_bytes += partition.stored_bytes;
        largest_bytes          = std::max(largest_bytes, partition.stored_bytes);
        smallest_bytes         = std::min(smallest_bytes, partition.stored_bytes);
        largest_restored_bytes = std::max(largest_restored_bytes, partition.restored_bytes);
      }
      if (state.slices == 0) { smallest_bytes = 0; }
      int64_t write_requests = 0;
      for (auto const& file : state.files) {
        write_requests += file.write_requests;
      }
      auto const& stage_c_rows = state.stage_c_rows;
      auto const rows_read = std::accumulate(stage_c_rows.begin(), stage_c_rows.end(), int64_t{0});
      bool const held      = static_cast<int32_t>(side) == _plan.held_side;
      GQE_LOG_INFO(
        "flowcsd_spill_operator op={} side={} signature={} partitions={} bytes={} resident_bytes={} "
        "resident_partitions={} projected_resident_partitions={} small_held_side={} "
        "max_partition_bytes={} min_partition_bytes={} slices={} write_requests={} restored_bytes={} "
        "max_partition_restored_bytes={} stage_c_partitions={} stage_c_rows={} "
        "spilled_probe_rows={} matched_probe_rows={} probe_filter={} probe_filter_bytes={} "
        "probe_rows_before_filter={} probe_rows_after_filter={} probe_filter_fpr={:.6g} "
        "probe_filter_build_ns={} probe_filter_apply_ns={} narrowed_bytes={} unnarrowed_bytes={} "
        "narrowed_columns={} dead_columns={} {}",
        _operator_id,
        side,
        _plan.signature,
        _plan.partitions,
        total_bytes,
        held ? _resident_bytes : 0,
        _resident_partitions,
        _projected_resident_partitions,
        small_held_side(0),
        largest_bytes,
        smallest_bytes,
        state.slices,
        write_requests,
        state.restored_bytes,
        largest_restored_bytes,
        _stage_c_partitions,
        rows_read,
        _spilled_probe_rows,
        matched_probe_rows,
        filter_kind,
        _probe_filter_peak_bytes,
        _probe_rows_before_filter,
        _probe_rows_after_filter,
        filter_false_positive_rate,
        _probe_filter_build_ns,
        _probe_filter_apply_ns,
        state.narrowed_bytes,
        state.unnarrowed_bytes,
        describe_narrowing(state),
        describe_dead_columns(state),
        _plan.description);
    }
  }

  // The narrowed columns of a side as "c<column>:<bytes>-><stored bytes>", ';'-separated, with
  // '/'-separated stored bytes when its input tasks needed different widths; "none" without any.
  static std::string describe_narrowing(side_state const& state)
  {
    if (state.stored_widths.empty()) { return "none"; }
    std::string description;
    for (auto const& [column, stored_widths] : state.stored_widths) {
      if (!description.empty()) { description += ";"; }
      auto const column_type = state.empty_input->view().column(column).type();
      description += "c" + std::to_string(column) + ":" +
                     std::to_string(cudf::size_of(column_type)) + "->";
      std::string widths;
      for (auto const width : stored_widths) {
        widths += (widths.empty() ? "" : "/") + std::to_string(width);
      }
      description += widths;
    }
    return description;
  }

  // The input columns written slices leave out, as "c<column>;c<column>", or "none".
  static std::string describe_dead_columns(side_state const& state)
  {
    if (state.dead_columns.empty()) { return "none"; }
    std::string description;
    for (auto const column : state.dead_columns) {
      if (!description.empty()) { description += ";"; }
      description += "c" + std::to_string(column);
    }
    return description;
  }

  // One `flowcsd_spill_probe_filter` line when the Bloom filter grew a chain (a filter past the
  // first) or the join stopped filtering (the chain would pass the filter size limit); both are rare.
  // Callers hold spill_mutex().
  void log_probe_filter_change(std::size_t bloom_filters_before, bool stopped_before) const
  {
    auto const bloom_filters = _probe_filter->bloom_filters();
    bool const grew          = bloom_filters > std::max<std::size_t>(bloom_filters_before, 1);
    bool const stopped       = _probe_filter->stopped() && !stopped_before;
    if (!grew && !stopped) { return; }
    GQE_LOG_INFO(
      "flowcsd_spill_probe_filter op={} signature={} event={} bloom_filters={} build_rows={} "
      "bytes={}",
      _operator_id,
      _plan.signature,
      stopped ? "stopped" : "grown",
      bloom_filters,
      _probe_filter->build_rows(),
      _probe_filter->device_bytes());
  }

  // The probe-side filter is no longer needed: free it and take it out of the query's budget.
  // Callers hold spill_mutex().
  void release_probe_filter_locked()
  {
    if (_probe_filter == nullptr) { return; }
    _probe_filter->release();
    registry().live_bytes -= _probe_filter_live_bytes;
    _probe_filter_live_bytes = 0;
  }

  // Close and delete the side's files (their I/O has finished).
  static void close_and_delete(side_state& state)
  {
    for (auto& file : state.files) {
      if (file.handle != nullptr) { cuFileHandleDeregister(file.handle); }
      if (file.fd >= 0) {
        ::close(file.fd);
        ::unlink(file.path.c_str());
      }
      file.handle = nullptr;
      file.fd     = -1;
    }
  }

  int64_t _operator_id;
  spill_operator_plan _plan;
  std::shared_ptr<spill_io> _io;
  std::vector<side_state> _sides;
  int32_t _resident_partitions;            // k: partitions 0 .. k-1 of the held side are resident
  int32_t _projected_resident_partitions;  // k projected at the held side's first task with rows
  bool _projected = false;
  int64_t _projected_held_bytes = 0;       // held side expected at the projection (0 = not yet)
  std::vector<int64_t> _stage_c_start_ns;  // per partition: its first stage-C read (0 = not yet)
  int64_t _stage_c_partitions  = 0;        // stage-C pipelines that ran
  int64_t _resident_bytes      = 0;        // held-side bytes kept in GPU memory, never written
  int64_t _live_resident_bytes = 0;        // of those, still counted in the query's budget
  int64_t _spilled_probe_rows  = 0;
  int64_t _matched_probe_rows  = 0;
  bool _match_known            = true;
  // Probe-side filter (joins that drop unmatched probe rows; null otherwise) and what it did.
  std::unique_ptr<probe_filter> _probe_filter;
  bool _probe_filter_applied        = false;  // it filtered probe rows in stage B
  int64_t _probe_filter_live_bytes  = 0;      // its bytes counted in the query's budget now
  int64_t _probe_filter_peak_bytes  = 0;
  int64_t _probe_filter_build_ns    = 0;
  int64_t _probe_filter_apply_ns    = 0;
  int64_t _probe_rows_before_filter = 0;      // probe rows entering stage B's split
  int64_t _probe_rows_after_filter  = 0;
};

std::shared_ptr<spill_set> make_spill_set(spill_operator_plan plan)
{
  return std::make_shared<spill_set>(std::move(plan));
}

namespace {

// Key columns of an input: views for column references, owned columns for computed keys.
struct evaluated_keys {
  evaluated_expressions columns;
  std::vector<cudf::data_type> types;
};

evaluated_keys evaluate_keys(cudf::table_view input,
                             optimization_parameters const& parameters,
                             std::vector<std::unique_ptr<expression>> const& key_exprs,
                             cudf::size_type key_column_offset)
{
  evaluated_keys keys;
  keys.columns = evaluate_expressions(
    parameters, input, utility::to_const_raw_ptrs(key_exprs), key_column_offset);
  for (auto const& key : keys.columns.evaluated_result) {
    keys.types.push_back(key.type());
  }
  return keys;
}

// Hash-partition `input` into `num_partitions` partitions. Keys that are input columns are hashed in
// place; computed keys ride along after the input columns. Both sides of a join use the same hash
// (cudf::hash_partition, MURMUR3), so matching keys meet in one partition.
partitioned_input partition_on_keys(cudf::table_view input,
                                    std::vector<std::unique_ptr<expression>> const& key_exprs,
                                    cudf::size_type key_column_offset,
                                    evaluated_keys const& keys,
                                    int32_t num_partitions)
{
  std::vector<cudf::column_view> columns(input.begin(), input.end());
  std::vector<cudf::size_type> key_indices;
  for (std::size_t key = 0; key < key_exprs.size(); key++) {
    if (key_exprs[key]->type() == expression::expression_type::column_reference) {
      auto const* reference = static_cast<column_reference_expression const*>(key_exprs[key].get());
      key_indices.push_back(reference->column_idx() - key_column_offset);
    } else {
      key_indices.push_back(static_cast<cudf::size_type>(columns.size()));
      columns.push_back(keys.columns.evaluated_result[key]);
    }
  }

  partitioned_input partitioned;
  {
    utility::nvtx_scoped_range partition_range("spill_hash_partition");
    std::tie(partitioned.table, partitioned.offsets) =
      cudf::hash_partition(cudf::table_view(columns), key_indices, num_partitions);
  }
  // cuDF 26.06 returns num_partitions + 1 offsets (partition.cpp:64-68).
  GQE_EXPECTS(static_cast<int32_t>(partitioned.offsets.size()) == num_partitions + 1,
              "cudf::hash_partition returned unexpected number of partition offsets");
  // Only the input columns are kept or spilled; the join or aggregation re-evaluates its keys.
  std::vector<cudf::size_type> input_column_indices(input.num_columns());
  std::iota(input_column_indices.begin(), input_column_indices.end(), 0);
  partitioned.input_columns = partitioned.table->view().select(input_column_indices);
  for (auto const column : input_column_indices) {
    partitioned.input_bytes += partitioned.table->get_column(column).alloc_size();
  }
  return partitioned;
}

}  // namespace

spill_partition_task::spill_partition_task(context_reference ctx_ref,
                                           int32_t task_id,
                                           int32_t stage_id,
                                           std::shared_ptr<task> input,
                                           std::vector<std::unique_ptr<expression>> key_exprs,
                                           cudf::size_type key_column_offset,
                                           std::shared_ptr<spill_set> spill,
                                           int32_t side)
  : task(ctx_ref, task_id, stage_id, {std::move(input)}, {}),
    _key_exprs(std::move(key_exprs)),
    _key_column_offset(key_column_offset),
    _spill(std::move(spill)),
    _side(side)
{
}

void spill_partition_task::execute()
{
  prepare_dependencies();
  utility::nvtx_scoped_range task_range("spill_partition_task");

  auto const input = *dependencies()[0]->result();
  // Partition time = key evaluation + hash partitioning; the probe-side filter (built here),
  // narrowing and packing are timed on their own.
  auto const evaluate_start_ns = monotonic_ns();
  auto const keys =
    evaluate_keys(input, get_query_context()->parameters, _key_exprs, _key_column_offset);
  auto partition_ns = monotonic_ns() - evaluate_start_ns;
  _spill->record_input(_side, input, keys.types);
  // A join's probe-side filter gets every build key before the build rows are partitioned.
  _spill->add_build_keys(_side, keys.columns.evaluated_result);
  auto const partition_start_ns = monotonic_ns();
  auto const partitioned =
    partition_on_keys(input, _key_exprs, _key_column_offset, keys, _spill->partitions());
  partition_ns += monotonic_ns() - partition_start_ns;
  counters().partition_ns += partition_ns;
  _spill->place_task(_side, partitioned);

  GQE_LOG_TRACE("Execute spill partition task: task_id={}, stage_id={}, side={}, input_rows={}.",
                task_id(),
                stage_id(),
                _side,
                input.num_rows());
  emit_result(std::make_unique<cudf::table>());
  remove_dependencies();
}

spill_probe_split_task::spill_probe_split_task(context_reference ctx_ref,
                                               int32_t task_id,
                                               int32_t stage_id,
                                               std::shared_ptr<task> input,
                                               std::vector<std::unique_ptr<expression>> key_exprs,
                                               cudf::size_type key_column_offset,
                                               std::shared_ptr<spill_set> spill,
                                               int32_t side)
  : task(ctx_ref, task_id, stage_id, {input}, {}),
    _input(std::move(input)),
    _key_exprs(std::move(key_exprs)),
    _key_column_offset(key_column_offset),
    _spill(std::move(spill)),
    _side(side)
{
}

void spill_probe_split_task::execute()
{
  prepare_dependencies();
  utility::nvtx_scoped_range task_range("spill_probe_split_task");

  auto const input      = *dependencies()[0]->result();
  auto const input_rows = input.num_rows();
  // Stage A is done, so k is final.
  auto const resident_partitions = _spill->resident_partitions();
  if (resident_partitions == _spill->partitions()) {
    // The whole build stayed resident: join the probe rows as they are, like the broadcast join
    // (the probe-side filter is not needed). `_input` keeps the producing task, and with it the
    // borrowed rows, alive.
    _spill->record_input(_side, input, {});
    _spill->pass_probe_rows(input_rows);
    GQE_LOG_TRACE("Execute spill probe split task: task_id={}, stage_id={}, side={}, rows={} (all "
                  "resident).",
                  task_id(),
                  stage_id(),
                  _side,
                  input_rows);
    emit_result(input);
    remove_dependencies();
    return;
  }

  // Probe rows that cannot find a build key are dropped first (probe-side filter). Of the rest,
  // rows of resident partitions join the resident build now and the others go to SSD for stage C.
  // Partition time = key evaluation + hash partitioning; filter, narrowing and packing are timed
  // apart.
  auto const& parameters       = get_query_context()->parameters;
  auto const evaluate_start_ns = monotonic_ns();
  auto keys         = evaluate_keys(input, parameters, _key_exprs, _key_column_offset);
  auto partition_ns = monotonic_ns() - evaluate_start_ns;
  _spill->record_input(_side, input, keys.types);
  auto const filtered_rows = _spill->filter_probe_rows(input, keys.columns.evaluated_result);
  auto const probe_rows    = filtered_rows != nullptr ? filtered_rows->view() : input;
  auto const partition_start_ns = monotonic_ns();
  if (filtered_rows != nullptr) {
    keys = evaluate_keys(probe_rows, parameters, _key_exprs, _key_column_offset);
  }
  auto partitioned =
    partition_on_keys(probe_rows, _key_exprs, _key_column_offset, keys, _spill->partitions());
  partition_ns += monotonic_ns() - partition_start_ns;
  counters().partition_ns += partition_ns;
  _spill->write_task(_side, partitioned, resident_partitions);
  auto const resident_rows = partition_rows(partitioned, 0, resident_partitions);
  _partitioned             = std::move(partitioned.table);
  _input.reset();  // the partitioned copy is all that is needed from here on

  GQE_LOG_TRACE(
    "Execute spill probe split task: task_id={}, stage_id={}, side={}, input_rows={}, "
    "filtered_rows={}, resident_rows={}.",
    task_id(),
    stage_id(),
    _side,
    input_rows,
    probe_rows.num_rows(),
    resident_rows.num_rows());
  emit_result(resident_rows);
  remove_dependencies();
}

spill_resident_build_task::spill_resident_build_task(
  context_reference ctx_ref,
  int32_t task_id,
  int32_t stage_id,
  std::vector<std::shared_ptr<task>> partition_tasks,
  std::shared_ptr<spill_set> spill,
  int32_t side)
  : task(ctx_ref, task_id, stage_id, std::move(partition_tasks), {}),
    _spill(std::move(spill)),
    _side(side)
{
}

// The resident build (this task's result) is freed with this task.
spill_resident_build_task::~spill_resident_build_task() { _spill->release_resident(); }

void spill_resident_build_task::execute()
{
  // The partition tasks finished in stage A; this only waits on their status.
  prepare_dependencies();
  utility::nvtx_scoped_range task_range("spill_resident_build_task");
  auto resident_build = _spill->take_resident(_side);
  GQE_LOG_TRACE("Execute spill resident build task: task_id={}, stage_id={}, side={}, rows={}.",
                task_id(),
                stage_id(),
                _side,
                resident_build->num_rows());
  emit_result(std::move(resident_build));
  remove_dependencies();
}

spill_read_task::spill_read_task(context_reference ctx_ref,
                                 int32_t task_id,
                                 int32_t stage_id,
                                 std::vector<std::shared_ptr<task>> partition_tasks,
                                 std::shared_ptr<spill_set> spill,
                                 int32_t side,
                                 int32_t partition)
  : task(ctx_ref, task_id, stage_id, std::move(partition_tasks), {}),
    _spill(std::move(spill)),
    _side(side),
    _partition(partition)
{
}

// Partition 0 may carry an aggregation's resident partitions; they are freed with this task.
spill_read_task::~spill_read_task()
{
  if (_partition == 0) { _spill->release_resident(); }
}

void spill_read_task::execute()
{
  // The partition tasks finished in an earlier stage; this only waits on their status.
  prepare_dependencies();
  utility::nvtx_scoped_range task_range("spill_read_task");
  auto partition_table = _spill->read(_side, _partition);

  GQE_LOG_TRACE(
    "Execute spill read task: task_id={}, stage_id={}, side={}, partition={}, output_size={}.",
    task_id(),
    stage_id(),
    _side,
    _partition,
    partition_table->num_rows());
  emit_result(std::move(partition_table));
  remove_dependencies();
}

spill_partition_done_task::spill_partition_done_task(context_reference ctx_ref,
                                                     int32_t task_id,
                                                     int32_t stage_id,
                                                     std::shared_ptr<task> input,
                                                     std::shared_ptr<spill_set> spill,
                                                     int32_t partition,
                                                     spill_match_rule match_rule,
                                                     std::vector<cudf::data_type> output_types)
  : task(ctx_ref, task_id, stage_id, {input}, {}),
    _input(std::move(input)),
    _spill(std::move(spill)),
    _partition(partition),
    _match_rule(match_rule),
    _output_types(std::move(output_types))
{
}

void spill_partition_done_task::execute()
{
  if (!_spill->start_stage_c_partition(_partition)) {
    // No spill data and nothing resident in this partition: its pipeline (reads and join or final
    // aggregation) would only produce an empty table, so it is released without running.
    std::vector<std::unique_ptr<cudf::column>> empty_columns;
    for (auto const& type : _output_types) {
      empty_columns.push_back(cudf::make_empty_column(type));
    }
    GQE_LOG_TRACE(
      "Skip spill partition pipeline: task_id={}, stage_id={}, partition={} (no spill data).",
      task_id(),
      stage_id(),
      _partition);
    emit_result(std::make_unique<cudf::table>(std::move(empty_columns)));
    _input.reset();
    remove_dependencies();
    return;
  }
  prepare_dependencies();
  auto const output = *dependencies()[0]->result();
  _spill->finish_partition(_partition, output.num_rows(), _match_rule);
  // `_input` keeps the join (or aggregation) task, and with it the borrowed rows, alive.
  emit_result(output);
  remove_dependencies();
}

}  // namespace gqe
