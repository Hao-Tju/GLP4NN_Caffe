#ifndef CPU_ONLY
#ifdef USE_PROF

#include <limits>
#include <string>
#include <fstream>
#include <sstream>

#include <cmath>
#include <cctype>
#include <ctime>

#include <boost/thread.hpp>

#include "caffe/util/async_tracker.hpp"
#include "caffe/util/info_log.hpp"

#define NS2MS 1000000.0

#define CHECK_KERNEL_RECORD(record)     {     \
  if (record->start == 0 or record->end == 0) {                   \
    LOG(FATAL) << "WARNING! Cannot record the START timestamp of kernel: " << record->name  \
              << ". Start timestamp: " << record->start << ", END timestamp: " << record->end;    \
  } \
}

#define MIN(first, second) (first < second ? first : second)
#define MAX(first, second) (first < second ? second : first)

namespace caffe {
  using std::string;
  using std::fstream;
  using std::stringstream;
  using std::time_t;
  using std::tm;

  int AsyncResTracker::static_kernel_counter_ = 0;
  int AsyncResTracker::profiling_device_id_ = -1;
  bool AsyncResTracker::profiler_flag_ = false;
  uint64_t AsyncResTracker::static_startTimestamp_ = 0;
  vector<Kernel_t>* AsyncResTracker::kernels_vec_ptr_ = NULL;
  //map<string, vector<Kernel_t> >* AsyncResTracker::name_kernel_ptr_ = NULL;
  vector<Timestamp_t>* AsyncResTracker::timestamp_vec_ptr_ = NULL;
  // Mutex used to guard the access to the asynchronous resource tracker.
  boost::mutex AsyncResTracker::profiler_mutex_;
  PROFTYPE AsyncResTracker::curr_prof_type_ = PROFTYPE::DEFAULT;

  // Global static asynchronous resource tracker.
  static AsyncResTracker* async_res_tracker_ = NULL;

  string str_tolower(string str) {
    std::transform(str.begin(), str.end(), str.begin(),
        [](unsigned char c){ return std::tolower(c); }
        );

    return str;
  }

  template <typename T>
  bool Compare(T val1, T val2) {
    return (val1 < val2);
  }

  string AsyncGetCurrentTime() {
    time_t curr_time = time(NULL);
    tm *local_time = localtime(&curr_time);
    stringstream temp_ss;
    temp_ss << local_time->tm_year << "-" << local_time->tm_mon << "-" << local_time->tm_mday;
    string result = temp_ss.str();
    temp_ss.str("");
    temp_ss.clear();

    return result;
  }

  const char* getActivityKindString(CUpti_ActivityOverheadKind kind) {
    switch (kind) {
      case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
        return "COMPILER OVERHEAD";
      case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
        return "BUFFER_FLUSH OVERHEAD";
      case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
        return "INSTRUMENTATION OVERHEAD";
      case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
        return "RESOURCE";
      default:
        break;
    }

    return "<unknown>";
  }

  const char* getActivityObjectKindString(CUpti_ActivityObjectKind kind) {
    switch (kind) {
      case CUPTI_ACTIVITY_OBJECT_PROCESS:
        return "PROCESS";
      case CUPTI_ACTIVITY_OBJECT_THREAD:
        return "THREAD";
      case CUPTI_ACTIVITY_OBJECT_DEVICE:
        return "DEVICE";
      case CUPTI_ACTIVITY_OBJECT_CONTEXT:
        return "CONTEXT";
      case CUPTI_ACTIVITY_OBJECT_STREAM:
        return "STREAM";
      default:
        break;
    }

    return "<unknown>";
  }

  AsyncResTracker& AsyncResTracker::Get() {
    if (async_res_tracker_ == NULL) {
      boost::mutex::scoped_lock instance_lock(profiler_mutex_);
      if (async_res_tracker_ == NULL) {
        async_res_tracker_ = new AsyncResTracker();
      }
    }

    return *async_res_tracker_;
  }

  AsyncResTracker::AsyncResTracker() {
    // Initialize the kernel_counter_ variable, which is used to record
    // the number of kernels profiled currently.
    this->kernel_counter_ = 0;
    this->kernel_launch_overhead_ = 0; // The launch overhead of a CUDA kernel.

    this->kernels_vec_.clear(); // Clear the kernels_vec_ vector.
    this->timestamp_vec_.clear(); // Clear the timestamp_vec_ vector.

    // Intialize the seg_tree_-related variables.
    this->next_node_idx_ = 0;
    this->tree_nodes_count_ = 0;
    this->seg_tree_ = NULL;
  }

  AsyncResTracker::~AsyncResTracker() {
    // Empty the information vector of kernels recorded and timestamps of them.
    if (!kernels_vec_.empty()) {
      this->kernels_vec_.clear();
    }
    if (!timestamp_vec_.empty()) {
      this->timestamp_vec_.clear();
    }

    // Empty the seg_tree_ which is used to calculate the idle time of the GPU device.
    if (seg_tree_ != NULL) {
      delete[] this->seg_tree_;
    }
    this->next_node_idx_ = 0;
    this->tree_nodes_count_ = 0;
    this->seg_tree_ = NULL;
  }

  void AsyncResTracker::InitAsyncResTracker(PROFTYPE prof_type) {
    // In default situation, the resource tracker is disabled.
    profiling_device_id_ = -1;
    profiler_flag_ = false;

    // Record current profiling type.
    // Enable tracking kernel information.
    LOG(INFO) << "Profiling Type: " << (prof_type == CONCURRENT ? "CONCURRENT" :
        (prof_type == DEFAULT ? "DEFAULT" : "SERIAL")) << ", Current profiling type: "
      << (curr_prof_type_ == CONCURRENT ? "CONCURRENT" : (curr_prof_type_ == DEFAULT ? "DEFAULT" : "SERIAL"));
    if ((prof_type == CONCURRENT) and (curr_prof_type_ != CONCURRENT)) {
      LOG(INFO) << "Change from Serial Profiling to Concurrent Profiling.";
      CHECK_CUPTI_ERROR(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL),
          "cuptiActivityDisable CUPTI_ACTIVITY_KIND_KERNEL");
      CHECK_CUPTI_ERROR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
          "cuptiActivityEnable CUPTI_ACTIVITY_KIND_CONRRENT_KERNEL")
    } else if ((curr_prof_type_ == CONCURRENT) and (prof_type != CONCURRENT)) {
      LOG(INFO) << "Change from Concurrent Profiling to Serial Profiling.";
      CHECK_CUPTI_ERROR(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
          "cuptiActivityDisable CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL");
      CHECK_CUPTI_ERROR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL),
          "cuptiActivityEnable CUPTI_ACTIVITY_KIND_KERNEL");
    } else if (curr_prof_type_ == DEFAULT and prof_type == DEFAULT) {
      //CHECK_CUPTI_ERROR(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
      //    "cuptiActivityDisenable CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL");
      CHECK_CUPTI_ERROR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
          "cuptiActivityEnable CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL");
    }
    curr_prof_type_ = prof_type;

    static bool config_flag = true;
    if (config_flag) {
      CHECK_CUPTI_ERROR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD), "cuptiActivityEnable CUPTI_ACTIVITY_KIND_OVERHEAD");

      // Register functions for requesting buffer or processing buffer.
      CHECK_CUPTI_ERROR(cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted), "cuptiActivityRegisterCallbacks");

      // Double the buffer size and the number of buffers.
      size_t bufferSize = 0, bufferValSize = sizeof(size_t);
      CHECK_CUPTI_ERROR(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &bufferValSize, &bufferSize), "cuptiActivityGetAttribute");
      bufferSize *= 2;
      CHECK_CUPTI_ERROR(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &bufferValSize, &bufferSize), "cuptiActivitySetAttribute");
      CHECK_CUPTI_ERROR(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP, &bufferValSize, &bufferSize), "cuptiActivityGetAttribut");
      bufferSize *= 2;
      CHECK_CUPTI_ERROR(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE_CDP, &bufferValSize, &bufferSize), "cuptiActivitySetAttribute");
      CHECK_CUPTI_ERROR(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &bufferValSize, &bufferSize), "cuptiActivityGetAttribute");
      bufferSize *= 2;
      CHECK_CUPTI_ERROR(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &bufferValSize, &bufferSize), "cuptiActivitySetAttribute");

      config_flag = false;
    }

    LOG(INFO) << "The initialization of the asynchronous resource tracker is COMPLETED!";
  }

  void AsyncResTracker::ProfilerStart(int device_id) {
    // Initialze the device needed to be profiled.
    profiling_device_id_ = device_id;

    // Reset kernel counter to 0;
    static_kernel_counter_ = 0;
    // Initialize the launch overhead to 0.
    kernel_launch_overhead_ = 0;

    // Initialize the vector storing kernels information.
    kernels_vec_ptr_ = &this->kernels_vec_;
    if (!kernels_vec_ptr_->empty()) {
      kernels_vec_ptr_->clear();
    }
    // Initialize the vector that store the kernels' timestamp to a object-specific value.
    timestamp_vec_ptr_ = &this->timestamp_vec_;
    if (!timestamp_vec_ptr_->empty()) {
      timestamp_vec_ptr_->clear();
    }

    // Get the start timestamp of profiling record, which is used to normalize kernel's timestampes.
    CHECK_CUPTI_ERROR(cuptiGetTimestamp(&static_startTimestamp_), "cuptiGetTimestamp");

    // Enable the GPU profiler after all preparations have been done.
    profiler_flag_ = true;
    LOG(INFO) << "START Profiling on Device " << profiling_device_id_ << " !";
  }

  void AsyncResTracker::ProfilerStop() {
    // Synchronization.
    CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    // Flush the CUPTI buffer.
    CHECK_CUPTI_ERROR(cuptiActivityFlushAll(0), "cuptiActivityFlushAll");

    // Record the number of kernels in the past profiling period.
    this->kernel_counter_ = static_kernel_counter_;
    // Stop the profiling.
    profiler_flag_ = false;
    LOG(INFO) << "STOP Profiling on Device " << profiling_device_id_ <<  "! TOTAL " << this->kernel_counter_ << " kernels are recorded!";
  }

  void CUPTIAPI AsyncResTracker::BufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    // If profiler_flag_ is set to false, then buffer will be setted to NULL, which represents
    // no kernels will be recorded.
    if (profiler_flag_) {
      uint8_t *temp_buf = new uint8_t[BUFFER_SIZE + ALIGN_SIZE];

      if (temp_buf == NULL) {
        LOG(FATAL) << "Error! Can not allocate new buffer for acivity collecting! OUT OF MEMORY!";
      }

      *size = BUFFER_SIZE;
      *buffer = ALIGN_BUFFER(temp_buf, ALIGN_SIZE);
      *maxNumRecords = 0;

      if (curr_prof_type_ != CONCURRENT) {
        stringstream temp_ss;
        temp_ss << (BUFFER_SIZE + ALIGN_SIZE) * sizeof(uint8_t) / 1024 << " KB";
        InfoLog::Get().RecordInfoLog("cupti_buffer", "CUPTI-BUFFER", temp_ss.str());
        temp_ss.str("");
        temp_ss.clear();
      }
    } else {
      *size = 0;
      *maxNumRecords = 0;
      *buffer = NULL;
    }
  }

  void CUPTIAPI AsyncResTracker::BufferCompleted(CUcontext context, uint32_t stream_id, uint8_t *buffer, size_t size, size_t validSize) {
    CUptiResult cupti_status;
    CUpti_Activity *record = NULL;

    if (validSize > 0 and buffer != NULL) {
      while (true) {
        // Get the next kernel recorded if exists.
        cupti_status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (cupti_status == CUPTI_SUCCESS) {
          // Parse the kernel execution information.
          ParseKernelConfig(record);
        } else if (cupti_status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
          break ;
        } else {
          CHECK_CUPTI_ERROR(cupti_status, "cuptiActivityGetNextRecord");
        }
      }

      // LOG the number of records dropped if exists.
      size_t dropped_records;
      CHECK_CUPTI_ERROR(cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped_records), "cuptiActivityGetNumDroppedRecords");
      if (dropped_records != 0) {
        LOG(INFO) << "Dropped " << dropped_records << " activity records!";
      }
    }

    // Free the allocated buffer.
    if (buffer != NULL) {
      delete[] buffer;
      buffer = NULL;
    }
  }

  void AsyncResTracker::ParseKernelConfig(CUpti_Activity *record) {
    // Only parse kernel information.
    if ((record->kind == CUPTI_ACTIVITY_KIND_KERNEL) or
        (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)) {
      //LOG(INFO) << (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ? "CUPTI_ACTIVITY_KIND_KERNEL" : "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL");
      CUpti_ActivityKernel3 *kernel_record = reinterpret_cast<CUpti_ActivityKernel3 *>(record);
      // Check whether the kernel record is valid.
      CHECK_KERNEL_RECORD(kernel_record);
      if (kernel_record->deviceId != profiling_device_id_) {
        LOG(INFO) << "Kernel " << kernel_record->name << " @(deviceID = " << kernel_record->deviceId << ") is discarded. Current device is " << profiling_device_id_ << ".";

        return ;
      }
      string kernel_name = kernel_record->name;
      //if (str_tolower(kernel_name).find("sync") == string::npos) {
      // Increment the number of kernels recorded.
      static_kernel_counter_ ++;

      Kernel_t kernel;
      uint64_t kernel_duration = 0;
      Timestamp_t timestamp;

      // Analyze the information of kernels recorded.
      kernel.name = timestamp.name = kernel_record->name;
      kernel.gridX = kernel_record->gridX;
      kernel.gridY = kernel_record->gridY;
      kernel.gridZ = kernel_record->gridZ;
      kernel.blockX = kernel_record->blockX;
      kernel.blockY = kernel_record->blockY;
      kernel.blockZ = kernel_record->blockZ;
      kernel.regPerThread = kernel_record->registersPerThread;
      kernel.smPerBlock = kernel_record->staticSharedMemory + kernel_record->dynamicSharedMemory;
      kernel_duration = kernel_record->end - kernel_record->start;

      int pos = FindKernelConfig(kernels_vec_ptr_, kernel);
      if (pos == -1) { // For the first time of recording a kernel.
        kernel.invocations = 1;
        kernel.duration = kernel_duration;
        kernel.average_exec_time = kernel_duration;
        kernels_vec_ptr_->push_back(kernel);
      } else { // Update the information of a kernel recorded already.
        kernels_vec_ptr_->at(pos).invocations ++;
        kernels_vec_ptr_->at(pos).duration += kernel_duration;
        kernels_vec_ptr_->at(pos).average_exec_time = std::ceil(kernels_vec_ptr_->at(pos).duration /
            static_cast<double>(kernels_vec_ptr_->at(pos).invocations));
      }
      LOG(INFO) << "END: " << kernel_record->end << ", START: " << kernel_record->start << ", DURATION: " << kernel_duration;

      timestamp.start = kernel_record->start - static_startTimestamp_;
      timestamp.end = kernel_record->end - static_startTimestamp_;
      timestamp.streamId = kernel_record->streamId;
      if (static_kernel_counter_ < timestamp_vec_ptr_->size()) {
        timestamp_vec_ptr_->at(static_kernel_counter_) = timestamp;
      } else {
        timestamp_vec_ptr_->push_back(timestamp);
      }
      //}
    } else if (record->kind == CUPTI_ACTIVITY_KIND_OVERHEAD) {
      CUpti_ActivityOverhead *overhead = reinterpret_cast<CUpti_ActivityOverhead *> (record);
      stringstream temp_ss;

      // Updated.
      LOG(INFO) << "OVERHEAD: " << getActivityKindString(overhead->overheadKind) << "," << static_cast<double>((overhead->end - overhead->start)) / NS2MS << "ms," << getActivityObjectKindString(overhead->objectKind) << std::endl;
      temp_ss << getActivityKindString(overhead->overheadKind) << "," << (overhead->end - overhead->start) / NS2MS << " ms," << getActivityObjectKindString(overhead->objectKind);

      InfoLog::Get().RecordInfoLog("cupti_overhead", "CUPTI-OVERHEAD", temp_ss.str());

      temp_ss.str("");
      temp_ss.clear();
    }
  }

  int AsyncResTracker::FindKernelConfig(const vector<Kernel_t> *kernel_vec_ptr, Kernel_t& kernel) {
    for (int i = 0; i < kernel_vec_ptr->size(); ++ i) {
      if (kernel_vec_ptr->at(i) == kernel) {
        return i;
      }
    }

    return -1;
  }

  vector<Kernel_t>& AsyncResTracker::GetKernelsRecorded() {
    return this->kernels_vec_;
  }

  uint64_t AsyncResTracker::GetKernelLaunchOverhead() {
    unsigned int min_invocations = std::numeric_limits<unsigned int>::max();
    for (auto kernel_rec : kernels_vec_) {
      min_invocations = MIN(min_invocations, kernel_rec.invocations);
      LOG(INFO) << "Kernel name: " << kernel_rec.name << " [" << kernel_rec.invocations << "].";
    }
    std::sort(timestamp_vec_.begin(), timestamp_vec_.end(), Compare<Timestamp_t>);

    unsigned int kernels_per_iter = timestamp_vec_.size() / min_invocations;
    // To avoid additional kernel recorded.

    uint64_t total_launch_overhead = 0;
    if (kernels_per_iter > 1) {
      for (int i = 0; i < min_invocations; ++ i) {
        for (int j = 1; j < kernels_per_iter; ++ j) {
          total_launch_overhead += (timestamp_vec_.at(i * kernels_per_iter + j).start - timestamp_vec_.at(i * kernels_per_iter + j - 1).end);
        }
      }
    } else {
      for (int i = 0; i < (min_invocations - 1); ++ i) {
        total_launch_overhead += (timestamp_vec_.at(i + 1).start - timestamp_vec_.at(i).end);
      }
    }

    LOG(INFO) << "total_launch_overhead = " << total_launch_overhead << "; kernels_per_iter = " << kernels_per_iter << "; min_invocations = " << min_invocations;
    if (kernels_per_iter > 1) {
      kernel_launch_overhead_ = total_launch_overhead / ((kernels_per_iter - 1) * min_invocations);
    } else {
      kernel_launch_overhead_ = total_launch_overhead / (min_invocations - 1);
    }

    stringstream temp_ss;
    temp_ss << "timestamp_vec_buffer," << sizeof(Timestamp_t) * timestamp_vec_.size() << "," << "kernel_temp_buffer," << sizeof(Kernel_t) * kernels_vec_.size();
    InfoLog::Get().RecordInfoLog("buffer_overhead", "TEMP-BUFFER-OVERHEAD", temp_ss.str());

    LOG(INFO) << "The launch overhead of a kernel is " << kernel_launch_overhead_ << "!";
    temp_ss.str("");
    temp_ss.clear();

    return kernel_launch_overhead_;
  }

  void AsyncResTracker::TreeBuild(SegTree_ptr seg_tree, int node_id, int start, int end) {
    int mid_1 = (start + end) >> 1;
    int mid_2 = mid_1;

    seg_tree[node_id].start = start;
    seg_tree[node_id].end = end;
    seg_tree[node_id].covered = 0;
    seg_tree[node_id].left = seg_tree[node_id].right = 0;

    if (start == end) {
      return ;
    }
    if (start == mid_1) {
      mid_1 = start;
      mid_2 = end;
    }

    seg_tree[node_id].left = next_node_idx_ ++;
    TreeBuild(seg_tree, seg_tree[node_id].left, start, mid_1);
    seg_tree[node_id].right = next_node_idx_ ++;
    TreeBuild(seg_tree, seg_tree[node_id].right, mid_2, end);

    return ;
  }

  void AsyncResTracker::TreeInsert(SegTree_ptr seg_tree, int current_node, int start, int end) {
    if ((start >= end) || (seg_tree[current_node].covered == 1)) {
      return ;
    }

    int start_val = seg_tree[current_node].start;
    int end_val = seg_tree[current_node].end;

    if (start_val == start && end_val == end) {
      seg_tree[current_node].covered = 1;
      return ;
    }

    int mid_val = (start_val + end_val) >> 1;
    if (end <= mid_val) {
      TreeInsert(seg_tree, seg_tree[current_node].left, start, end);
    } else if (start >= mid_val) {
      TreeInsert(seg_tree, seg_tree[current_node].right, start, end);
    } else {
      TreeInsert(seg_tree, seg_tree[current_node].left, start, mid_val);
      TreeInsert(seg_tree, seg_tree[current_node].right, mid_val, end);
    }

    return ;
  }

  uint64_t AsyncResTracker::TreeTraverse(const SegTree_ptr seg_tree, const vector<uint64_t>& time_vec, int node_id) {
    if (seg_tree[node_id].covered == 1) {
      return (time_vec[seg_tree[node_id].end]
          - time_vec[seg_tree[node_id].start]);
    }

    uint64_t left_value = 0, right_value = 0;
    if (seg_tree[node_id].left != 0) {
      left_value = TreeTraverse(seg_tree, time_vec, seg_tree[node_id].left);
    }
    if (seg_tree[node_id].right != 0) {
      right_value = TreeTraverse(seg_tree, time_vec, seg_tree[node_id].right);
    }

    return (left_value + right_value);
  }

  void AsyncResTracker::ComputeOccupancyRatio(const string layer_name, const unsigned int parallel_degree) {
    if (this->kernel_counter_ == 0) {
      LOG(INFO) << "There is no kernel recorded.";
      return ;
    }

    vector<uint64_t> time_vec;

    // Remove redundant timestamp from original record.
    for (int i = 0; i < this->kernel_counter_; ++ i) {
      vector<uint64_t>::iterator  start_iter = std::find(time_vec.begin(), time_vec.end(), timestamp_vec_.at(i).start);
      if (start_iter == time_vec.end()) {
        time_vec.push_back(timestamp_vec_.at(i).start);
      }
      vector<uint64_t>::iterator end_iter = std::find(time_vec.begin(), time_vec.end(), timestamp_vec_.at(i).end);
      if (end_iter == time_vec.end()) {
        time_vec.push_back(timestamp_vec_.at(i).end);
      }
    }

    std::sort(time_vec.begin(), time_vec.end(), Compare<uint64_t>);
    // Allocate memory for seg_tree.
    if (tree_nodes_count_ < (this->kernel_counter_ * 2 * 4)) {
      delete[] seg_tree_;
      seg_tree_ = NULL;

      tree_nodes_count_ = this->kernel_counter_ * 2 * 4;
      seg_tree_ = new SegTree_t[tree_nodes_count_];
      if (seg_tree_ == NULL) {
        LOG(FATAL) << "Failed to allocate memory for segment tree!";
      } else {
        LOG(INFO) << "Allocate " << tree_nodes_count_ << " nodes!";
      }
    }
    std::memset(seg_tree_, 0, tree_nodes_count_ * sizeof(SegTree_t));

    // Segment tree construction
    next_node_idx_ = 1;   // Index that tree nodes constructed should be stored from.
    TreeBuild(seg_tree_, 0, 0, time_vec.size() - 1);

    for (int i = 0; i < this->kernel_counter_; ++ i) {
      int start_pos = std::find(time_vec.begin(), time_vec.end(), timestamp_vec_.at(i).start)
                      - time_vec.begin();
      int end_pos = std::find(time_vec.begin(), time_vec.end(), timestamp_vec_.at(i).end)
                    - time_vec.begin();

      TreeInsert(seg_tree_, 0, start_pos, end_pos);
    }

    // Get the total busy time in a particular time range.
    uint64_t busy_time = TreeTraverse(seg_tree_, time_vec, 0);
    uint64_t total_time = time_vec.back() - time_vec.front();
    uint64_t idle_time = total_time - busy_time;

    double occupancy_ratio = static_cast<double>(busy_time) / static_cast<double>(total_time);
    LOG(INFO) << "Occupancy ratio of layer " << layer_name << " (#streams = " << parallel_degree << ") is " << occupancy_ratio << ". IDLE = " << idle_time << ", TOTAL=" << total_time;

    stringstream filename_ss, content_ss;
    filename_ss << layer_name << "_ParallelDeg_" << parallel_degree << "_occ";
    content_ss << layer_name << "," << parallel_degree << "," << occupancy_ratio << "," << total_time << "," << idle_time;
    InfoLog::Get().RecordInfoLog("Meanlingless_Value", filename_ss.str(), content_ss.str());
    filename_ss.str("");
    filename_ss.clear();
    content_ss.str("");
    filename_ss.clear();

    // Free memory allocated.
    time_vec.clear();

    return ;
  }

  void AsyncResTracker::TimestampLog(const string filename) const {
    stringstream temp_ss;

    const int step = 40;
    for (int i = 0; i < this->timestamp_vec_.size(); i += step) {
      temp_ss.str("");
      temp_ss.clear();
      for (int j = 0; j < step and (j + i) < this->timestamp_vec_.size(); ++ j) {
        const Timestamp_t *temp_timestamp = &timestamp_vec_[i + j];
        temp_ss << temp_timestamp->start << "," << temp_timestamp->streamId << ","
          << temp_timestamp->name << "\n";
        temp_ss << temp_timestamp->end << "," << temp_timestamp->streamId << ","
          << temp_timestamp->name << "\n\n\n";
      }

      InfoLog::Get().RecordInfoLog("Meaningless_Value", filename + "_timestamp", temp_ss.str());
    }
    temp_ss.str("");
    temp_ss.clear();
  }

  void AsyncResTracker::TempBufRelease() {
    if (!this->kernels_vec_.empty()) {
      this->kernels_vec_.clear();
    }
    if (!this->timestamp_vec_.empty()) {
      this->timestamp_vec_.clear();
    }
    if (seg_tree_ != NULL) {
      delete[] seg_tree_;
      seg_tree_ = NULL;

      tree_nodes_count_ = 0;
    }
  }
} /** namespace caffe **/

#endif /** USE_PROF **/
#endif /** CPU_ONLY settings **/
