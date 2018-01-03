#ifndef CPU_ONLY
#ifdef USE_PROF

#ifndef CAFFE_ASYNC_TRACKER_HPP_
#define CAFFE_ASYNC_TRACKER_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <cstring>
#include <vector>
#include <map>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <glog/logging.h>

#include <boost/thread/mutex.hpp>

#include "caffe/util/res_struct.hpp"

/**
 * @brief Macro used to check CUDA runtime errors, and print the corresponding error
 * string.
 */
#define CHECK_CUDA_ERROR(err, functionName) {                   \
  if (err != cudaSuccess) {                                     \
    LOG(FATAL) << __FILE__ << ":" << __LINE__ << ": error " <<   \
      err << " for CUDA Runtime API function '" << functionName \
      << "': " << cudaGetErrorString(err) << std::endl;         \
    exit (EXIT_FAILURE);                                        \
  }                                                             \
}

/**
 * @brief Macro used to check CUPTI runtime errors, and print the corresponding error
 * string.
 */
#define CHECK_CUPTI_ERROR(err, cuptifunc)     {                         \
  if (err != CUPTI_SUCCESS) {                                           \
    const char* errstr;                                                 \
    cuptiGetResultString(err, &errstr);                                 \
    LOG(FATAL) << __FILE__ << ":" << __LINE__ << ": Error " << errstr << \
      " for CUPTI API function '" << cuptifunc << "'." << std::endl;    \
    exit(EXIT_FAILURE);                                                 \
  }                                                                     \
}

/**
 * @brief Macro used to check CUDA driver errors, and print the corresponding error
 * string.
 */
#define CHECK_CU_ERROR(err, functionName) {                     \
  if (err != CUDA_SUCCESS) {                                    \
    const char* errstr;                                         \
    cuGetErrorString(err, &errstr);                             \
    LOG(FATAL) << __FILE__ << ":" << __LINE__ << ": error " <<   \
      err << " for CUDA Driver API function '" << functionName  \
      << "': " << errstr << std::endl;                          \
    exit (EXIT_FAILURE);                                        \
  }                                                             \
}

/**
 * @brief Disable the copy and assignment operation of a class operation.
 */
#define DISABLE_COPY_AND_ASSIGN(classname)                      \
private:                                                        \
  classname(const classname&);                                  \
  classname& operator=(const classname&)                        \

/**
 * @brief Macro used to represent the buffer size allocated for kernel recording.
 */
#define BUFFER_SIZE (32 * 1024)
#define ALIGN_SIZE 8      // \brief Alignment size.
/**
 * @brief Macro used to align profiling buffer.
 */
#define ALIGN_BUFFER(buffer, align_size)                        \
  ((reinterpret_cast<uintptr_t>(buffer) & (align_size - 1)) ? (buffer + align_size - (reinterpret_cast<uintptr_t>(buffer) & (align_size - 1))) : buffer)

// Standard built-in class used in AsyncResTracker.
using std::string;
using std::vector;
using std::map;
using std::fstream;
using std::stringstream;

namespace caffe {
  enum PROFTYPE {DEFAULT = 0, SERIAL = 1, CONCURRENT = 2};

  /**
   * @brief AsyncResTracker class.
   *
   * @details Class used to track kernels in a network layer.
   */
  class AsyncResTracker {
    public:
      /**
       * @brief   AsyncResTracker deconstructor.
       */
      ~AsyncResTracker();

      /**
       * @brief   Object access method.
       *
       * Singleton asynchronous resource tracker object.
       */
      static AsyncResTracker& Get();

      /**
       * @brief   Initialize CUPTI settings of a class object.
       */
      static void InitAsyncResTracker(PROFTYPE prof_type = SERIAL);

      /**
       * @brief   ProfilerLock function.
       *
       * Function used to lock a profiler.
       */
      void ProfilerLock() {
        this->profiler_mutex_.lock();
      }
      /**
       * @brief   ProfilerUnlock function.
       *
       * Function used to unlock a profiler.
       */
      void ProfilerUnlock() {
        this->profiler_mutex_.unlock();
      }

      /**
       * @brief   Kernel profiler starter.
       * @param[in] device_id    ID of the device needed to be profiled.
       *                         If device_id is lower than 0, then the
       *                         current device is profiled, otherwise
       *                         device device_id is profiled.
       */
      void ProfilerStart(int device_id = -1);
      /**
       * @brief   Kernel profiler stopper.
       */
      void ProfilerStop();

      /**
       * @brief   Buffer request function.
       *
       * Function used to handle buffer requests from CUPTI runtime. This function will allocate
       * buffers for CUPTI to store kernel records.
       *
       * @param[out] buffer     Buffer allocated for kernel profiling.
       * @param[out] size       Size of the buffer allocated.
       * @param[out] maxNumRecords The maximum number of records that should be placed in the buffer.
       *                           0 represents that the buffer is filled with as many records as
       *                           possible.
       */
      static void CUPTIAPI BufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
      /**
       * @brief   Buffer completion function.
       *
       * Function used by CUPTI to return a buffer of activity records. The buffer contains validSize
       * bytes of activity records which should be read using cuptiActivityGetNextRecord.
       *
       * @param[in] context     The context this buffer is associated with. This field is deprecated
       *                        as of CUDA 6.0 and will always be NULL.
       * @param[in] stream_id   CUDA stream ID.
       * @param[in] buffer      The activity record buffer.
       * @param[in] size
       * @param[in] validSize   Size of activity records in bytes.
       */
      static void CUPTIAPI BufferCompleted(CUcontext context, uint32_t stream_id, uint8_t *buffer, size_t size, size_t validSize);
      /**
       * @brief   Kernel record parse function.
       *
       * Parse the activity record and store the corresponding kernel name and kernel runtime
       * configuration.
       *
       * @param[in] record      A activity record from cuptiGetActivityGetNextRecord.
       */
      static void ParseKernelConfig(CUpti_Activity *record);
      /**
       * @brief   Kernel searching function.
       *
       * This function is used to check whether there is an identical kernel has been recorded.
       *
       * @param[in] kernel_vec_ptr  Kernel vector that stores kernels with the same key value.
       * @param[in] kernel          New kernel waiting for being processed.
       * @return  Index of the kernel in kernel_vec_ptr. It returns -1 if the kernel is not found.
       */
      static int FindKernelConfig(const vector<Kernel_t> *kernel_vec_ptr, Kernel_t& kernel);

      /**
       * @brief   GPU device setting function.
       * @param[in] device_id       New device needed to be profiled.
       */
      static void SetDevice(int device_id) {
        profiling_device_id_ = device_id;
      }

      /**
       * @brief   Get the vector of kernels recorded.
       * @return  Reference to the vector of kernels.
       */
      vector<Kernel_t>& GetKernelsRecorded();
      /**
       * @brief   Get the overhead of launching a kernel.
       * @return  The overhead of launching a kernel on the current device.
       */
      uint64_t GetKernelLaunchOverhead();

      /**
       * @brief   Build a segment tree.
       * @param[in] seg_tree    Pointer to the root of the segment tree.
       * @param[in] node_id     ID of the available tree node.
       * @param[in] start       The start number of the segment tree.
       * @param[in] end         The end number of the segment tree.
       */
      void TreeBuild(SegTree_ptr seg_tree, int node_id, int start, int end);
      /**
       * @brief   Insert a tree node.
       * @param[in] seg_tree      Pointer to the root of the segment tree.
       * @param[in] current_node  ID of the available tree node.
       * @param[in] start         The start number of the segment tree.
       * @param[in] end           The end number of the segment tree.
       */
      void TreeInsert(SegTree_ptr seg_tree, int current_node, int start, int end);
      /**
       * @brief   Traverse the segment tree.
       * @param[in] seg_tree      Pointer to the root of the segment tree.
       * @param[in] time_vec      Timestamps of kernels recorded.
       * @param[in] node_id       The current tree node being visited.
       * @return  The total busy time of the GPU device.
       */
      uint64_t TreeTraverse(const SegTree_ptr seg_tree, const vector<uint64_t>& time_vec, int node_id);
      /**
       * @brief   Compute the device occupancy ratio.
       * @param[in] layer_name      Name of the current layer.
       * @param[in] parallel_degree The current parallel_degree configuration.
       */
      void ComputeOccupancyRatio(const string layer_name, const unsigned int parallel_degree);
      /**
       * @brief   Record the kernel timestamp.
       *
       * This method is used to write kernel timestamps recorded for further analysis.
       *
       * @param[in] filename        Name of the log file.
       * @param[in] timestamp_ptr   Kernel timestamp vector.
       */
      void TimestampLog(const string filename) const;
      /**
       * @brief   Temp buffer release method.
       *
       * Method used to release temporary buffer allocated while analyzing kernels recorded.
       */
      void TempBufRelease();

    protected:
      // Boost mutex used to lock the profiler.
      static boost::mutex profiler_mutex_;
      // Static member variable used to count the number of recorded kernels.
      static int static_kernel_counter_;
      // Member variable used to store the number of recorded kernels.
      unsigned int kernel_counter_;

      // Static member variable used to identify the profiling device.
      static int profiling_device_id_;
      // Member variable used to identify whether start profiling.
      static bool profiler_flag_;
      // Record kind of the current activity profiled.
      static CUpti_ActivityKind cupti_act_kind_;

      // Static start timestamp of the profiling process.
      static uint64_t static_startTimestamp_;
      // Overhead used to launch a kernel.
      uint64_t kernel_launch_overhead_;

      // Static member variable used to point to kernel records.
      static vector<Kernel_t> *kernels_vec_ptr_;
      // Member variable used to maintain the recorded kernels.
      vector<Kernel_t> kernels_vec_;
      // Static member variable used to point to the vector storing kernel
      // timestamps.
      static vector<Timestamp_t> *timestamp_vec_ptr_;
      // Member variable used to store kernel timestamps.
      vector<Timestamp_t> timestamp_vec_;

      // The current profiling type: CONCURRENT or SERIAL.
      static PROFTYPE curr_prof_type_;

      // Member variable used to represent the next available tree node ID.
      // Only used to construct the segment tree.
      unsigned int next_node_idx_;
      // Member variable used to represent the total number of nodes in segment
      // tree seg_tree_.
      unsigned int tree_nodes_count_;
      // Member variable used to point to the root node of segment tree seg_tree_.
      SegTree_ptr seg_tree_;

      // File stream used to write idle interval in the total execution time.
      fstream idle_time_stream_;
      // File stream used to write the kernel profiling information.
      fstream kernel_stream_;

    private:
      // The private constructor to avoid duplicate instantiation.
      AsyncResTracker();

      DISABLE_COPY_AND_ASSIGN(AsyncResTracker);
  };
}   /** namespace caffe **/

#endif    /** CAFFE_ASYNC_TRACKER_HPP_ **/
#endif    /** USE_PROF **/
#endif    /** CPU_ONLY settings **/
