#ifndef CPU_ONLY
#ifdef USE_PROF

#include <sstream>
#include <fstream>
#include <ctime>

#include <boost/thread.hpp>
#include <glog/logging.h>

#include "caffe/util/info_log.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {
  using std::stringstream;
  using std::fstream;

  static boost::thread_specific_ptr<InfoLog> thread_info_log_;

  InfoLog& InfoLog::Get() {
    if (!thread_info_log_.get()) {
      thread_info_log_.reset(new InfoLog());
    }

    return *(thread_info_log_.get());
  }

  InfoLog::~InfoLog() {
    if (log_stream_.is_open()) {
      log_stream_.close();
    }
  }

  InfoLog::InfoLog() {
    CUDA_CHECK(cudaGetDevice(&this->device_id_));

    base_log_folder_ = "./LOG/";
    /*
    std::time_t curr_time = std::time(NULL);
    std::tm *timeinfo = localtime(&curr_time);

    stringstream temp_folder;
    temp_folder << timeinfo->tm_year << "_" << timeinfo->tm_mon << "_" << timeinfo->tm_mday;

    base_time_str_ = "./LOG" + temp_folder.str();
    */
  }

  void InfoLog::SetDevice(int device_id) {
    this->device_id_ = device_id;
  }

  void InfoLog::RecordInfoLog(string label_str, string log_type, uint64_t log_val) {
    if (this->log_stream_.is_open()) {
      this->log_stream_.close();
    }
    this->log_stream_.open(this->base_log_folder_ + log_type + ".csv", std::ios::out | std::ios::app);

    if (log_stream_.is_open()) {
      LOG(INFO) << "Failed to open log file: " << this->base_log_folder_ + log_type + ".csv";
    }

    log_stream_ << label_str << "," << this->device_id_ << "," << log_val << std::endl;

    log_stream_.close();
  }
}   /** namespace caffe **/

#endif    /** USE_PROF **/
#endif    /** CPU_ONLY **/
