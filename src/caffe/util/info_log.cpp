#ifndef CPU_ONLY

#include <sstream>
#include <fstream>
#include <ctime>

#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
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
  }

  void InfoLog::SetDevice(int device_id) {
    this->device_id_ = device_id;
  }

  void InfoLog::SetFolder(string net_folder) { 
    if (this->base_log_folder_.find(net_folder) != string::npos) {
      return ;
    }

    this->base_log_folder_ += net_folder;

    boost::filesystem::path log_dir(this->base_log_folder_.c_str());
    if (!boost::filesystem::create_directory(log_dir)) {
      LOG(FATAL) << "Failed to create folder: " << this->base_log_folder_;
    }

    this->base_log_folder_ += "/";
  }

  void InfoLog::RecordInfoLog(string label_str, string log_type, string log_val) {
    if (this->log_stream_.is_open()) {
      this->log_stream_.close();
    }
    this->log_stream_.open(this->base_log_folder_ + log_type + ".csv", std::ios::app);

    if (!log_stream_.is_open()) {
      LOG(INFO) << "Failed to open log file: " << this->base_log_folder_ + log_type + ".csv";
    }

    LOG(INFO) << "LOGGING: " << label_str << "," << this->device_id_ << "," << log_val << std::endl;
    log_stream_ << label_str << "," << this->device_id_ << "," << log_val << std::endl;

    log_stream_.close();
  }
}   /** namespace caffe **/

#endif    /** CPU_ONLY **/
