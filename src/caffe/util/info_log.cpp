#ifndef CPU_ONLY
#ifdef USE_PROF

#include <stringstream>
#include <ctime>

#include <boost/thread.hpp>
#include <glog/logging.h>

#include "caffe/util/info_log.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {
  using std::stringstream;
  using std::fstream;

  static boost::thread_specific_ptr<InfoLog> thread_info_log_;

  InfoLog& InfoLog<Dtype>::Get() {
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

  void InfoLog::RecordInfoLog(string layer_name, string loop_name, string log_type, uint64_t log_val) {
    if (this->log_stream_.is_open()) {
      this->log_stream_.close();
    }
    this->log_stream_.open(this->base_log_folder_ + log_type + ".csv", std::io::out | std::io::app);

    if (log_stream_.is_open()) {
      LOG(INFO) << "Failed to open log file: " << temp_prof_ss.str();
    }

    log_stream_ << layer_name << "," << loop_name << "," << this->device_id_ << "," << prof_time << std::endl;

    log_stream_.close();
  }
}   /** namespace caffe **/

#endif    /** USE_PROF **/
#endif    /** CPU_ONLY **/
