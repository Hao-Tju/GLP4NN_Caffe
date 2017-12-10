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
    /*
    if (log_stream_.is_open()) {
      log_stream_.close();
    }*/
    if (log_file_ != NULL) {
      fclose(log_file_);
    }
  }

  InfoLog::InfoLog() {
    CUDA_CHECK(cudaGetDevice(&this->device_id_));

    base_log_folder_ = "./LOG/";
    log_file_ = NULL;
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
    if (!boost::filesystem::exists(log_dir) and !boost::filesystem::create_directory(log_dir)) {
      LOG(FATAL) << "Failed to create folder: " << this->base_log_folder_;
    }

    this->base_log_folder_ += "/";
  }

  void InfoLog::RecordInfoLog(string label_str, string log_type, string log_val) {
    /*if (this->log_stream_.is_open()) {
      this->log_stream_.close();
    }*/
    if (log_file_ != NULL) {
      fclose(log_file_);
    }
    // this->log_stream_.open(this->base_log_folder_ + log_type + ".csv", std::ios::app);
    log_file_ = fopen((this->base_log_folder_ + log_type + ".csv").c_str(), "a");

    /*(
    if (!log_stream_.is_open()) {
      LOG(INFO) << "Failed to open log file: " << this->base_log_folder_ + log_type + ".csv";
    }
    */
    if (log_file_ == NULL) {
      LOG(FATAL) << "Failed to open log file: " << this->base_log_folder_ + log_type + ".csv";
    }

    /*
    switch(this->log_stream_.rdstate()) {
      case std::ios_base::badbit:
        LOG(FATAL) << "Irrecoverable stream error!";
        break;
      case std::ios_base::failbit:
        LOG(FATAL) << "Input/Output operation failed!";
        break;
      case std::ios_base::eofbit:
        LOG(FATAL) << "Associated input sequence has reached end-of-file!";
        break;
      default:
        break;
    } */

    //log_stream_.clear();
    if (!label_str.empty()) {
      LOG(INFO) << "LOGGING: " << this->device_id_ << "," << log_val.c_str() << std::endl;
      fprintf(log_file_, "%s\n", log_val.c_str());
    } else {
      LOG(INFO) << "LOGGING: " << label_str.c_str() << "," << this->device_id_ << "," << log_val.c_str() << std::endl;
      fprintf(log_file_, "%s,%i,%s\n", label_str.c_str(), this->device_id_, log_val.c_str());
    }
    fflush(log_file_);
    //log_stream_ << label_str << "," << this->device_id_ << "," << log_val << std::endl;
    //log_stream_.flush();

    fclose(log_file_);
    log_file_ = NULL;
    //log_stream_.close();
  }
}   /** namespace caffe **/

#endif    /** CPU_ONLY **/
