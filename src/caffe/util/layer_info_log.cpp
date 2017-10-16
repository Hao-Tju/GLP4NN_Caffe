#include "caffe/util/layer_info_log.hpp"

#define LogObjectMember(type, val) (Log<Dtype>::Get().val)

namespace caffe {
  template <typename Dtype>
  Log<Dtype>::~Log() {
    if (log_stream_.is_open()) {
      log_stream_.close();
    }
  }

  template <typename Dtype>
  Log<Dtype>& Log<Dtype>::Get() {
    static Log<Dtype> single_instance_;

    return single_instance_;
  }

  template <typename Dtype>
  void Log<Dtype>::SetFilename(string file_name) {
    LogObjectMember(Dtype, log_file_name_) = PREFIX_DIR + file_name;

    if (LogObjectMember(Dtype, log_stream_).is_open()) {
      LogObjectMember(Dtype, log_stream_).close();
    }

    LogObjectMember(Dtype, log_stream_).open(LogObjectMember(Dtype,log_file_name_).c_str(), std::fstream::in | std::fstream::app);

    return ;
  }

  template <typename Dtype>
  void Log<Dtype>::ConvParamInfoToLog(const LayerParameter& param,
      int batch_size, Blob<int>& conv_input_shape, int num_output, Blob<int>& kernel_shape,
      Blob<int>& stride, Blob<int>& pad) {
    if (!LogObjectMember(Dtype, log_stream_).is_open()) {
      LOG(FATAL) << "Cannot open LOG FILE: " << LogObjectMember(Dtype, log_file_name_);
    }

    static bool title_flag = true;
    if (title_flag) {
      LogObjectMember(Dtype, log_stream_) << "LayerName,LayerType,Engine,Parallel,BottomName,BatchSize,Picture,OutputChannels,Kernel,Stride,Pad" << endl;
      title_flag = false;
    }

    stringstream temp_ss;
    for (int i = 0; i < param.bottom_size(); ++ i) {
      temp_ss << param.name() << "," << param.type() << ",";

      if (param.convolution_param().engine() == ConvolutionParameter_Engine_CAFFE) {
        temp_ss << "CAFFE,";
      } else {
        temp_ss << "CUDNN,";
      }

      temp_ss << param.convolution_param().parallel_degree() << "," << param.bottom(i) << "," << batch_size << ",[";
      int count = conv_input_shape.count();
      for (int j = 0; j < count; ++ j) {
        if (j != (count - 1)) {
          temp_ss << conv_input_shape.cpu_data()[j] << " ";
        } else {
          temp_ss << conv_input_shape.cpu_data()[j];
        }
      }
      temp_ss << "]," << num_output << ",[";
      count = kernel_shape.count();
      for (int j = 0; j < count; ++ j) {
        if (j != (count - 1)) {
          temp_ss << kernel_shape.cpu_data()[j] << " ";
        } else {
          temp_ss << kernel_shape.cpu_data()[j];
        }
      }
      temp_ss << "],[";
      count = stride.count();
      for (int j = 0; j < count; ++ j) {
        if (j != (count - 1)) {
          temp_ss << stride.cpu_data()[j] << " ";
        } else {
          temp_ss << stride.cpu_data()[j];
        }
      }
      temp_ss << "],[";
      count = pad.count();
      for (int j = 0; j < count; ++ j) {
        if (j != (count - 1)) {
          temp_ss << pad.cpu_data()[j] << " ";
        } else {
          temp_ss << pad.cpu_data()[j];
        }
      }
      temp_ss << "]\n";

      LogObjectMember(Dtype, log_stream_) << temp_ss.str();
      temp_ss.str("");
      temp_ss.clear();
    }
  }

  template <typename Dtype>
  void Log<Dtype>::PoolParamInfoToLog(const LayerParameter& param,
      const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
}   /** namespace caffe **/
