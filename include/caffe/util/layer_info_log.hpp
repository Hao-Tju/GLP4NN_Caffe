#ifndef CAFFE_LAYER_INFO_LOG_HPP_
#define CAFFE_LAYER_INFO_LOG_HPP_

#include <string>
#include <sstream>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

// #define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet!"

namespace caffe {

  using std::fstream;
  using std::stringstream;
  using std::string;
  using std::endl;

  /**
   * \brief A class designed for logging corresponding network infomation.
   *
   * \tparam Dtype  Template class name.
   */
template <typename Dtype>
class Log
{
  public:
    ~Log();
    /**
     * \brief To get a Log object.
     *
     * \return A static Log object.
     */
    static Log& Get();

    /**
     * \brief Configure the log file.
     *
     * \param file_name
     */
    static void SetFilename(string file_name);

    /**
     * \brief Method used to write convolution layer parameter to log file.
     *
     * \tparam Dtype  Template class name.
     * \param param   Common parameter configuration of network layers.
     * \param bottom  Input blob of a convolution layer.
     */
    static void ConvParamInfoToLog(const LayerParameter& param,
        int batch_size, Blob<int>& conv_input_shape, int num_output, Blob<int>& kernel_shape,
        Blob<int>& stride, Blob<int>& pad);

    /**
     * \brief Method used to write pooling layer parameter to log file.
     *
     * \tparam Dtype  Template class name.
     * \param param   Common parameter configuration of network layers.
     * \param bottom  Input blob of the pooling layer.
     */
    static void PoolParamInfoToLog(const LayerParameter& param,
        const vector<Blob<Dtype>*>& bottom);

  protected:
    string log_file_name_;
    fstream log_stream_;

  private:
    /**
     * \brief Private constructor used to avoid duplicate instantiation.
     */
    Log() {};

    DISABLE_COPY_AND_ASSIGN(Log);
};  // class Log

// Instantiate Log class with float and double specifications.
// INSTANTIATE_CLASS(Log);
template class Log<float>;
template class Log<double>;

} /** namespace caffe **/

#endif    /** CAFFE_LAYER_INFO_LOG_HPP_ **/
