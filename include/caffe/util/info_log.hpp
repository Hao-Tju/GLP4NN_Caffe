#ifndef CPU_ONLY
#ifdef USE_PROF

#ifndef CAFFE_INFO_LOG_HPP_
#define CAFFE_INFO_LOG_HPP_

#include <string>
#include <sstream>
#include <fstream>

#include <cstdint>

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
// template <typename Dtype>
class InfoLog
{
  public:
    ~InfoLog();
    /**
     * \brief To get a Log object.
     *
     * \return A static Log object.
     */
    static InfoLog& Get();

    void SetDevice(int device_id);

    void RecordInfoLog(string layer_name, string loop_name, string log_type, uint64_t log_val);

  protected:
    int device_id_;
    string base_log_folder_;

    std::fstream log_stream_;

  private:
    /**
     * \brief Private constructor used to avoid duplicate instantiation.
     */
    InfoLog();

    DISABLE_COPY_AND_ASSIGN(InfoLog);
};  // class Log

// Instantiate Log class with float and double specifications.
// INSTANTIATE_CLASS(Log);
// template class InfoLog<float>;
// template class InfoLog<double>;

} /** namespace caffe **/

#endif    /** CAFFE_LAYER_INFO_LOG_HPP_ **/
#endif    /** USE_PROF **/
#endif    /** CPU_ONLY **/
