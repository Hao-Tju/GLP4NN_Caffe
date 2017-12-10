#ifndef CPU_ONLY

#ifndef CAFFE_INFO_LOG_HPP_
#define CAFFE_INFO_LOG_HPP_

#include <string>
#include <sstream>
#include <fstream>

#include <cstdio>
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

    void SetFolder(string net_folder);

    void RecordInfoLog(string label_str, string log_type, string log_val);

  protected:
    int device_id_;
    string base_log_folder_;

    std::fstream log_stream_;
    FILE *log_file_handle_;

  private:
    /**
     * \brief Private constructor used to avoid duplicate instantiation.
     */
    InfoLog();

    InfoLog(const InfoLog&);
    InfoLog& operator=(const InfoLog&);
};  // class Log

} /** namespace caffe **/

#endif    /** CAFFE_LAYER_INFO_LOG_HPP_ **/
#endif    /** CPU_ONLY **/
