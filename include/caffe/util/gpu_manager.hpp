#ifndef CPU_ONLY
#ifdef USE_PROF

#ifndef CAFFE_GPU_MANAGER_HPP_
#define CAFFE_GPU_MANAGER_HPP_

#include <boost/thread.hpp>
#include <glog/logging.h>

#include <cublas_v2.h>

#include "caffe/util/device_alternate.hpp"

#ifdef USE_CUDNN
#include "caffe/util/cudnn.hpp"
#endif    /** USE_CUDNN **/

// #define ObjectMember(val) (GpuStreamPool::Get().val)


#define DISABLE_COPY_AND_ASSIGN(classname)                      \
private:                                                        \
  classname(const classname&);                                  \
  classname& operator=(const classname&)                        \

#define GetCUDAConnNum(major, minor, num) {   \
  if (major == 3 && minor == 2) {             \
    num = 8;                                  \
  } else if ((major == 3 && minor == 0) ||    \
      (major == 5 && minor == 3)) {           \
    num = 8;                                  \
  } else {                                    \
    num = 16;                                 \
  }                                           \
}

#define GetMaxBlocksNum(major, minor)    {          \
  if (major == 2) {                                 \
    return 8;                                       \
  } else if (major == 3) {                          \
    return 16;                                      \
  }                                                 \
  return 32;                                        \
}

namespace caffe{
  class GpuStreamPool
  {
    public:
      ~GpuStreamPool();

      static GpuStreamPool& Get();

      void Reset();

      void SetPoolSize(size_t pool_size = 0);

      void SetDevice(const int device_id);

      inline unsigned int GetMaxNumOfStreams() {
        return GetCUDASettings(this->device_id_);
      }

      inline cudaStream_t cuda_stream(int stream_id = -1) {
        // LOG(INFO) << "Stream ID: " << stream_id;
        if (stream_id == -1) {
          return 0;
        }

        // return ObjectMember(streams_)[stream_id % ObjectMember(handle_num_)];
        return this->streams_[stream_id % handle_num_];
      }

      inline cublasHandle_t cublas_handle(int cublas_id = -1) {
        if (cublas_id == -1) {
          // return ObjectMember(default_cublas_handle_);
          return this->default_cublas_handle_;
        }

        // return ObjectMember(cublas_handles_)[cublas_id % ObjectMember(handle_num_)];
        return this->cublas_handles_[cublas_id % handle_num_];
      }

      inline unsigned int cublas_handle_num () {
        // return ObjectMember(handle_num_);
        return this->handle_num_;
      }

#ifdef USE_CUDNN
      inline cudnnHandle_t cudnn_handle(int cudnn_id = -1) {
        if (cudnn_id == -1) {
          // return ObjectMember(default_cudnn_handle_);
          return this->default_cudnn_handle_;
        }

        // return ObjectMember(cudnn_handles_)[cudnn_id % ObjectMember(handle_num_)];
        return this->cudnn_handles_[cudnn_id % handle_num_];
      }

      inline unsigned int cudnn_handle_num () {
        // return ObjectMember(handle_num_);
        return this->handle_num_;
      }
#endif

    protected:
      int device_id_;

      cudaDeviceProp device_prop_;

      // TODO: Convolution layer-specific parameters.
      int group_;
      // Manager of streams on all GPU devices.
      cudaStream_t* streams_;

      unsigned int handle_num_;

      // Manager of cuBLAS handler on all GPU devices in all available streams.
      cublasHandle_t* cublas_handles_;
      cublasHandle_t default_cublas_handle_;

#ifdef USE_CUDNN
      // Manager of cuDNN handler on all GPU devices in all available streams.
      cudnnHandle_t* cudnn_handles_;
      cudnnHandle_t default_cudnn_handle_;
#endif  /** USE_CUDNN **/

    private:
      size_t GetCUDASettings(const int device_id);
      GpuStreamPool();

    DISABLE_COPY_AND_ASSIGN(GpuStreamPool);
  };
}

#endif    /** CAFFE_GPU_MANAGER_HPP_ **/
#endif    /** USE_PROF **/
#endif    /** CPU_ONLY **/
