#ifndef CPU_ONLY

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
  if (major == 3 && (minor == 2 || minor == 5)) {             \
    num = 8;                                  \
  } else if ((major == 3 && minor == 0) ||    \
      (major == 5 && minor == 3)) {           \
    num = 8;                                  \
  } else {                                    \
    num = 16;                                 \
  }                                           \
}

#define GetMaxBlocksNum(major, minor)    {          \
  switch (major) {                                  \
  case 2:                                           \
    return 8;                                       \
  case 3:                                           \
    return 16;                                      \
  case 5:                                           \
    return 32;                                      \
  default:                                          \
    return 32;                                      \
  }                                                 \
}

namespace caffe{
  class GpuStreamPool
  {
    public:
      /**
       * @brief   GpuStreamPool Destructor
       */
      ~GpuStreamPool();

      /**
       * @brief   Object access method.
       *
       * Thread local context for GpuStreamPool. Each device has a unique GpuStreamPool
       * object to manage the connection between host and deivce.
       *
       * @return  The corresponding GpuStreamPool object.
       */
      static GpuStreamPool& Get();

      void Reset();

      /**
       * @brief   Pool size setting function.
       * @param[in] pool_size   The required pool size.
       */
      void SetPoolSize(size_t pool_size = 0);

      /**
       * @brief   Set the corresponding GPU device.
       * @param[in] device_id   The target GPU device.
       */
      void SetDevice(const int device_id);

      /**
       * @brief   Get the maximum number of GPU streams.
       */
      inline unsigned int GetMaxNumOfStreams() {
        return GetCUDASettings(this->device_id_);
      }

      /**
       * @brief   Get the kernel running CUDA stream.
       * @param[in] stream_id   ID of the selected stream_id that the kernel will be launched.
       */
      inline cudaStream_t cuda_stream(int stream_id = -1) {
        if (stream_id == -1 or this->handle_num_ == 0) {
          return 0;
        }

        // return ObjectMember(streams_)[stream_id % ObjectMember(handle_num_)];
        return this->streams_[stream_id % handle_num_];
      }

      /**
       * @brief   Get the cuBLAS kernel handler.
       * @param[in] cublas_id   ID of the selected cuBLAS handler.
       */
      inline cublasHandle_t cublas_handle(int cublas_id = -1) {
        if (cublas_id == -1 or this->handle_num_ == 0) {
          // return ObjectMember(default_cublas_handle_);
          return this->default_cublas_handle_;
        }

        // return ObjectMember(cublas_handles_)[cublas_id % ObjectMember(handle_num_)];
        return this->cublas_handles_[cublas_id % handle_num_];
      }

      /**
       * @brief   Get the number of cuBLAS handler.
       */
      inline unsigned int cublas_handle_num () {
        // return ObjectMember(handle_num_);
        return this->handle_num_;
      }

#ifdef USE_CUDNN
      /**
       * @brief   Get the cuDNN kernel handler.
       * @param[in] cudnn_id    ID of the selected cuDNN handler.
       */
      inline cudnnHandle_t cudnn_handle(int cudnn_id = -1) {
        if (cudnn_id == -1 or this->handle_num_ == 0) {
          // return ObjectMember(default_cudnn_handle_);
          return this->default_cudnn_handle_;
        }

        // return ObjectMember(cudnn_handles_)[cudnn_id % ObjectMember(handle_num_)];
        return this->cudnn_handles_[cudnn_id % handle_num_];
      }

      /**
       * @brief   Get the number of cuDNN handler.
       */
      inline unsigned int cudnn_handle_num () {
        // return ObjectMember(handle_num_);
        return this->handle_num_;
      }
#endif

    protected:
      int device_id_;   /*< The GPU device ID. */

      cudaDeviceProp device_prop_;    /*< The property of the GPU device. */

      // TODO: Convolution layer-specific parameters.
      int group_;   /*< The convolution operations group_. */
      // Manager of streams on all GPU devices.
      cudaStream_t* streams_;   /*< cudaStream_t array, used to manage CUDA stream. */

      unsigned int handle_num_;   /*< The number of CUDA streams. */

      // Manager of cuBLAS handler on all GPU devices in all available streams.
      cublasHandle_t* cublas_handles_;    /*< cublasHandle_t array. */
      cublasHandle_t default_cublas_handle_;    /*< The default cuBLAS handler binding to the default stream. */

#ifdef USE_CUDNN
      // Manager of cuDNN handler on all GPU devices in all available streams.
      cudnnHandle_t* cudnn_handles_;    /*< cudnnHandle_t array. */
      cudnnHandle_t default_cudnn_handle_;    /*< The default cuDNN handler binding to the default stream. */
#endif  /** USE_CUDNN **/

    private:
      size_t GetCUDASettings(const int device_id);
      GpuStreamPool();

    DISABLE_COPY_AND_ASSIGN(GpuStreamPool);
  };
}

#endif    /** CAFFE_GPU_MANAGER_HPP_ **/
#endif    /** CPU_ONLY **/
