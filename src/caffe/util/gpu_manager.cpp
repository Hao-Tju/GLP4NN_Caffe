#ifndef CPU_ONLY

#include <boost/thread.hpp>
#include <cstdlib>
#include <cstring>

#include "caffe/util/gpu_manager.hpp"

using std::malloc;
using std::realloc;
using std::free;

namespace caffe {
  // Make sure each thread can have different values.
  static boost::thread_specific_ptr<GpuStreamPool> thread_gsp_instance_;

  GpuStreamPool& GpuStreamPool::Get() {
    if (!thread_gsp_instance_.get()) {
      thread_gsp_instance_.reset(new GpuStreamPool());
    }

    return *(thread_gsp_instance_.get());
  }

  void GpuStreamPool::Reset() {
    thread_gsp_instance_.reset();
    return ;
  }

  void GpuStreamPool::SetPoolSize(size_t pool_size) {
    if (pool_size > GetCUDASettings(this->device_id_)) {
      LOG(INFO) << "Maximum concurrent streams between DEVICE " << this->device_id_
      << " and HOST is " << GetCUDASettings(this->device_id_) << ".";
      pool_size = GetCUDASettings(this->device_id_);
    }

    LOG(INFO) << "Pool size: " << handle_num_ << " ---> " << pool_size;
    if (this->handle_num_ >= pool_size) {
      return ;
    } else {
      cudaStream_t* temp_streams = NULL;
      cublasHandle_t* temp_cublas_handler = NULL;
#ifdef USE_CUDNN
      cudnnHandle_t* temp_cudnn_handler = NULL;
#endif
      if (this->handle_num_ != 0) {
        temp_streams = this->streams_;
        temp_cublas_handler = this->cublas_handles_;
#ifdef USE_CUDNN
        temp_cudnn_handler = this->cudnn_handles_;
#endif
      }

      this->streams_ = new cudaStream_t[pool_size];
      this->cublas_handles_ = new cublasHandle_t[pool_size];

      if (this->streams_ == NULL || this->cublas_handles_ == NULL) {
        LOG(FATAL) << "Failed to malloc GpuStreamPool member handler.";
      }

#ifdef USE_CUDNN
      this->cudnn_handles_ = new cudnnHandle_t[pool_size];
      if (this->cudnn_handles_ == NULL) {
        LOG(FATAL) << "Failed to malloc cuDNN handler.";
      }
#endif

      memcpy(this->streams_, temp_streams, sizeof(cudaStream_t) * this->handle_num_);
      memcpy(this->cublas_handles_, temp_cublas_handler,
          sizeof(cublasHandle_t) * this->handle_num_);
#ifdef USE_CUDNN
      memcpy(this->cudnn_handles_, temp_cudnn_handler,
          sizeof(cudnnHandle_t) * this->handle_num_);
#endif
      for (int i = this->handle_num_; i < pool_size; ++ i) {
        CUDA_CHECK(cudaStreamCreate(&this->streams_[i]));
        CUBLAS_CHECK(cublasCreate(&this->cublas_handles_[i]));
        CUBLAS_CHECK(cublasSetStream(this->cublas_handles_[i], this->streams_[i]));
#ifdef USE_CUDNN
        CUDNN_CHECK(cudnnCreate(&this->cudnn_handles_[i]));
        CUDNN_CHECK(cudnnSetStream(this->cudnn_handles_[i], this->streams_[i]));
#endif
      }

      if (this->handle_num_ != 0) {
        if (temp_streams != NULL) {
          delete[] temp_streams;
        }
        if (temp_cublas_handler != NULL) {
          delete[] temp_cublas_handler;
        }
#ifdef USE_CUDNN
        if (temp_cudnn_handler != NULL) {
          delete[] temp_cudnn_handler;
        }
#endif
      }

      this->handle_num_ = pool_size;
    }
  }

  GpuStreamPool::GpuStreamPool() {
    CUDA_CHECK(cudaGetDevice(&this->device_id_));
    this->handle_num_ = 0;
    this->streams_ = NULL;
    this->cublas_handles_ = NULL;
#ifdef USE_CUDNN
    this->cudnn_handles_ = NULL;
#endif

    CUBLAS_CHECK(cublasCreate(&default_cublas_handle_));
    CUBLAS_CHECK(cublasSetStream(default_cublas_handle_, 0));
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnCreate(&default_cudnn_handle_));
    CUDNN_CHECK(cudnnSetStream(default_cudnn_handle_, 0));
#endif
  }

  GpuStreamPool::~GpuStreamPool() {
    cudaSetDevice(this->device_id_);
    for (int i = 0; i < handle_num_; ++ i) {
      CUBLAS_CHECK(cublasDestroy(cublas_handles_[i]));
#ifdef USE_CUDNN
      CUDNN_CHECK(cudnnDestroy(cudnn_handles_[i]));
#endif
      CUDA_CHECK(cudaStreamDestroy(streams_[i]));
    }

    CUBLAS_CHECK(cublasDestroy(default_cublas_handle_));
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnDestroy(default_cudnn_handle_));
#endif

    if (this->handle_num_ != 0) {
      delete[] this->streams_;
      delete[] this->cublas_handles_;
#ifdef USE_CUDNN
      delete[] this->cudnn_handles_;
#endif
    }
    this->handle_num_ = 0;
  }

  size_t GpuStreamPool::GetCUDASettings(const int device_id) {
    size_t temp_handles_num = 0;
    const char* cuda_connections = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS");
    // CUDA_CHECK(cudaGetDevice(&this->device_id_));
    CUDA_CHECK(cudaGetDeviceProperties(&this->device_prop_, device_id));

    if (cuda_connections != NULL && strlen(cuda_connections) != 0) {
      temp_handles_num = static_cast<size_t>(std::atoi(cuda_connections));
    } else {
      GetCUDAConnNum(this->device_prop_.major,
          this->device_prop_.minor,
          temp_handles_num);
    }

    return temp_handles_num;
  }

  void GpuStreamPool::SetDevice(const int device_id) {
    if (device_id_ == device_id) {
      return ;
    } else {
      CUBLAS_CHECK(cublasDestroy(default_cublas_handle_));
#ifdef USE_CUDNN
      CUDNN_CHECK(cudnnDestroy(default_cudnn_handle_));
#endif

      if (this->handle_num_ != 0) {
        for (int i = 0; i < this->handle_num_; ++ i) {
          CUBLAS_CHECK(cublasDestroy(this->cublas_handles_[i]));
#ifdef USE_CUDNN
          CUDNN_CHECK(cudnnDestroy(this->cudnn_handles_[i]));
#endif
          CUDA_CHECK(cudaStreamDestroy(this->streams_[i]));
        }

        delete[] this->streams_;
        delete[] this->cublas_handles_;
#ifdef USE_CUDNN
        delete[] this->cudnn_handles_;
#endif
      }
    }

    CUDA_CHECK(cudaSetDevice(device_id));
    this->device_id_ = device_id;
    this->handle_num_ = 0;

    CUBLAS_CHECK(cublasCreate(&this->default_cublas_handle_));
    CUBLAS_CHECK(cublasSetStream(this->default_cublas_handle_, 0));
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnCreate(&this->default_cudnn_handle_));
    CUDNN_CHECK(cudnnSetStream(this->default_cudnn_handle_, 0));
#endif

    return ;
  }
}

#endif    /** CPU_ONLY **/
