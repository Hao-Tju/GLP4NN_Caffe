#include <boost/thread.hpp>
#include <cstdlib>

#include "caffe/util/gpu_manager.hpp"

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
  }

  void GpuStreamPool::SetPoolSize(size_t pool_size) {
    // thread_gsp_instance_.reset(new GpuStreamPool(pool_size));
  }

  GpuStreamPool::GpuStreamPool() {
    CUDA_CHECK(cudaGetDevice(&this->device_id_));
    this->handle_num_ = 0;

    CUBLAS_CHECK(cublasCreate(&default_cublas_handle_));
    CUBLAS_CHECK(cublasSetStream(default_cublas_handle_, 0));
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnCreate(&default_cudnn_handle_));
    CUDNN_CHECK(cudnnSetStream(default_cudnn_handle_, 0));
#endif
  }

  /*
  GpuStreamPool::GpuStreamPool(size_t pool_size) {
    CUDA_CHECK(cudaGetDevice(&this->device_id_));
    this->handle_num_ = 0;

    GetCUDASettings(this->device_id_);

    if (pool_size > this->handle_num_) {
      LOG(INFO) << "Oops! Number of streams requested is out of bounds! Maximum number of concurrent kernels supported is "
        << this->handle_num_ << " (" << pool_size << " requested).";

      pool_size = this->handle_num_;
    }

    if (this->handle_num_ != 0 &&
        this->handle_num_ < pool_size) {
      LOG(INFO) << "Maximum connections between CPU and GPU is " << this->handle_num_ << "! CANNOT establish " << pool_size << " concurrent connections!";
      LOG(INFO) << "Connections between CPU and GPU is setted as " << this->handle_num_;
    } else {
      this->handle_num_ = pool_size;
    }
    // Need to be completed.
  }
  */

  GpuStreamPool::~GpuStreamPool() {
    cudaSetDevice(this->device_id_);
    for (int i = 0; i < handle_num_; ++ i) {
      CUDA_CHECK(cudaStreamDestroy(streams_[i]));
      CUBLAS_CHECK(cublasDestroy(cublas_handles_[i]));
#ifdef USE_CUDNN
      CUDNN_CHECK(cudnnDestroy(cudnn_handles_[i]));
#endif
    }

    CUBLAS_CHECK(cublasDestroy(default_cublas_handle_));
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnDestroy(default_cudnn_handle_));
#endif

    if (streams_ != NULL) {
      delete[] streams_;
    }
    delete[] cublas_handles_;
#ifdef USE_CUDNN
    delete[] cudnn_handles_;
#endif
  }

  void GpuStreamPool::GetCUDASettings(const int device_id) {
    if (this->handle_num_ != 0) {
      return ;
    }

    const char* cuda_connections = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS");
    // CUDA_CHECK(cudaGetDevice(&this->device_id_));
    CUDA_CHECK(cudaGetDeviceProperties(&this->device_prop_, device_id));

    if (cuda_connections != NULL && strlen(cuda_connections) != 0) {
      this->handle_num_ = std::atoi(cuda_connections);
    } else {
      // this->handle_num_ = 0;
      GetCUDAConnNum(this->device_prop_.major,
          this->device_prop_.minor,
          this->handle_num_);
    }

    return ;
  }

  void GpuStreamPool::SetDevice(const int device_id) {
    // if (ObjectMember(device_id_) == device_id) {
    if (device_id_ == device_id) {
      return ;
    }
    /*
    const char* cuda_connections = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS");
    CUDA_CHECK(cudaGetDeviceProperties(&ObjectMember(device_prop_), device_id));

    ObjectMember(handle_num_) = std::atoi(cuda_connections);
    if (ObjectMember(handle_num_) == 0) {
      GetCUDAConnNum(ObjectMember(device_prop_).major,
          ObjectMember(device_prop_).minor,
          ObjectMember(handle_num_));
    }
    */

    CUDA_CHECK(cudaSetDevice(device_id));
    // ObjectMember(device_id_) = device_id;
    // ObjectMember(handle_num_) = 0;
    // ObjectMember(GetCUDASettings(device_id));
    this->device_id_ = device_id;
    this->handle_num_ = 0;
    GetCUDASettings(device_id);

    /*
    for (int i = 0; i < ObjectMember(handle_num_); ++ i) {
      CUDA_CHECK(cudaStreamDestroy(ObjectMember(streams_)[i]));
      CUBLAS_CHECK(cublasDestroy(ObjectMember(cublas_handles_)[i]));
#ifdef USE_CUDNN
      CUDNN_CHECK(cudnnDestroy(ObjectMember(cudnn_handles_)[i]));
#endif
    }
    */
    for (int i = 0; i < this->handle_num_; ++ i) {
      CUDA_CHECK(cudaStreamDestroy(this->streams_[i]));
      CUBLAS_CHECK(cublasDestroy(this->cublas_handles_[i]));
#ifdef USE_CUDNN
      CUDNN_CHECK(cudnnDestroy(this->cudnn_handles_[i]));
#endif
    }

    /*
    delete[] ObjectMember(streams_);
    delete[] ObjectMember(cublas_handles_);
#ifdef USE_CUDNN
    delete[] ObjectMember(cudnn_handles_);
#endif
    */
    delete[] this->streams_;
    delete[] this->cublas_handles_;
#ifdef USE_CUDNN
    delete[] this->cudnn_handles_;
#endif

    /*
    CUBLAS_CHECK(cublasDestroy(ObjectMember(default_cublas_handle_)));
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnDestroy(ObjectMember(default_cudnn_handle_)));
#endif
    */
    CUBLAS_CHECK(cublasDestroy(this->default_cublas_handle_));
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnDestroy(this->default_cudnn_handle_));
#endif

    /*
    ObjectMember(streams_) = new cudaStream_t[ObjectMember(handle_num_)];
    ObjectMember(cublas_handles_) = new cublasHandle_t[ObjectMember(handle_num_)];
#ifdef USE_CUDNN
    ObjectMember(cudnn_handles_) = new cudnnHandle_t[ObjectMember(handle_num_)];
#endif
    for (int i = 0; i < ObjectMember(handle_num_); ++ i) {
      CUDA_CHECK(cudaStreamCreate(&ObjectMember(streams_)[i]));
      CUBLAS_CHECK(cublasCreate(&ObjectMember(cublas_handles_)[i]));
      CUBLAS_CHECK(cublasSetStream(ObjectMember(cublas_handles_)[i],
            ObjectMember(streams_)[i]));
#ifdef USE_CUDNN
      CUDNN_CHECK(cudnnCreate(&ObjectMember(cudnn_handles_)[i]));
      CUDNN_CHECK(cudnnSetStream(ObjectMember(cudnn_handles_)[i],
            ObjectMember(streams_)[i]));
#endif
    }

    CUBLAS_CHECK(cublasCreate(&ObjectMember(default_cublas_handle_)));
    CUBLAS_CHECK(cublasSetStream(ObjectMember(default_cublas_handle_), 0));
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnCreate(&ObjectMember(default_cudnn_handle_)));
    CUDNN_CHECK(cudnnSetStream(ObjectMember(default_cudnn_handle_), 0));
#endif
    */
    this->streams_ = new cudaStream_t[this->handle_num_];
    this->cublas_handles_ = new cublasHandle_t[this->handle_num_];
#ifdef USE_CUDNN
    this->cudnn_handles_ = new cudnnHandle_t[this->handle_num_];
#endif
    for (int i = 0; i < this->handle_num_; ++ i) {
      CUDA_CHECK(cudaStreamCreate(&this->streams_[i]));
      CUBLAS_CHECK(cublasCreate(&this->cublas_handles_[i]));
      CUBLAS_CHECK(cublasSetStream(this->cublas_handles_[i],
            this->streams_[i]));
#ifdef USE_CUDNN
      CUDNN_CHECK(cudnnCreate(&this->cudnn_handles_[i]));
      CUDNN_CHECK(cudnnSetStream(this->cudnn_handles_[i],
            this->streams_[i]));
#endif
    }

    CUBLAS_CHECK(cublasCreate(&this->default_cublas_handle_));
    CUBLAS_CHECK(cublasSetStream(this->default_cublas_handle_, 0));
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnCreate(&this->default_cudnn_handle_));
    CUDNN_CHECK(cudnnSetStream(this->default_cudnn_handle_, 0));
#endif

    return ;
  }
}
