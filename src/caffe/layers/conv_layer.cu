#include <vector>

#include "caffe/layers/conv_layer.hpp"

DEFINE_int32(gemmOpt, 0,
    "Optional; loop unrolling flag.");

//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
DEFINE_int32(parallelDeg, 1,
    "Optional. static loop unrolling flag (>=1).");
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Modified by Hao Fu.
    int parallel_degree = 0;
#ifdef USE_PROF
    static bool folder_flag = true;
    KernelAnalyzer::Get().AnalyzerStart(this->layer_param().name(), "LOOP1", parallel_degree);
    if (parallel_degree) {
      this->SetColBufferNum(parallel_degree);
    }
#endif
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if (parallel_degree < FLAGS_parallelDeg) {
      parallel_degree = FLAGS_parallelDeg;
    }
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if (FLAGS_gemmOpt == 0) {
#ifdef USE_PROF
      if (folder_flag) {
        LOG(INFO) << "Now is doing UNOPTIMIZED execution!";
        InfoLog::Get().SetFolder("Unoptimized");
        folder_flag = false;
      }
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#else
      static bool gpu_pool_flag = false;
      this->SetColBufferNum(FLAGS_parallelDeg);
      if (!gpu_pool_flag) {
        gpu_pool_flag = true;
        GpuStreamPool::Get().SetPoolSize(parallel_degree);
      }
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#endif
      Dtype* bias, *bias_multiplier;
      if (this->bias_term_) {
        bias = this->blobs_[1]->mutable_gpu_data();
        bias_multiplier = this->bias_multiplier_.mutable_gpu_data();
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      for (int n = 0; n < this->num_; n += parallel_degree) {
        for (int k_idx = 0; k_idx < parallel_degree; ++ k_idx) {
          int idx = (n + k_idx);
          this->forward_gpu_gemm(bottom_data + idx * this->bottom_dim_, weight,
              top_data + idx * this->top_dim_, k_idx);

          if (this->bias_term_) {
            //const Dtype* bias = this->blobs_[1]->gpu_data();
            this->forward_gpu_bias(top_data + idx * this->top_dim_, bias, bias_multiplier, k_idx);
          }
        }
      }
    } else if (FLAGS_gemmOpt == 1) {
#ifdef USE_PROF
      if (folder_flag) {
        LOG(INFO) << "Now is doing OPTIMIZED_1 execution!";
        InfoLog::Get().SetFolder("Optimized_1");
        folder_flag = false;
      }
#endif
      Dtype* bias, *bias_multiplier;
      if (this->bias_term_) {
        bias = this->blobs_[1]->mutable_gpu_data();
        bias_multiplier = this->bias_multiplier_.mutable_gpu_data();
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      for (int n = 0; n < this->num_; n += parallel_degree) {
        for (int k_idx = 0; k_idx < parallel_degree; ++ k_idx) {
          this->forward_gpu_gemm(bottom_data + (n + k_idx) * this->bottom_dim_, weight,
              top_data + (n + k_idx) * this->top_dim_, k_idx);
        }

        if (this->bias_term_) {
          //const Dtype* bias = this->blobs_[1]->gpu_data();
          for (int k_idx = 0; k_idx < parallel_degree; ++ k_idx) {
            this->forward_gpu_bias(top_data + (n + k_idx) * this->top_dim_, bias, bias_multiplier, k_idx);
          }
        }
      }
    } else {
#ifdef USE_PROF
      if (folder_flag) {
        LOG(INFO) << "Now is doing OPTIMIZED_2 execution!";
        InfoLog::Get().SetFolder("Optimized_2");
        folder_flag = false;
      }
#endif
      Dtype* bias, *bias_multiplier;
      if (this->bias_term_) {
        bias = this->blobs_[1]->mutable_gpu_data();
        bias_multiplier = this->bias_multiplier_.mutable_gpu_data();
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      for (int n = 0; n < this->num_; n += parallel_degree) {
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_, 'y', parallel_degree);

        if (this->bias_term_) {
          //const Dtype* bias = this->blobs_[1]->gpu_data();
          for (int k_idx = 0; k_idx < parallel_degree; ++ k_idx) {
            this->forward_gpu_bias(top_data + (n + k_idx) * this->top_dim_, bias, bias_multiplier, k_idx);
          }
        }
      }
    }
#ifdef USE_PROF
    KernelAnalyzer::Get().AnalyzerStop();
    //LOG(INFO) << "End " << this->layer_param().name() << "_LOOP1:" << parallel_degree;
#endif
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
