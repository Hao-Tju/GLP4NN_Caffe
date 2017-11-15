#include <vector>

#include "caffe/layers/conv_layer.hpp"

DEFINE_int32(gemmOpt, 0,
    "Optional; loop unrolling flag.");

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
    static bool folder_flag = true;
#ifdef USE_PROF
    KernelAnalyzer::Get().AnalyzerStart(this->layer_param().name(), "LOOP1", parallel_degree);
    if (parallel_degree) {
      this->SetColBufferNum(parallel_degree);
    } else {
      parallel_degree = 1;
    }
#endif
    if (FLAGS_gemmOpt == 0) {
      if (folder_flag) {
        LOG(INFO) << "Now is doing UNOPTIMIZED execution!";
        InfoLog::Get().SetFolder("Unoptimized");
        folder_flag = false;
      }
      for (int n = 0; n < this->num_; n += parallel_degree) {
        for (int k_idx = 0; k_idx < parallel_degree; ++ k_idx) {
          this->forward_gpu_gemm(bottom_data + (n + k_idx) * this->bottom_dim_, weight,
              top_data + (n + k_idx) * this->top_dim_, k_idx);

          if (this->bias_term_) {
            const Dtype* bias = this->blobs_[1]->gpu_data();

            this->forward_gpu_bias(top_data + (n + k_idx) * this->top_dim_, bias, k_idx);
          }
        }
      }
    } else if (FLAGS_gemmOpt == 1) {
      if (folder_flag) {
        LOG(INFO) << "Now is doing OPTIMIZED_1 execution!";
        InfoLog::Get().SetFolder("Optimized_1");
        folder_flag = false;
      }
      for (int n = 0; n < this->num_; n += parallel_degree) {
        for (int k_idx = 0; k_idx < parallel_degree; ++ k_idx) {
          this->forward_gpu_gemm(bottom_data + (n + k_idx) * this->bottom_dim_, weight,
              top_data + (n + k_idx) * this->top_dim_, k_idx);
        }

        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();

          for (int k_idx = 0; k_idx < parallel_degree; ++ k_idx) {
            this->forward_gpu_bias(top_data + (n + k_idx) * this->top_dim_, bias, k_idx);
          }
        }
      }
    } else {
      if (folder_flag) {
        LOG(INFO) << "Now is doing OPTIMIZED_2 execution!";
        InfoLog::Get().SetFolder("Optimized_2");
        folder_flag = false;
      }
      for (int n = 0; n < this->num_; n += parallel_degree) {
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_, 'y', parallel_degree);

        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();

          for (int k_idx = 0; k_idx < parallel_degree; ++ k_idx) {
            this->forward_gpu_bias(top_data + (n + k_idx) * this->top_dim_, bias, k_idx);
          }
        }
      }
    }
#ifdef USE_PROF
    KernelAnalyzer::Get().AnalyzerStop();
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
