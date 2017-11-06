#include <vector>

#include "caffe/layers/conv_layer.hpp"

DEFINE_bool(gemmOpt, false,
    "Optional; loop unrolling flag.");

namespace caffe {

__global__ void sync() {}

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
    KernelAnalyzer::Get().AnalyzerStart(this->layer_param().name(), "LOOP1", parallel_degree);
    if (parallel_degree) {
      this->SetColBufferNum(parallel_degree);
    }
#endif
    static bool folder_flag = true;
    if (!FLAGS_gemmOpt) {
      if (folder_flag) {
        InfoLog::Get().SetFolder("Unoptimized");
        folder_flag = false;
      }
      for (int n = 0; n < this->num_; n ++) {
        int stream_id = parallel_degree ? n % parallel_degree : -1;
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_, stream_id);

        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();

          this->forward_gpu_bias(top_data + n * this->top_dim_, bias, stream_id);
        }
      }
    } else {
      if (folder_flag) {
        InfoLog::Get().SetFolder("Optimized");
        folder_flag = false;
      }
      if (parallel_degree == 0) {
        parallel_degree = 1;
      }
      for (int n = 0; n < this->num_; n += parallel_degree) {
        // int stream_id = parallel_degree ? n % parallel_degree : -1;
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_, 'y', parallel_degree);

        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();

          for (int k_idx = 1; k_idx <= parallel_degree; ++ k_idx) {
            this->forward_gpu_bias(top_data + n * this->top_dim_, bias, k_idx - 1);
          }
        }
      }
    }
#ifdef USE_PROF
    KernelAnalyzer::Get().AnalyzerStop();
    sync<<<1,1>>>();
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
