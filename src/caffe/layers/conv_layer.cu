#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

__global__ void sync_convs() {}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    int parallel_degree = 0;
    KernelAnalyzer::Get().AnalyzerStart(this->layer_param().name(), "LOOP1", parallel_degree);
    LOG(INFO) << "(Before Analyzing) Parallel Degree of " << this->layer_param().name() << "@"
      << "LOOP1 is: " << parallel_degree << ".";
    // for (int n = 0; n < this->num_; n += parallel_degree) {
    for (int n = 0; n < this->num_; n ++) {
      int stream_id = parallel_degree ? n % parallel_degree : -1;
      //for (int k = 0; k < parallel_degree && (n + k) < this->num_; ++ k) {
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_, stream_id);
      //}

      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();

        //for (int k = 0; k < parallel_degree && (n + k) < this->num_; ++ k) {
          this->forward_gpu_bias(top_data + n * this->top_dim_, bias, stream_id);
        //}
      }
    }
    KernelAnalyzer::Get().AnalyzerStop();
    std::cout << "(After Analyzing) Parallel Degree of " << this->layer_param().name() << "@"
      << "LOOP1 is: " << parallel_degree;
    /*
    if (this->parallel_degree_ <= 1) {
      for (int n = 0; n < this->num_; ++n) {
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();
          this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
        }
      }
    } else {
      // for (int n = 0; n < this->num_; ++ n) {
      for (int n = 0; n < this->num_; n += this->parallel_degree_) {
      //   int stream_id = n % this->parallel_degree_;
        for (int i = 0; i < this->parallel_degree_; ++ i) {
      //   this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
      //       top_data + n * this->top_dim_,
      //       stream_id);
          if ((n + i) < this->num_) {
            this->forward_gpu_gemm(bottom_data + (n+i) * this->bottom_dim_, weight,
                top_data + (n+i) * this->top_dim_,
                i);
          }
        }
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();
          for (int i = 0; i < this->parallel_degree_; ++ i) {
          // this->forward_gpu_bias(top_data + n * this->top_dim_, bias,
          //     stream_id);
            if ((n + i) < this->num_) {
              this->forward_gpu_bias(top_data + (n+i) * this->top_dim_, bias,
                  i);
            }
          }
        }
      }
      sync_convs<<<1,1>>>();
    }
    */
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
