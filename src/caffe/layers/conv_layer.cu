#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
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

template <typename Dtype>
__global__ void pos_kernel(const int n, const Dtype* a, Dtype* b) {
  CUDA_KERNEL_LOOP(index, n) {
    if (a[index] > 0)
      b[index] = a[index];
  }
}

template <typename Dtype>
__global__ void div_r_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    if (b[index] != 0)
      y[index] = a[index] / b[index];
    else
      y[index] = 0;
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_eb_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* W_data = this->blobs_[0]->gpu_data();
  Blob<Dtype> W_plus(this->blobs_[0]->shape());
  Dtype* W_plus_data = W_plus.mutable_gpu_data();
  caffe_gpu_set<Dtype>(W_plus.count(), Dtype(0), W_plus_data);
  pos_kernel<Dtype><<<CAFFE_GET_BLOCKS(W_plus.count()), CAFFE_CUDA_NUM_THREADS>>>(
        W_plus.count(), W_data, W_plus_data);
  
  Blob<Dtype> NN(top[0]->shape());
  Dtype* NN_data = NN.mutable_gpu_data();
  for (int i = 0; i < top.size(); ++i) {
    if (propagate_down[i]) {
      // compute the normalization factor by forwardpassing using W+
      const Dtype* bottom_data = bottom[i]->gpu_data();
      for (int n = 0; n < this->num_; ++n) {
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, W_plus_data,
              NN_data + n * this->top_dim_);
      }
      
      // do normalization
      const Dtype* top_diff = top[i]->gpu_diff();
      div_r_kernel<Dtype><<<CAFFE_GET_BLOCKS(NN.count()), CAFFE_CUDA_NUM_THREADS>>>(
            NN.count(), top_diff, NN_data, NN_data);
      
      // do backward pass
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_gemm(NN_data + n * this->top_dim_, W_plus_data,
            bottom_diff + n * this->bottom_dim_);
      }
      
      // multiply the bottom data
      caffe_gpu_mul<Dtype>(bottom[i]->count(), bottom[i]->gpu_diff(), bottom_data, bottom[i]->mutable_gpu_diff());
    }
  }

}
INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);
INSTANTIATE_LAYER_EB_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
