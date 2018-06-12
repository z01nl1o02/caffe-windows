#include <vector>

#include "caffe/layers/softmaxl2_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Softmaxl2LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
 
}

template <typename Dtype>
void Softmaxl2LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 
}

INSTANTIATE_LAYER_GPU_FUNCS(Softmaxl2LossLayer);

}  // namespace caffe
