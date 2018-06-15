#include <vector>

#include "caffe/layers/softmaxl2_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#define checkCudaErrors( a ) do { \
	    if (cudaSuccess != (a)) { \
	    fprintf(stderr, "Cuda runtime error in line %d of file %s \
	    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
	    exit(EXIT_FAILURE); \
	    } \
	    } while(0);

/////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
__global__ void get_max_of_each_sample_kernel(const Dtype* input, Dtype* output, int probDim, int batchSize)
{
	int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if(sampleIdx < batchSize)
	{
		Dtype maxVal = input[0];
		const Dtype* prob = input + sampleIdx * probDim;
		for (int probIdx = 1; probIdx < probDim; probIdx++)
		{
			if (maxVal < prob[probIdx])
				maxVal = prob[probIdx];
		}
		Dtype* out = output + sampleIdx * probDim;
		for (int probIdx = 0; probIdx < probDim; probIdx++)
		{
			out[probIdx] = maxVal;
		}
	}
	return;
}
template<typename Dtype> void get_max_of_each_sample(const Dtype* input, Dtype* output, int probDim, int batchSize);

template<>
void get_max_of_each_sample(const float* input, float* output, int probDim, int batchSize)
{
	int thread_per_block = 4;
	int block_num = (batchSize + thread_per_block - 1) / thread_per_block;
	get_max_of_each_sample_kernel<float> << <block_num, thread_per_block >> > (input, output, probDim, batchSize);
//	cudaDeviceSynchronize();
	return;
}
template<>
void get_max_of_each_sample(const double* input, double* output, int probDim, int batchSize)
{
	int thread_per_block = 4;
	int block_num = (batchSize + thread_per_block - 1) / thread_per_block;
	get_max_of_each_sample_kernel<double> << <block_num, thread_per_block >> > (input, output, probDim, batchSize);
//	cudaDeviceSynchronize();
	return;
}



/////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
__global__ void get_sum_of_each_sample_kernel(const Dtype* input, Dtype* output, int probDim, int batchSize)
{
	int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if(sampleIdx < batchSize)
	{
		const Dtype* prob = input + sampleIdx * probDim;
		Dtype sum = 0;
		for (int k = 0; k < probDim; k++) sum += prob[k];
		if (sum < FLT_MIN)
			sum = FLT_MIN;
		Dtype* out = output + sampleIdx * probDim;
		for (int k = 0; k < probDim; k++) out[k] = sum;
	}
	return;
}

template <typename Dtype>
void get_sum_of_each_sample(const Dtype* input, Dtype* output, int probDim, int batchSize);


template <>
void get_sum_of_each_sample(const float* input, float* output, int probDim, int batchSize)
{
	int thread_per_block = 4;
	int block_num = (batchSize + thread_per_block - 1) / thread_per_block;
	get_sum_of_each_sample_kernel<float> << <block_num, thread_per_block >> > (input, output, probDim, batchSize);
	return;

}


template <>
void get_sum_of_each_sample(const double* input, double* output, int probDim, int batchSize)
{
	int thread_per_block = 4;
	int block_num = (batchSize + thread_per_block - 1) / thread_per_block;
	get_sum_of_each_sample_kernel<double> << <block_num, thread_per_block >> > (input, output, probDim, batchSize);
	return;

}





////////////////////////////////////////////////////////////////////////////////////


template <typename Dtype>
__global__ void get_LabelSubSoftmax_of_each_sample_kernel(const Dtype* inputA, const Dtype* inputB, Dtype* output, int probDim, int batchSize)
{
	int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (sampleIdx < batchSize)
	{
		const int label = (int)(inputA[sampleIdx]);
		for (int probIdx = 0; probIdx < probDim; probIdx++)
		{
			Dtype val = 0;
			if (probIdx == label)
				val = 1 - inputB[sampleIdx * probDim + probIdx];
			else
				val = 0 - inputB[sampleIdx * probDim + probIdx];
			output[sampleIdx * probDim + probIdx] = val;
		}
	}
	return;
}

template<typename Dtype> void get_LabelSubSoftmax_of_each_sample(const Dtype* inputA, const Dtype* inputB, Dtype* output, int probDim, int batchSize);


template<> void get_LabelSubSoftmax_of_each_sample(const float* inputA, const float* inputB, float* output, int probDim, int batchSize)
{
	int thread_per_block = 4;
	int block_num = (batchSize + thread_per_block - 1) / thread_per_block;

	get_LabelSubSoftmax_of_each_sample_kernel<float> << <block_num, thread_per_block >> > (inputA, inputB, output, probDim, batchSize);
	return;
}


template<> void get_LabelSubSoftmax_of_each_sample(const double* inputA, const double* inputB, double* output, int probDim, int batchSize)
{
	int thread_per_block = 4;
	int block_num = (batchSize + thread_per_block - 1) / thread_per_block;

	get_LabelSubSoftmax_of_each_sample_kernel<double> << <block_num, thread_per_block >> > (inputA, inputB, output, probDim, batchSize);
	return;
}



//////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Dtype>
__global__ void get_loss_kernel(const Dtype* inputA, const Dtype* inputB, Dtype* output, int probDim, int batchSize )
{
	int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (sampleIdx < batchSize)
	{
		const int label = (int)(inputA[sampleIdx]);
		for (int probIdx = 0; probIdx < probDim; probIdx++)
		{
			Dtype val = 0;
			if (probIdx == label)
				val = 1 - inputB[sampleIdx * probDim + probIdx];
			else
				val = 0 - inputB[sampleIdx * probDim + probIdx];
			output[sampleIdx * probDim + probIdx] = val;
		}
	}
	return;
}

template<typename Dtype> void get_loss(const Dtype* inputA, const Dtype* inputB, Dtype* output, int probDim, int batchSize);

template<> void get_loss(const float* inputA, const float* inputB, float* output, int probDim, int batchSize)
{
	int thread_per_block = 4;
	int block_num = (batchSize + thread_per_block - 1) / thread_per_block;

	get_loss_kernel<float> << <block_num, thread_per_block >> > (inputA, inputB, output, probDim, batchSize);
}

template<> void get_loss(const double* inputA, const double* inputB, double* output, int probDim, int batchSize)
{
	int thread_per_block = 4;
	int block_num = (batchSize + thread_per_block - 1) / thread_per_block;

	get_loss_kernel<double> << <block_num, thread_per_block >> > (inputA, inputB, output, probDim, batchSize);
}




///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Dtype>
void Softmaxl2LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
	int batchSize = bottom[0]->count(0, 1);
	int probDim = bottom[0]->count(1);

	Blob<Dtype> tmpA, tmpB;

	tmpA.Reshape(bottom[0]->shape());
	tmpB.Reshape(bottom[0]->shape());

	/*
	for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
	{
		Dtype maxVal = bottom[0]->gpu_data()[0];
		const Dtype* prob = bottom[0]->gpu_data() + sampleIdx * probDim;
		for (int probIdx = 1; probIdx < probDim; probIdx++)
		{
			if (maxVal < prob[probIdx])
				maxVal = prob[probIdx];
		}
		caffe_set<Dtype>(probDim, maxVal, tmpA.mutable_gpu_data() + sampleIdx * probDim);
	}
	*/
//	int thread_per_block = 4;
//	int block_num = (batchSize + thread_per_block - 1) / thread_per_block;
	//get_max_of_each_sample<Dtype> << <block_num, thread_per_block >> > (bottom[0]->gpu_data(), tmpA.mutable_gpu_data(), probDim, batchSize);
	//cudaDeviceSynchronize();
	const Dtype* input = bottom[0]->gpu_data();
	get_max_of_each_sample<Dtype>(input, tmpA.mutable_gpu_data(), probDim, batchSize);

	caffe_gpu_sub<Dtype>(bottom[0]->count(), bottom[0]->gpu_data(), tmpA.gpu_data(), tmpB.mutable_gpu_data());
	caffe_gpu_exp<Dtype>(bottom[0]->count(), tmpB.gpu_data(), softmaxVals_.mutable_gpu_data()); //exp(x)

	/*
	for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
	{
		const Dtype* prob = softmaxVals_.gpu_data() + sampleIdx * probDim;
		Dtype sum = caffe_asum<Dtype>(probDim, prob);
		if (sum < FLT_MIN)
			sum = FLT_MIN;
		caffe_set<Dtype>(probDim, sum, tmpA.mutable_gpu_data() + sampleIdx * probDim);
	}
	*/
	get_sum_of_each_sample<Dtype>(softmaxVals_.gpu_data(), tmpA.mutable_gpu_data(), probDim, batchSize);

	caffe_gpu_div<Dtype>(bottom[0]->count(), softmaxVals_.gpu_data(), tmpA.gpu_data(), tmpB.mutable_gpu_data()); //normalizing sum to 1
	caffe_copy<Dtype>(bottom[0]->count(), tmpB.gpu_data(), softmaxVals_.mutable_gpu_data());


	get_LabelSubSoftmax_of_each_sample(bottom[1]->gpu_data(), softmaxVals_.gpu_data(), label_sub_softmaxVals_.mutable_gpu_data(), probDim, batchSize);


	get_loss(bottom[1]->gpu_data(), softmaxVals_.gpu_data(), label_sub_softmaxVals_.mutable_gpu_data(), probDim, batchSize);

	Dtype lossTmp;
	caffe_gpu_asum<Dtype>(label_sub_softmaxVals_.count(), label_sub_softmaxVals_.gpu_data(), &lossTmp);
	//caffe_gpu_scal<Dtype>(top[0]->count(), 1.0 / batchSize, top[0]->mutable_gpu_data());
	top[0]->mutable_cpu_data()[0] = lossTmp / batchSize; //why CPU data??????????????????????
	/*
	Dtype lossVal = 0;
	for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
	{
		const int label = (int)(bottom[1]->gpu_data()[sampleIdx]);
		for (int probIdx = 0; probIdx < probDim; probIdx++)
		{
			Dtype val = 0;
			if (probIdx == label)
				val = 1 - softmaxVals_.gpu_data()[sampleIdx * probDim + probIdx];
			else
				val = 0 - softmaxVals_.gpu_data()[sampleIdx * probDim + probIdx];
			label_sub_softmaxVals_.mutable_gpu_data()[sampleIdx * probDim + probIdx] = val;
			lossVal += val * val; //pow(softmax,2)
		}
	}
	top[0]->mutable_gpu_data()[0] = lossVal / batchSize;
	*/
	return;

}

template <typename Dtype>
void Softmaxl2LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Blob<Dtype> tmpA, tmpB, tmpC;
	tmpA.Reshape(bottom[0]->shape());
	tmpB.Reshape(bottom[0]->shape());
	tmpC.Reshape(bottom[0]->shape());
	caffe_gpu_mul<Dtype>(bottom[0]->count(), label_sub_softmaxVals_.gpu_data(), softmaxVals_.gpu_data(), tmpA.mutable_gpu_data()); //f(x) * (y - f(x))


	caffe_gpu_set<Dtype>(bottom[0]->count(), Dtype(1), tmpB.mutable_gpu_data());
	caffe_gpu_sub<Dtype>(bottom[0]->count(), tmpB.gpu_data(), softmaxVals_.gpu_data(), tmpC.mutable_gpu_data()); //1 - f(x)

	caffe_gpu_mul<Dtype>(bottom[0]->count(), tmpA.gpu_data(), tmpC.gpu_data(), tmpB.mutable_gpu_data()); //f(x) * (y - f(x)) * (1 - f(x))

	caffe_gpu_set<Dtype>(bottom[0]->count(), Dtype(-2), tmpA.mutable_gpu_data());
	caffe_gpu_mul<Dtype>(bottom[0]->count(), tmpA.gpu_data(), tmpB.gpu_data(), bottom[0]->mutable_gpu_diff());

	//Dtype lossW = top[0]->gpu_data()[0] / bottom[0]->count(0, 1); //batch size  //copy gpu memory to cpu memory
	Dtype lossTop;
	cudaMemcpy(&lossTop, top[0]->gpu_diff(), top[0]->count() * sizeof(Dtype), cudaMemcpyDeviceToHost);
	Dtype lossW = lossTop / bottom[0]->count(0, 1); //batch size
	caffe_gpu_scal<Dtype>(bottom[0]->count(), lossW, bottom[0]->mutable_gpu_diff());
	return;
}

INSTANTIATE_LAYER_GPU_FUNCS(Softmaxl2LossLayer);

}  // namespace caffe
