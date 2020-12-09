#include <torch/extension.h>

/* returns cosine similarity between vector x and each vector in matrix xs,
 * stores the result in out
 */
torch::Tensor cos(torch::Tensor x, torch::Tensor xs, torch::Tensor out) {
	return torch::matmul_out(out, x, xs.transpose(0, 1)).div_(xs.norm(2,1)).div_(x.norm());
}

/*
 * Performs onlinehd onepass learning algorithm.
 * x = the encoded input data points
 * y = the label of each data point
 * m = the initialized model (class hypervectors) to be trained in-place
 * lr = the learning rate
 */
torch::Tensor onepass(torch::Tensor x, torch::Tensor y, torch::Tensor m, float lr) {
	int n = x.size(0);
	int c = m.size(0);
	auto options = torch::TensorOptions()
		.dtype(x.dtype())
		.device(x.device());
	auto scores = torch::empty({c}, options);

	for (int i = 0; i < n; i++) {
		auto spl = x[i];
		auto lbl = y[i];
		scores = 1.0 - cos(spl, m, scores);
		auto prd = scores.argmin();
		m[lbl].add_(spl, (lr*scores[lbl]).item());
		m[prd].add_(spl, (-lr*scores[prd]).item());
	}
	return m;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("onepass", &onepass, "OnlineHD one pass training");
}
