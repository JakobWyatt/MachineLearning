//Copyright 2017 Jakob Wyatt
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//http ://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

#include "stdafx.h"
#include "nn.h"

#include <stdexcept>
#include <vector>
#include <utility>
#include <fstream>
#include <iterator>

#include "math.h"

//we will only error-check if the project is in debug mode
//we use the _DEBUG macro to check for this
namespace nn {

data::data(int flag) {
	switch (flag) {
	case mnisttest:
	{
		_data = mnisttestload();
	}
	case mnisttrain:
	{
		_data = mnisttrainload();
	}

	default:
	{
		throw std::invalid_argument("invalid flag");
	}
	
	}
}


std::vector<std::pair<math::matrix, math::matrix>> data::mnisttestload() {
	std::ifstream images("t10k-images.idx3-ubyte", std::ios::binary);
	std::vector<unsigned char> imagesvec((std::istreambuf_iterator<char>(images)), std::istreambuf_iterator<char>());
	
	std::ifstream labels("t10k-labels.idx1-ubyte", std::ios::binary);
	std::vector<unsigned char> labelsvec((std::istreambuf_iterator<char>(labels)), std::istreambuf_iterator<char>());

	std::vector<std::pair<math::matrix, math::matrix>> result(10000);
	std::vector<math::num> image(784);

	for (std::vector<std::pair<math::matrix, math::matrix>>::size_type i = 0; i != 10000; ++i) {
		result[i].second = math::matrix::onehotmatrix(10, 1, static_cast<math::matrix::size_type>(labelsvec[i + 8]), 0);
		for (std::vector<math::num>::size_type j = 0; j != 784; ++j) {
			image[j] = static_cast<math::num>(imagesvec[784 * i + 16 + j]) / 256;
		}
		result[i].first = math::matrix(image, 1);
	}

	return result;
}

std::vector<std::pair<math::matrix, math::matrix>> data::mnisttrainload() {
	std::ifstream images("train-images.idx3-ubyte", std::ios::binary);
	std::vector<unsigned char> imagesvec((std::istreambuf_iterator<char>(images)), std::istreambuf_iterator<char>());

	std::ifstream labels("train-labels.idx1-ubyte", std::ios::binary);
	std::vector<unsigned char> labelsvec((std::istreambuf_iterator<char>(labels)), std::istreambuf_iterator<char>());

	std::vector<std::pair<math::matrix, math::matrix>> result(60000);
	std::vector<math::num> image(784);

	for (std::vector<std::pair<math::matrix, math::matrix>>::size_type i = 0; i != 60000; ++i) {
		result[i].second = math::matrix::onehotmatrix(10, 1, static_cast<math::matrix::size_type>(labelsvec[i + 8]), 0);
		for (std::vector<math::num>::size_type j = 0; j != 784; ++j) {
			image[j] = static_cast<math::num>(imagesvec[784 * i + 16 + j]) / 256;
		}
		result[i].first = math::matrix(image, 1);
	}

	return result;
}

}