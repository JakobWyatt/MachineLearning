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
#include <memory>

#include "math.h"

//we will only error-check if the project is in debug mode
//we use the _DEBUG macro to check for this
namespace nn {

data::data(int flag) : _data(0) {
	switch (flag) {
	case mnisttest:
	{
		_data = mnisttestload();
		break;
	}
	case mnisttrain:
	{
		_data = mnisttrainload();
		break;
	}

	default:
	{
		throw std::invalid_argument("invalid flag");
		break;
	}
	
	}
}

data::size_type data::size() const {
	return this->_data.size();
}

const std::pair<math::matrix, math::matrix>& data::operator[](data::size_type element) const {
#ifdef _DEBUG
	if (element < 0 || element >= this->size()) {
		throw std::out_of_range("out of range");
	}
#endif

	return this->_data[element];
}

std::vector<std::pair<math::matrix, math::matrix>> data::mnisttestload() {
	std::ifstream images("t10k-images.idx3-ubyte", std::ios::binary);
	std::ifstream labels("t10k-labels.idx1-ubyte", std::ios::binary);

#ifdef _DEBUG
	if (!images.is_open() || !labels.is_open()) {
		throw std::runtime_error("could not open file");
	}
#endif

	std::vector<unsigned char> imagesvec((std::istreambuf_iterator<char>(images)), std::istreambuf_iterator<char>());
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
	std::ifstream labels("train-labels.idx1-ubyte", std::ios::binary);

#ifdef _DEBUG
	if (!images.is_open() || !labels.is_open()) {
		throw std::runtime_error("could not open file");
	}
#endif

	std::vector<unsigned char> imagesvec((std::istreambuf_iterator<char>(images)), std::istreambuf_iterator<char>());
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

nn::nn(std::initializer_list<layer*> layers) : _data(0) {
	std::initializer_list<layer*>::const_iterator end = layers.end();
	for (std::initializer_list<layer*>::const_iterator i = layers.begin(); i != end; ++i) {
		_data.push_back(std::move((*i)->clone()));
	}

#ifdef _DEBUG
	std::vector<std::unique_ptr<layer>>::size_type size = _data.size();
	if (size == 0) {
		throw std::invalid_argument("no layers were given");
	}
	for (std::vector<std::unique_ptr<layer>>::size_type i = 0; i != size - 1; ++i) {
		layer::size_type currentoutputwidth = _data[i]->outputwidth();
		layer::size_type nextinputwidth = _data[i + 1]->inputwidth();
		layer::size_type currentoutputheight = _data[i]->outputheight();
		layer::size_type nextinputheight = _data[i + 1]->inputheight();
		if (currentoutputwidth != nextinputwidth || currentoutputheight != nextinputheight) {
			throw std::invalid_argument("layer sizes do not match");
		}
	}
#endif
}

nn::size_type nn::size() const {
	return this->_data.size();
}

//preallocation is not used, as in this function, the output is only evaluated once
math::matrix nn::evaluate(const math::matrix& input) const {
#ifdef _DEBUG
	if (input.width() != this->_data[0]->inputwidth() || input.height() != this->_data[0]->inputheight()) {
		throw std::invalid_argument("input size is incompatible");
	}
#endif

	size_type size = this->size();
	math::matrix result(input);
	for (size_type i = 0; i != size; ++i) {
		result = this->_data[i]->evaluate(result);
	}

	return result;
}

sigmoid::sigmoid(size_type height, size_type width) : _height(height), _width(width) {
#ifdef _DEBUG
	if (height <= 0 || width <= 0) {
		throw std::invalid_argument("empty layer initialization");
	}
#endif
}

sigmoid::size_type sigmoid::inputwidth() const {
	return this->_width;
}

sigmoid::size_type sigmoid::inputheight() const {
	return this->_height;
}

sigmoid::size_type sigmoid::outputwidth() const {
	return this->_width;
}

sigmoid::size_type sigmoid::outputheight() const {
	return this->_height;
}

math::matrix sigmoid::evaluate(const math::matrix& input) const {
#ifdef _DEBUG
	if (this->inputheight() != input.height() || this->inputwidth() != input.width()) {
		throw std::invalid_argument("input matrix is incompatible");
	}
#endif

	return input(math::sigmoid);
}

std::unique_ptr<layer> sigmoid::clone() const {
	std::unique_ptr<layer> ptr(new sigmoid(*this));
	return std::move(ptr);
}

void sigmoid::evaluate(const math::matrix& input, math::matrix& output) const {
#ifdef _DEBUG
	if (this->inputheight() != input.height() || this->inputwidth() != input.width()) {
		throw std::invalid_argument("input matrix is incompatible");
	}
	if (this->outputheight() != output.height() || this->outputwidth() != output.width()) {
		throw std::invalid_argument("output matrix is incompatible");
	}
#endif

	math::matrix::function(math::sigmoid, input, output);
}

}