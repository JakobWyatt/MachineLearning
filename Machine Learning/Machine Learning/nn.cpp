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
#include <functional>
#include <algorithm>

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

	case XOR:
	{
		_data = generateXOR();
		break;
	}

	default:
	{
		throw std::invalid_argument("invalid flag");
		break;
	}
	
	}
}

data::data(std::vector<std::pair<math::matrix, math::matrix>> data) : _data(data) {
//empty function
//no need to error check as function is private
}

data::size_type data::size() const {
	return this->_data.size();
}

math::matrix::size_type data::inputheight() const {
	return this->_data[0].first.height();
}

math::matrix::size_type data::inputwidth() const {
	return this->_data[0].first.width();
}

math::matrix::size_type data::outputheight() const {
	return this->_data[0].second.height();
}

math::matrix::size_type data::outputwidth() const {
	return this->_data[0].second.width();
}

data data::shuffle() const {
	data result(*this);
	std::shuffle(result._data.begin(), result._data.end(), math::default_random_engine());
	return result;
}

data data::trim(size_type size) const {
	return data(std::vector<std::pair<math::matrix, math::matrix>>(this->_data.begin(), this->_data.begin() + size));
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

std::vector<std::pair<math::matrix, math::matrix>> data::generateXOR() {
	std::vector<std::pair<math::matrix, math::matrix>> result(5000);
	
	for (data::size_type i = 0; i != 5000; ++i) {
		result[i].first = math::matrix(2, 1, math::bernoullidist);

		//evaluate logical XOR
		if (result[i].first[0] == 1) {
			if (result[i].first[1] == 1) {
				result[i].second = math::matrix(1, 1, { 0 });
			}
			else {
				result[i].second = math::matrix(1, 1, { 1 });
			}
		}
		else {
			if (result[i].first[1] == 1) {
				result[i].second = math::matrix(1, 1, { 1 });
			}
			else {
				result[i].second = math::matrix(1, 1, { 0 });
			}
		}
	}

	return result;
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

void nn::train(const data& learningdata, math::num learningrate, data::size_type batchsize) {
	data::size_type batchnum = learningdata.size()/batchsize;
	nn::size_type nnsize = this->size();

#ifdef _DEBUG
	if (this->_data[0]->inputheight() != learningdata.inputheight() || this->_data[0]->inputwidth() != learningdata.inputwidth()) {
		throw std::invalid_argument("input data is incompatible");
	}
	if (this->_data[nnsize - 1]->outputheight() != learningdata.outputheight() || this->_data[nnsize - 1]->outputwidth() != learningdata.outputwidth()) {
		throw std::invalid_argument("output data is incompatible");
	}
#endif

	//get our minibatch pointers
	std::vector<void*> minibatchptr(this->allocateminibatch());
	//get our iteration pointers
	std::vector<void*> iterationptr(this->allocateiteration());
		
	//preallocate buffers
	math::matrix resultbuffer(this->_data[nnsize - 1]->outputheight(), this->_data[nnsize - 1]->outputwidth());
	math::matrix inputerrorbuffer(this->_data[0]->inputheight(), this->_data[0]->inputwidth());
	std::vector<math::matrix> buffervec;
	for (nn::size_type i = 0; i != nnsize; ++i) {
		math::matrix buffer(this->_data[i]->outputheight(), this->_data[i]->outputwidth());
		buffervec.push_back(buffer);
	}

	//iterate across our batches
	for (data::size_type i = 0; i != batchnum; ++i) {
		//iterate over a minibatch
		for (data::size_type j = 0; j != batchsize; ++j) {
			//feedforward
			this->_data[0]->feedforward(learningdata[i * batchsize + j].first, buffervec[0], iterationptr[0], minibatchptr[0]);
			for (nn::size_type k = 1; k != nnsize; ++k) {
				this->_data[k]->feedforward(buffervec[k - 1], buffervec[k], iterationptr[k], minibatchptr[k]);
			}
			//calculate difference between output and desired (aL - y)
			math::matrix::subtract(buffervec[nnsize - 1], learningdata[i * batchsize + j].second, resultbuffer);
			//backpropagate the error
			this->_data[nnsize - 1]->backprop(resultbuffer, buffervec[nnsize - 1], iterationptr[nnsize - 1], minibatchptr[nnsize - 1]);
			for (nn::size_type k = nnsize - 2; k != 0; --k) {
				this->_data[k]->backprop(buffervec[k], buffervec[k - 1], iterationptr[k], minibatchptr[k]);
			}
			this->_data[0]->backprop(buffervec[0], inputerrorbuffer, iterationptr[0], minibatchptr[0]);
		}
		this->update(minibatchptr, learningrate);
	}

	//deallocate our memory
	this->deallocateiteration(iterationptr);
	this->deallocateminibatch(minibatchptr);
}

data::size_type nn::test(const data& input, std::function<bool(const math::matrix&, const math::matrix&, math::matrix&)> compare) const {
	nn::size_type nnsize = this->size();
	data::size_type datasize = input.size();

#ifdef _DEBUG
	if (this->_data[0]->inputheight() != input.inputheight() || this->_data[0]->inputwidth() != input.inputwidth()) {
		throw std::invalid_argument("input data is incompatible");
	}
	if (this->_data[nnsize - 1]->outputheight() != input.outputheight() || this->_data[nnsize - 1]->outputwidth() != input.outputwidth()) {
		throw std::invalid_argument("output data is incompatible");
	}
#endif

	//preallocate buffers
	data::size_type numcorrect = 0;
	std::vector<math::matrix> buffervec;
	math::matrix resultbuffer(this->_data[nnsize - 1]->outputheight(), this->_data[nnsize - 1]->outputwidth());
	for (nn::size_type i = 0; i != nnsize; ++i) {
		math::matrix buffer(this->_data[i]->outputheight(), this->_data[i]->outputwidth());
		buffervec.push_back(buffer);
	}

	for (data::size_type i = 0; i != datasize; ++i) {
		this->_data[0]->evaluate(input[i].first, buffervec[0]);
		for (nn::size_type j = 1; j != nnsize; ++j) {
			this->_data[j]->evaluate(buffervec[j - 1], buffervec[j]);
		}
		if (compare(input[i].second, buffervec[nnsize - 1], resultbuffer)) {
			++numcorrect;
		}
	}

	return numcorrect;
}

math::num nn::cost(const data& input, std::function<math::num(const math::matrix& correct, const math::matrix& output)> cost) {
	nn::size_type nnsize = this->size();
	data::size_type datasize = input.size();

#ifdef _DEBUG
	if (this->_data[0]->inputheight() != input.inputheight() || this->_data[0]->inputwidth() != input.inputwidth()) {
		throw std::invalid_argument("input data is incompatible");
	}
	if (this->_data[nnsize - 1]->outputheight() != input.outputheight() || this->_data[nnsize - 1]->outputwidth() != input.outputwidth()) {
		throw std::invalid_argument("output data is incompatible");
	}
#endif

	//preallocate buffers
	math::num costsum = 0;
	std::vector<math::matrix> buffervec;
	for (nn::size_type i = 0; i != nnsize; ++i) {
		math::matrix buffer(this->_data[i]->outputheight(), this->_data[i]->outputwidth());
		buffervec.push_back(buffer);
	}

	for (data::size_type i = 0; i != datasize; ++i) {
		this->_data[0]->evaluate(input[i].first, buffervec[0]);
		for (nn::size_type j = 1; j != nnsize; ++j) {
			this->_data[j]->evaluate(buffervec[j - 1], buffervec[j]);
		}
		costsum += cost(input[i].second, buffervec[nnsize - 1]);
	}

	//reconsider this cost value, as it could be too small for a ML algorithm to pick up
	return costsum / datasize;
}

void nn::update(const std::vector<void*>& minibatch, math::num learningrate) {
	nn::size_type nnsize = this->size();

#ifdef _DEBUG
	if (nnsize != minibatch.size()) {
		throw std::invalid_argument("vec of minibatch ptr has incompatible size");
	}
#endif

	for (nn::size_type i = 0; i != nnsize; ++i) {
		this->_data[i]->update(minibatch[i], learningrate);
	}
}

std::vector<void*> nn::allocateminibatch() const {
	std::vector<void*> minibatchptr;
	nn::size_type nnsize = this->size();
	for (nn::size_type i = 0; i != nnsize; ++i) {
		minibatchptr.push_back(this->_data[i]->allocateminibatch());
	}

	return minibatchptr;
}

void nn::deallocateminibatch(const std::vector<void*>& minibatchptr) const {
	nn::size_type nnsize = this->size();

#ifdef _DEBUG
	if (nnsize != minibatchptr.size()) {
		throw std::invalid_argument("vec of minibatch ptr has incompatible size");
	}
#endif

	for (nn::size_type i = 0; i != nnsize; ++i) {
		this->_data[i]->deallocateminibatch(minibatchptr[i]);
	}
}

std::vector<void*> nn::allocateiteration() const {
	std::vector<void*> iterationptr;
	nn::size_type nnsize = this->size();
	for (nn::size_type i = 0; i != nnsize; ++i) {
		iterationptr.push_back(this->_data[i]->allocateiteration());
	}

	return iterationptr;
}

void nn::deallocateiteration(const std::vector<void*>& iterationptr) const {
	nn::size_type nnsize = this->size();

#ifdef _DEBUG
	if (nnsize != iterationptr.size()) {
		throw std::invalid_argument("vec of iteration ptr has incompatible size");
	}
#endif

	for (nn::size_type i = 0; i != nnsize; ++i) {
		this->_data[i]->deallocateiteration(iterationptr[i]);
	}
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

void* sigmoid::allocateminibatch() const {
	return nullptr;
}

void sigmoid::deallocateminibatch(void* minibatchptr) const {
	//empty virtual function
}

void* sigmoid::allocateiteration() const {
	return new math::matrix(this->inputheight(), this->inputwidth());
}

void sigmoid::deallocateiteration(void* iterationptr) const {
	delete static_cast<math::matrix*>(iterationptr);
}

void sigmoid::update(void* minibatchptr, math::num learningrate) {
	//empty virtual function
}

void sigmoid::feedforward(const math::matrix& input, math::matrix& output, void* iterationptr, void* minibatchptr) const {
#ifdef _DEBUG
	if (this->inputheight() != input.height() || this->inputwidth() != input.width()) {
		throw std::invalid_argument("input matrix is incompatible");
	}
	if (this->outputheight() != output.height() || this->outputwidth() != output.width()) {
		throw std::invalid_argument("output matrix is incompatible");
	}
#endif

	math::matrix* itptr = static_cast<math::matrix*>(iterationptr);
	*itptr = input;
	math::matrix::function(math::sigmoid, input, output);
}

void sigmoid::backprop(const math::matrix& errorin, math::matrix& errorout, void* iterationptr, void* minibatchptr) const {
#ifdef _DEBUG
	if (this->inputheight() != errorout.height() || this->inputwidth() != errorout.width()) {
		throw std::invalid_argument("error out has incompatible size");
	}
	if (this->outputheight() != errorin.height() || this->outputwidth() != errorin.width()) {
		throw std::invalid_argument("error in has incompatible size");
	}
#endif

	math::matrix* itptr = static_cast<math::matrix*>(iterationptr);
	math::matrix::function(math::sigmoidprime, *itptr, *itptr);
	math::matrix::hadamard(errorin, *itptr, errorout);
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

weights::weights(size_type inputheight, size_type outputheight, std::function<math::num()> func) : _data(outputheight, inputheight, func) {
#ifdef _DEBUG
	if (inputheight <= 0 || outputheight <= 0) {
		throw std::invalid_argument("empty weights initalization");
	}
#endif
}

weights::size_type weights::inputwidth() const {
	return 1;
}

weights::size_type weights::inputheight() const {
	return this->_data.width();
}

weights::size_type weights::outputwidth() const {
	return 1;
}

weights::size_type weights::outputheight() const {
	return this->_data.height();
}

math::matrix weights::evaluate(const math::matrix& input) const {
#ifdef _DEBUG 
	if (this->inputheight() != input.height() || this->inputwidth() != input.width()) {
		throw std::invalid_argument("input matrix is incompatible");
	}
#endif

	return this->_data * input;
}

std::unique_ptr<layer> weights::clone() const {
	std::unique_ptr<layer> ptr(new weights(*this));
	return std::move(ptr);
}

//the first pair member holds the accumalated derivatives,
//while the second is a buffer for backprop
void* weights::allocateminibatch() const {
	return new std::pair<math::matrix, math::matrix>
	(math::matrix(this->_data.height(), this->_data.width()),
	math::matrix(this->_data.height(), this->_data.width()));
}

void weights::deallocateminibatch(void* minibatchptr) const {
	delete static_cast<std::pair<math::matrix, math::matrix>*>(minibatchptr);
}

void* weights::allocateiteration() const {
	return new math::matrix(this->inputheight(), 1);
}

void weights::deallocateiteration(void* iterationptr) const {
	delete static_cast<math::matrix*>(iterationptr);
}

void weights::update(void* minibatchptr, math::num learningrate) {
	std::pair<math::matrix, math::matrix>* ptr = static_cast<std::pair<math::matrix, math::matrix>*>(minibatchptr);
	math::matrix::multiply(ptr->first, -learningrate, ptr->first);
	math::matrix::add(this->_data, ptr->first, this->_data);
	ptr->first = math::matrix(this->_data.height(), this->_data.width());
}

void weights::feedforward(const math::matrix& input, math::matrix& output, void* iterationptr, void* minibatchptr) const {
#ifdef _DEBUG
	if (this->inputheight() != input.height() || this->inputwidth() != input.width()) {
		throw std::invalid_argument("input matrix is incompatible");
	}
	if (this->outputheight() != output.height() || this->outputwidth() != output.width()) {
		throw std::invalid_argument("output matrix is incompatible");
	}
#endif

	math::matrix* itptr = static_cast<math::matrix*>(iterationptr);
	*itptr = input;
	math::matrix::multiply(this->_data, input, output);
}

void weights::backprop(const math::matrix& errorin, math::matrix& errorout, void* iterationptr, void* minibatchptr) const {
#ifdef _DEBUG
	if (this->inputheight() != errorout.height() || this->inputwidth() != errorout.width()) {
		throw std::invalid_argument("errorout has incompatible size");
	}
	if (this->outputheight() != errorin.height() || this->outputwidth() != errorin.width()) {
		throw std::invalid_argument("errorin has incompatible size");
	}
#endif

	//get our pointers
	math::matrix* itptr = static_cast<math::matrix*>(iterationptr);
	std::pair<math::matrix, math::matrix>* batchptr = static_cast<std::pair<math::matrix, math::matrix>*>(minibatchptr);
	
	//calculate the derivatives
	math::matrix::righttransposedmultiply(errorin, *itptr, batchptr->second);
	math::matrix::add(batchptr->second, batchptr->first, batchptr->first);

	//backprop the error
	math::matrix::lefttransposedmultiply(this->_data, errorin, errorout);
}

void weights::evaluate(const math::matrix& input, math::matrix& output) const {
#ifdef _DEBUG
	if (this->inputheight() != input.height() || this->inputwidth() != input.width()) {
		throw std::invalid_argument("input matrix is incompatible");
	}
	if (this->outputheight() != output.height() || this->outputwidth() != output.width()) {
		throw std::invalid_argument("output matrix is incompatible");
	}
#endif

	math::matrix::multiply(this->_data, input, output);
}

biases::biases(size_type height, size_type width) : _data(height, width) {
#ifdef _DEBUG
	if (height <= 0 || width <= 0) {
		throw std::invalid_argument("empty biases initalization");
	}
#endif
}

biases::size_type biases::inputwidth() const {
	return this->_data.width();
}

biases::size_type biases::inputheight() const {
	return this->_data.height();
}

biases::size_type biases::outputwidth() const {
	return this->_data.width();
}

biases::size_type biases::outputheight() const {
	return this->_data.height();
}

math::matrix biases::evaluate(const math::matrix& input) const {
#ifdef _DEBUG
	if (input.width() != this->inputwidth() || input.height() != this->inputheight()) {
		throw std::invalid_argument("input has incompatible size");
	}
#endif

	return input + this->_data;
}

std::unique_ptr<layer> biases::clone() const {
	std::unique_ptr<layer> ptr(new biases(*this));
	return std::move(ptr);
}

void* biases::allocateminibatch() const {
	return new math::matrix(this->_data.height(), this->_data.width());
}

void biases::deallocateminibatch(void* minibatchptr) const {
	delete static_cast<math::matrix*>(minibatchptr);
}

void* biases::allocateiteration() const {
	return nullptr;
}

void biases::deallocateiteration(void* iterationptr) const {
	//empty virtual function
}

void biases::update(void* minibatchptr, math::num learningrate) {
	math::matrix* batchptr = static_cast<math::matrix*>(minibatchptr);
	math::matrix::multiply(*batchptr, -learningrate, *batchptr);
	math::matrix::add(*batchptr, this->_data, this->_data);
	*batchptr = math::matrix(this->_data.height(), this->_data.width());
}

void biases::feedforward(const math::matrix& input, math::matrix& output, void* iterationptr, void* minibatchptr) const {
#ifdef _DEBUG
	if (this->inputheight() != input.height() || this->inputwidth() != input.width()) {
		throw std::invalid_argument("input matrix is incompatible");
	}
	if (this->outputheight() != output.height() || this->outputwidth() != output.width()) {
		throw std::invalid_argument("output matrix is incompatible");
	}
#endif

	math::matrix::add(input, this->_data, output);
}

void biases::backprop(const math::matrix& errorin, math::matrix& errorout, void* iterationptr, void* minibatchptr) const {
#ifdef _DEBUG
	if (this->inputheight() != errorout.height() || this->inputwidth() != errorout.width()) {
		throw std::invalid_argument("error out has incompatible size");
	}
	if (this->outputheight() != errorin.height() || this->outputwidth() != errorin.width()) {
		throw std::invalid_argument("error in has incompatible size");
	}
#endif

	math::matrix* batchptr = static_cast<math::matrix*>(minibatchptr);
	math::matrix::add(*batchptr, errorin, *batchptr);
	errorout = errorin;
}

void biases::evaluate(const math::matrix& input, math::matrix& output) const {
#ifdef _DEBUG
	if (this->inputheight() != input.height() || this->inputwidth() != input.width()) {
		throw std::invalid_argument("input matrix is incompatible");
	}
	if (this->outputheight() != output.height() || this->outputwidth() != output.width()) {
		throw std::invalid_argument("output matrix is incompatible");
	}
#endif

	math::matrix::add(input, this->_data, output);
}

}