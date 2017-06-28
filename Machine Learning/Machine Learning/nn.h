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

#ifndef GUARD_NN_H
#define GUARD_NN_H

#include <vector>
#include <utility>
#include <functional>
#include <memory>

#include "math.h"

namespace nn {

//stores the data we will use to train and test our neuralnet
class data {
public:
	typedef std::vector<std::pair<math::matrix, math::matrix>>::size_type size_type;

	//initializes our data given a flag
	data(int flag);

	//returns the size of the dataset
	size_type size() const;
	//returns the height of the input matrix

	//returns the width of the input matrix

	//returns the height of the output matrix

	//returns the width of the output matrix


	//common dataset flags
	enum datasets {
		//mnist dataset can be found here: http://yann.lecun.com/exdb/mnist/
		mnisttest,
		mnisttrain,
	};

private:
	std::vector<std::pair<math::matrix, math::matrix>> _data;

	//returns the mnist test data
	std::vector<std::pair<math::matrix, math::matrix>> mnisttestload();
	//returns the mnist training data
	std::vector<std::pair<math::matrix, math::matrix>> mnisttrainload();
};

//abstract base layer class
//all inherited layer classes SHOULD BE THREADSAFE
class layer {
public:
	typedef math::matrix::size_type size_type;
	friend class nn;

	//returns the input width of the layer
	virtual size_type inputwidth() const = 0;
	//returns the input height of the layer
	virtual size_type inputheight() const = 0;
	//returns the output width of the layer
	virtual size_type outputwidth() const = 0;
	//returns the output height of the layer
	virtual size_type outputheight() const = 0;

	//evaluates the output of a layer
	virtual math::matrix evaluate(const math::matrix& input) const = 0;

protected:
	//returns a pointer to a dynamically allocated copy of the object
	virtual std::unique_ptr<layer> clone() const = 0;

	//evaluates the output of a layer, and writes that output to a buffer
	virtual void evaluate(const math::matrix& input, math::matrix& output) const = 0;
};

//neuralnet class: interface for our layer classes
class nn {
public:
	typedef std::vector<std::unique_ptr<layer>>::size_type size_type;

	//initializes a neural network with an initializer list of layers
	nn(std::initializer_list<std::reference_wrapper<layer>> layers);

	//returns the number of layers in the neuralnet
	size_type size() const;

	//evaluates the output of the neuralnet
	math::matrix evaluate(const math::matrix& input) const;

private:
	std::vector<std::unique_ptr<layer>> _data;
};
/*
//this layer applies the sigmoid activation function
class sigmoid : public layer {
public:
	sigmoid(size_type height, size_type width);

	size_type inputwidth() const;
	size_type inputheight() const;
	size_type outputwidth() const;
	size_type outputheight() const;

	math::matrix evaluate(const math::matrix& input) const;

protected:
	std::unique_ptr<layer> clone() const;

	void evaluate(const math::matrix& input, math::matrix& output) const;

private:
	size_type _height;
	size_type _width;
}; */

}

#endif