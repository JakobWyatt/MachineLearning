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
#include <memory>
#include <functional>

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
	math::matrix::size_type inputheight() const;
	//returns the width of the input matrix
	math::matrix::size_type inputwidth() const;
	//returns the height of the output matrix
	math::matrix::size_type outputheight() const;
	//returns the width of the output matrix
	math::matrix::size_type outputwidth() const;

	//returns a copy of the dataset that has been randomly shuffled
	data shuffle() const;
	//returns a trimmed version of the dataset
	data trim(size_type size) const;

	//returns a const reference to the specified std::pair
	const std::pair<math::matrix, math::matrix>& operator[](size_type element) const;

	//common dataset flags
	enum datasets {
		//mnist dataset can be found here: http://yann.lecun.com/exdb/mnist/
		mnisttest,
		mnisttrain,
		XOR,
	};

private:
	std::vector<std::pair<math::matrix, math::matrix>> _data;

	//initializes a dataset given a vector of matrix pairs
	data(std::vector<std::pair<math::matrix, math::matrix>> data);

	//returns the mnist test data
	static std::vector<std::pair<math::matrix, math::matrix>> mnisttestload();
	//returns the mnist training data
	static std::vector<std::pair<math::matrix, math::matrix>> mnisttrainload();
	//returns a randomly generated set of XOR's
	static std::vector<std::pair<math::matrix, math::matrix>> generateXOR();
};

//abstract base layer class
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
	//dynamically allocates any memory the layer needs within a minibatch
	virtual void* allocateminibatch() const = 0;
	//deallocates this memory
	virtual void deallocateminibatch(void* minibatchptr) const = 0;
	//dynamically allocates any memory the layer needs within a training iteration
	virtual void* allocateiteration() const = 0;
	//deallocates this memory
	virtual void deallocateiteration(void* iterationptr) const = 0;

	//updates our neuralnet given a pointer to the data accumalated over the minibatch, and a learning rate
	virtual void update(void* minibatchptr, math::num learningrate) = 0;
	//evaluates the output of a layer, and prepares for a backprop
	virtual void feedforward(const math::matrix& input, math::matrix& output, void* iterationptr, void* minibatchptr) const = 0;
	//backpropagates the error through our network, and prepares for an update
	virtual void backprop(const math::matrix& errorin, math::matrix& errorout, void* iterationptr, void* minibatchptr) const = 0;
	//evaluates the output of a layer, and writes that output to a buffer
	virtual void evaluate(const math::matrix& input, math::matrix& output) const = 0;
};

//neuralnet class: interface for our layer classes
class nn {
public:
	typedef std::vector<std::unique_ptr<layer>>::size_type size_type;

	//initializes a neural network with an initializer list of layers
	nn(std::initializer_list<layer*> layers);

	//returns the number of layers in the neuralnet
	size_type size() const;

	//evaluates the output of the neuralnet
	math::matrix evaluate(const math::matrix& input) const;
	//trains the neuralnet given learning data, learning rate, and a batchsize
	//is threadsafe
	void train(const data& learningdata, math::num learningrate, data::size_type batchsize);
	//returns the number of successfully evaluated matricies from a data set
	//the second argument is a function that takes the real output and the nn output
	//and returns true if the output is deemed "correct",
	//as well as a buffer the size of the output matrix that can be written into if needs be
	data::size_type test(const data& input, std::function<bool(const math::matrix& correct, const math::matrix& output, math::matrix& buffer)> compare) const;
	//returns the average cost over a dataset
	math::num cost(const data& input, std::function<math::num(const math::matrix& correct, const math::matrix& output)> cost);

private:
	std::vector<std::unique_ptr<layer>> _data;

	//updates all the layers in a neuralnet
	void update(const std::vector<void*>& minibatch, math::num learningrate);

	//dynamically allocates any memory the layers need within a minibatch
	std::vector<void*> allocateminibatch() const;
	//deallocates this memory
	void deallocateminibatch(const std::vector<void*>& minibatchptr) const;
	//dynamically allocates any memory the layers need within a training iteration
	std::vector<void*> allocateiteration() const;
	//deallocates this memory
	void deallocateiteration(const std::vector<void*>& iterationptr) const;
};

//this layer applies the sigmoid activation function
class sigmoid : public layer {
public:
	//initializes a sigmoid layer with a height and width
	sigmoid(size_type height, size_type width);

	//returns the input width of the layer
	virtual size_type inputwidth() const;
	//returns the input height of the layer
	virtual size_type inputheight() const;
	//returns the output width of the layer
	virtual size_type outputwidth() const;
	//returns the output height of the layer
	virtual size_type outputheight() const;

	//evaluates the output of a layer
	virtual math::matrix evaluate(const math::matrix& input) const;

protected:
	//returns a pointer to a dynamically allocated copy of the object
	virtual std::unique_ptr<layer> clone() const;
	//dynamically allocates any memory the layer needs within a minibatch
	virtual void* allocateminibatch() const;
	//deallocates this memory
	virtual void deallocateminibatch(void* minibatchptr) const;
	//dynamically allocates any memory the layer needs within a training iteration
	virtual void* allocateiteration() const;
	//deallocates this memory
	virtual void deallocateiteration(void* iterationptr) const;

	//updates our neuralnet given a pointer to the data accumalated over the minibatch, and a learning rate
	virtual void update(void* minibatchptr, math::num learningrate);
	//evaluates the output of a layer, and prepares for a backprop
	virtual void feedforward(const math::matrix& input, math::matrix& output, void* iterationptr, void* minibatchptr) const;
	//backpropagates the error through our network, and prepares for an update
	virtual void backprop(const math::matrix& errorin, math::matrix& errorout, void* iterationptr, void* minibatchptr) const;
	//evaluates the output of a layer, and writes that output to a buffer
	virtual void evaluate(const math::matrix& input, math::matrix& output) const;

private:
	size_type _height;
	size_type _width;
};

//this layer applies a weights matrix
class weights : public layer {
public:
	//initializes a weights layer with an input size, output size, and a function
	//this function determines how the weights matrix will be filled
	weights(size_type inputheight, size_type outputheight, std::function<math::num()> func = math::standarddist);

	//returns the input width of the layer
	virtual size_type inputwidth() const;
	//returns the input height of the layer
	virtual size_type inputheight() const;
	//returns the output width of the layer
	virtual size_type outputwidth() const;
	//returns the output height of the layer
	virtual size_type outputheight() const;

	//evaluates the output of a layer
	virtual math::matrix evaluate(const math::matrix& input) const;

protected:
	//returns a pointer to a dynamically allocated copy of the object
	virtual std::unique_ptr<layer> clone() const;
	//dynamically allocates any memory the layer needs within a minibatch
	virtual void* allocateminibatch() const;
	//deallocates this memory
	virtual void deallocateminibatch(void* minibatchptr) const;
	//dynamically allocates any memory the layer needs within a training iteration
	virtual void* allocateiteration() const;
	//deallocates this memory
	virtual void deallocateiteration(void* iterationptr) const;

	//updates our neuralnet given a pointer to the data accumalated over the minibatch, and a learning rate
	virtual void update(void* minibatchptr, math::num learningrate);
	//evaluates the output of a layer, and prepares for a backprop
	virtual void feedforward(const math::matrix& input, math::matrix& output, void* iterationptr, void* minibatchptr) const;
	//backpropagates the error through our network, and prepares for an update
	virtual void backprop(const math::matrix& errorin, math::matrix& errorout, void* iterationptr, void* minibatchptr) const;
	//evaluates the output of a layer, and writes that output to a buffer
	virtual void evaluate(const math::matrix& input, math::matrix& output) const;

private:
	math::matrix _data;
};

//this layer applies a bias matrix
class biases : public layer {
public:
	//initializes a sigmoid layer with a height and width
	biases(size_type height, size_type width);

	//returns the input width of the layer
	virtual size_type inputwidth() const;
	//returns the input height of the layer
	virtual size_type inputheight() const;
	//returns the output width of the layer
	virtual size_type outputwidth() const;
	//returns the output height of the layer
	virtual size_type outputheight() const;

	//evaluates the output of a layer
	virtual math::matrix evaluate(const math::matrix& input) const;

protected:
	//returns a pointer to a dynamically allocated copy of the object
	virtual std::unique_ptr<layer> clone() const;
	//dynamically allocates any memory the layer needs within a minibatch
	virtual void* allocateminibatch() const;
	//deallocates this memory
	virtual void deallocateminibatch(void* minibatchptr) const;
	//dynamically allocates any memory the layer needs within a training iteration
	virtual void* allocateiteration() const;
	//deallocates this memory
	virtual void deallocateiteration(void* iterationptr) const;

	//updates our neuralnet given a pointer to the data accumalated over the minibatch, and a learning rate
	virtual void update(void* minibatchptr, math::num learningrate);
	//evaluates the output of a layer, and prepares for a backprop
	virtual void feedforward(const math::matrix& input, math::matrix& output, void* iterationptr, void* minibatchptr) const;
	//backpropagates the error through our network, and prepares for an update
	virtual void backprop(const math::matrix& errorin, math::matrix& errorout, void* iterationptr, void* minibatchptr) const;
	//evaluates the output of a layer, and writes that output to a buffer
	virtual void evaluate(const math::matrix& input, math::matrix& output) const;

private:
	math::matrix _data;
};

}

#endif