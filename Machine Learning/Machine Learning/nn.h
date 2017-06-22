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

#include "math.h"

namespace nn {

//stores the data we will use to train and test our neuralnet
class data {
public:
	data(int flag);

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

//neuralnet class: interface for our layer classes
class nn {
public:

private:

};

//abstract base layer class
class layer {
public:

private:

};

//sigmoid layer
class sigmoidlayer {
public:
	
private:

};

}

#endif