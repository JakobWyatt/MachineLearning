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
#include "math.h"

#include <vector>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <random>
#include <cmath>

//we will only error-check if the project is in debug mode
//we use the _DEBUG macro to check for this
namespace math {

matrix::matrix() : _data(0), _width(0) {

}

matrix::matrix(matrix::size_type height, matrix::size_type width) : _data(height * width), _width(width) {
#ifdef _DEBUG
	if (width <= 0 || height <= 0) {
		throw std::invalid_argument("empty matrix initialization");
	}
#endif
}

//we init to zero's->fill using index instead of empty init->reserve->push_back
//this is because, under performance tests, the method below is approx 3.5x faster
matrix::matrix(matrix::size_type height, matrix::size_type width, std::function<num()> func) : matrix(height, width) {
#ifdef _DEBUG
	if (width <= 0 || height <= 0) {
		throw std::invalid_argument("empty matrix initialization");
	}
#endif

	matrix::size_type size = this->size();
	for (matrix::size_type i = 0; i != size; ++i) {
		this->_data[i] = func();
	}
}

matrix::matrix(const std::vector<num>& data, matrix::size_type width) : _data(data), _width(width) {
#ifdef _DEBUG
	if (width <= 0 || data.size() <= 0) {
		throw std::invalid_argument("empty matrix initialization");
	}
#endif
}

matrix matrix::onehotmatrix(matrix::size_type height, matrix::size_type width, matrix::size_type row, matrix::size_type column) {
	matrix result(height, width);
	result(row, column) = 1;
	return result;
}

//TODO: make different rows line up
std::ostream& operator<<(std::ostream& cout, const matrix& toprint) {
	matrix::size_type height = toprint.height();
	matrix::size_type width = toprint.width();
	for (matrix::size_type i = 0; i != height; ++i) {
		for (matrix::size_type j = 0; j != width; ++j) {
			cout << toprint(i, j) << " ";
		}
		cout << "\n";
	}
	return cout;
}

const num& matrix::operator()(matrix::size_type row, matrix::size_type column) const {
#ifdef _DEBUG
	if (this->width() <= column || this->height() <= row || row < 0 || column < 0) {
		throw std::out_of_range("out of range");
	}
#endif

	return this->_data[this->width() * row + column];
}

num& matrix::operator()(matrix::size_type row, matrix::size_type column) {
#ifdef _DEBUG
	if (this->width() <= column || this->height() <= row || row < 0 || column < 0) {
		throw std::out_of_range("out of range");
	}
#endif

	return this->_data[this->width() * row + column];
}

const num& matrix::operator[](matrix::size_type element) const {
#ifdef _DEBUG
	if (element < 0 || element >= this->size()) {
		throw std::out_of_range("out of range");
	}
#endif

	return this->_data[element];
}

num& matrix::operator[](matrix::size_type element) {
#ifdef _DEBUG
	if (element < 0 || element >= this->size()) {
		throw std::out_of_range("out of range");
	}
#endif

	return this->_data[element];
}

matrix matrix::operator*(const matrix& rhs) const {
#ifdef _DEBUG
	if (this->width() != rhs.height()) {
		throw std::invalid_argument("matrix dimensions are incompatible");
	}
#endif

	matrix::size_type lhsheight = this->height();
	matrix::size_type rhswidth = rhs.width();

	matrix result(lhsheight, rhswidth);
	for (matrix::size_type i = 0; i != lhsheight; ++i) {
		for (matrix::size_type j = 0; j != rhswidth; ++j) {
			result(i, j) = dotproduct(*this, rhs, i, j);
		}
	}

	return result;
}

matrix matrix::operator()(std::function<num(num)> func) const {
	matrix result(this->height(), this->width());
	matrix::size_type size = this->size();
	for (matrix::size_type i = 0; i != size; ++i) {
		result[i] = func(this->operator[](i));
	}

	return result;
}

void matrix::multiply(const matrix& lhs, const matrix& rhs, matrix& buffer) {
#ifdef _DEBUG
	if (lhs.width() != rhs.height()) {
		throw std::invalid_argument("matrix dimensions are incompatible");
	}
	if (buffer.height() != lhs.height() || buffer.width() != rhs.width()) {
		throw std::invalid_argument("buffer matrix has incompatible size");
	}
	if (&lhs == &buffer) {
		throw std::invalid_argument("buffer matrix is the same object as the left hand side argument");
	}
	if (&rhs == &buffer) {
		throw std::invalid_argument("buffer matrix is the same object as the right hand side argument");
	}
#endif

	matrix::size_type lhsheight = lhs.height();
	matrix::size_type rhswidth = rhs.width();

	matrix result(lhsheight, rhswidth);
	for (matrix::size_type i = 0; i != lhsheight; ++i) {
		for (matrix::size_type j = 0; j != rhswidth; ++j) {
			buffer(i, j) = dotproduct(lhs, rhs, i, j);
		}
	}
}

void matrix::function(std::function<num(num)> func, const matrix& input, matrix& buffer) {
#ifdef _DEBUG
	if (buffer.height() != input.height() || buffer.width() != input.width()) {
		throw std::invalid_argument("buffer matrix has incompatible size");
	}
#endif
	
	matrix::size_type size = input.size();
	for (matrix::size_type i = 0; i != size; ++i) {
		buffer[i] = func(input[i]);
	}
}

num matrix::dotproduct(const matrix& first, const matrix& second, matrix::size_type row, matrix::size_type column) {
	num sum = 0;
	matrix::size_type vectorlength = first.width();
	for (matrix::size_type k = 0; k != vectorlength; ++k) {
		sum += first(row, k) * second(k, column);
	}
	return sum;
}

matrix::size_type matrix::height() const {
	return this->size()/this->width();
}

matrix::size_type matrix::width() const {
	return this->_width;
}

matrix::size_type matrix::size() const {
	return this->_data.size();
}

//TODO: remove use of global static while maintaining efficiency
num standarddist() {
	static std::random_device rd;
	static std::default_random_engine re(rd());
	static std::normal_distribution<num> nd(0, 1);
	return nd(re);
}

num sigmoid(num input) {
	return (1/(1 + std::exp(-input)));
}

}