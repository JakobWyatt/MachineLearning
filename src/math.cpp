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

#include "math.h"

#include <vector>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <random>
#include <cmath>
#include <algorithm>
#include <utility>

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

matrix::matrix(size_type height, size_type width, std::initializer_list<num> initializerlist) : _data(initializerlist), _width(width) {
#ifdef _DEBUG
	if (width <= 0 || height <= 0) {
		throw std::invalid_argument("empty matrix initialization");
	}
	if (width * height != initializerlist.size()) {
		throw std::invalid_argument("initializer list is incompatible");
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

matrix matrix::operator*(math::num scalar) const {
	matrix result(this->height(), this->width());
	matrix::size_type size = this->size();
	for (matrix::size_type i = 0; i != size; ++i) {
		result[i] = this->operator[](i) * scalar;
	}

	return result;
}

matrix operator*(num scalar, const matrix& rhs) {
	matrix result(rhs.height(), rhs.width());
	matrix::size_type size = rhs.size();
	for (matrix::size_type i = 0; i != size; ++i) {
		result[i] = rhs[i] * scalar;
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

matrix matrix::operator+(const matrix& rhs) const {
#ifdef _DEBUG
	if (this->width() != rhs.width() || this->height() != rhs.height()) {
		throw std::invalid_argument("matrix dimensions do not match");
	}
#endif

	matrix result(this->height(), this->width());
	matrix::size_type size = this->size();
	for (matrix::size_type i = 0; i != size; ++i) {
		result[i] = this->operator[](i) + rhs[i];
	}

	return result;
}

matrix matrix::operator-(const matrix& rhs) const {
#ifdef _DEBUG
	if (this->width() != rhs.width() || this->height() != rhs.height()) {
		throw std::invalid_argument("matrix dimensions do not match");
	}
#endif

	matrix result(this->height(), this->width());
	matrix::size_type size = this->size();
	for (matrix::size_type i = 0; i != size; ++i) {
		result[i] = this->operator[](i) - rhs[i];
	}

	return result;
}


matrix::const_iterator matrix::max() const {
	return std::max_element(this->begin(), this->end());
}

matrix::iterator matrix::max() {
	return std::max_element(this->begin(), this->end());
}

matrix::const_iterator matrix::begin() const {
	return this->_data.begin();
}

matrix::iterator matrix::begin() {
	return this->_data.begin();
}

matrix::const_iterator matrix::end() const {
	return this->_data.end();
}

matrix::iterator matrix::end() {
	return this->_data.end();
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

matrix matrix::lefttransposedmultiply(const matrix& lhs, const matrix& rhs) {
#ifdef _DEBUG
	if (lhs.height() != rhs.height()) {
		throw std::invalid_argument("matrix dimensions are incompatible");
	}
#endif

	matrix::size_type lhsheight = lhs.height();
	matrix::size_type lhswidth = lhs.width();
	matrix::size_type rhswidth = rhs.width();
	num sum = 0;

	matrix result(lhs.width(), rhs.width());
	for (matrix::size_type i = 0; i != lhswidth; ++i) {
		for (matrix::size_type j = 0; j != rhswidth; ++j) {
			for (matrix::size_type k = 0; k != lhsheight; ++k) {
				sum += lhs(k, i) * rhs(k, j);
			}
			result(i, j) = sum;
			sum = 0;
		}
	}

	return result;
}

void matrix::lefttransposedmultiply(const matrix& lhs, const matrix& rhs, matrix& buffer) {
#ifdef _DEBUG
	if (lhs.height() != rhs.height()) {
		throw std::invalid_argument("matrix dimensions are incompatible");
	}
	if (lhs.width() != buffer.height() || rhs.width() != buffer.width()) {
		throw std::invalid_argument("buffer matrix dimensions are incompatible");
	}
	if (&lhs == &buffer) {
		throw std::invalid_argument("buffer matrix is the same object as the left hand side argument");
	}
	if (&rhs == &buffer) {
		throw std::invalid_argument("buffer matrix is the same object as the right hand side argument");
	}
#endif

	matrix::size_type lhsheight = lhs.height();
	matrix::size_type lhswidth = lhs.width();
	matrix::size_type rhswidth = rhs.width();
	num sum = 0;

	for (matrix::size_type i = 0; i != lhswidth; ++i) {
		for (matrix::size_type j = 0; j != rhswidth; ++j) {
			for (matrix::size_type k = 0; k != lhsheight; ++k) {
				sum += lhs(k, i) * rhs(k, j);
			}
			buffer(i, j) = sum;
			sum = 0;
		}
	}
}

matrix matrix::righttransposedmultiply(const matrix& lhs, const matrix& rhs) {
#ifdef _DEBUG
	if (lhs.width() != rhs.width()) {
		throw std::invalid_argument("matrix dimensions are incompatible");
	}
#endif

	matrix::size_type lhsheight = lhs.height();
	matrix::size_type rhsheight = rhs.height();
	matrix::size_type lhswidth = lhs.width();
	num sum = 0;

	matrix result(lhsheight, rhsheight);
	for (matrix::size_type i = 0; i != lhsheight; ++i) {
		for (matrix::size_type j = 0; j != rhsheight; ++j) {
			for (matrix::size_type k = 0; k != lhswidth; ++k) {
				sum += lhs(i, k) * rhs(j, k);
			}
			result(i, j) = sum;
			sum = 0;
		}
	}

	return result;
}

void matrix::righttransposedmultiply(const matrix& lhs, const matrix& rhs, matrix& buffer) {
#ifdef _DEBUG
	if (lhs.width() != rhs.width()) {
		throw std::invalid_argument("matrix dimensions are incompatible");
	}
	if (lhs.height() != buffer.height() || rhs.height() != buffer.width()) {
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
	matrix::size_type rhsheight = rhs.height();
	matrix::size_type lhswidth = lhs.width();
	num sum = 0;

	for (matrix::size_type i = 0; i != lhsheight; ++i) {
		for (matrix::size_type j = 0; j != rhsheight; ++j) {
			for (matrix::size_type k = 0; k != lhswidth; ++k) {
				sum += lhs(i, k) * rhs(j, k);
			}
			buffer(i, j) = sum;
			sum = 0;
		}
	}
}

void matrix::multiply(const matrix& lhs, num scalar, matrix& buffer) {
#ifdef _DEBUG
	if (lhs.width() != buffer.width() || lhs.height() != buffer.height()) {
		throw std::invalid_argument("buffer matrix has incompatible size");
	}
#endif

	matrix::size_type size = lhs.size();
	for (matrix::size_type i = 0; i != size; ++i) {
		buffer[i] = lhs[i] * scalar;
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

void matrix::add(const matrix& lhs, const matrix& rhs, matrix& buffer) {
#ifdef _DEBUG
	if (lhs.width() != rhs.width() || lhs.height() != rhs.height()) {
		throw std::invalid_argument("matrix dimensions do not match");
	}
	if (buffer.width() != lhs.width() || buffer.height() != lhs.height()) {
		throw std::invalid_argument("buffer matrix has incompatible size");
	}
#endif

	matrix::size_type size = lhs.size();
	for (matrix::size_type i = 0; i != size; ++i) {
		buffer[i] = lhs[i] + rhs[i];
	}
}

void matrix::subtract(const matrix& lhs, const matrix& rhs, matrix& buffer) {
#ifdef _DEBUG
	if (lhs.width() != rhs.width() || lhs.height() != rhs.height()) {
		throw std::invalid_argument("matrix dimensions do not match");
	}
	if (buffer.width() != lhs.width() || buffer.height() != lhs.height()) {
		throw std::invalid_argument("buffer matrix has incompatible size");
	}
#endif

	matrix::size_type size = lhs.size();
	for (matrix::size_type i = 0; i != size; ++i) {
		buffer[i] = lhs[i] - rhs[i];
	}
}

matrix matrix::hadamard(const matrix& lhs, const matrix& rhs) {
#ifdef _DEBUG
	if (lhs.width() != rhs.width() || lhs.height() != rhs.height()) {
		throw std::invalid_argument("matrix dimensions do not match");
	}
#endif

	matrix result(lhs.height(), lhs.width());
	matrix::size_type size = lhs.size();
	for (matrix::size_type i = 0; i != size; ++i) {
		result[i] = lhs[i] * rhs[i];
	}

	return result;
}

void matrix::hadamard(const matrix& lhs, const matrix& rhs, matrix& buffer) {
#ifdef _DEBUG
	if (lhs.width() != rhs.width() || lhs.height() != rhs.height()) {
		throw std::invalid_argument("matrix dimensions do not match");
	}
	if (buffer.width() != lhs.width() || buffer.height() != lhs.height()) {
		throw std::invalid_argument("buffer matrix has incompatible size");
	}
#endif

	matrix::size_type size = lhs.size();
	for (matrix::size_type i = 0; i != size; ++i) {
		buffer[i] = lhs[i] * rhs[i];
	}
}

bool matrix::comparemax(const matrix& lhs, const matrix& rhs, matrix& buffer) {
#ifdef _DEBUG
	if (lhs.height() != rhs.height() || lhs.width() != rhs.width()) {
		throw std::invalid_argument("matrix sizes do not match");
	}
#endif

	return lhs.max() - lhs.begin() == rhs.max() - rhs.begin();
}

bool matrix::comparebool(const matrix& correct, const matrix& totest, matrix& buffer) {
#ifdef _DEBUG
	if (correct.size() != 1 || totest.size() != 1) {
		throw std::invalid_argument("arguments must be singular matricies");
	}
	if (correct[0] != 0 && correct[0] != 1) {
		throw std::invalid_argument("first argument must contain bool value");
	}
#endif

	if (correct[0] == 0 && totest[0] < 0.5) {
		return true;
	} else if (correct[0] == 1 && totest[0] >= 0.5) {
		return true;
	}
	else {
		return false;
	}
}

num matrix::quadraticcost(const math::matrix& y, const math::matrix& aL) {
#ifdef _DEBUG
	if (y.height() != aL.height()) {
		throw std::invalid_argument("vectors are incompatible");
	}
	if (y.width() != 1 || aL.width() != 1) {
		throw std::invalid_argument("arguments are not vectors");
	}
#endif

	matrix::size_type size = y.size();
	num squaresum = 0;
	for (matrix::size_type i = 0; i != size; ++i) {
		squaresum += std::pow(y[i] - aL[i], 2);
	}

	return squaresum * 0.5;
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

std::default_random_engine default_random_engine() {
	std::random_device rd;
	std::default_random_engine re(rd());
	return re;
}

//TODO: remove use of global static while maintaining efficiency
//not really a standard dist but this gives us better results for now
num standarddist() {
	static std::random_device rd;
	static std::default_random_engine re(rd());
	static std::normal_distribution<num> nd(0, 0.6);
	return nd(re);
}

bool bernoullidist() {
	static std::random_device rd;
	static std::default_random_engine re(rd());
	static std::bernoulli_distribution bd(0.5);
	return bd(re);
}

num sigmoid(num input) {
	return (1/(1 + std::exp(-input)));
}

num sigmoidprime(num input) {
	return sigmoid(input) * (1 - sigmoid(input));
}

}