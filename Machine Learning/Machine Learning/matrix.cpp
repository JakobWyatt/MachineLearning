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
#include "matrix.h"

#include <vector>
#include <iostream>
#include <stdexcept>
#include <functional>

//we will only error-check if the project is in debug mode
//we use the _DEBUG macro to check for this
namespace matrix {

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

const matrix::num& matrix::operator()(matrix::size_type row, matrix::size_type column) const {
#ifdef _DEBUG
	if (this->width() < column || this->height() < row || row < 0 || column < 0) {
		throw std::out_of_range("out of range");
	}
#endif

	return this->_data[this->width() * row + column];
}

matrix::num& matrix::operator()(matrix::size_type row, matrix::size_type column) {
#ifdef _DEBUG
	if (this->width() < column || this->height() < row || row < 0 || column < 0) {
		throw std::out_of_range("out of range");
	}
#endif

	return this->_data[this->width() * row + column];
}

//seperated dot product function
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

matrix::num matrix::dotproduct(const matrix& first, const matrix& second, matrix::size_type row, matrix::size_type column) const {
	matrix::num sum = 0;
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

}