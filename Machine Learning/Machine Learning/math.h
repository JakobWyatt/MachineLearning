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

#ifndef GUARD_MATH_H
#define GUARD_MATH_H

#include <vector>
#include <iostream>
#include <functional>

namespace math {

//we typedef double num for flexibility
//in future, if we want to port to GPU, float instead of double will give us better performance
typedef double num;

//row-major matrix class - interface of std::vector
//this class uses zero-indexing unless otherwise stated
class matrix {
public:
	typedef std::vector<num>::size_type size_type;

	//default constructor
	matrix();
	//initializes a matrix with specified dimensions, all elements being set to 0
	matrix(size_type height, size_type width);
	//initializes a matrix with specified dimensions, and fills this matrix using the given function
	matrix(size_type height, size_type width, std::function<num()> func);
	//initializes a matrix given a data vector and a width
	matrix(std::vector<num> data, size_type width);

	//initializes a matrix such that all elements are zero except for one specified element, set to one
	static matrix onehotmatrix(size_type height, size_type width, size_type row, size_type column);

	//prints the matrix to the output stream
	friend std::ostream& operator<<(std::ostream& cout, const matrix& toprint);
	//returns a const reference to the specified element of the matrix
	const num& operator()(size_type row, size_type column) const;
	//returns a reference to the specified element of the matrix
	num& operator()(size_type row, size_type column);
	//multiplies two matricies together and returns the result
	matrix operator*(const matrix& rhs) const;

	//multiplies two matricies together and writes the result to a buffer
	static void multiply(const matrix& lhs, const matrix& rhs, matrix& buffer);

	//returns the height of the matrix
	size_type height() const;
	//returns the width of the matrix
	size_type width() const;
	//returns the size of the matrix
	size_type size() const;

private:
	std::vector<num> _data;
	size_type _width;

	//seperate dotproduct function for matrix multiplication
	static matrix::num dotproduct(const matrix& first, const matrix& second, matrix::size_type row, matrix::size_type column);
};

//returns a random num from the standard distribution (mean 0, SD 1)
num standarddist();

}

#endif