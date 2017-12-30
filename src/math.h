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
#include <random>
#include <utility>

namespace math {

//we typedef double num for flexibility
//in future, if we want to port to GPU, float instead of double will give us better performance
typedef double num;

//row-major matrix class - interface of std::vector
//this class uses zero-indexing unless otherwise stated
class matrix {
public:
	typedef std::vector<num>::size_type size_type;
	typedef std::vector<num>::iterator iterator;
	typedef std::vector<num>::const_iterator const_iterator;

	//default constructor
	matrix();
	//initializes a matrix with specified dimensions, all elements being set to 0
	matrix(size_type height, size_type width);
	//initializes a matrix with specified dimensions, and fills this matrix using the given function
	matrix(size_type height, size_type width, std::function<num()> func);
	//initializes a matrix given a data vector and a width
	matrix(const std::vector<num>& data, size_type width);
	//initializes a matrix given a height, width, and initializer list
	matrix(size_type height, size_type width, std::initializer_list<num> initializerlist);

	//initializes a matrix such that all elements are zero except for one specified element, set to one
	static matrix onehotmatrix(size_type height, size_type width, size_type row, size_type column);

	//prints the matrix to the output stream
	friend std::ostream& operator<<(std::ostream& cout, const matrix& toprint);
	//returns a const reference to the specified element of the matrix
	const num& operator()(size_type row, size_type column) const;
	//returns a reference to the specified element of the matrix
	num& operator()(size_type row, size_type column);
	//returns a const reference to the specified data element
	const num& operator[](size_type element) const;
	//returns a reference to the specified data element
	num& operator[](size_type element);
	//multiplies two matricies together and returns the result
	matrix operator*(const matrix& rhs) const;
	//multiplies the matrix by a scalar value
	matrix operator*(math::num scalar) const;
	friend matrix operator*(num scalar, const matrix& rhs);
	//applies a function to every element in the matrix
	matrix operator()(std::function<num (num)> func) const;
	//adds two matricies together and returns the result
	matrix operator+(const matrix& rhs) const;
	//subtracts two matricies and returns the result
	matrix operator-(const matrix& rhs) const;

	//returns a const iterator to the max element in the matrix
	const_iterator max() const;
	//returns a iterator to the max element in the matrix
	iterator max();
	//returns a const iterator to the first element in the matrix
	const_iterator begin() const;
	//returns a iterator to the first element in the matrix
	iterator begin();
	//returns a const iterator to one past the last element in the matrix
	const_iterator end() const;
	//returns a iterator to one past the last element in the matrix
	iterator end();

	//multiplies two matricies together and writes the result to a buffer
	static void multiply(const matrix& lhs, const matrix& rhs, matrix& buffer);

	//TODO: test other methods of transposed multiply for efficiency
	//multiplies two matricies together, with the first matrix viewed as transposed
	static matrix lefttransposedmultiply(const matrix& lhs, const matrix& rhs);
	//multiplies two matricies together, with the first matrix viewed as transposed. result is written to a buffer
	static void lefttransposedmultiply(const matrix& lhs, const matrix& rhs, matrix& buffer);
	//multiplies two matricies together, with the second matrix viewed as transposed
	static matrix righttransposedmultiply(const matrix& lhs, const matrix& rhs);
	//multiplies two matricies together, with the second matrix viewed as transposed. result is written to a buffer
	static void righttransposedmultiply(const matrix& lhs, const matrix& rhs, matrix& buffer);

	//multiplies a matrix by a scalar and writes the result to a buffer
	static void multiply(const matrix& lhs, num scalar, matrix& buffer);
	//applies a function to every element in a matrix and writes the result to a buffer
	static void function(std::function<num(num)> func, const matrix& input, matrix& buffer);
	//adds two matricies together and writes the result to a buffer
	static void add(const matrix& lhs, const matrix& rhs, matrix& buffer);
	//subtracts two matricies and writes the result to a buffer
	static void subtract(const matrix& lhs, const matrix& rhs, matrix& buffer);
	//finds the hadamard product of two matricies
	static matrix hadamard(const matrix& lhs, const matrix& rhs);
	//finds the hadamard product of two matricies and writes the result to a buffer
	static void hadamard(const matrix& lhs, const matrix& rhs, matrix& buffer);
	//returns true if the max element in lhs has the same position as the max element in rhs
	//doesnt do anything with buffer
	static bool comparemax(const matrix& lhs, const matrix& rhs, matrix& buffer);
	static bool comparemax(const matrix& lhs, const matrix& rhs) { matrix m; return comparemax(lhs, rhs, m);}
	//takes single element matricies for bool comparision
	//first argument must be either 0 or 1
	static bool comparebool(const matrix& correct, const matrix& totest, matrix& buffer);
	static bool comparebool(const matrix& correct, const matrix& totest) {matrix m; return comparebool(correct, totest, m);};
	//finds the quadratic cost of two vectors
	static num quadraticcost(const math::matrix& y, const math::matrix& aL);

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
	static num dotproduct(const matrix& first, const matrix& second, matrix::size_type row, matrix::size_type column);
};

//returns a randomly initialized instance of the default random engine
std::default_random_engine default_random_engine();
//returns a random num from the standard distribution (mean 0, SD 1)
//(NOT ACTUALLY A STANDARD DISTRIBUTION RIGHT NOW)
num standarddist();
//returns a 50/50 bool
bool bernoullidist();
//returns the sigmoid function of a number
num sigmoid(num input);
//returns the derivative of the sigmoid function
//if we have performance issues, we could change the form of the equation
num sigmoidprime(num input);

}

#endif