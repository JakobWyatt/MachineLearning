//Copyright 2017 Jakob Wyatt
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
////Copyright 2017 Jakob Wyatt
//
//http ://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.
// http ://www.apache.org/licenses/LICENSE-2.0

#include "stdafx.h"

#include <vector>
#include <iostream>
#include <chrono>
#include <random>

#include "math.h"
#include "nn.h"

//creates a random integer between -5 and 5 inclusive (for testing)
double randomint() {
	std::random_device rd;
	std::default_random_engine re(rd());
	std::uniform_int_distribution<int> intdist(-5, 5);
	return intdist(re);
}

int main() {


	//generic code for testing speeds
	/*
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	math::matrix a(40, 750, math::standarddist);
	math::matrix b(750, 1, math::standarddist);
	for (int i = 0; i != 1000; ++i) {
		math::matrix l = a * b;
	}
	std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
	std::chrono::steady_clock::duration duration1 = t2 - t1;
	std::cout << duration1.count() << "\n";

	std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
	math::matrix c(40, 750, math::standarddist);
	math::matrix d(750, 1, math::standarddist);
	math::matrix e(40, 1);
	for (int j = 0; j != 1000; ++j) {
		math::matrix::multiply(c, d, e);
	}
	std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
	std::chrono::steady_clock::duration duration2 = t4 - t3;
	std::cout << duration2.count();
	*/

	return 0;
}