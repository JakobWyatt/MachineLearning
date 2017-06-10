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

#include "matrix.h"

//creates a random integer between 0 and 10 inclusive (for testing)
double randomint() {
	std::random_device rd;
	std::default_random_engine re(rd());
	std::uniform_int_distribution<int> intdist(0, 10);
	return intdist(re);
}

int main() {


	return 0;
}