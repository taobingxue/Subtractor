#pragma once

#include <windows.h>
#include <time.h>

using namespace std;

class Timer {
public:
	inline Timer():start() {}
	inline void reset() {
		start = clock();
	}
	inline double getTime() const {
		return (clock() - start - 0.0) / CLOCKS_PER_SEC;
	}

private:
	clock_t start;
};