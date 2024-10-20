#pragma once

#include <iostream>

struct X {
	~X() { std::cout << std::endl; }
};

#define LOG (X(), std::cout << "\n[IR Marker Tracking: " << __FILE__ << " " << __LINE__ << "] ")