#pragma once
#include <unknwn.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>

// OpenCV
#include <opencv2/core.hpp> // for mats
#include <opencv2/features2d.hpp> // for blob detection
#include <opencv2/imgproc.hpp> // for thresholding
#include <opencv2/calib3d.hpp>

#include <queue>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <wchar.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <cmath>
#include <DirectXMath.h>
#include <vector>
#include<winrt/Windows.Perception.Spatial.h>
#include<winrt/Windows.Perception.Spatial.Preview.h>