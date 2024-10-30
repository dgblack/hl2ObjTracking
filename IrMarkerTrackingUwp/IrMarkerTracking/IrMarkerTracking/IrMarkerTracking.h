#pragma once

#include <winapifamily.h>
#define WINAPI_FAMILY WINAPI_PARTITION_APP

#include "log.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <thread>
#include <mutex>

class __declspec(dllexport) IrTracker {
public:
	enum class DetectionMode {
		Contour,
		Blob
	};
	enum class LogLevel {
		Silent,
		Verbose,
		VeryVerbose
	};

	struct IrDetection {
		std::vector<Eigen::Vector3d> points;
		std::vector<Eigen::Vector2i> imCoords;
		std::vector<float> markerDiameters;
		Eigen::Isometry3d devicePose;
	};

	/// Create an IrTracker object.
	/// @param width the width of the input image in pixels
	/// @param height the height of the input image in pixels
	IrTracker(const int width, const int height, LogLevel logLevel = LogLevel::Silent);

	/// Set the rectangular region of const interest in the image in which to search for the markers.
	/// @param xMax the column of the left edge of the ROI
	/// @param xMax the column of the right edge of the ROI
	/// @param yMin the row of the top edge of the ROI
	/// @param yMax the row of the bottom edge of the ROI
	Eigen::Vector4i setROI(int xMin, int xMax, int yMin, int yMax);

	/// Parameter settings
	/// @param minArea the minimum area blob to consider as a potential marker (units of pixels)
	/// @param maxArea the maximum area blob to consider as a potential marker (units of pixels)
	/// @param binThreshold the threshold for image binarization (values > binThreshold are set to 255 and others are set to 0)
	/// @param convexity the minimum convexity blob to consider as a potential marker, measured as (area) / (area of convex hull)
	/// @param circularity the minimum circularity blob to consider as a potential marker, measured as (area) / (area of minimum enclosing circle)
	/// @param detectMode use contours or blob detection. Blob detection is much slower but has fewer outliers
	void setSearchParams(const int minArea, const int maxArea, const int binThreshold, const float convexity, const float circularity, const DetectionMode detectMode);

	/// minArea is the minimum area blob to consider as a potential marker (units of pixels)
	void setMinArea(const int minArea);

	/// maxArea is the maximum area blob to consider as a potential marker (units of pixels)
	void setMaxArea(const int maxArea);

	/// Set the threshold for the binarization of the IR image in which the IR markers are segmented
	void setBinaryThreshold(const int binThreshold);

	/// Set the minimum convexity blob to consider as a potential marker, measured as 0 < (area) / (area of convex hull) <= 1
	void setMinConvexity(const float convexity);

	/// Set the minimum circularity blob to consider as a potential marker, measured as 0 < (area) / (area of minimum enclosing circle) <= 1
	void setMinCircularity(const float circularity);

	/// Use contour or blob detection. Blob detection is much slower but has fewer outliers
	void setDetectionMode(DetectionMode mode);

	/// Set the matrix device_T_depthCam by which a point, x_depthCam, in the IR/depth camera frame, is transformed to the device frame: y_device = device_T_depthCam * x_depthCam
	/// This is, for example, for a MR headset for which the depth camera frame and device frame in the world are not identical
	void setCameraExtrinsics(const Eigen::Isometry3d& device_T_depth);

	/// Register a function which is used to compute the camera intrinsics. It should take a point in the image coordinates and map it to the camera unit plane
	/// @param uv the (column, row) coordinates of a point in the image, in pixels
	/// @param xy This array is filled in when the function is called. The point is at position depth * (xy[0], xy[1], 1) in the camera frame
	void setCameraIntrinsics(std::function<void(const std::array<double, 2>&, std::array<double, 2>&)> intrinsics);

	/// Set the settings of the camera, including the region of interest and near/far clipping planes
	/// @param roiUpperRow the row index of the upper limit of the camera's region of interest
	/// @param roiLowerRow the row index of the lower limit of the camera's region of interest
	/// @param roiLeftCol the column index of the left limit of the camera's region of interest
	/// @param roiRightCol the column index of the right limit of the camera's region of interest
	/// @param nearClipPlane the depth value of the near clipping plane
	/// @param farClipPlane the depth value of the far clipping plane
	void setCameraBoundaries(int roiUpperRow, int roiLowerRow, int roiLeftCol, int roiRightCol, int nearClipPlane, int farClipPlane);

	/// Find potential marker positions in the world frame by segmenting them in the IR image, determining the corresponding depth in the depth map, and transforming
	/// this 3-vector to the world frame using the world_T_depthCam matrix previously set using setWorld_T_depthMatrix. If this matrix is not set, it defaults to identity,
	/// so the points are in the depth/IR camera frame.
	/// @param irIm the infrared reflectivity image in 16-bit grayscale in which the reflective markers are to be segmented
	/// @param depthMap the depth map in 16-bit grayscale, which must be colocated with the IR camera
	/// @param devicePose the pose of the device relative to the world when the images were captured
	std::shared_ptr<IrTracker::IrDetection> findKeypointsWorldFrame(std::unique_ptr<std::vector<uint16_t>> irIm, std::unique_ptr<std::vector<uint16_t>> depthMap, const Eigen::Isometry3d& devicePose);

	/// Return the log level verbosity setting
	LogLevel getLogLevel();

	int getWidth();
	int getHeight();

private:
	// CV settings
	int m_minArea = 2;
	int m_maxArea = 1000;
	int m_binThreshold = 222;
	float m_convexity = 0.75f;
	float m_circularity = 0.6f;
	DetectionMode m_mode = DetectionMode::Contour;

	// Input image settings
	int m_imWidth;
	int m_imHeight;
	Eigen::Isometry3d m_device_T_depth;

	// Region of Interest for search
	int m_xMaxCrop = 511;
	int m_yMaxCrop = 511;
	int m_xMinCrop = 0;
	int m_yMinCrop = 0;

	// Camera settings
	int m_depthCamRoiLowerRow = 511;
	int m_depthCamRoiUpperRow = 0;
	int m_depthCamRoiLeftCol = 0;
	int m_depthCamRoiRightCol = 512;
	int m_depthNearClip = 0;
	int m_depthFarClip = 65535;

	// Debugging
	LogLevel m_logLevel;

	// Thread safety
	std::mutex m_paramMutex;

	std::function<void(const std::array<double, 2>&, std::array<double, 2>&)> m_imagePointToCameraUnitPlane;
	bool blobDetect(cv::Mat& im, std::vector<cv::KeyPoint>& keypoints);
	bool contourDetect(cv::Mat& im, std::vector<cv::KeyPoint>& keypoints);
};