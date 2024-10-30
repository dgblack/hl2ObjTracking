#include "IrTracker.h"

using namespace Eigen;

IrTracker::IrTracker(int width, int height, LogLevel logLevel)
    : m_imWidth(width)
    , m_imHeight(height)
    , m_logLevel(logLevel)
{
    m_device_T_depth.setIdentity();

    m_xMaxCrop = width - 1;
    m_yMaxCrop = height - 1;

    // Default to normalized coordinates
    m_imagePointToCameraUnitPlane = [width, height](const std::array<double, 2>& uv, std::array<double, 2>& xy) {
        xy[0] = uv[0] / width;
        xy[1] = uv[1] / height;
        };
}

Vector4i IrTracker::setROI(int xMin, int xMax, int yMin, int yMax) {
    // Keep within the bounds of the camera
    if (xMin < m_depthCamRoiLeftCol) xMin = m_depthCamRoiLeftCol;
    if (yMin < m_depthCamRoiUpperRow) yMin = m_depthCamRoiUpperRow;
    if (xMax > m_depthCamRoiRightCol) xMax = m_depthCamRoiRightCol;
    if (yMax > m_depthCamRoiLowerRow) yMax = m_depthCamRoiLowerRow;

    if (xMax - xMin <= 10 || yMax - yMin <= 10) {
        std::cerr << "ROI width or height too small" << std::endl;
        return Vector4i{ m_xMinCrop, m_xMaxCrop, m_yMinCrop, m_yMaxCrop };
    }

    if (m_logLevel == LogLevel::VeryVerbose)
        LOG << "Set ROI to " << xMin << ", " << xMax << ", " << yMin << ", " << yMax;

    std::scoped_lock<std::mutex> l(m_paramMutex);
    m_xMinCrop = xMin;
    m_yMinCrop = yMin;
    m_xMaxCrop = xMax;
    m_yMaxCrop = yMax;

    return Vector4i{ xMin, xMax, yMin, yMax };
}

void IrTracker::setSearchParams(int minArea, int maxArea, int binThreshold, float convexity, float circularity, DetectionMode detectMode) {
    if (m_logLevel == LogLevel::VeryVerbose)
        LOG << "Setting search parameters";
    std::scoped_lock<std::mutex> l(m_paramMutex);
    m_minArea = minArea;
    m_maxArea = maxArea;
    m_binThreshold = binThreshold;
    m_convexity = convexity;
    m_circularity = circularity;
    m_mode = detectMode;
}

void IrTracker::setMinArea(const int minArea)
{
    std::scoped_lock<std::mutex> l(m_paramMutex);
    m_minArea = minArea;
}

void IrTracker::setMaxArea(const int maxArea)
{
    std::scoped_lock<std::mutex> l(m_paramMutex);
    m_maxArea = maxArea;
}

void IrTracker::setBinaryThreshold(const int binThreshold)
{
    std::scoped_lock<std::mutex> l(m_paramMutex);
    m_binThreshold = binThreshold;
}

void IrTracker::setMinConvexity(const float convexity)
{
    std::scoped_lock<std::mutex> l(m_paramMutex);
    m_convexity = convexity;
}

void IrTracker::setMinCircularity(float circularity) {
    std::scoped_lock<std::mutex> l(m_paramMutex);
    m_circularity = circularity;
}

void IrTracker::setDetectionMode(DetectionMode mode) {
    std::scoped_lock<std::mutex> l(m_paramMutex);
    m_mode = mode;
}

void IrTracker::setCameraExtrinsics(const Eigen::Isometry3d& device_T_depth) {
    std::scoped_lock<std::mutex> l(m_paramMutex);
    m_device_T_depth = device_T_depth;
}

void IrTracker::setCameraIntrinsics(std::function<void(const std::array<double, 2>&, std::array<double, 2>&)> intrinsics) {
    m_imagePointToCameraUnitPlane = std::move(intrinsics);
}

void IrTracker::setCameraBoundaries(int roiUpperRow, int roiLowerRow, int roiLeftCol, int roiRightCol, int nearClipPlane, int farClipPlane) {
    m_depthCamRoiUpperRow = roiUpperRow;
    m_depthCamRoiLowerRow = roiLowerRow;
    m_depthCamRoiLeftCol = roiLeftCol;
    m_depthCamRoiRightCol = roiRightCol;
    m_depthNearClip = nearClipPlane;
    m_depthFarClip = farClipPlane;

    // Update the ROI so it cuts off at the correct boundaries if it is currently searching the whole image
    if (m_xMaxCrop == m_imWidth - 1 && m_yMaxCrop == m_imHeight - 1 && m_xMinCrop == 0 && m_yMinCrop == 0) {
        setROI(m_xMinCrop, m_xMaxCrop, m_yMinCrop, m_yMaxCrop);
    }


    if (m_logLevel == LogLevel::VeryVerbose)
        LOG << "Top row " << m_depthCamRoiUpperRow << ", bottom row " << m_depthCamRoiLowerRow << ", left col " << m_depthCamRoiLeftCol << ", right col " << m_depthCamRoiRightCol << ", near " << m_depthNearClip << ", far " << m_depthFarClip;
}

IrTracker::LogLevel IrTracker::getLogLevel() {
    return m_logLevel;
}
int IrTracker::getWidth() {
    return m_imWidth;
}
int IrTracker::getHeight() {
    return m_imHeight;
}

std::shared_ptr<IrTracker::IrDetection> IrTracker::findKeypointsWorldFrame(std::unique_ptr<std::vector<uint16_t>> reflIm, std::unique_ptr<std::vector<uint16_t>> depthMap, const Eigen::Isometry3d& devicePose) {
    auto detection = std::make_shared<IrDetection>();
    detection->devicePose = Isometry3d(devicePose);

    // Freeze the parameters for now
    int xMinCrop, yMinCrop, xMaxCrop, yMaxCrop, width, height;
    Isometry3d device_T_depth;
    {
        std::scoped_lock<std::mutex> l(m_paramMutex);
        xMinCrop = m_xMinCrop;
        yMinCrop = m_yMinCrop;
        xMaxCrop = m_xMaxCrop;
        yMaxCrop = m_yMaxCrop;
        width = m_imWidth;
        height = m_imHeight;
        device_T_depth = Isometry3d(m_device_T_depth);
    }

    if (m_logLevel == LogLevel::VeryVerbose)
        LOG << "Creating CV mat with dimensions " << width << " x " << height;

    const std::vector<int> sz = { width, height };
    cv::Mat im = cv::Mat(sz, CV_16UC1, reflIm->data(), 0);

    if (im.empty()) {
        return detection;
    }

    if (m_logLevel == LogLevel::VeryVerbose)
        LOG << "Cropping image to ROI";

    // Scale contrast so it uses whole range
    // Also crop the image to the ROI
    const std::vector<int> croppedSz = { yMaxCrop - yMinCrop + 1, xMaxCrop - xMinCrop + 1 };
    cv::Mat im8b = cv::Mat(croppedSz, CV_8UC1);
    for (int x = xMinCrop; x <= xMaxCrop; x++) {
        for (int y = yMinCrop; y <= yMaxCrop; y++) {
            uint16_t el = im.at<uint16_t>(y, x);
            int xc = x - xMinCrop;  int yc = y - yMinCrop;
            if (el > 1000)
                im8b.at<uchar>(yc, xc) = 255;
            else
                im8b.at<uchar>(yc, xc) = (uchar)(el * 255 / 1000);
        }
    }

    if (m_logLevel == LogLevel::VeryVerbose)
        LOG << "Cropped image to (" << xMinCrop << ", " << yMinCrop << "),  (" << xMaxCrop << ", " << yMaxCrop << ")";

    bool success = false;
    std::vector<cv::KeyPoint> keypoints;
    auto start = std::chrono::steady_clock::now();
    if (m_mode == DetectionMode::Contour)
        success = contourDetect(im8b, keypoints);
    else // Use blob detection
        success = blobDetect(im8b, keypoints);
    if (m_logLevel == LogLevel::VeryVerbose)
        LOG << "Elapsed while searching: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count() << " us";

    // Map to 3D
    if (success) {
        if (m_logLevel == LogLevel::VeryVerbose) {
            LOG << "Found " << keypoints.size() << " points";
            LOG << "Extrinsics: " << device_T_depth.matrix();
        }
        // Save the 3D points
        for (size_t i = 0; i < keypoints.size(); i++) {
            // Point in image array
            double x = keypoints[i].pt.x + xMinCrop;
            double y = keypoints[i].pt.y + yMinCrop;
            int xInt = static_cast<int>(std::round(x));
            int yInt = static_cast<int>(std::round(y));

            // Consider only points in the camera's uniform region of interest
            if (y < m_depthCamRoiUpperRow || y > m_depthCamRoiLowerRow || x < m_depthCamRoiLeftCol || x > m_depthCamRoiRightCol) {
                if (m_logLevel == LogLevel::Verbose)
                    LOG << "Actually out of bounds";
                //continue;
            }

            // Unmap points from camera plane
            std::array<double, 2> xy = { 0, 0 };
            std::array<double, 2> uv = { x , y };
            m_imagePointToCameraUnitPlane(uv, xy);
            auto pointOnUnitPlane = Vector3d(xy[0], xy[1], 1);

            if (m_logLevel == LogLevel::VeryVerbose)
                LOG << "(" << uv[0] << ", " << uv[1] << ") mapped to (" << xy[0] << ", " << xy[1] << ")";

            // Find depth at point
            auto depth = (*depthMap)[(int)(yInt * m_imWidth + xInt)];
            if (m_logLevel == LogLevel::VeryVerbose)
                LOG << "Depth: " << depth;
            if (depth < m_depthNearClip || depth > m_depthFarClip) continue;
            Vector3d pt = ((double)depth) / 1000.0 * pointOnUnitPlane.normalized();

            // Apply transformation
            Vector3d pointInHl = device_T_depth * pt;
            if (m_logLevel == LogLevel::VeryVerbose) {
                LOG << "In camera frame: " << pt;
                LOG << "In HoloLens frame: " << pointInHl;
            }

            // Also find the diameter of the blob in the world coordinates
            auto left = (x - keypoints[i].size / 2 > 0) ? x - keypoints[i].size / 2 : 0;
            auto right = (x + keypoints[i].size / 2 < 512) ? x + keypoints[i].size / 2 : 511;
            std::array<double, 2> xyL = { 0, 0 };
            std::array<double, 2> uvL = { left , y };
            m_imagePointToCameraUnitPlane(uvL, xyL);
            std::array<double, 2> xyR = { 0, 0 };
            std::array<double, 2> uvR = { right , y };
            m_imagePointToCameraUnitPlane(uvR, xyR);
            auto leftPoint = Vector3d(xyL[0], xyL[1], 1);
            auto rightPoint = Vector3d(xyR[0], xyR[1], 1);
            leftPoint = ((double)depth) / 1000.0 * leftPoint.normalized();
            rightPoint = ((double)depth) / 1000.0 * rightPoint.normalized();
            double diam = (rightPoint - leftPoint).norm();

            // Save the points
            detection->points.push_back(pointInHl);
            detection->imCoords.emplace_back(xInt, yInt);
            detection->markerDiameters.push_back(diam);
        }
    }
    else {
        if (m_logLevel == LogLevel::Verbose)
            LOG << "Did not find any markers";
    }

    if (m_logLevel == LogLevel::Verbose)
        LOG << "Returning IR detection with " << detection->points.size() << " points";

    return detection;
}

bool IrTracker::contourDetect(cv::Mat& im, std::vector<cv::KeyPoint>& keypoints) {
    // Use findContours
    cv::Mat imBW;
    threshold(im, imBW, m_binThreshold, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(imBW, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {

        // Compute area
        double area = cv::contourArea(contours[i]);

        // Compute convexity
        std::vector<cv::Point> cvxHull;
        cv::convexHull(contours[i], cvxHull);
        double cvxArea = cv::contourArea(cvxHull);
        double cvxity = area / cvxArea;

        // Compute circularity
        cv::Point2f centre; float radius;
        cv::minEnclosingCircle(contours[i], centre, radius);
        double circy = area / (3.14159265 * radius * radius);

        // Filter by area, convexity, circularity
        if (area < m_minArea || area > m_maxArea || cvxity < m_convexity || circy < m_circularity)
            continue;

        // Find contour centroids and save as keypoints
        float x = 0; float y = 0;
        for (size_t j = 0; j < contours[i].size(); j++) {
            x += contours[i][j].x;
            y += contours[i][j].y;
        }
        x /= contours[i].size();
        y /= contours[i].size();

        cv::KeyPoint p(x, y, 0);
        p.size = radius * 2;
        keypoints.push_back(p);
    }

    return !keypoints.empty();
}

bool IrTracker::blobDetect(cv::Mat& im, std::vector<cv::KeyPoint>& keypoints) {
    // Define parameters
    cv::SimpleBlobDetector::Params params;
    params.filterByColor = true; // Filter by brightness. Markers should be bright
    params.blobColor = 255;
    params.minThreshold = 10;
    params.maxThreshold = 255;
    params.filterByArea = true;
    params.minArea = m_minArea; // size in pixels
    params.maxArea = m_maxArea;
    params.filterByConvexity = true;
    params.minConvexity = m_convexity;

    params.minDistBetweenBlobs = 1;
    params.filterByInertia = false;
    params.filterByCircularity = false;

    // Create detector
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

    // Detect keypoints
    detector->detect(im, keypoints);

    return !keypoints.empty();
}
