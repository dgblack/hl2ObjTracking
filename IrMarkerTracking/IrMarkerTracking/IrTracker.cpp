#include "IrTracker.h"

using namespace Eigen;

IrTracker::IrTracker(int width, int height, LogLevel logLevel)
    : m_imWidth(width)
    , m_imHeight(height)
    , m_logLevel(logLevel)
{
    m_device_T_depth.setIdentity();

    m_wCrop = width;
    m_hCrop = height;

    // Default to normalized coordinates
    m_imagePointToCameraUnitPlane = [width, height](const std::array<double, 2>& uv, std::array<double, 2>& xy) {
        xy[0] = uv[0] / width;
        xy[1] = uv[1] / height;
        };
}

void IrTracker::setROI(int x, int y, int w, int h) {
    // Sanity checks
    if (w <= 10 || h <= 10) {
        std::cerr << "ROI width or height too small" << std::endl;
        return;
    }
    if (x < 0) x = 0;
    if (y < 0) y = 0;

    // Top left corner
    int xl = x - (int)(w / 2);
    int yl = y - (int)(w / 2);

    // More sanity checks
    if (xl < 0) xl = 0;
    if (yl < 0) yl = 0;
    if (xl + w >= m_imWidth) w = m_imWidth - xl - 1;
    if (yl + h >= m_imHeight) h = m_imHeight - yl - 1;

    if (m_logLevel == LogLevel::VeryVerbose)
        LOG << "Set ROI to " << xl << ", " << yl << ", " << w << ", " << h;

    std::scoped_lock<std::mutex> l(m_paramMutex);
    m_xCrop = xl;
    m_yCrop = yl;
    m_wCrop = w;
    m_hCrop = h;
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

    if (m_logLevel == LogLevel::VeryVerbose)
        LOG << "Top row " << m_depthCamRoiUpperRow << ", bottom row " << m_depthCamRoiLowerRow << ", left col " << m_depthCamRoiLeftCol << ", right col " << m_depthCamRoiRightCol << ", near " << m_depthNearClip << ", far " << m_depthFarClip;
}

IrTracker::LogLevel IrTracker::getLogLevel() {
    return m_logLevel;
}

std::shared_ptr<IrTracker::IrDetection> IrTracker::findKeypointsWorldFrame(std::unique_ptr<std::vector<uint16_t>> reflIm, std::unique_ptr<std::vector<uint16_t>> depthMap, const Eigen::Isometry3d& devicePose) {
    auto detection = std::make_shared<IrDetection>();
    detection->devicePose = Isometry3d(devicePose);

    // Freeze the parameters for now
    int xCrop, yCrop, wCrop, hCrop, width, height;
    Isometry3d device_T_depth;
    {
        std::scoped_lock<std::mutex> l(m_paramMutex);
        xCrop = m_xCrop;
        yCrop = m_yCrop;
        wCrop = m_wCrop;
        hCrop = m_hCrop;
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

    // Scale contrast so it uses whole range
    // Also crop the image to the ROI
    const std::vector<int> croppedSz = { wCrop, hCrop };
    cv::Mat im8b = cv::Mat(croppedSz, CV_8UC1);
    for (int x = xCrop; x < xCrop + wCrop; x++) {
        for (int y = yCrop; y < yCrop + hCrop; y++) {
            uint16_t el = im.at<uint16_t>(y, x);
            int xc = x - xCrop;  int yc = y - yCrop;
            if (el > 1000)
                im8b.at<uchar>(yc, xc) = 255;
            else
                im8b.at<uchar>(yc, xc) = (uchar)(el * 255 / 1000);
        }
    }

    if (m_logLevel == LogLevel::VeryVerbose)
        LOG << "Cropped image to " << wCrop << " x " << hCrop << " at [" << xCrop << ", " << yCrop << "]";

    bool success = false;
    std::vector<cv::KeyPoint> keypoints;
    if (m_mode == DetectionMode::Contour)
        success = contourDetect(im8b, keypoints);
    else // Use blob detection
        success = blobDetect(im8b, keypoints);

    // Map to 3D
    if (success) {
        if (m_logLevel == LogLevel::VeryVerbose)
            LOG << "Found " << keypoints.size() << " points";
        // Save the 3D points
        for (size_t i = 0; i < keypoints.size(); i++) {
            // Point in image array
            double x = keypoints[i].pt.x + xCrop;
            double y = keypoints[i].pt.y + yCrop;
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
                LOG << "Extrinsics: " << device_T_depth.matrix();
                LOG << "In camera frame: " << pt;
                LOG << "In HoloLens frame: " << pointInHl;
            }

            // Save the points
            detection->points.push_back(pointInHl);
            detection->imCoords.emplace_back(xInt, yInt);
        }
    }
    else {
        setROI(256, 256, 512, 512);
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
        keypoints.push_back(cv::KeyPoint(x, y, 0));
    }

    return (contours.size() > 0);
}

bool IrTracker::blobDetect(cv::Mat& im, std::vector<cv::KeyPoint>& keypoints) {
    // Define parameters
    cv::SimpleBlobDetector::Params params;
    params.filterByColor = true; // Filter by brightness. Markers should be bright
    params.blobColor = 255;
    params.minThreshold = 10;
    params.maxThreshold = 255;
    params.filterByArea = true;
    params.minArea = 2; // size in pixels
    params.maxArea = 1000;
    params.filterByConvexity = true;
    params.minConvexity = 0.8f;

    params.minDistBetweenBlobs = 1;
    params.filterByInertia = false;
    params.filterByCircularity = false;

    // Create detector
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

    // Detect keypoints
    detector->detect(im, keypoints);
    if (keypoints.empty())
        return false;
    return true;
}
