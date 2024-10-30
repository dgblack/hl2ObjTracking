#include "PoseTracker.h"

PoseTracker::PoseTracker(std::shared_ptr<IrTracker> irTracker, std::vector<Vector3d> geometry, float markerDiameter, bool unity)
    : m_irTracker(irTracker)
    , m_markerGeom(geometry)
    , m_markerDiameter(markerDiameter)
    , m_unity(unity)
{
    m_logLevel = irTracker->getLogLevel();

    m_runDetectionThread = true;
    m_detectionThread = std::make_shared<std::thread>([this]() { detectionThreadFunction(); });
    m_nPoints = geometry.size();

    m_width = m_irTracker->getWidth();
    m_height = m_irTracker->getHeight();
    m_roi = { 0, m_width - 1, 0, m_height - 1 };

    // Create lists of permutations for brute force matching
    m_perm5 = recursiveCreateCombo(std::vector<int>{ 0, 1, 2, 3, 4 });
    m_perm4 = recursiveCreateCombo(std::vector<int>{ 0, 1, 2, 3 });
    m_perm3 = recursiveCreateCombo(std::vector<int>{ 0, 1, 2 });

    m_objectPose = IRPose();

    // Create the matrix of distances between markers
    m_markerDiffs.setZero(m_nPoints, m_nPoints);
    for (int i = 0; i < m_nPoints; i++) {
        if (m_logLevel == IrTracker::LogLevel::VeryVerbose)
            LOG << "Geom: " << m_markerGeom[i][0] << ", " << m_markerGeom[i][1] << ", " << m_markerGeom[i][2];
        for (int j = 0; j < m_nPoints; j++) {
            m_markerDiffs(i, j) = (m_markerGeom[i] - m_markerGeom[j]).norm();
            // Also save the greatest distance
            if (m_markerDiffs(i, j) > m_maxPointDistance) m_maxPointDistance = m_markerDiffs(i, j);
        }
    }

    for (int i = 0; i < 4; i++)
        m_lastPoints.emplace_back(0, 0, 0);
}

PoseTracker::~PoseTracker() {
    m_runDetectionThread = false;

    if (m_detectionThread)
        m_detectionThread->join();

    while (m_detectionQ.size_approx() > 0) {
        if (!m_detectionQ.pop()) break;
    }
}

Matrix4d PoseTracker::getPose() {
    std::scoped_lock<std::mutex> lock(m_poseMutex);
    m_hasNewPose = false;
    return m_objectPose.pose.matrix();
}

PoseTracker::IRPose PoseTracker::getLastMeasurement() {
    std::scoped_lock<std::mutex> lock(m_poseMutex);
    m_hasNewPose = false;
    return m_objectPose;
}

bool PoseTracker::hasRecentPose(uint64_t milliseconds) {
    return time() - m_lastMeasTime <= milliseconds;
}

bool PoseTracker::hasNewPose() {
    return m_hasNewPose;
}

void PoseTracker::update(const Isometry3d& world_T_cam, std::unique_ptr<std::vector<uint16_t>> irIm, std::unique_ptr<std::vector<uint16_t>> depthMap) {
    // Launch the keypoint search asynchronously and store the future to check later
    //auto fut = std::async(std::launch::async, [this](const Isometry3d& T, std::unique_ptr<std::vector<uint16_t>> ir, std::unique_ptr<std::vector<uint16_t>> depth) { return m_irTracker->findKeypointsWorldFrame(std::move(ir), std::move(depth), T); }, world_T_cam, std::move(irIm), std::move(depthMap));
    //m_detectionQ.try_enqueue(std::move(fut));

    // Actually just pass the images to the queue
    m_detectionQ.emplace(std::move(irIm), std::move(depthMap), world_T_cam);
}

void PoseTracker::setSmoothing(const float smoothing)
{
    m_smoothing = smoothing;
}

void PoseTracker::setJumpSettings(bool filterJumps, const float jumpThresholdMetres, const int numFramesUntilSet) {
    m_filterJumps = filterJumps;
    m_jumpThreshold = jumpThresholdMetres;
    m_numJumpFrames = numFramesUntilSet;
}

void PoseTracker::detectionThreadFunction() {
    using namespace std::chrono_literals;
    while (m_runDetectionThread) {
        // Try to pop images from the queue
        SensorPacket detec(nullptr, nullptr, Isometry3d::Identity());
        if (m_detectionQ.try_dequeue(detec)) {
            auto detection = m_irTracker->findKeypointsWorldFrame(std::move(detec.irIm), std::move(detec.depthMap), detec.devicePose);

            if (detection->points.empty()) {
                setRoi(m_roi[0] / 1.8, m_roi[1] * 1.25, m_roi[2] / 1.8, m_roi[3] * 1.25, m_roiBuffer);
                std::this_thread::sleep_for(5ms);
                continue;
            }

            if (preprocessMarkerDetection(detection))
                processMarkerDetection(detection);
        }
        else {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(5ms);
        }
    }
}

double absval(double val) {
    return (val < 0) ? -val : val;
}

bool PoseTracker::preprocessMarkerDetection(std::shared_ptr<IrTracker::IrDetection> detection) {
    // Remove obvious outliers that are too close, too far, or completely the wrong size
    size_t i = 0;
    size_t init = detection->points.size();
    while (i < detection->points.size()) {
        if (absval(detection->points[i][2]) > 2 || absval(detection->points[i][2]) < 0.05) {// || absval(detection->markerDiameters[i] - m_markerDiameter) > m_markerDiameter / 2) {
            if (m_logLevel == IrTracker::LogLevel::VeryVerbose)
                LOG << "Removing outlier point [" << detection->points[i][0] << ", " << detection->points[i][1] << ", " << detection->points[i][2] << "] with diameter " << detection->markerDiameters[i];
            detection->points.erase(detection->points.begin() + i);
            detection->imCoords.erase(detection->imCoords.begin() + i);
            detection->markerDiameters.erase(detection->markerDiameters.begin() + i);
        }
        else {
            // Look for loners in terms of size and position
            /*int numSimilarSize = 0;
            int numClosePosition = 0;
            for (int j = 0; j < detection->points.size(); j++) {
                if (i == j) continue;

                float ratio = detection->markerDiameters[i] / detection->markerDiameters[j];
                if (ratio > 0.5 && ratio < 2)
                    numSimilarSize++;

                double d = (detection->points[i] - detection->points[j]).norm();
                if (d < m_maxPointDistance * 1.5)
                    numClosePosition++;
            }
            if (numSimilarSize < 2 || numClosePosition < 2) {
                if (m_logLevel == IrTracker::LogLevel::VeryVerbose) {
                    if (numSimilarSize < 2)
                        LOG << "Removing outlier in size [" << detection->points[i][0] << ", " << detection->points[i][1] << ", " << detection->points[i][2] << "] with diameter " << detection->markerDiameters[i];
                    else if (numClosePosition < 2)
                        LOG << "Removing outlier in position [" << detection->points[i][0] << ", " << detection->points[i][1] << ", " << detection->points[i][2] << "] with diameter " << detection->markerDiameters[i];
                    else
                        LOG << "Removing outlier in size and position [" << detection->points[i][0] << ", " << detection->points[i][1] << ", " << detection->points[i][2] << "] with diameter " << detection->markerDiameters[i];
                }
                detection->points.erase(detection->points.begin() + i);
                detection->imCoords.erase(detection->imCoords.begin() + i);
                detection->markerDiameters.erase(detection->markerDiameters.begin() + i);
                continue;
            }*/

            // Convert coordinates to left-handed for Unity
            if (m_unity) {
                auto tmp = detection->points[i][0];
                detection->points[i][0] = -detection->points[i][1];
                detection->points[i][1] = -tmp;
            }

            // Convert to world frame
            detection->points[i] = detection->devicePose * detection->points[i];

            i++;
        }
    }

    if (m_logLevel == IrTracker::LogLevel::VeryVerbose)
        LOG << "Filtering kept " << detection->points.size() << " of " << init << " points";

    // Not enough points left over? Increase the size of the ROI again
    if (detection->points.size() < 2) {
        setRoi(m_roi[0] / 1.8, m_roi[1] * 1.25, m_roi[2] / 1.8, m_roi[3] * 1.25, m_roiBuffer);
        return false;
    }

    return true;
}

void PoseTracker::processMarkerDetection(std::shared_ptr<IrTracker::IrDetection> detection) {
    // ----------------------------------- Try to find point correspondences ---------------------------------------------
    // If the ith element of idxs equals j, then the ith measured point corresponds to the jth known point
    if (m_logLevel == IrTracker::LogLevel::VeryVerbose) {
        std::string s = ""; int i = 0;
        for (const auto& v : detection->points) {
            s += std::to_string(v[0]) + ", " + std::to_string(v[1]) + ", " + std::to_string(v[2]) + "; Diameter: " + std::to_string(detection->markerDiameters[i]) + "\n";
            i++;
        }
        LOG << s;
    }

    // Use Euclidean matching first
    std::vector<int> idxs(detection->points.size());
    bool success = tryEuclideanMatching(detection->points, idxs);

    // Check if we got enough inliers - if we have mostly outliers, we can't do anything else so just return
    // I've never seen this condition triggered
    if (success && badMatch(idxs)) {
        if (m_logLevel == IrTracker::LogLevel::Verbose)
            LOG << "Failed with bad match";
        poseCalcFailed();
        return;
    }

    // Check that the fit is reasonable - not fitting to some outliers
    if (success) {
        double err = absval(meanMatchError(detection->points, idxs));
        success = (err < m_matchThreshold);
        if (!success) {
            // The match was possibly wrong so make everything unknown
            for (size_t i = 0; i < idxs.size(); i++)
                if (idxs[i] > 0)
                    idxs[i] = -2;

            if (m_logLevel == IrTracker::LogLevel::Verbose)
                LOG << "Euclidean matching failed with match error " << err;
        }
        else if (m_logLevel == IrTracker::LogLevel::Verbose) LOG << "SUCCESS: Euclidean Matching with error " << err;
    }

    // Everything failed so far - try brute force
    if (!success) {
        if (m_logLevel == IrTracker::LogLevel::Verbose)
            LOG << "Failed Euclidean matching";

        // Remove outliers
        std::vector<Vector3d> mps;
        std::vector<int> idxsMinusOutliers;
        for (size_t i = 0; i < detection->points.size(); i++)
        {
            if (idxs[i] != -1)
            {
                mps.push_back(detection->points[i]);
                idxsMinusOutliers.push_back(idxs[i]);
            }
        }

        // If more than 5, this becomes too slow. If <= 2, this is garbage anyway
        if (mps.size() <= 5 && mps.size() > 2) {
            double err = bruteForceMatch(mps, idxs, idxsMinusOutliers, false);
            success = (err < m_matchThreshold);
            if (m_logLevel == IrTracker::LogLevel::Verbose) {
                if (!success) LOG << "Failed brute force with error " << err;
                else LOG << "SUCCESS: brute force with error" << err;
            }
        }
    }

    // Try filling in any missing markers based on the ones we know
    if (success) {
        double err = fillInMissing(detection->points, idxs);
        success = (err < m_matchThreshold && err > 0);
        if (m_logLevel == IrTracker::LogLevel::Verbose) {
            if (!success) LOG << "Failed with error " << err << " after filling in missing"; // Never happens with 0.005 thresh
            else LOG << "SUCCESS: Filled in missing with error " << err;
        }
    }

    // Look for markers bouncing around badly
    if (success && m_filterJumps) {
        success = noBadJumps(detection->points, idxs);
        if (!success && m_logLevel == IrTracker::LogLevel::Verbose) LOG << "Detected illegal jump";
    }

    // Calculate and return the resulting pose from the match
    if (success)
    {
        // We found a good match
        if (m_logLevel == IrTracker::LogLevel::Verbose)
            LOG << "GOT POSE";
        m_lastPoseTime = time();
        setPoseFromResult(detection, idxs);
        m_hasNewPose = true;
        return;
    }
    if (m_logLevel == IrTracker::LogLevel::Verbose)
        LOG << "FAILED POSE COMPUTATION";
    poseCalcFailed();
}

void PoseTracker::poseCalcFailed()
{
    // Nothing worked. Don't use this sample
}

void PoseTracker::setPoseFromResult(std::shared_ptr<IrTracker::IrDetection> mes, const std::vector<int> idxs)
{
    Isometry3d p = transformFromIdxs(mes->points, idxs);

    // First order low-pass filter
    m_lastRot = slerp(m_lastRot, Quaterniond(p.rotation()), 1 - m_smoothing);
    m_lastPos = m_lastPos * m_smoothing + p.translation() * (1 - m_smoothing);

    // Set the current pose in a thread-safe way
    std::scoped_lock<std::mutex> lock(m_poseMutex);
    m_objectPose.pose = m_objectPose.pose.fromPositionOrientationScale(m_lastPos, m_lastRot, Vector3d(1, 1, 1));
    m_objectPose.imageCoords = std::vector<Vector2i>(mes->imCoords);
    m_objectPose.markerPositions = std::vector<Vector3d>(mes->points);

    // Also update the search area
    int minX = m_width; int minY = m_height; int maxX = 0; int maxY = 0;
    std::vector<Vector2i> markerImPoints;
    for (size_t j = 0; j < mes->imCoords.size(); j++)
    {
        if (idxs.size() > j && idxs[j] != -1)
        {
            if (mes->imCoords[j][0] < minX) minX = mes->imCoords[j][0];
            if (mes->imCoords[j][1] < minY) minY = mes->imCoords[j][1];
            if (mes->imCoords[j][0] > maxX) maxX = mes->imCoords[j][0];
            if (mes->imCoords[j][1] > maxY) maxY = mes->imCoords[j][1];
        }
    }

    if (minX == 0 && minY == 0 && maxX == m_width && maxY == m_height)
    {
        //Debug.Log("No inlying points found");
        // Don't reset ROI here. It should only happen in one place, otherwise it gets out of hand. See the else for kps.Count < 0
    }
    else
    {
        setRoi(minX, maxX, minY, maxY, m_roiBuffer);
        //updateRoi(minX, maxX, minY, maxY);
    }
}

void PoseTracker::setRoi(int minX, int maxX, int minY, int maxY, int buffer) {
    m_lastRoiTime = time();
    m_roi = m_irTracker->setROI(minX-buffer, maxX+buffer, minY-buffer, maxY+buffer);
}

void PoseTracker::updateRoi(int minX, int maxX, int minY, int maxY) {
    Vector4i newRoi = { minX, maxX, minY, maxY };

    // If the ROI was not set, set it immediately to the new one but with a larger buffer
    if (m_roi[0] == 0 && m_roi[1] == m_width - 1 && m_roi[2] == 0 && m_roi[3] == m_height - 1) {
        setRoi(minX, maxX, minY, maxY, m_roiBuffer * 2);
    }

    // Find the centres of the ROIs
    Vector2d newRoiCentre = { (minX + maxX) / 2.0, (minY + maxY) / 2.0 };
    Vector2d oldRoiCentre = { (m_roi[0] + m_roi[1]) / 2.0, (m_roi[2] + m_roi[3]) / 2.0 };

    // Compute the velocity of the ROI
    auto t = time();
    auto dt = t - m_lastRoiTime;
    m_lastRoiTime = t;
    Vector2d newVel = (newRoiCentre - oldRoiCentre) / dt;

    // Smooth the velocity a bit
    Vector2d vel = m_roiSmoothing * m_roiVel + (1 - m_roiSmoothing) * newVel;

    // Extrapolate the predicted position of the next ROI, assuming velocity and framerate stay constant with slight deceleration
    Vector2d offset = dt * m_roiDeceleration * vel;
    setRoi(minX + offset[0], maxX + offset[0], minY + offset[1], maxY + offset[1], m_roiBuffer);
}

double PoseTracker::fillInMissing(std::vector<Vector3d>& mes, std::vector<int>& idxs)
{
    // Find how many indices were matched
    int nPos = 0;
    int nUA = 0;
    for (size_t i = 0; i < idxs.size(); i++)
    {
        if (idxs[i] >= 0) nPos++;
        else if (idxs[i] == -2) nUA++;
    }

    if (nPos == 4) // All points are accounted for. Just calculate the fit error
    {
        // Find the transform
        std::vector<Vector3d> mesPts;
        std::vector<Vector3d> expPts;
        for (size_t i = 0; i < idxs.size(); i++)
        {
            if (idxs[i] != -1)
            {
                mesPts.push_back(mes[i]);
                expPts.push_back(m_markerGeom[idxs[i]]);
            }
        }
        Isometry3d M = findTransformFromPoints(mesPts, expPts);

        // Compute the error
        double err = 0;
        for (size_t i = 0; i < mesPts.size(); i++)
        {
            err += (mesPts[i] - multPoint3x4(M, expPts[i])).norm();
        }
        return err / mesPts.size();
    }
    else if (nPos == 3) // All but 1 accounted for. Should be easy to find the fourth
    {
        // Find the transform
        std::vector<Vector3d> mesPts;
        std::vector<Vector3d> expPts;
        for (size_t i = 0; i < idxs.size(); i++)
        {
            if (idxs[i] >= 0)
            {
                mesPts.push_back(mes[i]);
                expPts.push_back(m_markerGeom[idxs[i]]);
            }
        }
        Isometry3d M = findTransformFromPoints(mesPts, expPts);

        // Determine which point is missing
        int missing = -1;
        for (size_t i = 0; i < 4; i++)
        {
            if (!contains(idxs, i)) { missing = i; break; }
        }

        if (missing != -1)
        {
            // If this point was actually missing, add it
            if (nUA == 0)
            {
                // push_back the missing point to the lists
                Vector3d pt = multPoint3x4(M, m_markerGeom[missing]);
                mesPts.push_back(pt);
                mes.push_back(pt); // push_back to the referenced list, which affects the call to ukf.OnNewPointSample
                expPts.push_back(m_markerGeom[missing]);

                // push_back the new index to the end of the idxs list as well to match the new point
                idxs.push_back(missing);
            }
            // Otherwise, if we had this point and it was just unmatched, change it
            else if (nUA == 1)
            {
                // Find the unmatched point
                int idx = -1;
                for (size_t i = 0; i < idxs.size(); i++)
                    if (idxs[i] == -2) idx = i;

                // push_back it / change it in the various lists
                if (idx != -1)
                {
                    Vector3d pt = multPoint3x4(M, m_markerGeom[missing]);
                    mesPts.push_back(pt);
                    expPts.push_back(m_markerGeom[missing]);
                    mes[idx] = pt;
                    idxs[idx] = missing;
                }
            }
        }

        // Compute the error
        double err = 0;
        for (size_t i = 0; i < mesPts.size(); i++)
        {
            err += (mesPts[i] - multPoint3x4(M, expPts[i])).norm();
        }
        return err / mesPts.size() * ((missing < 0) ? -1 : 1);
    }
    else if (nPos == 2 && nUA == 1)
    {
        // We know which marker is ok, we just don't know what geometry point that fits to
        // There are only two options, so try both and compare the error
        // Determine which geometry points are unassigned
        std::vector<int> missing = { -1, -1 };
        int count = 0;
        for (size_t i = 0; i < 4; i++)
        {
            if (!contains(idxs, i)) { missing[count] = i; count++; }
        }

        // For each unassigned geometry point, assign it and check the fit error
        double minErr = 99999999;
        int minIdx = -1;
        for (size_t ua = 0; ua < 2; ua++)
        {
            // Find the transform
            std::vector<Vector3d> mesPts;
            std::vector<Vector3d> expPts;
            for (size_t i = 0; i < idxs.size(); i++)
            {
                if (idxs[i] >= 0)
                {
                    mesPts.push_back(mes[i]);
                    expPts.push_back(m_markerGeom[idxs[i]]);
                }
                else if (idxs[i] == -2)
                {
                    mesPts.push_back(mes[i]);
                    expPts.push_back(m_markerGeom[missing[ua]]);
                }
            }
            Isometry3d M = findTransformFromPoints(mesPts, expPts);

            // Compute the error
            double err = 0;
            for (size_t i = 0; i < mesPts.size(); i++)
            {
                err += (mesPts[i] - multPoint3x4(M, expPts[i])).norm();
            }
            err /= mesPts.size();

            // Compare the error
            if (err < minErr)
            {
                minErr = err;
                minIdx = missing[ua];
            }
        }

        if (minErr < m_matchThreshold)
        {
            // If the new error is acceptable, assign the new index 
            for (size_t i = 0; i < idxs.size(); i++)
                if (idxs[i] == -2) idxs[i] = minIdx;

            // Try to fill in the final point
            return fillInMissing(mes, idxs);
        }
        else return -1;
    }
    else return -1;
}

bool PoseTracker::tryEuclideanMatching(const std::vector<Vector3d>& mesPts, std::vector<int>& idxs)
{
    // ----------------------- Euclidean Outlier Removal -----------------------------------
    // Remove outliers from mesPts and look for possible matches
    idxs = euclideanVotingOutliers(mesPts);

    bool hasRedundant = false;
    int numUnassigned = 0;
    int numAssigned = 0;
    for (size_t i = 0; i < idxs.size(); i++)
    {
        if (idxs[i] == -2) numUnassigned++;
        else if (idxs[i] != -1) numAssigned++;
        for (size_t j = i + 1; j < idxs.size(); j++)
        {
            if (idxs[i] != -1 && idxs[i] == idxs[j])
            {
                hasRedundant = true;
                idxs[i] = -2;
                idxs[j] = -2;
                break;
            }
        }
    }

    // Assign the single unknown point to the single remaining geometry point (if possible)
    if (!hasRedundant && numAssigned == m_nPoints - 1 && numUnassigned == 1)
    {
        for (int i = 0; i < m_nPoints; i++)
        {
            if (!contains(idxs, i))
            {
                // This geometry point is not included yet
                // Find the unassigned measured point
                for (size_t j = 0; j < idxs.size(); j++)
                {
                    // And set it to this geometry point
                    if (idxs[j] == -2)
                    {
                        idxs[j] = i;
                        numAssigned++;
                        break;
                    }
                }

                break;
            }
        }
    }

    // If the point correspondence is already all good, return it
    if (!hasRedundant && numAssigned == m_nPoints) return true;
    return false;
}

std::vector<int> PoseTracker::euclideanVotingOutliers(const std::vector<Vector3d>& mesPts)
{
    if (mesPts.size() == 0) return std::vector<int>();

    size_t np = mesPts.size();
    size_t mp = m_nPoints;
    std::vector<int> idxs(np);

    // LOG << "Creating " << np.ToString() << "x" << np.ToString() << " diffs matrix";
    // Create inter-point distance matrix
    // The same matrix for the known geometry is in ukf.geom.diffs
    MatrixXd D; D.setZero(np, np);
    for (size_t i = 0; i < np; i++)
    {
        for (size_t j = i + 1; j < np; j++)
        {
            D(i, j) = (mesPts[i] - mesPts[j]).norm();
            D(j, i) = D(i, j);
        }
    }

    // Voting matrix
    MatrixXd V; V.setZero(mp, np);
    for (size_t i = 0; i < mp; i++)
    {
        for (size_t j = i + 1; j < mp; j++)
        {
            // Find closest inter-point distance
            int bestK = -1;
            int bestL = -1;
            double bestDist = 9999999;
            for (size_t k = 0; k < np; k++)
            {
                for (size_t l = k + 1; l < np; l++)
                {
                    double dist = absval(D(k, l) - m_markerDiffs(i, j));
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestK = k; bestL = l;
                    }
                }
            }

            // Vote on the corresponding indices
            if (bestK != -1 && bestL != -1)
                V(i, bestK)++; V(i, bestL)++; V(j, bestK)++; V(j, bestL)++;
        }
    }

    // Find outliers with few votes and likely point correspondences
    std::vector<bool> leq1(np);
    for (size_t i = 0; i < np; i++)
    {
        // Find sum of votes for this measured point
        // As well as the most-matched known point, if there is an obvious max
        leq1[i] = true;
        int sm = 0;
        int mostMatches = 0;
        int mostMatched = -1;
        bool isRedundant = false;
        for (size_t j = 0; j < mp; j++)
        {
            sm += static_cast<int>(V(j, i));
            if (V(j, i) > mostMatches)
            {
                mostMatches = static_cast<int>(V(j, i));
                mostMatched = j;
                isRedundant = false;
            }
            // Check if two points have the same number of matches
            else if (V(j, i) == mostMatches)
                isRedundant = true;

            if (V(j, i) > 1) leq1[i] = false;
        }

        if ((double)sm / mp < m_voteThreshold)
            idxs[i] = -1;
        else if (!isRedundant)
            idxs[i] = mostMatched;
        else idxs[i] = -2;
    }

    // Check if we're still including an outlier and an included column of votes has only 1s
    // and no other included column had only 1s. In this case we can flag that column as an outlier
    int nInliers = 0; int nLeq1 = 0; int leq1Idx = -1;
    for (size_t i = 0; i < np; i++)
    {
        if (idxs[i] != -1) nInliers++;
        if (leq1[i] && idxs[i] != -1) { nLeq1++; leq1Idx = i; }
    }
    if (leq1Idx != -1 && nInliers > static_cast<int>(mp) && nLeq1 == 1)
        idxs[leq1Idx] = -1;

    return idxs;
}

double PoseTracker::bruteForceMatch(const std::vector<Vector3d>& mesPts, std::vector<int>& idxsWithOutliers, std::vector<int>& idxs, bool useFixedIdxs)
{
    size_t np = mesPts.size();
    double bestErr = 9999999;
    int bestPermIdx = -1;

    // Generate list of permutations
    std::vector<std::vector<int>> permutations;
    if (np <= 4) permutations = m_perm4;
    // Outliers should be removed already. If not, this becomes much slower. > 6 becomes infeasibly slow
    else if (np == 5) permutations = m_perm5;
    else
    {
        if (m_logLevel == IrTracker::LogLevel::Verbose)
            LOG << "Too many outliers - exiting brute force match";
        return 100;
        //int[] ints = new int[np];
        //for (size_t i = 0; i < np; i++)
        //    ints[i] = i;
        //permutations = RecursiveCreateCombo(ints);
    }

    // Try all the permutations
    int count = 0;
    for (const auto& perm : permutations)
    {
        if (useFixedIdxs)
        {
            // Only consider the permutation if it matches the known fixed indices 
            bool valid = true;
            for (size_t i = 0; i < idxs.size(); i++)
            {
                if (idxs[i] >= 0 && idxs[i] != perm[i])
                {
                    valid = false;
                    break;
                }
            }
            if (!valid)
            {
                count++;
                continue;
            }
        }

        // Make the lists of points in the new order
        std::vector<Vector3d> exp;
        std::vector<Vector3d> mes;
        int nInclude = 0;
        for (size_t i = 0; i < np; i++)
        {
            // if perm[i] >= 4, mes[i] is an outlier so we ignore it
            if (perm[i] < m_nPoints)
            {
                nInclude++;
                mes.push_back(mesPts[i]);
                exp.push_back(m_markerGeom[perm[i]]);
            }
            //exp.push_back(m_markerGeom[perm[i]]);
        }

        // Calculate transform
        Isometry3d M = findTransformFromPoints(mes, exp);

        // Calculate the error
        double err = 0;
        for (size_t i = 0; i < mes.size(); i++)
            err += (multPoint3x4(M, exp[i]) - mes[i]).norm();
        err /= nInclude;

        // Save if minimal
        if (err < bestErr)
        {
            bestErr = err;
            bestPermIdx = count;
        }

        count++;
    }

    if (bestPermIdx == -1) return 999999;

    // Mark the outliers with -1
    std::vector<int> bestPerm = permutations[bestPermIdx];
    int j = 0;
    for (size_t i = 0; i < idxsWithOutliers.size(); i++)
    {
        if (idxsWithOutliers[i] == -1) continue;
        if (bestPerm[j] < 4)
            idxsWithOutliers[i] = bestPerm[j];
        else idxsWithOutliers[i] = -1;
        j++;
    }
    return bestErr;
}

Isometry3d PoseTracker::transformFromIdxs(const std::vector<Vector3d>& mes, const std::vector<int>& idxs)
{
    std::vector<Vector3d> mesPts;
    std::vector<Vector3d> expPts;
    for (size_t i = 0; i < idxs.size(); i++)
    {
        if (idxs[i] != -1)
        {
            mesPts.push_back(mes[i]);
            expPts.push_back(m_markerGeom[idxs[i]]);
        }
    }

    return findTransformFromPoints(mesPts, expPts);
}

Isometry3d PoseTracker::findTransformFromPoints(const std::vector<Vector3d>& mes, const std::vector<Vector3d>& exp)
{
    std::vector<Vector3d> _mes(mes.size());
    std::vector<Vector3d> _exp(exp.size());

    // Find average of each cluster of points
    Vector3d mesAvg(0, 0, 0);
    Vector3d expAvg(0, 0, 0);
    for (size_t i = 0; i < mes.size(); i++)
    {
        mesAvg += mes[i];
        expAvg += exp[i];
    }
    mesAvg /= mes.size();
    expAvg /= exp.size();

    // Move the expected points so the centroids are at the origin
    for (size_t i = 0; i < mes.size(); i++)
    {
        _exp[i] = exp[i] - expAvg;
        _mes[i] = mes[i] - mesAvg;
    }

    // Now the difference should be a pure rotation
    Matrix3d R = kabsch(_mes, _exp);
    Vector3d t = mesAvg - (R * expAvg);

    // Create homogeneous transformation matrix
    Matrix4d pse = Matrix4d::Identity();
    pse.topLeftCorner<3, 3>() = R;
    pse.topRightCorner<3, 1>() = t;
    /*for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            pse(i, j) = R(i, j);
        }
    }*/
    /*for (size_t j = 0; j < 3; j++)
    {
        pse(j, 3) = t[j];
    }*/

    return Isometry3d(pse);
}

Matrix3d PoseTracker::kabsch(const std::vector<Vector3d>& mes, const std::vector<Vector3d>& exp)
{
    const int n = mes.size();
    if (n != exp.size()) {
        LOG << "ERROR: size mismatch in Kabsch-Umeyma algorithm";
        return Matrix3d::Identity();
    }

    // Create matrices
    MatrixXd At; At.setZero(3, n); // 3 x n
    MatrixXd B;  B.setZero(n, 3);  // n x 3
    for (size_t i = 0; i < exp.size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
            At(j, i) = exp[i][j];
    }
    for (size_t i = 0; i < mes.size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
            B(i, j) = mes[i][j];
    }

    // Find rotation matrix (orthogonal, determinant 1) to map pointsFrom to pointsTo with minimum RMSE
    // Assumes the centroid of r is at [0,0,0]
    // Covariance H = A'*B
    Matrix3d H = At * B;

    // SVD of covariance matrix
    Matrix3d U, V;
    svd(H, U, V);

    // Find determinant of VU'
    Matrix3d Ut = U.transpose();
    Matrix3d VUt = V * Ut;
    auto d = VUt.determinant();

    // Check determinant (should be 1, but it's numerical, so won't be precisely 1)
    if (absval(d - 1) < 0.01f)
        return VUt;
    else
    {
        Matrix3d sign = Matrix3d::Identity();
        sign(2, 2) = -1; // d

        return V * sign * Ut;
    }
}

bool PoseTracker::svd(const Matrix3d& mat, Matrix3d& U, Matrix3d& V)
{
    // Calculate SVD
    auto jacobiSvd = JacobiSVD<Matrix3d>(mat, ComputeFullV | ComputeFullU);
    if (jacobiSvd.computeU() && jacobiSvd.computeV()) {
        U = jacobiSvd.matrixU();
        V = jacobiSvd.matrixV();
        return true;
    }
    else return false;
}

bool PoseTracker::noBadJumps(const std::vector<Vector3d>& mes, std::vector<int> idxs)
{
    if (m_nBadJumps > m_numJumpFrames)
    {
        // Been "bad" for a while so it probably actually should be this way
        for (size_t i = 0; i < mes.size(); i++)
        {
            if (idxs[i] >= 0 && idxs[i] < m_nPoints)
            {
                m_lastPoints[idxs[i]] = mes[i];
            }
        }
        m_nBadJumps = 0;
        return true;
    }

    // Check if points are reasonably close to their last locations
    bool hasBad = false;
    for (size_t i = 0; i < idxs.size(); i++)
    {
        if (idxs[i] >= 0 && idxs[i] < m_nPoints)
        {
            double dist = (mes[i] - m_lastPoints[idxs[i]]).norm();
            if (dist > m_jumpThreshold && m_nBadJumps >= 0)
            {
                hasBad = true;
            }
            else
            {
                // Update the good point positions
                m_lastPoints[idxs[i]] = mes[i];
            }
        }
    }

    if (m_nBadJumps == -1) m_nBadJumps = 0;
    if (hasBad) m_nBadJumps++;
    else m_nBadJumps = 0;

    return !hasBad;
}

double PoseTracker::meanMatchError(const std::vector<Vector3d>& mesPts, const std::vector<int>& idxs)
{
    // Make the lists of points in the new order
    std::vector<Vector3d> exp;
    std::vector<Vector3d> mes;
    for (size_t i = 0; i < mesPts.size(); i++)
    {
        // Remove outliers
        if (idxs[i] >= 0)
        {
            mes.push_back(mesPts[i]);
            exp.push_back(m_markerGeom[idxs[i]]);
        }
    }

    // Calculate transform
    Isometry3d M = findTransformFromPoints(mes, exp);

    // Calculate the error
    double err = 0;
    for (size_t i = 0; i < mes.size(); i++)
        err += (multPoint3x4(M, exp[i]) - mes[i]).norm();

    return err / mesPts.size();
}

bool PoseTracker::contains(const std::vector<int>& vec, int val)
{
    for (size_t v : vec)
        if (v == val) return true;
    return false;
}

bool PoseTracker::badMatch(const std::vector<int>& lst)
{
    int nGood = 0;
    for (size_t i : lst)
        if (i != -1) nGood++;
    return nGood <= 2;
}

Vector3d PoseTracker::multPoint3x4(const Matrix4d& M, const Vector3d& v)
{
    return M.topLeftCorner<3, 3>() * v + M.topRightCorner<3, 1>();
}

Vector3d PoseTracker::multPoint3x4(const Isometry3d& M, const Vector3d& v)
{
    return M.rotation() * v + M.translation();
}

Quaterniond PoseTracker::slerp(const Quaterniond& q1, const Quaterniond& q2, const float& t)
{
    return q1.slerp(t, q2);
}

uint64_t PoseTracker::time()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::vector<std::vector<int>> PoseTracker::recursiveCreateCombo(const std::vector<int>& ns)
{
    if (ns.size() == 2)
    {
        std::vector<std::vector<int>> l;
        l.push_back(std::vector<int>{ ns[0], ns[1] });
        l.push_back(std::vector<int>{ ns[1], ns[0] });
        return l;
    }
    else
    {
        std::vector<std::vector<int>> arrs;

        // Take each element to be the first element, arrange the remaining elements
        for (size_t i = 0; i < ns.size(); i++)
        {
            // Create list not including chosen first element
            std::vector<int> newList = std::vector<int>(ns.size() - 1);
            int count = 0;
            for (size_t j = 0; j < ns.size(); j++)
            {
                if (j != i)
                {
                    newList[count] = ns[j];
                    count++;
                }
            }

            // Add the chosen first element to all the subarrs and add to master list
            for (auto lst : recursiveCreateCombo(newList))
            {
                lst.insert(lst.begin(), ns[i]);
                arrs.push_back(lst);
            }
        }

        return arrs;
    }
}