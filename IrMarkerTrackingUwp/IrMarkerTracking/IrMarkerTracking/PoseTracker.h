#pragma once
#include "IrMarkerTracking.h"
#include "readerwriterqueue.h"
#include <future>

using namespace Eigen;

class __declspec(dllexport) PoseTracker {
public:
	explicit PoseTracker(std::shared_ptr<IrTracker> irTracker, std::vector<Vector3d> geometry, bool unity = false);
	~PoseTracker();

	/// Update the tracking with a new set of images and the device pose
	/// @param world_T_cam Pose of the camera in the world when the images were captured</param>
	/// @param irIm Infrared reflectivity image for segmenting the markers. Ownership of the pointer passes to this function</param>
	/// @param depthMap Depth map for obtaining the depth value at the segmented locations. Ownership of the pointer passes to this function</param>
	void update(const Isometry3d& world_T_cam, std::unique_ptr<std::vector<uint16_t>> irIm, std::unique_ptr<std::vector<uint16_t>> depthMap);

	/// Return the most recent computed pose of the tracked object
	Matrix4d getPose();

	/// Set the smooothing (1 means it never updates and 0 means no smoothing)
	void setSmoothing(const float smoothing);

	/// Set jump detection settings
	/// @param jumpThresholdMetres if a point's position changes more than this amount (in metres) between frames, it is considered to have jumped and is ignored
	/// @param numFramesUntilSet if a point jumped but is still there after this many frames, the new position is probably correct so we accept it
	void setJumpSettings(const float jumpThresholdMetres, const int numFramesUntilSet);

private:
	/// The IR tracker which actually interacts with the images and finds the keypoints
	std::shared_ptr<IrTracker> m_irTracker;

	/// The marker points in local coordinates, in units of metres. Should be arranged so the centroid of the points is the origin
	std::vector<Vector3d> m_markerGeom;
	MatrixXd m_markerDiffs;

	/// Mutex to synchronize calls to the resulting pose matrix
	std::mutex m_poseMutex;
	std::mutex m_camMutex;

	/// The current computed object pose
	Isometry3d m_objectPose = Isometry3d::Identity();

	/// Lockless, concurrent queue to store futures of ongoing IR detections
	moodycamel::ReaderWriterQueue<std::future<std::shared_ptr<IrTracker::IrDetection>>> m_detectionQ;

	/// Thread the continuously looks for new detections in the images and then handles the further processing
	std::shared_ptr<std::thread> m_detectionThread;

	/// Atomic flag to control the detection thread
	std::atomic<bool> m_runDetectionThread = true;

	/// Some things for the specific tracking implementation
	double m_matchThreshold = 0.008;
	int m_voteThreshold = 1;
	bool m_hasPose = false;
	bool m_hadPose = false;
	long m_framesSinceLastPose = 0;
	bool m_extrapolate = false;
	int m_nPoints;
	float m_roiBuffer = 1.25f;

	// Corrections for specific devices used with Unity (left-handed coordinates)
	bool m_unity;
	IrTracker::LogLevel m_logLevel;

	// Avoid big, sudden jumps
	std::vector<Vector3d> m_lastPoints;
	int m_nBadJumps = -1;
	float m_jumpThreshold = 0.1f; // A point moving more than this is considered to have jumped
	int m_numJumpFrames = 3; // If the jump remains in place for more than this number of frames, we take it to be the new position

	// Smoothing
	long long m_lastMeasTime = 0;
	float m_smoothing = 0.9f;
	Quaterniond m_lastRot = Quaterniond::Identity();
	Vector3d m_lastPos = Vector3d(0, 0, 0);

	// Permutations for brute force
	std::vector<std::vector<int>> m_perm3;
	std::vector<std::vector<int>> m_perm4;
	std::vector<std::vector<int>> m_perm5;

	/// The loop function that runs in the detection thread
	void detectionThreadFunction();

	/// Perform some preprocessing, discarding obvious outliers, etc.
	bool preprocessMarkerDetection(std::shared_ptr<IrTracker::IrDetection> detection);

	/// Perform processing on the detected keypoints to obtain the object pose
	void processMarkerDetection(std::shared_ptr<IrTracker::IrDetection> detection);

	// Functions for the pose computation
	double fillInMissing(std::vector<Vector3d>& mes, std::vector<int>& idxs);
	bool tryEuclideanMatching(const std::vector<Vector3d>& mesPts, std::vector<int>& idxs);
	std::vector<int> euclideanVotingOutliers(const std::vector<Vector3d>& mesPts);
	double bruteForceMatch(const std::vector<Vector3d>& mesPts, std::vector<int>& idxsWithOutliers, std::vector<int>& idxs, bool useFixedIdxs);
	void poseCalcFailed();
	Matrix3d kabsch(const std::vector<Vector3d>& mes, const std::vector<Vector3d>& exp);
	bool svd(const Matrix3d& mat, Matrix3d& U, Matrix3d& V);
	bool noBadJumps(const std::vector<Vector3d>& mes, std::vector<int> idxs);
	void setPoseFromResult(const std::shared_ptr<IrTracker::IrDetection> mes, const std::vector<int> idxs);
	Isometry3d transformFromIdxs(const std::vector<Vector3d>& mes, const std::vector<int>& idxs);
	Isometry3d findTransformFromPoints(const std::vector<Vector3d>& mes, const std::vector<Vector3d>& exp);
	double meanMatchError(const std::vector<Vector3d>& mesPts, const std::vector<int>& idxs);
	bool contains(const std::vector<int>& vec, int val);
	bool badMatch(const std::vector<int>& lst);
	Vector3d multPoint3x4(const Matrix4d& M, const Vector3d& v);
	Vector3d multPoint3x4(const Isometry3d& M, const Vector3d& v);
	Quaterniond slerp(const Quaterniond& q1, const Quaterniond& q2, const float& t);
	std::vector<std::vector<int>> recursiveCreateCombo(const std::vector<int>& ns);

	/// Current timestamp in milliseconds
	long long time();
};