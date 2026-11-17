#pragma once

#include "ByteTrack/Rect.h"
#include "ByteTrack/KalmanFilter.h"

#include <cstddef>

namespace byte_track
{
enum class STrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3,
};

/// Rate of change of the box, in pixels per second.
struct Velocity
{
    float x = 0.f;       // box centre
    float y = 0.f;       // box centre
    float height = 0.f;
};

class STrack
{
public:
    /// Class-aware matching needs the detection's class label travelling
    /// with the track; default 0 keeps source-level compatibility with
    /// callers that don't care about class.
    STrack(const Rect<float>& rect, const float& score,
           const int& label = 0,
           const float& kalman_pos_weight = 1.f / 20.f,
           const float& kalman_vel_weight = 1.f / 160.f,
           const double& frame_period_ms = 1000.0 / 30.0);
    ~STrack();

    const Rect<float>& getRect() const;
    const STrackState& getSTrackState() const;

    const bool& isActivated() const;
    const float& getScore() const;
    const int&  getLabel() const;
    const size_t& getTrackId() const;
    const size_t& getFrameId() const;
    const size_t& getStartFrameId() const;
    const size_t& getTrackletLength() const;

    /// Kalman velocity of the box. Zero until the track has been predicted and updated at least once.
    Velocity getVelocity() const;

    /// now_ms is the time of the frame the call belongs to. Kalman steps are measured from it in units of the
    /// nominal frame period, so tracks follow real time when frames arrive at irregular intervals.
    void activate(const size_t& frame_id, const size_t& track_id, double now_ms);
    void reActivate(const STrack &new_track, const size_t &frame_id, double now_ms, const int &new_track_id = -1);

    void predict(double now_ms);

    /// Sets getRect() to the Kalman state, so IoU association compares detections with the predicted box rather
    /// than the last measured one.
    void usePredictedRect();
    void update(const STrack &new_track, const size_t &frame_id, double now_ms);

    void markAsLost();
    void markAsRemoved();

private:
    KalmanFilter kalman_filter_;
    KalmanFilter::StateMean mean_;
    KalmanFilter::StateCov covariance_;

    Rect<float> rect_;
    STrackState state_;

    bool is_activated_;
    float score_;
    int   label_;
    size_t track_id_;
    size_t frame_id_;
    size_t start_frame_id_;
    size_t tracklet_len_;
    double frame_period_ms_;
    double state_ms_;        // time the Kalman state refers to

    void updateRect();
};
}
