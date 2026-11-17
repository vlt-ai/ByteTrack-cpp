#include "ByteTrack/STrack.h"

#include <algorithm>
#include <cstddef>

byte_track::STrack::STrack(const Rect<float>& rect, const float& score,
                           const int& label,
                           const float& kalman_pos_weight,
                           const float& kalman_vel_weight,
                           const double& frame_period_ms) :
    kalman_filter_(kalman_pos_weight, kalman_vel_weight),
    mean_(),
    covariance_(),
    rect_(rect),
    state_(STrackState::New),
    is_activated_(false),
    score_(score),
    label_(label),
    track_id_(0),
    frame_id_(0),
    start_frame_id_(0),
    tracklet_len_(0),
    frame_period_ms_(frame_period_ms),
    state_ms_(0.0)
{
}

byte_track::STrack::~STrack()
{
}

const byte_track::Rect<float>& byte_track::STrack::getRect() const
{
    return rect_;
}

const byte_track::STrackState& byte_track::STrack::getSTrackState() const
{
    return state_;
}

const bool& byte_track::STrack::isActivated() const
{
    return is_activated_;
}
const float& byte_track::STrack::getScore() const
{
    return score_;
}

const int& byte_track::STrack::getLabel() const
{
    return label_;
}

const size_t& byte_track::STrack::getTrackId() const
{
    return track_id_;
}

const size_t& byte_track::STrack::getFrameId() const
{
    return frame_id_;
}

const size_t& byte_track::STrack::getStartFrameId() const
{
    return start_frame_id_;
}

const size_t& byte_track::STrack::getTrackletLength() const
{
    return tracklet_len_;
}

byte_track::Velocity byte_track::STrack::getVelocity() const
{
    const float per_s = static_cast<float>(1000.0 / frame_period_ms_);
    return Velocity{mean_[4] * per_s, mean_[5] * per_s, mean_[7] * per_s};
}

void byte_track::STrack::activate(const size_t& frame_id, const size_t& track_id, double now_ms)
{
    state_ms_ = now_ms;
    kalman_filter_.initiate(mean_, covariance_, rect_.getXyah());

    updateRect();

    state_ = STrackState::Tracked;
    if (frame_id == 1)
    {
        is_activated_ = true;
    }
    track_id_ = track_id;
    frame_id_ = frame_id;
    start_frame_id_ = frame_id;
    tracklet_len_ = 0;
}

void byte_track::STrack::reActivate(const STrack &new_track, const size_t &frame_id, double now_ms,
                                    const int &new_track_id)
{
    // A measurement update makes the state current at now_ms even when the track was not predicted first.
    state_ms_ = std::max(state_ms_, now_ms);
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());

    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    label_ = new_track.getLabel();
    if (0 <= new_track_id)
    {
        track_id_ = new_track_id;
    }
    frame_id_ = frame_id;
    tracklet_len_ = 0;
}

void byte_track::STrack::predict(double now_ms)
{
    if (state_ != STrackState::Tracked)
    {
        mean_[7] = 0;
    }
    if (now_ms > state_ms_)
    {
        kalman_filter_.predict(mean_, covariance_, static_cast<float>((now_ms - state_ms_) / frame_period_ms_));
        state_ms_ = now_ms;
    }
}

void byte_track::STrack::usePredictedRect()
{
    updateRect();
}

void byte_track::STrack::update(const STrack &new_track, const size_t &frame_id, double now_ms)
{
    state_ms_ = std::max(state_ms_, now_ms);
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());

    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    label_ = new_track.getLabel();
    frame_id_ = frame_id;
    tracklet_len_++;
}

void byte_track::STrack::markAsLost()
{
    state_ = STrackState::Lost;
}

void byte_track::STrack::markAsRemoved()
{
    state_ = STrackState::Removed;
}

void byte_track::STrack::updateRect()
{
    rect_.width() = mean_[2] * mean_[3];
    rect_.height() = mean_[3];
    rect_.x() = mean_[0] - rect_.width() / 2;
    rect_.y() = mean_[1] - rect_.height() / 2;
}
