#pragma once

#include "Eigen/Dense"

#include "ByteTrack/Rect.h"

namespace byte_track
{
class KalmanFilter
{
public:
    using DetectBox = Xyah<float>;

    using StateMean = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;
    using StateCov = Eigen::Matrix<float, 8, 8, Eigen::RowMajor>;

    using StateHMean = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
    using StateHCov = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

    KalmanFilter(const float& std_weight_position = 1. / 20,
                 const float& std_weight_velocity = 1. / 160);

    void initiate(StateMean& mean, StateCov& covariance, const DetectBox& measurement);

    void predict(StateMean& mean, StateCov& covariance);

    /// Advances the state by dt nominal frame periods: positions move by velocity * dt and the motion noise
    /// variance scales with dt, so one step of 2 moves as far as two steps of 1. dt <= 0 leaves the state alone.
    void predict(StateMean& mean, StateCov& covariance, float dt);

    void update(StateMean& mean, StateCov& covariance, const DetectBox& measurement);

private:
    float std_weight_position_;
    float std_weight_velocity_;

    Eigen::Matrix<float, 4, 8, Eigen::RowMajor> update_mat_;

    void project(StateHMean &projected_mean, StateHCov &projected_covariance,
                 const StateMean& mean, const StateCov& covariance);
};
}