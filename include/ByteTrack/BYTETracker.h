#pragma once

#include "ByteTrack/STrack.h"
#include "ByteTrack/lapjv.h"
#include "ByteTrack/Object.h"

#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <vector>

namespace byte_track
{
class BYTETracker
{
public:
    using STrackPtr = std::shared_ptr<STrack>;

    /// All tunable knobs, grouped so the call site doesn't have to
    /// memorize argument order. All fields have sane defaults matching the
    /// upstream ByteTrack paper.
    struct Params
    {
        int   frame_rate                = 30;
        int   track_buffer              = 30;             // frames @ 30 FPS
        float track_thresh              = 0.5f;           // global high/low pool split
        float high_thresh               = 0.6f;           // global new-track confirm gate
        float match_thresh              = 0.8f;           // 1st-association IoU cost cap
        float match_thresh_second       = 0.5f;           // 2nd association (was hardcoded)
        float match_thresh_unconfirmed  = 0.7f;           // unconfirmed match (was hardcoded)
        float kalman_pos_weight         = 1.0f / 20.0f;
        float kalman_vel_weight         = 1.0f / 160.0f;
        bool  class_aware_match         = false;          // gate IoU across class
        // Per-class overrides keyed by `Object.label`. Empty → fall back
        // to the global `track_thresh` / `high_thresh`.
        std::map<int, float> track_thresh_per_class;
        std::map<int, float> high_thresh_per_class;
        // Hard caps on the tracked / lost track buffers. Zero = unbounded
        // (pre-cap behaviour). When exceeded, the oldest entries by
        // `getFrameId()` are dropped. Acts as a memory panic-stop; should
        // be set well above any expected scene density.
        size_t max_tracked              = 0;
        size_t max_lost                 = 0;
    };

    /// New API: pass the full Params struct.
    explicit BYTETracker(const Params& params);

    /// Legacy 5-arg constructor — preserved so the upstream unit tests in
    /// third_party/.../test/test_BYTETracker.cpp keep compiling unchanged.
    BYTETracker(const int& frame_rate,
                const int& track_buffer,
                const float& track_thresh = 0.5f,
                const float& high_thresh = 0.6f,
                const float& match_thresh = 0.8f);

    ~BYTETracker();

    std::vector<STrackPtr> update(const std::vector<Object>& objects);

    /// Current occupancy of the tracked / lost track buffers. Used by
    /// the caller to publish memory-observability gauges without having
    /// to expose the underlying vectors.
    size_t getTrackedCount() const { return tracked_stracks_.size(); }
    size_t getLostCount()    const { return lost_stracks_.size(); }
    size_t getRemovedCount() const { return removed_stracks_.size(); }

private:
    std::vector<STrackPtr> jointStracks(const std::vector<STrackPtr> &a_tlist,
                                        const std::vector<STrackPtr> &b_tlist) const;

    std::vector<STrackPtr> subStracks(const std::vector<STrackPtr> &a_tlist,
                                      const std::vector<STrackPtr> &b_tlist) const;

    void removeDuplicateStracks(const std::vector<STrackPtr> &a_stracks,
                                const std::vector<STrackPtr> &b_stracks,
                                std::vector<STrackPtr> &a_res,
                                std::vector<STrackPtr> &b_res) const;

    void linearAssignment(const std::vector<std::vector<float>> &cost_matrix,
                          const int &cost_matrix_size,
                          const int &cost_matrix_size_size,
                          const float &thresh,
                          std::vector<std::vector<int>> &matches,
                          std::vector<int> &b_unmatched,
                          std::vector<int> &a_unmatched) const;

    std::vector<std::vector<float>> calcIouDistance(const std::vector<STrackPtr> &a_tracks,
                                                    const std::vector<STrackPtr> &b_tracks) const;

    std::vector<std::vector<float>> calcIous(const std::vector<Rect<float>> &a_rect,
                                             const std::vector<Rect<float>> &b_rect) const;

    /// When `params_.class_aware_match` is true, gate `dists[i][j]` to
    /// FLT_MAX whenever a_tracks[i].label != b_tracks[j].label so the
    /// Hungarian solver cannot match across class.
    void gateCrossClass(std::vector<std::vector<float>>& dists,
                        const std::vector<STrackPtr>& a_tracks,
                        const std::vector<STrackPtr>& b_tracks) const;

    double execLapjv(const std::vector<std::vector<float> > &cost,
                     std::vector<int> &rowsol,
                     std::vector<int> &colsol,
                     bool extend_cost = false,
                     float cost_limit = std::numeric_limits<float>::max(),
                     bool return_cost = true) const;

    /// `params_.track_thresh_per_class[label]` if present, else
    /// `params_.track_thresh`.
    float trackThreshFor(int label) const;

    /// `params_.high_thresh_per_class[label]` if present, else
    /// `params_.high_thresh`.
    float highThreshFor(int label) const;

private:
    const Params  params_;
    const size_t  max_time_lost_;

    size_t frame_id_;
    size_t track_id_count_;

    std::vector<STrackPtr> tracked_stracks_;
    std::vector<STrackPtr> lost_stracks_;
    std::vector<STrackPtr> removed_stracks_;
};
}
