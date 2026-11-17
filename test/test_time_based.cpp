#include "ByteTrack/BYTETracker.h"
#include "ByteTrack/KalmanFilter.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace
{
using byte_track::BYTETracker;
using byte_track::Object;
using byte_track::Rect;

// 5 FPS nominal: a 200 ms frame period.
BYTETracker::Params params_5fps()
{
    BYTETracker::Params p;
    p.frame_rate = 5;
    p.track_buffer = 60;
    p.match_on_prediction = true;
    return p;
}

Object person_at(float x)
{
    return Object(Rect<float>(x, 100.f, 50.f, 100.f), 0, 0.9f);
}

// The id of the single track the tracker outputs, or 0 when it outputs none.
size_t only_id(const std::vector<BYTETracker::STrackPtr>& out)
{
    return out.size() == 1 ? out[0]->getTrackId() : 0;
}

TEST(KalmanTimeStep, OneStepOfTwoMovesAsFarAsTwoStepsOfOne)
{
    byte_track::KalmanFilter kf;
    byte_track::KalmanFilter::StateMean a, b;
    byte_track::KalmanFilter::StateCov ca, cb;
    kf.initiate(a, ca, Rect<float>(0.f, 0.f, 50.f, 100.f).getXyah());
    a(4) = 3.f;  // x velocity, per nominal frame
    a(5) = -2.f;
    b = a;
    cb = ca;

    kf.predict(a, ca, 2.f);
    kf.predict(b, cb, 1.f);
    kf.predict(b, cb, 1.f);

    for (int i = 0; i < 8; ++i) EXPECT_NEAR(a(i), b(i), 1e-4) << "state " << i;
}

TEST(KalmanTimeStep, ZeroStepLeavesTheStateAlone)
{
    byte_track::KalmanFilter kf;
    byte_track::KalmanFilter::StateMean m;
    byte_track::KalmanFilter::StateCov c;
    kf.initiate(m, c, Rect<float>(0.f, 0.f, 50.f, 100.f).getXyah());
    m(4) = 3.f;
    const auto m0 = m;
    const auto c0 = c;

    kf.predict(m, c, 0.f);

    EXPECT_EQ(m, m0);
    EXPECT_EQ(c, c0);
}

// A person walks 100 px/s, 2 box widths per second. Frames come every 200 ms, then a 1 s motion gap: the box moves
// 100 px, so a one-frame prediction (+20 px) no longer overlaps it while a 1 s prediction does.
TEST(TimeBasedTracking, KeepsTheTrackAcrossAMotionGap)
{
    BYTETracker tracker(params_5fps());
    int64_t t = 0;
    size_t id = 0;
    for (int i = 0; i < 20; ++i, t += 200)
    {
        id = only_id(tracker.update({person_at(0.1f * t)}, t));
    }
    ASSERT_NE(id, 0u);
    t += 800;  // 1000 ms since the last frame
    EXPECT_EQ(only_id(tracker.update({person_at(0.1f * t)}, t)), id);
}

TEST(TimeBasedTracking, FrameSteppingLosesTheSameWalk)
{
    BYTETracker tracker(params_5fps());
    int64_t t = 0;
    size_t id = 0;
    for (int i = 0; i < 20; ++i, t += 200)
    {
        id = only_id(tracker.update({person_at(0.1f * t)}));
    }
    ASSERT_NE(id, 0u);
    t += 800;
    EXPECT_NE(only_id(tracker.update({person_at(0.1f * t)})), id);
}

TEST(TimeBasedTracking, MatchingOnTheLastBoxLosesTheSameWalk)
{
    auto p = params_5fps();
    p.match_on_prediction = false;
    BYTETracker tracker(p);
    int64_t t = 0;
    size_t id = 0;
    for (int i = 0; i < 20; ++i, t += 200)
    {
        id = only_id(tracker.update({person_at(0.1f * t)}, t));
    }
    ASSERT_NE(id, 0u);
    t += 800;
    EXPECT_NE(only_id(tracker.update({person_at(0.1f * t)}, t)), id);
}

TEST(TimeBasedTracking, ReportsVelocityInPixelsPerSecondWhateverTheFramePeriod)
{
    for (const int64_t period : {200, 500, 1000})
    {
        BYTETracker tracker(params_5fps());
        BYTETracker::STrackPtr last;
        for (int64_t t = 0; t <= 20 * period; t += period)
        {
            const auto out = tracker.update({person_at(0.01f * t)}, t);  // 10 px/s
            ASSERT_EQ(out.size(), 1u) << "period " << period << " t " << t;
            last = out[0];
        }
        EXPECT_NEAR(last->getVelocity().x, 10.f, 1.5f) << "period " << period;
        EXPECT_NEAR(last->getVelocity().y, 0.f, 1.f) << "period " << period;
    }
}

TEST(TimeBasedTracking, AStillPersonHasNoVelocity)
{
    BYTETracker tracker(params_5fps());
    BYTETracker::STrackPtr last;
    for (int64_t t = 0; t <= 4000; t += 200)
    {
        last = tracker.update({person_at(10.f)}, t).at(0);
    }
    EXPECT_NEAR(last->getVelocity().x, 0.f, 1e-3);
    EXPECT_NEAR(last->getVelocity().y, 0.f, 1e-3);
}

TEST(TimeBasedTracking, RepeatedOrBackwardTimestampsDoNotMoveTracks)
{
    BYTETracker tracker(params_5fps());
    size_t id = 0;
    for (int64_t t = 0; t <= 1000; t += 200) id = only_id(tracker.update({person_at(10.f)}, t));
    EXPECT_EQ(only_id(tracker.update({person_at(10.f)}, 1000)), id);
    EXPECT_EQ(only_id(tracker.update({person_at(10.f)}, 900)), id);
    EXPECT_EQ(only_id(tracker.update({person_at(10.f)}, 1200)), id);
}

}  // namespace
