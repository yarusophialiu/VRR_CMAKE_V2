#pragma once
#include <vector>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <functional>

class ScalarViterbi {
public:
    using StartProbFn = std::function<float(int state, int bitrate)>;
    ScalarViterbi(const std::vector<int>& states, float alpha = 0.9f, int maxJump = -1);

    void setObservations(const std::vector<int>& obs);
    void run(int initialVal);
    int getBestFinalState(int prevDisplayState) const;
    const std::vector<int>& getBestPath() const;
    float getBestPathProb() const;
    void reset();
    void setBitrate(int bitrate);
    void setStartProbFunction(StartProbFn fn);

private:
    std::vector<int> mStates;
    std::vector<int> mObservations;
    float mAlpha;
    int mMaxJump;
    int mBitrate = 0;
    StartProbFn mStartProbFn;

    std::vector<std::unordered_map<int, float>> V;
    std::unordered_map<int, std::vector<int>> path;
    std::vector<int> bestPath;
    float bestProb;

    float startProb(int state) const;
    float transitionProb(int prev, int curr, int initialVal) const;
    float emissionProb(int state, int observation) const;
};
