#include "ScalarViterbi.h"

ScalarViterbi::ScalarViterbi(const std::vector<int>& states, float alpha, int maxJump)
    : mStates(states), mAlpha(alpha), mMaxJump(maxJump), bestProb(0.0f)
{
    // Default to uniform
    mStartProbFn = [this](int, int) {
        return 1.0f / static_cast<float>(mStates.size());
    };
}

void ScalarViterbi::setObservations(const std::vector<int>& obs) {
    mObservations = obs;
}

void ScalarViterbi::reset() {
    mObservations.clear();
    V.clear();
    path.clear();
    bestPath.clear();
    bestProb = 0.0f;
}

void ScalarViterbi::setBitrate(int bitrate) {
    mBitrate = bitrate;
}


void ScalarViterbi::setStartProbFunction(StartProbFn fn) {
    mStartProbFn = fn;
}


float ScalarViterbi::startProb(int state) const {
    return 1.0f / static_cast<float>(mStates.size());  // uniform
}

float ScalarViterbi::transitionProb(int prev, int curr, int initialVal) const {
    if (mMaxJump > 0 && std::abs(initialVal - curr) >= mMaxJump)
        return 0.f;

    float stayProb = 0.75f;
    float switchProb = (1.0f - stayProb) / (mStates.size() - 1);
    return (prev == curr) ? stayProb : switchProb;
}

float ScalarViterbi::emissionProb(int state, int observation) const {
    float beta = (1.0f - mAlpha) / static_cast<float>(mStates.size() - 1);
    return (state == observation) ? mAlpha : beta;
}

void ScalarViterbi::run(int initialVal) {
    int T = static_cast<int>(mObservations.size());
    if (T == 0) return;

    V.clear();
    V.resize(T);
    path.clear();

    // Initialization
    for (int s : mStates) {
        // float prob = startProb(s) * emissionProb(s, mObservations[0]);
        float prob = mStartProbFn(s, mBitrate) * emissionProb(s, mObservations[0]);

        V[0][s] = prob;
        path[s] = { s };
    }

    // Recursion
    for (int t = 1; t < T; ++t) {
        std::unordered_map<int, std::vector<int>> newPath;

        for (int curr : mStates) {
            float best_prob = -1.0f;
            int bestPrev = -1;

            for (int prev : mStates) {
                float prob = V[t - 1][prev] *
                             transitionProb(prev, curr, initialVal) *
                             emissionProb(curr, mObservations[t]);

                if (prob > best_prob) {
                    best_prob = prob;
                    bestPrev = prev;
                }
            }

            V[t][curr] = best_prob;
            newPath[curr] = path[bestPrev];
            newPath[curr].push_back(curr);
        }

        path = std::move(newPath);
    }

    // Termination
    bestProb = -1.0f;
    int bestFinal = -1;

    for (int s : mStates) {
        if (V[T - 1][s] > bestProb) {
            bestProb = V[T - 1][s];
            bestFinal = s;
        }
    }

    bestProb = std::round(bestProb * 10000.0f) / 10000.0f;
    bestPath = path[bestFinal];

}

int ScalarViterbi::getBestFinalState(int prevDisplayState) const {
    int last_state = bestPath.empty() ? -1 : bestPath.back();
    // if (last_state == -1 || mMaxJump < 0) {
    //     return last_state;
    // }
    // int diff = last_state - prevDisplayState;
    // if (std::abs(diff) <= mMaxJump) {
    //     return last_state;
    // } else {
    //     return prevDisplayState + (diff > 0 ? mMaxJump : -mMaxJump);
    // }
    return last_state;
}

const std::vector<int>& ScalarViterbi::getBestPath() const {
    return bestPath;
}

float ScalarViterbi::getBestPathProb() const {
    return bestProb;
}
