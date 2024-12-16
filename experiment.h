

#include <string>


class ExperimentStimulus {

    public:

    std::string sceneName;
    int sceneIndex;
    int pathIndex;
    int bitrate;
    int resolution; // Set to -1 for variable resolution
    int framerate; //Set to -1 for variable frame-rate
};


class ExperimentCondition {

    public:

    ExperimentStimulus stimulus1;
    ExperimentStimulus stimulus2;
};
