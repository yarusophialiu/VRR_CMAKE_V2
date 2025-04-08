

#include <string>
#include <iostream>
#include <vector>
#include <map>

class ExperimentStimulus {

    public:

    std::string sceneName;
    int sceneIndex;
    std::string pathName;
    int bitrate;
    int resolution; // Set to -1 for variable resolution
    int framerate; //Set to -1 for variable frame-rate
    float speed;
    float DROPJODSCALE;

    ExperimentStimulus()
        : sceneName(""), sceneIndex(0), pathName(""), bitrate(0), resolution(-1), framerate(-1), speed(0.0f), DROPJODSCALE(0.0f) {}


    ExperimentStimulus(const std::string& scene, const std::string& p, int br, int res, int fps, float spd, float dropScale, int sceneIdx)
    : sceneName(scene), sceneIndex(sceneIdx), pathName(p), bitrate(br),
        resolution(res), framerate(fps), speed(spd), DROPJODSCALE(dropScale) {}
};


class ExperimentCondition {

    public:

    ExperimentStimulus stimulus1;
    ExperimentStimulus stimulus2;
    // default constructor
    ExperimentCondition() = default;

    // Explicit Constructor
    ExperimentCondition(const ExperimentStimulus& s1, const ExperimentStimulus& s2)
        : stimulus1(s1), stimulus2(s2) {}
};



class ExperimentManager {
public:
    std::map<std::string, std::vector<ExperimentStimulus>> sceneStimuli;
    // std::vector<ExperimentCondition> mConditions;

    // Add a stimulus to a specific scene
    void addStimulus(const ExperimentStimulus& stimulus) {
        sceneStimuli[stimulus.sceneName].push_back(stimulus);
    }


    // Generate conditions by pairing stimuli and pushing ExperimentCondition objects into mConditions
    void generateConditions(std::vector<ExperimentCondition>& mConditions) {
        mConditions.clear();  // Ensure it's empty before adding new conditions

        for (const auto& [scene, stimuli] : sceneStimuli) {
            int n = stimuli.size();
            std::cout << "Generating conditions for scene: " << scene << " with " << n << " stimuli.\n";

            if (n < 2) {
                std::cerr << "Warning: Not enough stimuli in scene " << scene << " to create conditions.\n";
                continue;  // Skip this scene if there aren't at least 2 stimuli
            }

            for (int i = 0; i < n - 1; i += 2) {
                ExperimentCondition condition(stimuli[i], stimuli[i + 1]);
                mConditions.push_back(condition);
                std::cout << "Added Condition: (" << stimuli[i].pathName << ", " << stimuli[i + 1].pathName << ")\n";
            }

            // If there's an **odd** number of stimuli, pair the last one with the previous one
            if (n % 2 == 1) {
                ExperimentCondition condition(stimuli[n - 2], stimuli[n - 1]);
                mConditions.push_back(condition);
                std::cout << "Added Condition (odd case): (" << stimuli[n - 2].pathName << ", " << stimuli[n - 1].pathName << ")\n";
            }
        }
        std::cout << "Total conditions generated: " << mConditions.size() << "\n";
    }


    // // Print all conditions
    // void printConditions() const {
    //     for (const auto& condition : mConditions) {
    //         std::cout << "Condition: \n"
    //                   << "  Stimulus 1: Scene " << condition.stimulus1.sceneName
    //                   << ", Bitrate: " << condition.stimulus1.bitrate
    //                   << ", Speed: " << condition.stimulus1.speed << "\n"
    //                   << "  Stimulus 2: Scene " << condition.stimulus2.sceneName
    //                   << ", Bitrate: " << condition.stimulus2.bitrate
    //                   << ", Speed: " << condition.stimulus2.speed << "\n";
    //     }
    // }
};