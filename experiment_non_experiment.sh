#!/bin/bash
# run through git bash

# BITRATES=(8000)
# BITRATES=(8000)
# BITRATES=(3500 4000 4500 5000 5500 6000 6500 7000)
BITRATES=(2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000)
# BITRATES=(500 1000 1500)
BITRATES=(1500)

FRAMERATE=(30 40 50 60 70 80 90 100 110 120)
# FRAMERATE=(30 60)
FRAMERATE=(166)

# resolutions=("1920x1080" "1536x864" "1280x720" "960x540" "854x480" "640x360") # "1200x676"
resolutions=("1920x1080" "1536x864" "1280x720" "854x480" "640x360")
resolutions=("1920x1080")

speeds=(1 2 3)
paths=(1 2 3 4 5)
segs=(1 2 3)


speeds=(1)
paths=(1)
segs=(2)


# start a new git bash !!!!! important as git bash might not run the lastet code in this file
# run this line in git
# /c/RFL/Falcor2/Falcor/Source/Samples/EncodeDecode


# scenes=("crytek_sponza" "suntemple_statue" "room" "lost_empire" "suntemple" "bedroom", "bistro")
scenes=("crytek_sponza" "suntemple_statue" "bistro" "room" "lost_empire" "living_room" "suntemple" "sibenik" "bedroom" "gallery")

# scenes=("room" "suntemple" "suntemple_statue" "lost_empire" "bedroom")
scenes=("bistro")


for sceneval in "${scenes[@]}"; do
  for path in "${paths[@]}"; do
    for seg in "${segs[@]}"; do
      # Construct the key and value
       scene="${sceneval}_path${path}_seg${seg}"
       scenepath="${sceneval}/path${path}_seg${seg}.fbx"

        # for scene in "${!scenedict[@]}"; do
        echo "scene: $scene"
        echo "seg: $seg"
        # echo "scenedict: ${scenedict[$scene]}"
        for speedInput in "${speeds[@]}"; do
            echo "speed: $speedInput"
            for resolution in "${resolutions[@]}"; do
                IFS='x' read -r width height <<< "$resolution"
                echo "Processing resolution: $resolution"
                echo "Width: $width, Height: $height"
                for framerate in "${FRAMERATE[@]}"; do
                    for bitrate in "${BITRATES[@]}"; do
                        # Print current bitrate
                        echo "============================ Setting bitrate to: $bitrate ============================"
                        echo "Bitrate: $bitrate, Resolution: $height, FRAMERATE: $framerate"

                        # to generate frames for training, only use encodedeco.cpp, generate_truevals.py and data_prepare.py in VRR_cvvdp
                        /c/RFL/Falcor2/Falcor/build/windows-vs2022/bin/Debug/EncodeDecode.exe "$bitrate" "$framerate" "$width" "$height" "$scene" "$speedInput" "$scenepath"


                        # # generate decode and reference videos; change True in playBMP.py to generate reference videos
                        # python /d/VRR/VRR_cvvdp/playBMP.py "$bitrate" "$height" "$framerate" "$scene" "$speedInput" "$sceneval"

                        # python /c/RFL/VRR/VRRML_Data/frames.py "$bitrate" "$height" "$framerate" "$scene" "$speed"
                        # wait
                    done
                done
            done
        done
        # done
    done
  done
done


# /c/RFL/Falcor2/Falcor/Source/Samples/EncodeDecode

# gcc EncodeDecode.cpp -o outputfile.exe -I "../../../../Source/Falcor/"
# C:/Users/15142/source/repos/Falcor/Falcor/build/Source/PerceptualRendering/EncodeDecode

# run this line in git bash
# C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode

# C:/Users/15142/new/Falcor/build/windows-vs2022/bin/Debug
