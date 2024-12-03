#!/bin/bash
# run through git bash

# BITRATES=(8000)
# BITRATES=(8000)
# BITRATES=(3500 4000 4500 5000 5500 6000 6500 7000)
BITRATES=(2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000)
# BITRATES=(2000 2500 3000 3500 4000 4500)
BITRATES=(8000)

FRAMERATE=(30 40 50 60 70 80 90 100 110 120)
# FRAMERATE=(30)
FRAMERATE=(30)

# resolutions=("1920x1080" "1536x864" "1280x720" "960x540" "854x480" "640x360") # "1200x676"
resolutions=("1920x1080" "1536x864" "1280x720" "854x480" "640x360")
resolutions=("640x360")


# paths=(1 2 3 4 5)
# segs=(1 2 3)
# speeds=(1 2 3)


paths=(1)
segs=(1)
speeds=(1)
# 23, 31 have problems


scenes=("crytek_sponza" "suntemple_statue" "bistro" "room" "lost_empire" "living_room" "suntemple" "sibenik" "bedroom" "gallery")
# scenes=("bistro" "crytek_sponza" "gallery"  "living_room" "sibenik") # "breakfast_room"
scenes=("suntemple_statue")


# run this line in git bash
# set DELETEBMP in playBMP.py to True
for sceneval in "${scenes[@]}"; do
 for path in "${paths[@]}"; do
    for seg in "${segs[@]}"; do
      # Construct the key and value
    #    scene="${sceneval}_path${path}_seg${seg}"
    #    scenepath="${sceneval}/path${path}_seg${seg}.fbx"
       scene="suntemple_statue03"
       scenepath="${sceneval}/suntemple_statue01.fbx"
    #    echo "scene: $scene" "seg: $seg" "scenedict: ${scenedict[$scene]}"
        for speed in "${speeds[@]}"; do
            echo "speed: $speed"
            for resolution in "${resolutions[@]}"; do
                IFS='x' read -r width height <<< "$resolution"
                for framerate in "${FRAMERATE[@]}"; do
                    for bitrate in "${BITRATES[@]}"; do
                        echo "=========================== Setting bitrate to: $bitrate ==========================="
                        echo "Bitrate: $bitrate, Resolution: $height, FRAMERATE: $framerate"

                        # to generate frames for training, only use encodedeco.cpp, generate_truevals.py and data_prepare.py in VRR_cvvdp
                        "/c/Users/15142/new/Falcor/build/windows-vs2022/bin/Debug/EncodeDecode.exe" "$bitrate" "$framerate" "$width" "$height" "$scene" "$speed" "$scenepath"

                        # # generate decode and reference videos; change True in playBMP.py to generate reference videos
                        # # python "/c/Users/15142/OneDrive - University of Cambridge/Desktop/VRR/VRR_CVVDP/playBMP.py" "$bitrate" "$height" "$framerate" "$scene" "$speed"
                        # python "/c/Users/15142/Projects/VRR/VRR_CVVDP/playBMP.py" "$bitrate" "$height" "$framerate" "$scene" "$speed"

                        # # find max and min motion vector and save in a file
                        # # empty recycle bin
                        # python "/c/Users/15142/Projects/VRR/VRR_Motion/encode_motion.py" "$bitrate" "$framerate" "$width" "$height" "$scene" "$speed" "$sceneval"


                        # # compute velocity of a video sequence and save in a file
                        # # empty recycle bin
                        # python "/c/Users/15142/Projects/VRR/VRR_Motion/compute_velocity.py" "$bitrate" "$framerate" "$width" "$height" "$sceneval" "$path" "$seg" "$speed"

                        # # rename folder
                        # python "/c/Users/15142/OneDrive - University of Cambridge/Desktop/VRR/VRRML_Create_Data/frames.py" "$bitrate" "$height" "$framerate" "$scene" "$speed"
                        # wait
                    done
                done
            done
        done
    done
  done
done



# run this line in git bash
# C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode

# C:/Users/15142/new/Falcor/build/windows-vs2022/bin/Debug




# declare -A scenedict

# # Add key-value pairs to the dictionary
# scenedict=(
#     # scene_name: path_name
#     # ["bistropath_one"]="Bistro/Bistro/BistroInterior.fbx"
#     # # ["bistro"]="Bistro/Bistro/bistro_path2.fbx"
#     # ["bistropath_three"]="Bistro/Bistro/bistro_path3.fbx"
#     # ["suntemple"]="SunTemple/SunTemple.pyscene"
#     # ["room"]="test_scenes/grey_and_white_room/grey_and_white_room.fbx"
#     # ["suntemple_statue"]="SunTemple/statue.fbx"
#     ["crytek_sponza_path5_seg3"]="crytek_sponza/path5_seg3.fbx"
#     # ["suntemple_path1_seg1"]="SunTemple/path1_seg1.fbx"
#     # ["suntemple_statue_path5_seg3"]="SunTemple/path5_seg3.fbx" # pathtwo_segtwo pathone3 pathtwo_segone
#     # ["suntemple_statue_test"]="SunTemple/test.fbx" # pathtwo_segtwo pathone3 pathtwo_segone
#     # ["suntemple_statue_test2"]="SunTemple/test2.fbx" # pathtwo_segtwo pathone3 pathtwo_segone
#     # ["suntemple_statue_freeze"]="SunTemple/stat/ue_freeze4.fbx"
#     # ["bistro_glasses"]="Bistro/Bistro/glasses.fbx"
#     # ["paint"]="test_scenes/grey_and_white_room/paint.fbx"
#     # ["sponza-fast21"]="sponza/sponza21.fbx"
#     # ["lost-empire"]="lost-empire/le5.fbx"
#     # ["breakfast_room_two"]="breakfast_room/breakfastroom2.fbx"
#     # ["lost_empire_three"]="lost-empire/lost_empire3.fbx"
#     # ["sponza_three"]="crytek_sponza/sponza_light3.fbx"
# )


# # if generate motion, only need EncodeDecode.exe
# # change refBaseFilePath = "D:/VRR-frame/room-fast/refOutputBMP/" in encodedecode.h
# # rt.slan getreflectioncolor
# for scene in "${!scenedict[@]}"; do
#     echo "scene: $scene"
#     echo "scenedict: ${scenedict[$scene]}"
#     for speed in "${speeds[@]}"; do
#         echo "speed: $speed"

#         for resolution in "${resolutions[@]}"; do
#             IFS='x' read -r width height <<< "$resolution"

#             echo "Processing resolution: $resolution"
#             echo "Width: $width, Height: $height"
#             for framerate in "${FRAMERATE[@]}"; do
#                 for bitrate in "${BITRATES[@]}"; do
#                     # Print current bitrate
#                     echo "============================ Setting bitrate to: $bitrate ============================"
#                     echo "Bitrate: $bitrate, Resolution: $height, FRAMERATE: $framerate"

#                     # # to generate frames for training, only use encodedeco.cpp, generate_truevals.py and data_prepare.py in VRR_cvvdp
#                     # /c/Users/15142/new/Falcor/build/windows-vs2022/bin/Debug/EncodeDecode.exe "$bitrate" "$framerate" "$width" "$height" "$scene" "$speed" "${scenedict[$scene]}"

#                     # # generate decode and reference videos; change True in playBMP.py to generate reference videos
#                     # python "/c/Users/15142/OneDrive - University of Cambridge/Desktop/VRR/VRR_CVVDP/playBMP.py" "$bitrate" "$height" "$framerate" "$scene" "$speed"

#                     # # rename folder
#                     # python "/c/Users/15142/OneDrive - University of Cambridge/Desktop/VRR/VRRML_Create_Data/frames.py" "$bitrate" "$height" "$framerate" "$scene" "$speed"
#                     # wait
#                 done
#             done
#         done
#     done
# done


# gcc EncodeDecode.cpp -o outputfile.exe -I "../../../../Source/Falcor/"
# C:/Users/15142/source/repos/Falcor/Falcor/build/Source/PerceptualRendering/EncodeDecode

# run this line in git bash
# C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode

# C:/Users/15142/new/Falcor/build/windows-vs2022/bin/Debug
