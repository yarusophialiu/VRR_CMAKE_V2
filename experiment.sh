# #!/bin/bash
# # run through git bash

# paths=(1 2 3 4 5)
# segs=(1 2 3)

paths=(3)
segs=(2)


# scenes=("crytek_sponza" "suntemple_statue" "room" "lost_empire" "suntemple" "bedroom", "bistro")
# scenes=("crytek_sponza" "suntemple_statue" "bistro" "room" "lost_empire" "living_room" "suntemple" "sibenik" "bedroom" "gallery")
# scenes=("lost_empire" "room" "sibenik" "suntemple" "suntemplestatue")

# scenes=("gallery" "bistro" "crytek_sponza" "bedroom" "living_room" )
scenes=("bedroom")


for sceneval in "${scenes[@]}"; do
  for path in "${paths[@]}"; do
    for seg in "${segs[@]}"; do
      # Construct the key and value
       scene="${sceneval}_path${path}_seg${seg}"
       scenepath="${sceneval}/path${path}_seg${seg}.fbx"

        # to generate frames for training, only use encodedeco.cpp, generate_truevals.py and data_prepare.py in VRR_cvvdp
        "/c/Users/15142/new/Falcor/build/windows-vs2022/bin/Debug/EncodeDecode.exe" "$scene" "$scenepath"
    done
  done
done


# run this line in git bash
# C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode

# gcc EncodeDecode.cpp -o outputfile.exe -I "../../../../Source/Falcor/"
# C:/Users/15142/source/repos/Falcor/Falcor/build/Source/PerceptualRendering/EncodeDecode

# "/c/Users/15142/new/Falcor/build/windows-vs2022/bin/Debug/EncodeDecode.exe"
# /c/RFL/Falcor2/Falcor/Source/Samples/EncodeDecode

# /c/Users/15142/Projects/VRR/VRR_CVVDP/playBMP.py

# C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode/ComputeVelocity.cs.slang
# C:/RFL/Falcor2/Falcor/Source/Samples/EncodeDecode/ComputeVelocity.cs.slang