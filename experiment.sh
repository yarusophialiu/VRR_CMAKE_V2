#!/bin/bash
# run through git bash

speeds=(2)

# scenes=("crytek_sponza" "suntemple_statue" "bistro" "room" "lost_empire" "living_room" "suntemple" "sibenik" "bedroom" "gallery")

# sibenik 05, 06, 12
# vokseliaspawn 01, 02, 03, 04
# sponza  03 05 07

for speed in "${speeds[@]}"; do
    "/c/Users/15142/new/Falcor/build/windows-vs2022/bin/Debug/EncodeDecode.exe" "$speed"
done

# run this line in git bash
# C:/Users/15142/new/Falcor/Source/Samples/EncodeDecode

# C:/Users/15142/new/Falcor/build/windows-vs2022/bin/Debug
