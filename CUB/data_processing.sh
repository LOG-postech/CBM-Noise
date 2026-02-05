#!/bin/bash

# dataset configuration
DIR= # Your path here

# (Step 1) standard dataset
# mkdir -p ${DIR}/CBM-Noise/noiseDATA/CUB/basic/
# python3 data_processing.py \
#     -save_dir ${DIR}/CBM-Noise/noiseDATA/CUB/basic/ \
#     -data_dir ${DIR}/DATA/CUB_200_2011

# (Step 2) majority voting
# python3 generate_new_data.py \
#     MajorityVoting \
#     --out_dir ${DIR}/CBM-Noise/noiseDATA/CUB/mjvote \
#     --data_dir ${DIR}/CBM-Noise/noiseDATA/CUB/basic

# (Step 3) inject noise
# noise_rate=0.4  # 0.1, 0.2, 0.3, 0.4
# noise_loc=both  # concept, class, both
# python3 add_noise.py \
#     --out_dir ${DIR}/CBM-Noise/noiseDATA/CUB/mjvote_${noise_loc}_${noise_rate} --data_dir ${DIR}/CBM-Noise/noiseDATA/CUB/mjvote \
#     --noise_rate ${noise_rate} \
#     --exp ${noise_loc}

# noise_rates=(0.1 0.2 0.3 0.4)
# noise_locs=("class" "concept" "both")
# for noise_rate in ${noise_rates[@]}; do
#     for noise_loc in ${noise_locs[@]}; do
#         python3 add_noise.py \
#             --out_dir ${DIR}/CBM-Noise/noiseDATA/CUB/mjvote_${noise_loc}_${noise_rate} --data_dir ${DIR}/CBM-Noise/noiseDATA/CUB/mjvote \
#             --noise_rate ${noise_rate} \
#             --exp ${noise_loc}
#     done
# done