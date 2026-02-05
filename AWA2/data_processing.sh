#!/bin/bash

# dataset configuration
DIR= # Your path here

# (Step 1) standard dataset
# mkdir ${DIR}/CBM-Noise/noiseDATA/AWA2
# mkdir ${DIR}/CBM-Noise/noiseDATA/AWA2/mjvote
# python3 gen_awa2_split.py \
#     -save_dir ${DIR}/CBM-Noise/noiseDATA/AWA2/mjvote \
#     -data_dir ${DIR}/DATA/awa2 

# (Step 2) inject noise
# noise_rate=0.1      # 0.1, 0.2, 0.3, 0.4
# noise_loc="both"    # concept, class, both
# python3 add_noise.py \
#     --out_dir ${DIR}/CBM-Noise/noiseDATA/AWA2/mjvote_${noise_loc}_${noise_rate} \
#     --data_dir ${DIR}/CBM-Noise/noiseDATA/AWA2/mjvote \
#     --image_dir ${DIR}/DATA/awa2 \
#     --noise_rate ${noise_rate} \
#     --exp ${noise_loc}

# noise_rates=(0.1 0.2 0.3 0.4)
# noise_locs=("concept" "class" "both")
# for noise_rate in ${noise_rates[@]}; do
#     for noise_loc in ${noise_locs[@]}; do
#         python3 add_noise.py \
#             --out_dir ${DIR}/CBM-Noise/noiseDATA/AWA2/mjvote_${noise_loc}_${noise_rate} \
#             --data_dir ${DIR}/CBM-Noise/noiseDATA/AWA2/mjvote \
#             --image_dir ${DIR}/DATA/awa2 \
#             --noise_rate ${noise_rate} \
#             --exp ${noise_loc}
#     done
# done