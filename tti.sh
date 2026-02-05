#!/bin/bash
# Configuration
FOLDER_DIR=#your path       
DATASET=cub                 # cub, awa2
NOISE_LOC=concept           # concept, class, both
NOISE=0.0
OPTIMIZER=SGD               # SGD, SAM
CRITERION=ucp               # rand, ucp, lcp, cctp, ectp, eudtp / rand_wrong
RHO=0.1
XTOC_EPOCH=1000
CTOY_EPOCH=500
SAVE_STEP=50
BACKBONE=inception_v3       # resnet18_cub, inception_v3(-use_aux: xtoc only), resnet101, vit_b16, vit_l16

conda activate cbm

# Use mjvote folder when noise is 0
if [ ${NOISE} == 0.0 ]; then
    DATA_DIR=mjvote
else
    DATA_DIR=mjvote_${NOISE_LOC}_${NOISE}
fi

# Append RHO to DIR when optimizer is SAM
if [ ${OPTIMIZER} == "SGD" ]; then
    DIR=RESULT/${DATASET}/independent/mjvote/${NOISE_LOC}_${NOISE}/${BACKBONE}/${OPTIMIZER}
else
    DIR=RESULT/${DATASET}/independent/mjvote/${NOISE_LOC}_${NOISE}/${BACKBONE}/${OPTIMIZER}/rho_${RHO}
fi

# Dataset-specific settings
if [ ${DATASET} == "cub" ]; then
    NOISE_DATA_DIR=${FOLDER_DIR}/CBM/robustCBM/noiseDATA/CUB
    ATTRIBUTES=112
    CLASSES=200
elif [ ${DATASET} == "awa2" ]; then
    NOISE_DATA_DIR=${FOLDER_DIR}/CBM/robustCBM/noiseDATA/AWA2
    ATTRIBUTES=85
    CLASSES=50
fi

# Training loop for different seeds
for SEED in 1 2 3; do
    LOG_DIR=${DIR}/seed${SEED}/tti/${CRITERION}
    if [ -d "$LOG_DIR" ]; then
        echo "Oops! The results exist at ${LOG_DIR} (so skip this job)"
    else
        python3 CUB/tti.py \
            -dataset ${DATASET} \
            -log_dir ${LOG_DIR} \
            -optimizer ${OPTIMIZER} \
            -model_dirs ${DIR}/seed${SEED}/XtoC/best_model_${SEED}.pth \
            -model_dirs2 ${DIR}/seed${SEED}/CtoY/best_model_${SEED}.pth \
            -use_attr \
            -bottleneck \
            -criterion ${CRITERION} \
            -n_trials 1 \
            -use_invisible \
            -level 'i+s' \
            -class_level \
            -use_sigmoid \
            -no_intervention_when_invisible \
            -predicate_dir ${FOLDER_DIR}/DATA/awa2 \
            -n_attributes ${ATTRIBUTES} \
            -n_classes ${CLASSES} \
            -data_dir ${NOISE_DATA_DIR}/mjvote \
            -noise ${NOISE} \
            -noise_loc ${NOISE_LOC} \
            -backbone ${BACKBONE}
    fi
done

# Deactivate environment
conda deactivate