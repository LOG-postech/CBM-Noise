#!/bin/bash
# Configuration
FOLDER_DIR=#your path       
DATASET=awa2                # cub, awa2
NOISE_LOC=concept           # concept, class, both
NOISE=0.0
OPTIMIZER=SGD               # SGD, SAM
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
    NOISE_DATA_DIR=${FOLDER_DIR}/CBM-Noise/noiseDATA/CUB
    ATTRIBUTES=112
    CLASSES=200
elif [ ${DATASET} == "awa2" ]; then
    NOISE_DATA_DIR=${FOLDER_DIR}/CBM-Noise/noiseDATA/AWA2
    ATTRIBUTES=85
    CLASSES=50
fi

# Training loop for different seeds
for SEED in 1 2 3; do
    LOG_DIR=${DIR}/seed${SEED}/XtoC/
    if [ -d "$LOG_DIR" ]; then
        echo "Oops! The results exist at ${LOG_DIR} (so skip this job)"
    else
        python experiments.py \
            ${DATASET} \
            Concept_XtoC \
            --seed ${SEED} \
            -log_dir ${LOG_DIR} \
            -e ${XTOC_EPOCH} \
            -save_step ${SAVE_STEP} \
            -optimizer ${OPTIMIZER} \
            -pretrained \
            -use_attr \
            -use_aux \
            -weighted_loss multiple \
            -data_dir ${NOISE_DATA_DIR}/${DATA_DIR} \
            -val_dir ${NOISE_DATA_DIR}/mjvote \
            -predicate_dir ${FOLDER_DIR}/DATA/awa2 \
            -n_attributes ${ATTRIBUTES} \
            -n_classes ${CLASSES} \
            -normalize_loss \
            -b 64 \
            -weight_decay 0.00004 \
            -lr 0.01 \
            -scheduler_step 1000 \
            -bottleneck \
            -rho ${RHO} \
            -noise ${NOISE} \
            -noise_loc ${NOISE_LOC} \
            -backbone ${BACKBONE} \
            -cbm_type independent
    fi

    LOG_DIR=${DIR}/seed${SEED}/CtoY/
    if [ -d "$LOG_DIR" ]; then
        echo "Oops! The results exist at ${LOG_DIR} (so skip this job)"
    else
        python3 experiments.py \
            ${DATASET} \
            Independent_CtoY \
            --seed ${SEED} \
            -log_dir ${LOG_DIR} \
            -e ${CTOY_EPOCH} \
            -save_step ${SAVE_STEP} \
            -optimizer ${OPTIMIZER} \
            -use_attr \
            -data_dir ${NOISE_DATA_DIR}/${DATA_DIR} \
            -val_dir ${NOISE_DATA_DIR}/mjvote \
            -predicate_dir ${FOLDER_DIR}/DATA/awa2 \
            -n_attributes ${ATTRIBUTES} \
            -n_classes ${CLASSES} \
            -no_img \
            -b 64 \
            -weight_decay 0.00005 \
            -lr 0.001 \
            -rho ${RHO} \
            -scheduler_step 1000 \
            -noise ${NOISE} \
            -noise_loc ${NOISE_LOC} \
            -backbone ${BACKBONE} \
            -cbm_type independent
    fi

    LOG_DIR=${DIR}/seed${SEED}/inference/
    if [ -d "$LOG_DIR" ]; then
        echo "Oops! The results exist at ${LOG_DIR} (so skip this job)"
    else
        python3 CUB/inference.py \
            -dataset ${DATASET} \
            -log_dir ${LOG_DIR} \
            -model_dirs ${DIR}/seed${SEED}/XtoC/best_model_${SEED}.pth \
            -model_dirs2 ${DIR}/seed${SEED}/CtoY/best_model_${SEED}.pth \
            -eval_data test \
            -use_attr \
            -n_attributes ${ATTRIBUTES} \
            -n_classes ${CLASSES} \
            -data_dir ${NOISE_DATA_DIR}/mjvote \
            -predicate_dir ${FOLDER_DIR}/DATA/awa2 \
            -bottleneck \
            -use_sigmoid \
            -rho ${RHO} \
            -noise ${NOISE} \
            -optimizer ${OPTIMIZER} \
            -noise_loc ${NOISE_LOC} \
            -backbone ${BACKBONE} \
            -cbm_type independent
    fi
done

# Deactivate environment
conda deactivate