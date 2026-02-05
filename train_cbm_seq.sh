#!/bin/bash
# Configuration
FOLDER_DIR=#your path       
DATASET=cub                 # cub, awa2
NOISE_LOC=concept           # concept, class, both
NOISE=0.0
OPTIMIZER=SGD               # SGD, SAM
RHO=0.1
XTOC_EPOCH=1000
CTOY_EPOCH=1000
SAVE_STEP=50
BACKBONE=inception_v3       # resnet18_cub, inception_v3(-use_aux: ctoy only), resnet101, vit_b16, vit_l16

conda activate cbm

# Use mjvote folder when noise is 0
if [ ${NOISE} == 0.0 ]; then
    DATA_DIR=mjvote
else
    DATA_DIR=mjvote_${NOISE_LOC}_${NOISE}
fi

# Append RHO to DIR when optimizer is SAM
if [ ${OPTIMIZER} == "SGD" ]; then
    DIR=RESULT/${DATASET}/seq/mjvote/${NOISE_LOC}_${NOISE}/${BACKBONE}/${OPTIMIZER}
    XTOC_DIR=RESULT/${DATASET}/independent/mjvote/${NOISE_LOC}_${NOISE}/${BACKBONE}/${OPTIMIZER}
else
    DIR=RESULT/${DATASET}/seq/mjvote/${NOISE_LOC}_${NOISE}/${BACKBONE}/${OPTIMIZER}/rho_${RHO}
    XTOC_DIR=RESULT/${DATASET}/independent/mjvote/${NOISE_LOC}_${NOISE}/${BACKBONE}/${OPTIMIZER}/rho_${RHO}
fi

# Dataset-specific settings
if [ ${DATASET} == "cub" ]; then
    NOISE_DATA_DIR=${FOLDER_DIR}/CBM-Noise/noiseDATA/CUB
    CONCEPT_DATA_DIR=${FOLDER_DIR}/CBM-Noise/conceptDATA/CUB
    ATTRIBUTES=112
    CLASSES=200
elif [ ${DATASET} == "awa2" ]; then
    NOISE_DATA_DIR=${FOLDER_DIR}/CBM-Noise/noiseDATA/AWA2
    CONCEPT_DATA_DIR=${FOLDER_DIR}/CBM-Noise/conceptDATA/AWA2
    ATTRIBUTES=85
    CLASSES=50
fi

# Create CONCEPT_DATA_DIR if it does not exist
if [ ! -d "${CONCEPT_DATA_DIR}" ]; then
    echo "Directory ${CONCEPT_DATA_DIR} does not exist. Creating now..."
    mkdir -p "${CONCEPT_DATA_DIR}"
fi

# Training loop for different seeds
for SEED in 1 2 3; do
    LOGIT_DIR=${CONCEPT_DATA_DIR}/mjvote_${NOISE_LOC}_${NOISE}_${BACKBONE}_${OPTIMIZER}_seed${SEED}
    if [ -d "$LOGIT_DIR" ]; then
        echo "Oops! The results exist at ${LOGIT_DIR} (so skip this job)"
    else
        if [ ${DATASET} == "cub" ]; then
            python CUB/generate_new_data.py \
                ExtractConcepts \
                --model_path ${XTOC_DIR}/seed${SEED}/XtoC/best_model_${SEED}.pth \
                --data_dir ${NOISE_DATA_DIR}/${DATA_DIR} \
                --out_dir ${LOGIT_DIR} \
                --gpu_loc_data ${FOLDER_DIR}/DATA/ \
                --backbone ${BACKBONE}
        elif [ ${DATASET} == "awa2" ]; then
            python AWA2/generate_new_data.py \
                ExtractConcepts \
                --model_path ${XTOC_DIR}/seed${SEED}/XtoC/best_model_${SEED}.pth \
                --data_dir ${NOISE_DATA_DIR}/${DATA_DIR} \
                --out_dir ${LOGIT_DIR} \
                --gpu_loc_data ${FOLDER_DIR}/DATA/ \
                --backbone ${BACKBONE}
        fi
    fi
    
    LOG_DIR=${DIR}/seed${SEED}/seq_CtoY/
    if [ -d "$LOG_DIR" ]; then
        echo "Oops! The results exist at ${LOG_DIR} (so skip this job)"
    else
        python experiments.py \
            ${DATASET} \
            Sequential_CtoY \
            --seed ${SEED} \
            -log_dir ${LOG_DIR} \
            -e ${CTOY_EPOCH} \
            -save_step ${SAVE_STEP} \
            -optimizer ${OPTIMIZER} \
            -pretrained \
            -use_attr \
            -use_aux \
            -data_dir ${LOGIT_DIR} \
            -val_dir ${NOISE_DATA_DIR}/mjvote \
            -predicate_dir ${FOLDER_DIR}/DATA/awa2 \
            -n_attributes ${ATTRIBUTES} \
            -n_classes ${CLASSES} \
            -no_img \
            -b 64 \
            -weight_decay 0.00004 \
            -lr 0.001 \
            -scheduler_step 1000 \
            -rho ${RHO} \
            -noise ${NOISE} \
            -noise_loc ${NOISE_LOC} \
            -backbone ${BACKBONE} \
            -cbm_type sequential
    fi

    LOG_DIR=${DIR}/seed${SEED}/inference/
    if [ -d "$LOG_DIR" ]; then
        echo "Oops! The results exist at ${LOG_DIR} (so skip this job)"
    else
        python3 CUB/inference.py \
            -dataset ${DATASET} \
            -log_dir ${LOG_DIR} \
            -model_dirs ${XTOC_DIR}/seed${SEED}/XtoC/best_model_${SEED}.pth \
            -model_dirs2 ${DIR}/seed${SEED}/seq_CtoY/best_model_${SEED}.pth \
            -eval_data test \
            -use_attr \
            -n_attributes ${ATTRIBUTES} \
            -data_dir ${NOISE_DATA_DIR}/mjvote \
            -predicate_dir ${FOLDER_DIR}/DATA/awa2 \
            -bottleneck \
            -feature_group_results \
            -rho ${RHO} \
            -noise ${NOISE} \
            -optimizer ${OPTIMIZER} \
            -noise_loc ${NOISE_LOC} \
            -backbone ${BACKBONE} \
            -cbm_type sequential
    fi
done

# Deactivate environment
conda deactivate