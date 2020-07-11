# !/bin/sh
source /etc/profile.d/modules.sh
echo "\$SCRIPT_NAME: $0"

source activate meta_cognition
echo "enverionment"
conda info -e

CSV_FILE=(
"/mnt/aoni02/matsunaga/dense/RESULT/4000/train_img_list_4000.txt"
)

OUTPUT=(
"/mnt/aoni02/matsunaga/MCDropout/early_stage/vgg/4000"
)


EPOCH='20'
BATCHSIZE='16'
LR='1e-3'
TFBOARD='True'

DR_RATE=(0.3 0.5)

MODEL='0'

for j in 0 1 ;do
    echo ${DR_RATE[j]}
    for i in 0 1;do
        echo ${CSV_FILE[i]}
        SAVE_DIR=${OUTPUT[i]}/${DR_RATE[j]}
        CSV_INPUT=${CSV_FILE[i]}


        python -u train.py \
        --txt $CSV_INPUT \
        --output $SAVE_DIR \
        --epoch $EPOCH \
        --batchsize $BATCHSIZE \
        --lr $LR \
        --tfboard $TFBOARD \
        --dr_rate ${DR_RATE[j]} \
        --model $MODEL

    done
done