# !/bin/sh
source /etc/profile.d/modules.sh
echo "\$SCRIPT_NAME: $0"

source activate meta_cognition
echo "enverionment"
conda info -e

INPUT="/mnt/aoni02/matsunaga/Dataset/200313_global-model_include_garbage/train"


OUTPUT="/mnt/aoni02/matsunaga/MCDropout"

INCLUDE_GARBAGE='include_garbage'


EPOCH='20'
BATCHSIZE='16'
LR='1e-3'
TFBOARD='True'
NUM_CLASS="3"

DR_RATE="
0.1
0.3
0.5
"

# MODEL=(0, 1) #0:VGG 1,Dense
MODEL=(
"VGG16"
"Dense161_finetune"
)

for dr_rate in ${DR_RATE[@]} ;do
    echo ${DR_RATE[j]}
    for model in ${MODEL[@]};do
        echo $INPUT
        SAVE_DIR=$OUTPUT/$model/$INCLUDE_GARBAGE/$dr_rate
        echo $SAVE_DIR

        if [ $model == "VGG16" ];then
            model_mode="0"
        else
            model_mode="1"
        fi
        echo $model
        echo $model_mode
        echo $NUM_CLASS

        python -u train.py \
        --input $INPUT \
        --output $SAVE_DIR \
        --epoch $EPOCH \
        --batchsize $BATCHSIZE \
        --lr $LR \
        --tfboard $TFBOARD \
        --dr_rate $dr_rate \
        --model $model_mode \
        --n_cls $NUM_CLASS

    done
done