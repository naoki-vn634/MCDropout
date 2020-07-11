# !/bin/bash

source /etc/profile.d/modules.sh
echo "\$SCRIPT_NAME: $0"

source activate meta_cognition
echo "enverionment"
conda info -e

MODEL='0'
INPUT='/mnt/aoni02/matsunaga/10_cropped-images/all'

DR_RATE=(0.3 0.5)
N_DROP=(10 100)
MODEL=('dense' 'vgg')
N_TRAIN=(1000 2000 3000 4500 5000 10000 15000)
DENSE='dense'
VGG='vgg'

CSV_ROOT='/mnt/aoni02/matsunaga'
CSV_PATH='RESULT'


OUTPUT='/mnt/aoni02/matsunaga/MCDropout/early_stage'

for n_drop in ${N_DROP[@]};do
    for model in ${MODEL[@]};do
        for dr_rate in ${DR_RATE[@]};do
            for n_train in ${N_TRAIN[@]};do
                WRONG_CSV=$CSV_ROOT/$model/$CSV_PATH/$n_train
                SAVE_DIR=$OUTPUT/$model/$n_train/$dr_rate
                echo $WRONG_CSV
                echo $SAVE_DIR
                if [ $model == 'dense' ]; then
                    model_ver='1'
                else 
                    model_ver='0'
                fi
                echo $model_ver
                echo $n_drop
                echo $dr_rate
                echo $n_train

                python eval.py \
                --model $model_ver \
                --input $INPUT \
                --output $SAVE_DIR \
                --weight $SAVE_DIR \
                --dr_rate $dr_rate \
                --n_drop $n_drop \
                --wrong_path $WRONG_CSV
            done        
        done
    done
done