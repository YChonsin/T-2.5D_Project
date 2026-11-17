dsname=$1
modelname=$2

python ./src/transfer_learning/fine_tuning.py \
    --net TIR_3d_4TL_drop_0_1 \
    --dpath ${dsname} \
    --pname ${modelname} \
    --gpunum 1 \
    --batchSize 1 \
    --dtype tl \
    --epoch 10
