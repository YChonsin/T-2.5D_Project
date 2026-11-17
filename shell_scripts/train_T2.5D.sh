dsname=$1
modelname=$2

python ./src/transfer_learning/tl_2d_3d.py \
    --net T25D \
    --dpath ${dsname} \
    --pname ${modelname} \
    --gpunum 1 \
    --batchSize 1 \
    --epoch 20
