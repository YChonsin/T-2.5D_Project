dataset=$1

python ./src/dataset_processing/dp2d.py \
    --df ${dataset} \
    --tnum 6000 --vnum 1500 \
    --shape 40 40 --stride 20 20

python ./src/dataset_processing/dp3d.py \
    --df ${dataset} \
    --tnum 6000 --vnum 1500 \
    --shape 40 40 16 --stride 16 16 8