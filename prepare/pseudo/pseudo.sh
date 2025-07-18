#! /bin/bash

python prepare_unlabeled_data.py --image_folder \
    ../collections/raw/VIPCUP2020/aip-cup-2020/vipcup2020/vipcup2020/images/train/ \
    ../collections/raw/VIPCUP2020/aip-cup-2020/vipcup2020/vipcup2020/images/val/ \
    ../collections/raw/VIPCUP2020/vip-cup-2020/vip_cup_2020/fisheye-day-30062020/images/train/ \
    ../collections/raw/fisheye8k/test/images/ \
    ../collections/raw/fisheye8k/train/images/ \
    ../collections/raw/fisheye8k/images \
    --output_folder aip_vip_fisheye/images \
    --p_num 64 \

cd VNPT

# last is the number of gpus
./inference_codetr_parser.sh ../../../data/collections/unlabeled/aip_vip_fisheye/images 1

# # those bellow only need one gpu
./inference_yolor_parser.sh ../../../data/collections/unlabeled/aip_vip_fisheye/images
./inference_yolov9_parser.sh ../../../data/collections/unlabeled/aip_vip_fisheye/images
./inference_internimage_parser.sh ../../../data/collections/unlabeled/aip_vip_fisheye/images

# cd infer

python fuse_results.py 
mv final.json pseudo.json

