#!/usr/bin/env bash
# python3 anno_coco2voc.py --anno_file /media/cilab/disk_1/Talen/HeatMap/Datasets/COCO/annotations2014/instances_train2014.json \
#                          --type instance \
#                          --output_dir /media/cilab/disk_1/Talen/HeatMap/WSL_CNN/Data/COCO/instance_train_annotation_2014

# python3 anno_coco2voc.py --anno_file /media/cilab/disk_1/Talen/HeatMap/Datasets/COCO/annotations2014/instances_val2014.json \
#                          --type instance \
#                          --output_dir /media/cilab/disk_1/Talen/HeatMap/WSL_CNN/Data/COCO/instance_val_annotation_2014

python3 anno_coco2voc.py --anno_file /media/cilab/disk_1/Talen/HeatMap/Datasets/COCO/annotations2017/instances_val2017.json \
                         --type instance \
                         --output_dir /media/cilab/disk_1/Talen/HeatMap/WSL_CNN/Data/COCO/instance_val_annotation_2017

