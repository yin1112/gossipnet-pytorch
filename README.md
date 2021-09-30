# Learning non-maximum suppression for object detection

This is the code for the paper  
_Learning non-maximum suppression. Jan Hosang, Rodrigo Benenson, Bernt Schiele. CVPR 2017._

You can find the project page with downloads here: https://mpi-inf.mpg.de/learning-nms

## Software Requirements
- torch : 1.7.1.post2
- numpy : 1.19.5
- protoc : libprotoc 3.13.0

## First

Run `make` in the home directory to compile protobufs

## Second

Download the files according to the addresses in `./data/readme` and `./data/coco/annotations/README` files

## Third

run `python train.py --config=experiments/coco_person/conf.yaml` 

