# Learning non-maximum suppression for object detection

This is the code for the paper  
_Learning non-maximum suppression. Jan Hosang, Rodrigo Benenson, Bernt Schiele. CVPR 2017._

You can find the project page with downloads here: https://mpi-inf.mpg.de/learning-nms

# Some optimization
We have done some optimization on the original author's code (in the production of manual features, we have compressed the size of multi category features). This optimization has greatly improved the multiclass AP while keeping the map from getting worse .

The following is the validation data after the 300000 training round.
(tensorflow) 

| Original method  | mAP |  multiclass AP |
| ------------- | ------------- | ------------- |
| Original method  | 44.3  | 37.8 |
| After optimization  | 44.0 | 47.9 |

The following is the validation data after the 20000 training round.(I haven't recorded it yet. I'll make it up later ..)
(pytorch) 
| Original method  | mAP |  multiclass AP |
| ------------- | ------------- | ------------- |
| Original method  | 40.55 | 36.05 |
| After optimization  | 41.93 | 46.26 |

# Software Requirements
- torch : 1.7.1.post2
- numpy : 1.19.5
- protoc : libprotoc 3.13.0
- scipy : 1.1.0
# First

Run `make` in the home directory to compile protobufs

# Second

Download the files according to the addresses in `./data/README` and `./data/coco/annotations/README` files

# Third

run `python train.py --config=experiments/coco_person/conf.yaml` 

