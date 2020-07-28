# MSFFN
MultiSpectral Feature Fusion Network for object detection or pedestrian detection based on KAIST Multispectral Pedestrian Detection Benchmark

## Download KAIST dataset
Download KAIST Multispectral Pedestrian Detection Benchmark [[KAIST](http://multispectral.kaist.ac.kr)]

```
Extract all of these tars into one directory and rename them, which should have the following basic structure.
```

### Kaist datasets path

1) data/dataset/kaist/annos

2) data/dataset/kaist/images

2.1) data/dataset/kaist/images/lwir

2.2) data/dataset/kaist/images/visible

3) data/dataset/kaist/imgsets

3.1) data/dataset/kaist/imgsets/train.txt

3.2) data/dataset/kaist/imgsets/val.txt                     

### Make kaist train and val annotation

```bashrc
$ python scripts/annotation.py

Then edit your `core/config.py` to make some necessary configurations
```

	__C.YOLO.CLASSES = "data/classes/pedestrian.names"
	
	__C.TRAIN.ANNOT_PATH = "data/dataset/pedestrian_train.txt"
	
	__C.TEST.ANNOT_PATH = "data/dataset/pedestrian_val.txt"


## Train KAIST dataset
Two files are required as follows:

- data/classes/pedestrian.names
	
	```
	person
	```

- data/dataset/pedestrian_train.txt

	```
	data/dataset/kaist/images/visible/set03_V001_I00909.jpg data/dataset/kaist/images/lwir/set03_V001_I00909.jpg 323,319,345,273,0 287,215,301,249,0 279,222,288,244,0 1,240,36,441,0
	```

- data/dataset/pedestrian_val.txt

	```
	data/dataset/kaist/images/visible/set00_V008_I00627.jpg data/dataset/kaist/images/lwir/set00_V008_I00627.jpg 385,228,408,285,0
	```

### Train method

```bashrc
$ python train.py
$ tensorboard --logdir data/log/train
```

### Evaluate method

```bashrc
$ python evaluate.py
```

### mAP

```bashrc
$ python evaluate.py
$ cd mAP
$ python main.py
```
