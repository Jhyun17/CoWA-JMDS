# Confidence score Weighting Adaptation using the JMDS (CoWA-JMDS)

This repository is the official implementation of ["Confidence Score for Source-Free Unsupervised Domain Adaptation"](https://proceedings.mlr.press/v162/lee22c/lee22c.pdf), accepted in ICML 2022.
This code refers to [SHOT](https://github.com/tim-learn/SHOT) implementation.

### Environments

Fix 'prefix' in environment.yaml file.
```bash
prefix: /home/[your_username]/anaconda3/env/CoWA
```
Then create the environment.
```bash
$ conda env create --file environment.yaml
```

### Datasets
You can download datasets here:

- [VISDA-2017](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)

### Training

After downloading the datasets, create following files and directories in this directory.
```bash
$ mkdir ./data

data
└── VISDA-C
    ├── train_list.txt
    └── validation_list.txt
```

Each list.txt file has (image_path, class index) pairs
```Example
./data/VISDA-C/train_list.txt

line 1 : /home/[username]/data/VisDA-2017/train/aeroplane/src_2_02691156_4def53f149137451b0009f08a96f38a9__44_349_150.png 0
line 2 : /home/[username]/data/VisDA-2017/train/aeroplane/src_1_02691156_5d0d3f54c5d9dd386a1aee7416e39fad__180_236_150.png 0
...
```

Then run a script file.
```bash
$ chmod +x run_visda.sh
$ ./run_visda.sh
```
