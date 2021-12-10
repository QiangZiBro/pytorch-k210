# Pytorch K210

## TL;DR

This repo intends to introduce a complete routine for deploying deep learning models to k210 board, using the MNIST digit recognition as an example.



## Pipeline

```
Pytorch --> ONNX --> NNCASE --> KModel --> K210 SDK / MicroPython
```



## Tutorials

You can find the tutorial [here](http://qiangzibro.com/2021/12/01/cvaio/) (简体中文), bon appetite!



## Goals

- [ ] Use Micropython to load KModel
- [ ] Use C SDK to load the KModel
- [ ] Add object detection pipeline



## Useful links

- Gravati Open Dataset https://gas.graviti.cn/dataset/qiangzibro/MNIST