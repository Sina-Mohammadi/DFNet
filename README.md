# DFNet
Keras code for our paper "DFNet: Discriminative feature extraction and integration network for salient object detection"

Our paper can be found at: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197619303252) & [arXiv](https://arxiv.org/abs/2004.01573).

## !!News!!(April 2020) 
Our new salient object detection model accepted by Pattern Recognition can be found at: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320320301072) & [arXiv](https://arxiv.org/abs/1911.13168) & [GitHub](https://github.com/Mehrdad-Noori/CAGNet)

## Saliency Maps
You can download the pre-computed saliency maps from: [Google Drive](https://drive.google.com/open?id=17nwK9Q8sfVjCMcGC5rtnQiy9fQIeIjo_) & [Baidu](https://pan.baidu.com/s/1XT7_oYit81ZI5QdBfYYh1Q) (extraction code: e2g7) for datasets DUTS-TE,  ECSSD, DUT-OMRON, PASCAL-S, HKU-IS, SOD, THUR15K.

## Framework
![image](https://github.com/Sina-Mohammadi/DFNet/blob/master/figures/Framework.png)

#### Modules:
<p align="center"><img src="https://github.com/Sina-Mohammadi/DFNet/blob/master/figures/Modules.PNG" img align="center" width="780" height="460"></p>


## Comparison with the state-of-the-art
1- Quantitative comparison

![image](https://github.com/Sina-Mohammadi/DFNet/blob/master/figures/Quantitative%20Comparison.PNG)


2- Qualitative comparison

![image](https://github.com/Sina-Mohammadi/DFNet/blob/master/figures/Visual%20Comparison.png)

## Our Sharpening Loss vs. Cross-entropy Loss visual comparison
Our Sharpening Loss guides the network to output saliency maps with higher certainty and less blurry salient objects which are much closer to the ground truth compared to the Cross-entropy Loss.

<p align="center"><img src="https://github.com/Sina-Mohammadi/DFNet/blob/master/figures/Sharpenning%20Loss%20vs.%20Cross-entropy%20Loss.png" width="420" height="500"></p>

## Usage
If you want to train the model with VGG16 Backbone, you can run

```
python main.py --batch_size=8 --Backbone_model "VGG16"
```

You can also try one of the following three options as the *Backbone_model*: "ResNet50" or "NASNetMobile" or "NASNetLarge"

In addition to *batch_size* and *Backbone_model*, you can set these training configurations: *learning_rate, epochs, train_set_directory, save_directory, use_multiprocessing, show_ModelSummary*

## Citation
```
@article{noori2020dfnet,
  title={DFNet: Discriminative feature extraction and integration network for salient object detection},
  author={Noori, Mehrdad and Mohammadi, Sina and Majelan, Sina Ghofrani and Bahri, Ali and Havaei, Mohammad},
  journal={Engineering Applications of Artificial Intelligence},
  volume={89},
  pages={103419},
  year={2020},
  publisher={Elsevier}
}
```
