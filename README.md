# Style Transfer based on GAN network

This is a python implementation of adaptive gan based on TensorFlow，which is used to realized style transfer.


## Requirement

- TensorFlow 1.12

- matplotlib

- py-opencv 3.42

- numpy


## Training

To get started with training you can run
```
python GANTrain.py
```
Notice that I used `tf.app.flags` to control the options shown below, you need to edit them in `GANTrain.py`:

 - `img_content` : the location of the content images saved, should be directory and will load all images in the directory, including images in subdirectory.

 - `imgB` : the location of the style images saved.

 - `checkpoint` : the location of ckpt model and TensorBoard summary saved.`../checkout` by default.

 - `Norm` : the norm method to use.Could be `BATCH` or `INSTANCE`.`INSTANCE` by default.

 - `learning_rate` : the initial learning rate. `2e-4` by default.

 - `start_step` : the start step if using `linear_decay`. `300000` by default, which means learning rate remains unchanged during first 300000 steps, the start to reduce linearly.

 - `end_step` : the end step if using `linear_decay`. `300000` by default, which means learning rate should be 0 after 300000 steps.

 - `max_to_keep` : the number of saved model kept on the same time. `10` by default.

- `summary_iter` : the interval training steps of every summary step. `10` by default, which means summary every 10 steps.

- `save_iter` : the interval training steps of every save step. `200` by default, which means save model every 200 steps.

- `batch_size` : the batch size of training. `1` by default. This parameter depend on GPU memory.

- `multi_threads` : the number of threads used to load images. `5` by default.

- `discr` : the weight of D_loss. `1` by default.

- `img` : the weight of img_loss. `100` by default.

- `feature` : the weight of feature_loss. `100` by default.

- `ngf`: the number of gen filters in generater layer. `32` by default.

- `ndf`: the number of gen filters in discriminator layer. `64` by default.

- `win_rate`: the value used to choose to train dis or gen. `0.8` by default.

- `img_size`: the input size of the GAN. `768` by default.

- `together_step`: the number of steps that train dis and gen together. `0` by default.


## Testing

To get started with testing you can run
```
python GANTest.py
```
Notice that I used `tf.app.flags` to control the options shown below, you need to edit them in `GANTest.py`:

 - `input` : the location of the input image saved.

 - `output` : the location of the result saved.

 - `checkpoint` : the location of ckpt model.`../checkout` by default.

 - `Norm` : the norm method to use.Could be `BATCH` or `INSTANCE`.`INSTANCE` by default.

- `batch_size` : the batch size of testing. `1` by default. This parameter depend on GPU memory.

- `ngf`: the number of gen filters in generater layer. `32` by default.

- `ndf`: the number of gen filters in discriminator layer. `64` by default.

- `img_size`: the input size of the GAN. `768` by default.

## Note

1. Original paper: https://arxiv.org/abs/1807.10201

2. Author's code: https://github.com/CompVis/adaptive-style-transfer

3. Content images used for training: [Places365-Standard high-res train mages (105GB)](http://data.csail.mit.edu/places/places365/train_large_places365standard.tar).  

## Result

I just tried the transfer between pictures and Picasso's painting.

<img src="./result/IMG_2071.jpg" style="zoom:50%">

<img src="./result/IMG_2075.jpg" style="zoom:50%">

<img src="./result/IMG_2076.jpg" style="zoom:50%">

<img src="./result/IMG_2072.jpg" style="zoom:50%">

<img src="./result/IMG_2077.jpg" style="zoom:50%">

<img src="./result/IMG_2074.jpg" style="zoom:50%">

<img src="./result/IMG_2078.jpg" style="zoom:50%">

<img src="./result/IMG_2073.jpg" style="zoom:50%">

