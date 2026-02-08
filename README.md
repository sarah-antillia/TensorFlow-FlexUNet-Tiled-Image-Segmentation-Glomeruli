<h2>TensorFlow-FlexUNet-Tiled-Image-Segmentation-Glomeruli (2026/02/09)</h2>
Sarah T.  Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Glomeruli</b> based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass), 
and a 512x512 pixels
<a href="https://drive.google.com/file/d/1ppNI4YDEbzZiaU0ew23mYHbrFFX2BdL0/view?usp=sharing">
<b>Tiled-Glomeruli-ImageMask-Dataset.zip</b></a> which was derived by us from <br><br>
<a href="https://www.kaggle.com/datasets/baesiann/glomeruli-hubmap-external-1024x1024">
Glomeruli (HuBMAP external) 1024x1024 </b><br></a>
External data for HuBMAP competition (2020)
<br><br>
<b>Divide-and-Conquer Strategy</b><br>
Since the images and masks in the training dataset are large 1024x1024 pixels,
we adopted the following <b>Divide-and-Conquer Strategy</b> for building the segmentation model.
<br><br>
<b>1. Tiled Image and Mask Dataset</b><br>
We generated a 512x512 pixels tiledly-split dataset from the original <b>Glomeruli</b>.<br><br>
<b>2. Train Model by Tiled ImageMask Dataset</b><br>
We trained and validated the TensorFlowFlexUNet model by using the Tiled-Glomeruli-ImageMask-Dataset.
<br><br>
<b>3. Tiled Image Segmentation</b><br>
We applied our Tiled-Image Segmentation inference method to predict Glomeruli regioins 
 for the mini_test images with a resolution of 1024x1024 pixels.<br><br>
<hr>
<b>Actual Image Segmentation for Glomeruli Images of 1024x1024 pixels </b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br><br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/images/10003.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/masks/10003.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test_output_tiled/10003.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/images/10149.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/masks/10149.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test_output_tiled/10149.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/images/10183.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/masks/10183.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test_output_tiled/10183.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<a href="https://www.kaggle.com/datasets/baesiann/glomeruli-hubmap-external-1024x1024">
Glomeruli (HuBMAP external) 1024x1024 </b><br></a>
External data for HuBMAP competition (2020)
<br><br>
The following explanation was taken from the above kaggle web site<br><br>

<b>About Dataset</b><br>
DATA WAS TAKEN FROM <a href="https://data.mendeley.com/datasets/k7nvtgn2x6/3">Data for glomeruli characterization in histopathological image
</a> MANUALLY LABELED AND TILED TO 1024x1024 PNGS
<br><br>
<b>Data for glomeruli characterization in histopathological images</b><br>
Published: 05-02-2020<br>
Version 3<br>
DOI: 10.17632/k7nvtgn2x6.3<br>
Contributors: Gloria Bueno, Lucia Gonzalez-Lopez, Marcial García-Rojo, Arvydas Laurinavicius<br>
License: CC BY 4.0<br>
<br>
<b>Description</b><br>
The data presented here is part of the whole slide imaging (WSI) datasets generated in European project AIDPATH. <br>
This data is also related to the research paper entitle 
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0169260719311381">
Glomerulosclerosis identification in whole slide images using semantic segmentation</a>, 
published in Computer Methods and Programs in Biomedicine Journal (DOI: 10.1016/j.cmpb.2019.105273) . <br>
In that article, different methods based on deep learning for glomeruli segmentation and their classification into normal and sclerotic 
glomerulous are presented and discussed. These data will encourage research on artificial intelligence (AI) methods, create 
and compare fresh algorithms, and measure their usability in quantitative nephropathology.
<br><br>
Parameters for data collection: Tissue samples were collected with a biopsy needle having an outer diameter between 100μm 
and 300μm. Afterwards, paraffin blocks were prepared using tissue sections of 4μm and stained using Periodic acid–Schiff (PAS). <br>
Then, images at 20x magnification were selected.
<br><br>
Description of data collection: The tissue samples were scanned at 20x with a Leica Aperio ScanScope CS scanner.
<br>
<br>
<h3>
2 Glomeruli ImageMask Dataset
</h3>
 If you would like to train this Glomeruli Segmentation model by yourself,
please down load our dataset <a href="https://drive.google.com/file/d/1ppNI4YDEbzZiaU0ew23mYHbrFFX2BdL0/view?usp=sharing">
<b>Tiled-Glomeruli-ImageMask-Dataset.zip</b>
</a> on the google drive,
expand the downloaded , and put it under <b>./dataset/</b> to be.
<pre>
./dataset
└─Glomeruli
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Glomeruli Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Glomeruli/Glomeruli_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br><br>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Glomeruli/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Glomeruli/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Glomeruli TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Glomeruli/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Glomeruli and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 2
base_filters   = 16
base_kernels  = (11,11)
num_layers    = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for Glomeruli 1+1classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;Glomeruli 1+1 classes,  
;                      Glomeruli:red
rgb_map={(0,0,0):0,(255,0,0):1,}
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer     = False
epoch_change_infer_dir =  "./epoch_change_infer"
epoch_change_tiled_infer     = True
epoch_change_tiled_infer_dir =  "./epoch_change_tiled_infer"
</pre>
By using this epoch_change_tiled_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Glomeruli/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (15,16,17)</b><br>
<img src="./projects/TensorFlowFlexUNet/Glomeruli/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (30,31,32)</b><br>
<img src="./projects/TensorFlowFlexUNet/Glomeruli/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was terminated at epoch 32.<br><br>
<img src="./projects/TensorFlowFlexUNet/Glomeruli/asset/train_console_output_at_epoch32.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Glomeruli/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Glomeruli/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Glomeruli/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Glomeruli/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Glomeruli</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for Glomeruli.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Glomeruli/asset/evaluate_console_output_at_epoch32.png" width="880" height="auto">
<br><br>Image-Segmentation-Glomeruli

<a href="./projects/TensorFlowFlexUNet/Glomeruli/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Glomeruli/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.0485
dice_coef_multiclass,0.9785
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Glomeruli</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Glomeruli.<br>
<pre>
>./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Glomeruli/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Glomeruli/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Glomeruli/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for  Glomeruli  Images of 1024x1024 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br>

<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/images/10064.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/masks/10064.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test_output_tiled/10064.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/images/10150.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/masks/10150.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test_output_tiled/10150.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/images/10164.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/masks/10164.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test_output_tiled/10164.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/images/10177.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/masks/10177.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test_output_tiled/10177.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/images/10183.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/masks/10183.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test_output_tiled/10183.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/images/10224.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test/masks/10224.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Glomeruli/mini_test_output_tiled/10224.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. A General Pipeline for Glomerulus Whole-Slide Image Segmentation</b><br>
Quan Huu Cap<br>
<a href="https://arxiv.org/html/2411.04782v2">https://arxiv.org/html/2411.04782v2</a>
<br>
<br>
<b>2. Enhancing glomeruli segmentation through cross-species pre-training</b><br>
Paolo Andreini,  Simone Bonechi, Giovanna Maria Dimitri<br>
<a href="https://www.sciencedirect.com/science/article/pii/S0925231223010706">https://www.sciencedirect.com/science/article/pii/S0925231223010706</a>
<br>
<br>
<b>3. Integrated model for segmentation of glomeruli in kidney images</b><br>
Gurjinder Kaur,  Meenu Garg, Sheifali Gupta<br>
<a href="https://www.sciencedirect.com/science/article/pii/S2667241324000211">
https://www.sciencedirect.com/science/article/pii/S2667241324000211</a>
<br>
<br>
<b>4.Tiled-ImageMask-Dataset-BCNB</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-BCNB">
https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-BCNB</a>
<br>
<br>
