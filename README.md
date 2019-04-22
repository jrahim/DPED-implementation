# DPED-implementation

This is an implementation of the DPED project for the COMP3314: Machine Learning course at HKU.

Details and paper for the project can be found (http://people.ee.ethz.ch/~ihnatova/ "here").

# Training
To train you will first require the dped dataset from the project webpage, which should be extracted to a folder named "dataset" in the directory of this repo. You will also require a pre-trained VGG19 model which can be downloaded from ("https://drive.google.com/file/d/0BwOLOmqkYj-jMGRwaUR2UjhSNDQ/view?usp=sharing" here). The model file should be placed in a folder named "vgg_pretrained". You are all set to go now!

To train the model using this code:
`python train.py "<phone model>"`

Available phone models are: "iphone", "blackberry", and "sony"

For example, for training on iphone dataset samples:
`python train.py "iphone"`

# Testing

For testing on custom images, create the following path: "test/custom_images", then run the following:

`python test.py "<phone model>" --testing_dir "test" --run_img "<directory of custom images>"`

The output images will be stored in test/custom_images
