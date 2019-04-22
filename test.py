from scipy.misc import imsave

import argparse
from utils import *
from dataloader import *
from model import *
import tensorflow as tf

parser = argparse.ArgumentParser(description="testing options")

parser.add_argument("phone_model", type=str, help="phone model to test")

parser.add_argument("--num_epochs", type=int, help="number of epochs to train network for", default=100000)
parser.add_argument("--use_bn", help="use batch nor or not. default is True", type=bool, default=False)
parser.add_argument("--batch_size", type=int, help="batch size for each epoch", default=5)
parser.add_argument("--test_size", type=float, help="fraction for test size", default=0.10)
parser.add_argument("--dataset_dir", type=str, help="directory for the SSID dataset only",
                    default="dataset/dped")
parser.add_argument("--res", type=int, help="resolution of images", default=100)
parser.add_argument("--augment", help="use data augmentation or not. default is True", type=bool, default=True)
parser.add_argument("--w_content_loss", type=float, help="weightage for the content loss based on the VGG19",
                    default=0.5)
parser.add_argument("--w_adversarial_loss", type=float, help="weightage for discriminator guesses", default=1.0)
parser.add_argument("--w_tv_loss", type=float, help="weightage for total variance", default=1.0)
parser.add_argument("--w_pixel_loss", type=float, help="weightage for loss based on difference from GT", default=1.0)
parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.002)
parser.add_argument("--vgg_dir", type=str, help="directory for trained VGG 19 model",
                    default="vgg_pretrained/imagenet-vgg-verydeep-19.mat")
parser.add_argument("--content_layer", type=str, help="content layer to use in VGG 19 net", default="relu5_4")
parser.add_argument("--checkpoint_dir", type=str, help="directory for storing model checkpoints", default="checkpoints")
parser.add_argument("--testing_dir", type=str, help="directory for storing testing images", default="testing")
parser.add_argument("--num_files_to_load", type=int, help="number of images to load", default=None)
parser.add_argument("--load_checkpoint", type=bool, help="load checkpoint or not", default=True)
parser.add_argument("--test_mode", type=bool, help="testing mode or not", default=True)
parser.add_argument("--test_patches", type=int, help="patches (1) or full (0)", default=1)
parser.add_argument("--num_tests", type=int, help="number of tests to run", default=1)
parser.add_argument("--epoch_to_load", type=int, help="epoch num to load (use multiples of 1000)", default=None)

parser.add_argument("--run_img", type=str, help="run_img", default=None)

config = parser.parse_args()

if __name__ == '__main__':
    tf.reset_default_graph()
    sess = tf.Session()
    config.checkpoint_dir = "./" + config.phone_model + "_" + config.checkpoint_dir
    if not os.path.exists(config.checkpoint_dir):
        print("making ckpt dir: ", config.checkpoint_dir)
        os.makedirs(config.checkpoint_dir)
    if config.run_img:
        config.testing_dir = os.path.join("./" + config.testing_dir, "custom_images")
    elif config.test_patches:
        config.testing_dir = os.path.join("./" + config.testing_dir, config.phone_model, "patches")
    else:
        config.testing_dir = os.path.join("./" + config.testing_dir, config.phone_model, "full_size")
    data_loader = DataLoader(config)
    model = Model(sess, config, data_loader)
    if config.run_img:
        print(config.testing_dir)
        inputs, rets, gts = model.test()
        for i in range(len(inputs)):
            imsave(os.path.join(config.testing_dir, str(i)+"_input.jpg"), postprocess(inputs[i]))
            imsave(os.path.join(config.testing_dir, str(i)+"_output.jpg"), postprocess(rets[i]))
    else:
        for counter in range(config.num_tests):
            inputs, rets, gts = model.test()
            counter2 = 0
            for input, ret, gt in zip(inputs, rets, gts):
                imsave(os.path.join(config.testing_dir, str(counter2)+"_input.jpg"), postprocess(input))
                imsave(os.path.join(config.testing_dir, str(counter2)+"_output.jpg"), postprocess(ret))
                if config.test_patches:
                    imsave(os.path.join(config.testing_dir, str(counter2)+"_gt.jpg"), gt)
                counter2 += 1

