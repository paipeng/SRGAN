import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from tqdm import tqdm
import torchvision
from torchvision.transforms import ToTensor, ToPILImage, CenterCrop

from PIL import Image
import matplotlib.pyplot as plt

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Authorization')
parser.add_argument('--crop_size', default=180, type=int, required=False, help='training images crop size')

parser.add_argument('-i', '--input', metavar='input-image-path', type=str, required=True, help='input image path')
parser.add_argument('-r', '--reference', metavar='reference-image-path', type=str, required=True, help='reference image path')
parser.add_argument('-g', '--gray', metavar='G', type=bool, required=False, default=False, help='Grayscale image')
parser.add_argument('-m', '--model_name', default='netD_epoch_2_100.pth', type=str, help='Discriminator model epoch name')

def load_image(image_path, crop_size):
    image = Image.open(image_path)
    if image.mode != 'L':
        image = image.convert('L')
    width, height = image.size   # Get dimensions

    new_width = crop_size
    new_height = crop_size
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    image = Variable(ToTensor()(image)).unsqueeze(0)
    
    if torch.cuda.is_available():
        image = image.cuda()
    return image

def show_images(input_image, reference_image, d_loss, output):
    i = torchvision.transforms.ToPILImage()(input_image[0])
    r = torchvision.transforms.ToPILImage()(reference_image[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(i, cmap='gray')
    ax1.set_title('input image')
    #hide x-axis
    ax1.get_xaxis().set_visible(False)
    #hide y-axis 
    ax1.get_yaxis().set_visible(False)
    ax2.imshow(r, cmap='gray')
    ax2.set_title(f'Discriminator dloss: {d_loss}')
    
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    plt.axis('off')
    plt.show()
    extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    print(f'save figure to {output}')
    fig.savefig(output)

if __name__ == '__main__':
    opt = parser.parse_args()
    print(f'input image name: {opt.input}')
    print(f'input image name: {opt.reference}')
    CROP_SIZE = opt.crop_size
    
    netD = Discriminator().eval()
    netD.load_state_dict(torch.load(opt.model_name))

    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    
    if torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'
        netD.cuda()
    
    optimizerD = optim.Adam(netD.parameters())

    input_image = load_image(opt.input, CROP_SIZE)
    reference_image = load_image(opt.reference, CROP_SIZE)


    #netD.zero_grad()
    real_out = netD(input_image).mean()
    print(real_out.item())
    fake_out = netD(reference_image).mean()
    print(fake_out)
    d_loss = 1 - real_out + fake_out
    #d_loss.backward(retain_graph=True)
    #optimizerD.step()
    print(d_loss.item())

    show_images(input_image, reference_image, d_loss.item(), 'authori.png')
        
    
