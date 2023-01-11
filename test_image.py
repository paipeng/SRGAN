import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, required=True, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_2_100.pth', type=str, help='generator model epoch name')
parser.add_argument('-c', '--crop', default=False, type=bool, help='crop image')
parser.add_argument('-g', '--gray', default=False, type=bool, help='convert image to grayscale')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name
CROP_WIDTH = 452
CROP_HEIGHT = 380

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME))
else:
    model.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))
print(f'input image name: {IMAGE_NAME}')
image = Image.open(IMAGE_NAME)
if opt.crop:
    image = image.resize((640, 640))
    new_width = CROP_WIDTH
    new_height = CROP_HEIGHT
    width, height = image.size   # Get dimensions
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))

if opt.gray:
    if(image.mode != 'L'):
        image = image.convert('L')

with torch.no_grad():
    image = Variable(ToTensor()(image)).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    start = time.process_time()
    out = model(image)
    elapsed = (time.process_time() - start)
    print('cost: ' + str(elapsed) + ' s')
    out_img = ToPILImage()(out[0].data.cpu())
    if opt.crop:
        vd = int((640 -CROP_HEIGHT)/2)
        hd = int((640 -CROP_WIDTH)/2)
        out_img = add_margin(out_img, vd, hd, vd, hd, 255)
    out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME + '.bmp')
