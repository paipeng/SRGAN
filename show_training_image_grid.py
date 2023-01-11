import torchvision
import torch
import argparse
import matplotlib.pyplot as plt

from data_utils import TrainDatasetFromFolder
"""
batch_tensor = torch.randn(*(10, 3, 256, 256)) # (N, C, H, W)
# make grid (2 rows and 5 columns) to display our 10 images
grid_img = torchvision.utils.make_grid(batch_tensor, nrow=2, padding=50)

print(grid_img.shape)

# reshape and plot (because matplotlib needs channel as the last dimension)
plt.imshow(grid_img.permute(1, 2, 0))

torchvision.utils.save_image(grid_img, 'filename.png')

"""

def show_image(images):
    images = images.numpy()
    images = images.transpose((1,2,0))
    print(images)
    plt.imshow(images)
    plt.show()

def generate_grid(train_loader):
    #torch_tensor = torch.tensor(train_loader['targets'].values)
    #torch_tensor = []
    #device = torch.device('cpu')
    #for batch, (X, y) in enumerate(train_loader):
    #    X = X.to(device)
    #    torch_tensor.append(X)
    blur_images, targets = next(iter(train_loader))
    grid_img = torchvision.utils.make_grid(targets, nrow=4, padding=10)
    #print(grid_img.shape)
    torchvision.utils.save_image(grid_img, 'filename.png')

    # print("grid:", grid)
    #img = torchvision.transforms.ToPILImage()(grid_img)
    #img.show()
    #show_image(grid_img)

    grid_img = torchvision.utils.make_grid(blur_images, nrow=4, padding=10)
    #print(grid_img.shape)
    torchvision.utils.save_image(grid_img, 'filename_blur.png')



parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=180, type=int, help='training images crop size')
parser.add_argument('-d', '--traindir', metavar='D', type=str, required=True, help='train data pictures folder path')

if __name__ == '__main__':
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    
    train_set = TrainDatasetFromFolder(opt.traindir, crop_size=CROP_SIZE, upscale_factor=0, gray=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=1, batch_size=32, shuffle=True)
    
    generate_grid(train_loader=train_loader)
