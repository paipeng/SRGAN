from PIL import Image
import matplotlib.pyplot as plt
import argparse

def main(lr_file, hr_file, srgan_file, output):
    lr = Image.open(lr_file)
    hr = Image.open(hr_file)

    srgan = Image.open(srgan_file)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 10))
    ax1.imshow(lr, cmap='gray')
    ax1.set_title('downscale 2x: (iq: 4.31/nano: 172)')
    #hide x-axis
    ax1.get_xaxis().set_visible(False)
    #hide y-axis 
    ax1.get_yaxis().set_visible(False)
    ax2.imshow(srgan, cmap='gray')
    ax2.set_title('SRGAN (iq: 5.77/nano: 175)')
    ax3.imshow(hr, cmap='gray')
    ax3.set_title('Original (iq: 5.58/nano: 178)')

    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    plt.axis('off')
    plt.show()
    extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SRGAN module')
    parser.add_argument('-lr', '--lr', metavar='L', type=str, required=True, help='low resolution image file')
    parser.add_argument('-hr', '--hr', metavar='H', type=str, required=True, help='style image file')
    parser.add_argument('-s', '--srgan', metavar='srgan result image', type=str, required=True, help='srgan result image name')
    parser.add_argument('-o', '--output', metavar='Output', type=str, required=True, help='output image name')
    
    args = parser.parse_args()

    main(args.lr, args.hr, args.srgan, args.output)