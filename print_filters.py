#!/home/claudio/Apps/anaconda3/envs/PyTorch/bin/python

# builtins
import glob, os, time, argparse
from pathlib import Path
from PIL import Image
# project dependencies
from model import DeepVO
# external dependencies
import numpy as np
import torch
from matplotlib import pyplot as plt

def plot_filters_single_channel(t, im_r=False, fname='filters.png'):
    #kernels depth * number of kernels
    # nplots = t.shape[0]*t.shape[1]
    nplots = t.shape[0]*3
    ncols = 12
    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)
    count = 0
    fig = plt.figure(figsize=(ncols, nrows))
    #looping through all the kernels in each channel
    for i in range(t.shape[0]):
        r = range(3,6) if im_r else range(3)
        for j in r:
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, name), dpi=100)
    plt.show()

# parse passed arguments
argparser = argparse.ArgumentParser(description="Visualizer of DeepVO Filters")
argparser.add_argument('model', type=str, help="path to trained model")
argparser.add_argument('out', type=str, help="path where plots will be saved")
argparser.add_argument('--img_w', '-img_w', type=int, default=608, help="heigth of input images to the model")
argparser.add_argument('--img_h', '-img_h', type=int, default=160, help="width of input images to the model")
argparser.add_argument('--no_bnorm', '-no_bnorm', action='store_false', help="set to disable batch normalization layers")
argparser.add_argument('--flownet', '-flownet', action='store_true', help="visualize initial flownet filters")
args = argparser.parse_args()

# main
if __name__ == '__main__':

    # prepare directory structure
    load_model_path = args.model
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # load model
    model = DeepVO(args.img_h, args.img_w, args.no_bnorm)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(load_model_path))
    else:
        model.load_state_dict(torch.load(load_model_path, map_location={'cuda:0': 'cpu'}))
    print('load model from: ', load_model_path)

    # load flownet weights
    if args.flownet:
        flownet_path = '/home/claudio/Projects/DeepVO-pytorch/pretrained/flownets_EPE1.951.pth.tar'
        print('load conv weights from pretrained FlowNet model ({})'.format(flownet_path))
        if use_cuda:
            pretrained_w = torch.load(flownet_path)
        else:
            pretrained_w = torch.load(flownet_path, map_location='cpu')
        # Use conv-layer-weights of FlowNet for DeepVO conv-layers
        model_dict = model.state_dict()
        update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)

        # # compare initial weights with trained ones
        # flownet_w = model.conv1[0].weight.data.numpy().copy()
        # comp = np.allclose(trained_w, flownet_w, atol=1e-3)
        # print(comp)

    print(model)
    print(model.conv1[0].weight.data.shape)

    name = 'flownet.png' if args.flownet else 'filters.png'
    plot_filters_single_channel(model.conv1[0].weight.data, im_r=True, fname=name)

