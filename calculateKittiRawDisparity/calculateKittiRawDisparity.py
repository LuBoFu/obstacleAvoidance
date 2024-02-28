# from msilib import sequence
import sys
sys.path.append('core')
DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from igev_stereo import IGEVStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2
import h5py
import skimage.io

def load_image(imfile):
    # 下面有一个float()转换，此处为何要转换为uint8？
    img = np.array(Image.open(imfile)).astype(np.uint8)    
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    # 为何要添加一个大小为1的维？
    return img[None].to(DEVICE)

def calculateDisparity(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    outputDir = Path(args.outputDir)
    outputDir.mkdir(exist_ok=True)

    with h5py.File(args.kittiRawIndex, "a") as f:
        sequenceNumber = f["sequenceNumber"][()]
        for seq in range(sequenceNumber):
            sequenceName = "seq_" + str(seq)
            leftImagePaths  = f[sequenceName + "/leftImages"] [()]
            rightImagePaths = f[sequenceName + "/rightImages"] [()]
            assert len(leftImagePaths) == len(rightImagePaths), "lengthes of left and right image paths are not equal"            
            # The following list will be used to store the paths of disparity data file.
            disparityFilePaths = []
            for frame in range(len(leftImagePaths)):
                print(f"processing frame {frame}/{len(leftImagePaths)} of sequence {seq}")
                leftImagePath  = leftImagePaths[frame]
                rightImagePath = rightImagePaths[frame]            
                # print(leftImagePath, rightImagePath)
                with torch.no_grad():
                    leftImage  = load_image(leftImagePath)
                    rightImage = load_image(rightImagePath)
                    assert leftImage.shape == rightImage.shape, "shapes of the left and right images are different"
                    # print("original image shape: ", leftImage.shape)
                    
                    # padding the input images so that their height and weight are dividable by 32
                    padder = InputPadder(leftImage.shape, divis_by=32)
                    leftImage, rightImage = padder.pad(leftImage, rightImage)
                    # print("padded image shape: ", leftImage.shape)

                    # calculate the disparity image.
                    disp = model(leftImage, rightImage, iters=args.valid_iters, test_mode=True)                    
                    disp = padder.unpad(disp)                    
                    disp = disp.cpu().numpy().squeeze()
                    # print("disparity image shape: ",  disp.shape)
                    
                    # save as an image file which can show the disparity effect more clearly.
                    disparityFileDir = args.outputDir + "/" + sequenceName
                    Path(disparityFileDir).mkdir(exist_ok=True, parents=True)
                    disparityEffectFilePath = disparityFileDir + "/" + "frame_" + str(frame) + "_effect.png"
                    plt.imsave(disparityEffectFilePath, disp, cmap='jet')

                    # save as a data file which stores the disparity data accurately.
                    disp = np.round(disp * 256).astype(np.uint16)
                    disparityFilePath = disparityFileDir +  "/frame_" + str(frame) + ".png"
                    skimage.io.imsave(disparityFilePath, disp)
                    
                    disparityFilePaths.append(disparityFilePath)            

            # append to the H5 index file.
            f.create_dataset(sequenceName + "/disparityImages", data=disparityFilePaths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # If an argument is referred by any code, don't delete the corresponding following line.
    parser.add_argument('--restore_ckpt',  help="path to the pretrained model")   
    parser.add_argument('--kittiRawIndex', help="path to the index file for kitti raw dataset")
    parser.add_argument('--outputDir',     help="directory to output disparity images")
    
    # Arguments referred by the algorithm
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    
    args = parser.parse_args()

    Path(args.outputDir).mkdir(exist_ok=True, parents=True)

    calculateDisparity(args)
