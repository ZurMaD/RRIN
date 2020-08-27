import argparse
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from model import Net
import numpy as np


import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
#from PIL import Image
from skimage.io import imread

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


def list_path(p_dir):
    list_files = [f for f in listdir(p_dir) if isfile(join(p_dir, f))]
    list_files = sorted(list_files, key=lambda x: int(os.path.splitext(x)[0]))
    return list_files


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Video Frame Interpolation via Residue Refinement')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--testpath', default='',
                        help='Path to your test folder')
    parser.add_argument('--subfolder', default='',
                        help='Path to the folder inside your test path. For example: "input","input_x2","input_x4" ')
    parser.add_argument('--multiplier', default=2,
                        help='How many times you want to interpolate frames?. Use integer more than zero! and less than 6.')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    path = args.testpath
    sub_folder = args.subfolder
    multiplier = args.multiplier
    transform = transforms.ToTensor()

    model = Net()
    state = torch.load('pretrained_model.pth.tar')
    model.load_state_dict(state, strict=True)
    model = model.cuda()
    model.eval()

    # PARAMETERS
    print('---------- MAIN ----------', '\n',
          'Start Time:', datetime.now(), '\n',
          'Frames Founded:', len(list_path(path)), '\n',
          'Final frames:', len(list_path(path))*multiplier, '\n',
          # 'Remain time estimated:',len(list_path(path))*multiplier*0.15,'\n
          '--------------------------')

    # CREATE MKDIR
    if multiplier <= 0:
        sub_folder = path+'input'
        if(not os.path.exists(sub_folder)):
            os.mkdir(sub_folder)
        print(datetime.now(), 'Select more than 0 in duplicate_frames')
        print(datetime.now(), 'Creating folder named: {}'.format(sub_folder))
    else:
        # CREATE FOLDERS
        for m in range(multiplier):
            sub_folder = path+'input_x{}'.format(2**(m+1))
            if(not os.path.exists(sub_folder)):
                os.mkdir(sub_folder)
            print(datetime.now(), 'Creating folder named: {}'.format(sub_folder))

    # MULTIPLIER
    for a in range(multiplier):

        if a == 0:
            temp_sub_folder = 'input'
            temp_sub_folder_dest = 'input_x2'
            mydir = path+'input'
            onlyfiles = list_path(mydir)
        else:
            temp_sub_folder = 'input_x{}'.format(2**(a))
            temp_sub_folder_dest = 'input_x{}'.format(2**(a+1))
            mydir = path+temp_sub_folder
            onlyfiles = list_path(mydir)

        print(datetime.now(), 'Multiplier', a, 'DIR: ...'+mydir[-30:])

        # INFERENCE
        for b in range(len(onlyfiles)-1):

            img1 = path+temp_sub_folder+'/'+onlyfiles[b]
            img2 = path+temp_sub_folder+'/'+onlyfiles[b+1]

            ######################
            #    DO THE MODEL    #
            ######################

            with torch.no_grad():

                i1 = Image.open(img1)
                i2 = Image.open(img2)
                i1 = transform(i1).unsqueeze(0).cuda()
                i2 = transform(i2).unsqueeze(0).cuda()

                if i1.size(1) == 1:
                    i1 = i1.expand(-1, 3, -1, -1)
                    i2 = i2.expand(-1, 3, -1, -1)

                _, _, H, W = i1.size()
                H_, W_ = int(np.ceil(H/32)*32), int(np.ceil(W/32)*32)
                pader = torch.nn.ReplicationPad2d([0, W_-W, 0, H_-H])
                i1, i2 = pader(i1), pader(i2)

                # 1.png -> 10.png
                new_img1 = path+temp_sub_folder_dest+'/' + \
                    onlyfiles[b][:-4]+'0'+onlyfiles[b][-4:]
                # 1.png + 2.png -> 11.png
                new_img3 = path+temp_sub_folder_dest+'/' + \
                    onlyfiles[b][:-4]+'1'+onlyfiles[b][-4:]
                # 2.png -> 20.png
                new_img2 = path+temp_sub_folder_dest+'/' + \
                    onlyfiles[b+1][:-4]+'0'+onlyfiles[b+1][-4:]

                output = model(i1, i2)
                output = output[0, :, 0:H, 0:W].squeeze(0).cpu()
                output = transforms.functional.to_pil_image(output)
                i1.save(new_img1)
                output.save(new_img3)
                i1.save(new_img2)

                print(datetime.now(), new_img3)


if __name__ == '__main__':
    main()
