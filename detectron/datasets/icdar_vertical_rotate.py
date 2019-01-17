import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np


ori_img_root = 'data/icdar/icdar15/train/'
ori_gt_root = ori_img_root + 'anno/'

img_save_dir = 'data/icdar/icdar15/train_aug/'
gt_save_dir = img_save_dir + 'anno/'


def RotateClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip(trans_img, 1)
    return new_img

def RotateAntiClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip( trans_img, 0 )
    return new_img


for _, _, files in os.walk(ori_img_root):
    for file_name in files:
        if file_name.split('.')[-1]=='jpg':

            print(file_name)
            
            im = cv2.imread(ori_img_root + file_name)
            im_width = im.shape[1]
            im_height = im.shape[0]

            img_index = int(file_name.split('_')[1].split('.')[0])

            is_clockwise = (img_index % 2)==0

            # 1, rotate image
            if is_clockwise:
                rot_img = RotateClockWise90(im)
            else:
                rot_img = RotateAntiClockWise90(im)
            
            cv2.imwrite(img_save_dir + 'img_' + str(1000 + img_index) + '.jpg', rot_img)

            dbg = 0
            if dbg:
                im_plt = rot_img[:,:,(2,1,0)]
                plt.imshow(im_plt)

            gt_file = open(ori_gt_root + 'gt_img_' + str(img_index)+'.txt', 'r')
            gtlines = gt_file.readlines()
            gt_file.close()

            new_gt_file = open(gt_save_dir + 'gt_img_' + str(1000 + img_index) + '.txt', 'w')

            # rotate points
            for line in gtlines:
                if '\xef\xbb\xbf'  in line:
                    line = line.replace('\xef\xbb\xbf','')

                line_seg = line.split(',')
                points = map(int, line_seg[:8])
                word = line_seg[-1]
 
                if is_clockwise:
                    new_px = im_height - np.array(points[1::2])
                    new_py = np.array(points[::2])
                    new_line = str(new_px[3]) + ',' + str(new_py[3]) + ',' +\
                                str(new_px[0]) + ',' + str(new_py[0]) + ',' +\
                                str(new_px[1]) + ',' + str(new_py[1]) + ',' +\
                                str(new_px[2]) + ',' + str(new_py[2]) + ',' + word
                else:
                    new_px = np.array(points[1::2])
                    new_py = im_width - np.array(points[::2])

                    new_line = str(new_px[1]) + ',' + str(new_py[1]) + ',' +\
                                str(new_px[2]) + ',' + str(new_py[2]) + ',' +\
                                str(new_px[3]) + ',' + str(new_py[3]) + ',' +\
                                str(new_px[0]) + ',' + str(new_py[0]) + ',' + word

                new_gt_file.write(new_line)
                
                if dbg:
                    for i in range(new_px.shape[0]):
                        plt.gca().add_patch(plt.Circle( (new_px[i], new_py[i] ), 1, edgecolor='r', fill=True, linewidth=1))

            if dbg:
                plt.show()

            new_gt_file.close()
            
                
                



