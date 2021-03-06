import json as js
import os
import cv2
import numpy as np
import math

inno_dict_all = {
    'info':{'url': 'icdar', 'version': '1.0', 
            'year': '2015', 'contributor':'icdar', 'data_created':'2015'},
    
    'images':[],

    'licenses':[{'url':'icdar', 'id':'xxxxx', 'name':'icdar'}],

    'type':'instances',

    'annotations':[],

    'categories': [{"supercategory": "Text", "id": 1, "name": "text_block"}, {"supercategory": "Text", "id": 2, "name": "confused"}]
}

dataset_dir = './data/icdar/icdar15/'

index = 0
state = 'test'
if state =='train':
    instance_id = 100000
elif state == 'test':
    instance_id = 500000

for _, _, files in os.walk(dataset_dir + state + '/'):
    for file_name in files:
        if file_name.split('.')[-1]=='jpg':

            # 'images' part
            img_dict = {'license':'xxx', 
                        'url':'xxx',
                        'date_capture':'xxx'}

            img_dict['file_name'] = file_name

            img = cv2.imread(dataset_dir + state +'/'+ file_name)
            img_dict['width'] = img.shape[1]
            img_dict['height'] = img.shape[0]

            img_index = int(file_name.split('_')[1].split('.')[0])
            if state == 'train':
                img_dict['id'] = img_index + 10000
            elif state == 'test':
                img_dict['id'] = img_index + 20000

            inno_dict_all['images'].append(img_dict)
            print img_dict

            # annotations part
            gt_file = open(dataset_dir + state+'/anno/gt_img_'+str(img_index)+'.txt', 'r')
            gtlines = gt_file.readlines()
            gt_file.close()
            for line in gtlines:
                # init instance
                anno_dict ={'segmentation':[],
                            'area':0.0,
                            'iscrowd':0,
                            'image_id':0,
                            'bbox':[],
                            'category_id':1,
                            'id':0}
                if '\xef\xbb\xbf'  in line:
                    line = line.replace('\xef\xbb\xbf','') 

                skip = 0

                word = line.split(',')[-1]
                
                if word == '###\r\n':
                    # skip = 1
                    anno_dict['category_id'] = 2  #confused words

                str_points = line.split(',')[:8]
                points = map(int, str_points)

                # caculate polygon
                polygon = []
                # 1
                step = 0
                if (points[2]-points[0]) != 0:
                    deltay = float(points[3]-points[1])/(points[2]-points[0])
                    for i in range(points[0],points[2]):
                        polygon.append(i)
                        polygon.append(round(points[1] + deltay*step, 2))
                        step = step + 1
                else:
                    skip = 1 
                #2
                step= 0
                if (points[5]-points[3])!=0:
                    deltax = float(points[4]-points[2])/(points[5]-points[3])
                    for i in range(points[3],points[5]):
                        polygon.append(round(points[2] + deltax*step, 2))
                        polygon.append(i)
                        step = step + 1
                else:
                    skip = 1
                
                # 3 
                step = 0
                if (points[4]-points[6]) !=0:
                    deltay = float(points[7]-points[5])/(points[4]-points[6])
                    for i in range(points[6],points[4]):
                        polygon.append(points[4] - i + points[6])
                        polygon.append(round(points[5] + deltay*step, 2))
                        step = step + 1
                else:
                    skip = 1
                
                #4
                step= 0
                if (points[7]-points[1]) != 0:
                    deltax = float(points[0]-points[6])/(points[7]-points[1])
                    for i in range(points[1],points[7]):
                        polygon.append(round(points[6] + deltax*step, 2))
                        polygon.append(points[7] - i + points[1])
                        step = step + 1
                else:
                    skip = 1

                anno_dict['segmentation'].append(polygon)
                # print anno_dict['segmentation']
                
                # caculate area
                points_offset = points[2:]
                points_offset.extend(points[:2])
                square = map(lambda (a,b) : (a-b)*(a-b), zip(points, points_offset))
                edge = map(lambda(x,y):round(math.sqrt(x+y), 2), zip(square[::2], square[1::2]))
                hypotenuse = round(math.sqrt(pow(points[2] - points[6],2) + pow(points[3]-points[7], 2)), 2)

                p1 = 0.5*(edge[0] + edge[3] + hypotenuse)
                if p1* (p1-edge[0])* (p1-edge[3])* (p1-hypotenuse) > 0:
                    area1 = math.sqrt(p1* (p1-edge[0])* (p1-edge[3])* (p1-hypotenuse))  
                else:
                    area1 = 0 

                p2 = 0.5*(edge[2] + edge[1] + hypotenuse)         
                if p2* (p2-edge[2])* (p2-edge[1])* (p2-hypotenuse) > 0: 
                    area2 = math.sqrt(p2* (p2-edge[2])* (p2-edge[1])* (p2-hypotenuse))
                else:
                    area2 = 0

                anno_dict['area'] = round(area1 + area2, 2)
            
                anno_dict['image_id'] = img_dict['id']

                px = points[::2]
                py = points[1::2]
                anno_dict['bbox'] = [min(px), min(py), max(px)-min(px), max(py)-min(py)]

                anno_dict['id'] = instance_id
                instance_id = instance_id + 1

                if skip == 0:
                    inno_dict_all['annotations'].append(anno_dict)
                
            index = index + 1


print index
print instance_id

with open(dataset_dir + 'anno_'+state+'_icdar15.json', 'w') as jsonfile:
    js.dump(inno_dict_all,jsonfile)
    print 'finish!'    

