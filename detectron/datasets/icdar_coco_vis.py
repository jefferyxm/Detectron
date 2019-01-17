from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import matplotlib.pyplot as plt
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

state = 'train'

data_dir = './data/icdar/icdar15/'
anno_file = data_dir + 'annotations/anno_' + state + '_icdar15.json'

coco = COCO(anno_file)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print 'COCO categories: \n\n', ' '.join(nms)

nms = set([cat['supercategory'] for cat in cats])
print 'COCO supercategories: \n', ' '.join(nms)

catIds = coco.getCatIds(catNms=['text_block'])
imgIds = coco.getImgIds(catIds=catIds )

print imgIds

cnt = 0
for i in imgIds:
    
    cnt = cnt + 1
    if cnt >1000:
        img = coco.loadImgs(i)[0]
        print img
        I = io.imread(data_dir + state + '/' + img['file_name'])
        plt.figure(); plt.axis('off')
        # plt.imshow(I)
        # plt.show()

        plt.imshow(I); plt.axis('off')
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        coco.showAnns(anns)
        for ann in anns:
            bbox = ann['bbox']
            plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], edgecolor='r', fill=False, linewidth=2))
        plt.show()
        # plt.savefig('/home/xiem/tmp/gt/val/gt_'+img['file_name'])
        # plt.savefig('/home/xiem/tmp/gt2/train/gt_'+img['file_name'])
        # plt.show()
