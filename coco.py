import numpy as np

from pycocotools.coco import COCO
import PIL.Image
import shutil

TYPE = 'val'

FOLDER = TYPE + '2017'
ANN_FILE = 'annotations/instances_{}.json'.format(FOLDER)

coco = COCO(annotation_file=ANN_FILE)

img_ids = coco.getImgIds(catIds=1) # All image ids with person
print('There are originally', len(img_ids), 'pictures with people in them.')

total_area = 0

def should_keep(img):
	"""
	We only keep images that have 1 person in them
	Anything else would be less important to train on
	If lb < area of person < ub, we keep the image and save to disk
	Otherwise, we throw it away
	"""

	lb = 20
	ub = 70

	annIds = coco.getAnnIds(imgIds=img['id'])
	anns = coco.loadAnns(annIds)
	people_count = np.sum([ann['category_id'] == 1 for ann in anns])
	
	if people_count != 1:
		return False
		
	anns = list(filter(lambda ann: ann['category_id'] == 1, anns))

	ann1 = anns[0]
	img1 = coco.loadImgs(ann1['image_id'])[0]
	img1_area = img1['height'] * img1['width']
	img1_ratio = ann1['area'] / img1_area * 100

	if not(lb < img1_ratio < ub):
		return False 
		
	mask = coco.annToMask(anns[0])

	mask_save = PIL.Image.fromarray(mask)
	
	mask_save.save('{}_mask/mask/{}'.format(TYPE, img['file_name']))
	shutil.copy2('{}2017/{}'.format(TYPE, img['file_name']), '{}_img/img/{}'.format(TYPE, img['file_name']))

	global total_area
	total_area += img1_ratio

	return True

cnt = 0
for i in img_ids:
	img = coco.loadImgs(i)[0]
	cnt += should_keep(img)

total_area /= cnt
print('There are {} images after filtering.'.format(cnt))
print('The average area is {}'.format(total_area))
