from lvis_vis import LVISVis
from lvis.lvis import LVIS
json_path = "data/coco/annotations/instances_val2017.json.LVIS"
image_dir = "/mnt/data2/wlt/open_set_new/OpenSet-dev/data/coco/val2017"
lvis_gt = LVIS(json_path)
lvis_vis = LVISVis(lvis_gt,img_dir = image_dir)
random = 1
# import ipdb;ipdb.set_trace()
image_id = lvis_gt.dataset['images'][random]['id']
fg=lvis_vis.vis_img(image_id,show_boxes=True,show_classes=True,show_segms=False)
fg.savefig('./temp.jpg')



