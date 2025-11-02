import json
import shutil
import os

ALLOWED_CLASSES = {"banana", "apple", "orange"}

def coco_to_object_list(coco_json_path):
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    cat_map = {cat['id']: cat['name'] for cat in coco['categories']}
    img_ann_map = {}

    for ann in coco['annotations']:
        img_id = ann['image_id']
        obj_class = cat_map[ann['category_id']]

        if obj_class not in ALLOWED_CLASSES:
            continue

        if img_id not in img_ann_map:
            img_ann_map[img_id] = []

        x_min, y_min, width, height = ann['bbox']
        obj = {
            "class": obj_class,
            "objectCenterX": int(x_min + width / 2),
            "objectCenterY": int(y_min + height / 2),
            "boundingBoxWidth": int(width),
            "boundingBoxHeight": int(height)
        }
        img_ann_map[img_id].append(obj)

    output = {}
    for img in coco['images']:
        objects = img_ann_map.get(img['id'], [])
        if objects:
            output[img['file_name']] = objects

    return output

def copy_filtered_images(data_dict, source_dir, dest_dir="images"):
    os.makedirs(dest_dir, exist_ok=True)
    for img_name in data_dict.keys():
        src_path = os.path.join(source_dir, img_name)
        dst_path = os.path.join(dest_dir, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: {src_path} does not exist")
