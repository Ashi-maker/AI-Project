import json   # to read and write JSON files
import shutil # to copy or move files
import os     # to work with file paths and folders
import cv2    # OpenCV library for image processing


# We only care about these three classes
ALLOWED_CLASSES = {"banana", "apple", "orange"}


def coco_to_object_list(coco_json_path):
    # Step 1: Open the COCO annotation file and load it as a Python dictionary
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # Step 2: Create a mapping from category IDs to their names
    # COCO stores objects by numeric IDs, so we need this to get readable names
    cat_map = {cat['id']: cat['name'] for cat in coco['categories']}
    
    # This dictionary will hold lists of objects for each image ID
    img_ann_map = {}

    # Step 3: Loop through all annotations and extract the objects we care about
    for ann in coco['annotations']:
        img_id = ann['image_id']            # get the image ID for this annotation
        obj_class = cat_map[ann['category_id']]  # convert category ID to name

        # Skip any object that is not in our allowed classes
        if obj_class not in ALLOWED_CLASSES:
            continue

        # Make sure there's a list to store objects for this image
        if img_id not in img_ann_map:
            img_ann_map[img_id] = []

        # COCO bounding box format: [x_min, y_min, width, height]
        x_min, y_min, width, height = ann['bbox']

        # Convert to center-based coordinates (required for YOLO)
        obj = {
            "class": obj_class,
            "objectCenterX": int(x_min + width / 2),
            "objectCenterY": int(y_min + height / 2),
            "boundingBoxWidth": int(width),
            "boundingBoxHeight": int(height)
        }

        # Add this object to the image's list
        img_ann_map[img_id].append(obj)

    # Step 4: Create a dictionary that maps image file names to their objects
    output = {}
    for img in coco['images']:
        objects = img_ann_map.get(img['id'], [])
        if objects:
            output[img['file_name']] = objects

    # Return a dictionary where keys are image file names and values are object lists
    return output


def copy_filtered_images(data_dict, source_dir, dest_dir="images"):
    # Make sure the destination folder exists
    os.makedirs(dest_dir, exist_ok=True)

    # Loop through all filtered images and copy them
    for img_name in data_dict.keys():
        src_path = os.path.join(source_dir, img_name)  # original location
        dst_path = os.path.join(dest_dir, img_name)    # where we want to copy it
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            # Warn if the source image does not exist
            print(f"Warning: {src_path} does not exist")


def draw_bounding_boxes(image_folder, annotations, save_folder=None):
    """
    Draw bounding boxes on images for visualization.

    image_folder: folder with images
    annotations: dictionary output from coco_to_object_list
    save_folder: if given, save images with boxes there; otherwise, display them
    """
    # If saving images, make sure the folder exists
    os.makedirs(save_folder, exist_ok=True) if save_folder else None

    for img_name, objects in annotations.items():
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            continue

        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Draw all bounding boxes for this image
        for obj in objects:
            # Convert center coordinates back to top-left for drawing
            x = int(obj['objectCenterX'] - obj['boundingBoxWidth']/2)
            y = int(obj['objectCenterY'] - obj['boundingBoxHeight']/2)
            w = obj['boundingBoxWidth']
            h = obj['boundingBoxHeight']
            cls = obj['class']

            # Draw the rectangle and class name
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, cls, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save or display the image
        if save_folder:
            cv2.imwrite(os.path.join(save_folder, img_name), img)
        else:
            cv2.imshow('Image', img)
            cv2.waitKey(0)

    if not save_folder:
        cv2.destroyAllWindows()


def save_yolo_labels(annotations, image_folder, save_folder="labels"):
    """
    Convert COCO-style object lists to YOLO-format .txt files.

    annotations: output of coco_to_object_list
    image_folder: path to images (needed to normalize coordinates)
    save_folder: folder to save YOLO .txt labels
    """
    os.makedirs(save_folder, exist_ok=True)

    # YOLO requires class numbers instead of names
    class_map = {"apple": 0, "banana": 1, "orange": 2}

    for img_name, objects in annotations.items():
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            continue

        # Read the image to get its width and height
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        lines = []
        for obj in objects:
            # Map class name to number
            cls_idx = class_map[obj['class']]

            # Normalize bounding box coordinates (0-1) for YOLO
            x_center = obj['objectCenterX'] / w
            y_center = obj['objectCenterY'] / h
            width    = obj['boundingBoxWidth'] / w
            height   = obj['boundingBoxHeight'] / h

            # Each line in YOLO format: class x_center y_center width height
            lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save label file with same name as image
        label_file = os.path.join(save_folder, os.path.splitext(img_name)[0] + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(lines))
