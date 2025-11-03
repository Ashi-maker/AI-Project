from utils import coco_to_object_list, copy_filtered_images, draw_bounding_boxes, save_yolo_labels
from pprint import pprint
import os
import random
import shutil

def split_train_val(image_folder="images", label_folder="labels", train_ratio=0.8):
    """
    Splits images and corresponding YOLO labels into train and val folders.
    
    Only images with labels are moved, preventing FileNotFoundError.
    """
    os.makedirs(os.path.join(image_folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(image_folder, "val"), exist_ok=True)
    os.makedirs(os.path.join(label_folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(label_folder, "val"), exist_ok=True)

    all_images = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))]
    # Keep only images that have a corresponding label
    images_with_labels = [
        img for img in all_images 
        if os.path.exists(os.path.join(label_folder, os.path.splitext(img)[0] + ".txt"))
    ]

    if not images_with_labels:
        print("No images with labels found. Check your label generation step!")
        return

    random.shuffle(images_with_labels)

    train_count = int(len(images_with_labels) * train_ratio)
    train_images = images_with_labels[:train_count]
    val_images = images_with_labels[train_count:]

    # Move images and labels into train and val folders
    for img_name in train_images:
        shutil.move(os.path.join(image_folder, img_name), os.path.join(image_folder, "train", img_name))
        label_name = os.path.splitext(img_name)[0] + ".txt"
        shutil.move(os.path.join(label_folder, label_name), os.path.join(label_folder, "train", label_name))

    for img_name in val_images:
        shutil.move(os.path.join(image_folder, img_name), os.path.join(image_folder, "val", img_name))
        label_name = os.path.splitext(img_name)[0] + ".txt"
        shutil.move(os.path.join(label_folder, label_name), os.path.join(label_folder, "val", label_name))

    print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}")

if __name__ == "__main__":
    # Convert COCO annotations (only apple, banana, orange)
    data = coco_to_object_list("annotations.json")

    # Print some info about filtered images
    print(f"Found {len(data)} images with banana/apple/orange:\n")
    for img_name, objects in data.items():
        print(f"{img_name}:")
        pprint(objects)
        print()

    # Copy filtered images from COCO train folder to local 'images/' folder
    coco_train_path = "/Users/ashigupta/Downloads/COCO2017/train2017"  # adjust your path
    copy_filtered_images(data, coco_train_path)
    print("\nFiltered images copied to 'images/' folder!")

    # Draw bounding boxes on copied images for visualization/debug (optional)
    draw_bounding_boxes("images", data, save_folder="outputs")
    print("\nBounding boxes drawn on images (saved in 'outputs/')!")

    # Generate YOLO-format labels for training
    save_yolo_labels(data, image_folder="images", save_folder="labels")
    print("\nYOLO label files saved to 'labels/' folder!")

    # Split images and labels into train and val sets
    split_train_val(image_folder="images", label_folder="labels", train_ratio=0.8)
    print("\nTrain/validation split done! Check 'images/train', 'images/val', 'labels/train', 'labels/val'")
