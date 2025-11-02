from utils import coco_to_object_list, copy_filtered_images
from pprint import pprint

if __name__ == "__main__":
    # 1️⃣ Convert annotations (filtered)
    data = coco_to_object_list("annotations.json")

    # 2️⃣ Print some info
    print(f"Found {len(data)} images with banana/apple/orange:\n")
    for img_name, objects in data.items():
        print(f"{img_name}:")
        pprint(objects)
        print()

    # 3️⃣ Copy only filtered images frcd ~/codingvscode/AI\ Projectom COCO train folder
    coco_train_path = "/Users/ashigupta/Downloads/COCO2017/train2017"  # adjust your path
    copy_filtered_images(data, coco_train_path)
    print("\nFiltered images copied to 'images/' folder!")
