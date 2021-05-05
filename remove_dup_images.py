import pandas as pd
from pathlib import Path
import os

data = pd.read_csv("/home/alexander/projects/car_data_analysis/final_project_data/deduped_data.csv")
valid_ids = data["id"]
assert len(valid_ids) == len(set(valid_ids))
valid_ids = set(valid_ids)

image_dir = Path("/home/alexander/projects/car_data_analysis/final_project_data/images")
image_names = image_dir.glob("*.jpeg")
image_names = [x.stem for x in image_names]
assert len(image_names) == len(set(image_names))
image_names = set(image_names)

if image_names == valid_ids:
    print("All good")
else:
    images_to_remove = image_names - valid_ids
    images_to_remove = [image_dir/f"{stem}.jpeg" for stem in images_to_remove]

    for file in images_to_remove:
        os.rename(file, file.parents[1]/f"dup_images/{file.name}")

    print(f"Removed {len(images_to_remove)} images")