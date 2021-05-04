"""
Could dramatically speed up script through parallelising. And/or async on writes
"""

import io
from pathlib import Path
import pickle as pkl
import re
import csv
from multiprocessing import Pool

from PIL import Image

# DESC_SAVE_PATH
STRUC_SAVE_PATH = "final_project_data/data.csv"

data_dir = Path('/mnt/data/car_auction_data')

data_files =  list(data_dir.glob('backup*'))
data_files = sorted(data_files)
# If I wanted a subsample here is where I would take it
data_files = data_files[:]
tot_files = len(data_files)

# After extraction how do we process keys
pre_process = {
    'mileage': lambda x: float(re.sub('[^(0-9|.)]','',x)),
    'manufactured-year' : lambda x: int(x.split(' ')[0]),
    'engine-size' : lambda x: float(re.sub('[^(0-9|.)]','',x)),
    'owners' : lambda x: int(x.split(' ')[0]),
    'price' : lambda x:x,
    'desc' : lambda x:x,
    'make' : lambda x:x,
    'id': lambda x:x,
}

# Mapping between (price, desc) to files where it appears
desc_price_pairs_seen = dict()
new_rows = []
duplicates_count = 0
id = 0

# Going through all backups made
for f_idx, f in enumerate(data_files):
    data = pkl.load(open(f,'rb'))
    tot_file_data = len(data)

    for d_idx, row in enumerate(data):
        # Extract relevant data from this ad
        # id needed for tracking results through various experiments
        to_add = {
            "mileage":row['vehicle']['keyFacts'].get("mileage", None),
            "manufactured-year":row['vehicle']['keyFacts'].get("manufactured-year", None),
            "engine-size":row['vehicle']['keyFacts'].get("engine-size", None),
            "owners":row['vehicle']['keyFacts'].get("owners", None),
            "desc":row["description"], 
            "price":row["price_float"],
            "make":row['vehicle']['make'],
            "id":id
        }
        id += 1
        # The data structure we will use to store keys of interest
        # Check if this data is duplicate of anything seen so far
        # done with a check on description and price tuple
        dup_check = (to_add["desc"], to_add["price"])
        if dup_check in desc_price_pairs_seen.keys():
            desc_price_pairs_seen[dup_check].append((f, d_idx))
            duplicates_count += 1
        else:
            desc_price_pairs_seen[dup_check] = [(f,d_idx)]
            # If not duplicate we treat this ad as worthy of being further processed
            for key in pre_process:
                if to_add[key] is not None:
                    to_add[key] = pre_process[key](to_add[key])
            if len(row["images"]) > 0:
                # only want to record if we also have an image associated with it
                new_rows.append(to_add)
                raw_im = row["images"][0]
                im_f = io.BytesIO(raw_im)
                im = Image.open(im_f).convert('RGB')
                im.save(open(f'final_project_data/images/{id}.jpeg', "wb"), "JPEG", quality=85)

        # Print progress
        print(f"\r Loaded File {f_idx+1}/{tot_files} | Row {d_idx+1}/{tot_file_data}  | Duplicates seen {duplicates_count}          ", end="")


# Write log on what has been duplicated
with open("duplicate_summary_v2.txt", "w") as f:
    for k in desc_price_pairs_seen:
        if len(desc_price_pairs_seen[k]) > 1:
            f.write(f'{k}:{desc_price_pairs_seen[k]} \n')

# Write out pickle compatible with previous experiments - no need to do any special processing as we extract it later
# pkl.dump(new_rows,open(DESC_SAVE_PATH, "wb"))

# Write out structured data in csv format for loading in R
with open(STRUC_SAVE_PATH, "w") as f:
    writer = csv.DictWriter(f, fieldnames=new_rows[0].keys() )
    writer.writeheader()
    writer.writerows(new_rows)