"""
Could dramatically speed up script through parallelising. And/or async on writes
"""

from datetime import datetime
import io
from pathlib import Path
import pickle as pkl
import re
import csv
from multiprocessing import Manager
import tqdm

from PIL import Image

# DESC_SAVE_PATH
STRUC_SAVE_PATH = "final_project_data/data.csv"
SOURCE_DATA_PATH = '/mnt/data/car_auction_data'

def process_file(args: tuple) -> list:
    f_idx = args[0]
    filepath = args[1]
    files_processed= args[2]
    duplicates_seen = args[3]
    duplicates_tracker = args[4]
    tot_files = args[5]

    duplicates_count = 0
    new_rows = []

    data = pkl.load(open(filepath,'rb'))
    
    for d_idx, row in enumerate(data):
        # Extract relevant data from this ad
        # id needed for tracking results through various experiments
        id = f"{f_idx}-{d_idx}"
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
        # The data structure we will use to store keys of interest
        # Check if this data is duplicate of anything seen so far
        # done with a check on description and price tuple
        dup_check = (to_add["desc"], to_add["price"])
        if dup_check in duplicates_tracker.keys():
            duplicates_tracker[dup_check].append((filepath, d_idx))
            duplicates_count += 1
        else:
            duplicates_tracker[dup_check] = [(filepath,d_idx)]
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

    # with files_processed.get_lock():
    #     files_processed.value += 1
    # with duplicates_seen.get_lock():
    #     duplicates_seen.value += duplicates_count
    return new_rows

if __name__ == "__main__":
    data_dir = Path(SOURCE_DATA_PATH)

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
   
    with Manager() as man:
        files_processed = man.Value('i', 0)
        duplicates_seen = man.Value('i', 0)
        duplicates_tracker = man.dict()
        flat_new_rows = []
        print("Entering process pool")
        print(datetime.now())
        with man.Pool(processes=6) as pool:
            args = [(i, f, files_processed, duplicates_seen, duplicates_tracker, tot_files) for i, f in enumerate(data_files)]
            for result in tqdm.tqdm(pool.imap_unordered(process_file, args), total=len(args)):
                flat_new_rows.extend(result)
    
    # flat_new_rows = [item for sublist in new_rows_iter for item in sublist]
    print(datetime.now())
    print(f"\nMultiprocessing done, retrieved {flat_new_rows} ads")
    
    # Write log on what has been duplicated
    # with open("duplicate_summary_v2.txt", "w") as f:
    #     for k in desc_price_pairs_seen:
    #         if len(desc_price_pairs_seen[k]) > 1:
    #             f.write(f'{k}:{desc_price_pairs_seen[k]} \n')

    # Write out pickle compatible with previous experiments - no need to do any special processing as we extract it later
    # pkl.dump(new_rows,open(DESC_SAVE_PATH, "wb"))

    # Write out structured data in csv format for loading in R
    with open(STRUC_SAVE_PATH, "w") as f:
        writer = csv.DictWriter(f, fieldnames=flat_new_rows[0].keys() )
        writer.writeheader()
        writer.writerows(flat_new_rows)