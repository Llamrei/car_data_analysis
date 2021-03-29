from pathlib import Path
import pickle as pkl

data_dir = Path('/mnt/data/car_auction_data')

data_files =  list(data_dir.glob('backup*'))
tot_files = len(data_files)
# Load interrupted transcription
# Remove files we've already seen from the data_files

new_rows = []
for f_idx, f in enumerate(data_files):
    data = pkl.load(open(f,'rb'))
    tot_file_data = len(data)
    # Could vectorise but with such small data size it is not worth it
    for d_idx, row in enumerate(data):
        new_rows.append({"desc":row["description"], "price":row["price_float"]})
        print(f"\r Loaded File {f_idx+1}/{tot_files} | Row {d_idx+1}/{tot_file_data}            ", end="")

pkl.dump(new_rows,open("complete_car_data.pickle", "wb"))