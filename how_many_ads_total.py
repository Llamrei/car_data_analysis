from pathlib import Path

backup_dir = Path('/mnt/data/car_auction_data/')
print(len(list(backup_dir.glob("backup*"))))
total = 0
for file in backup_dir.glob("backup*"):
    end = int(str(file).split("_")[-1].split(".")[0])
    start = int(str(file).split("_")[-2])
    print(f"\r{file} {start} {end}             ")
    total += end - start 

print(total)