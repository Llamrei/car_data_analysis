import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

result_folder = Path(sys.argv[1])
method = result_folder.stem
result_files = sorted(result_folder.glob("*.csv"))

runtimes = [result_files[0].stem.split("_")[-1] + "\n"]
all_results = pd.read_csv(result_files[0])
i = 0
all_results["exp"] = i

for result_file in tqdm(result_files[1:]):
    i += 1
    next_exp = pd.read_csv(result_file)
    next_exp["exp"] = i
    all_results = pd.concat([all_results, next_exp])
    runtimes.append(result_file.stem.split("_")[-1] + '\n')

all_results[all_results["test"] == 1].to_csv(result_folder.parent / f"{method}_test.csv", index=False)
all_results[all_results["test"] == 0].to_csv(result_folder.parent / f"{method}_train.csv", index=False)

with open(result_folder.parent / f"{method}_runtimes.csv", "w") as f:
    f.write("seconds\n")
    f.writelines(runtimes)
print("done")


