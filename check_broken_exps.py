from pathlib import Path
from re import T
from tqdm import tqdm
import pandas as pd
import sys

results_dir = Path(sys.argv[1])
results = []
broken_exps = []
warning_exps = []
runtimes = []
broken = False
methods = {"RF", "NLP", "NN", "GLM", "LM", "CNN"}
for exp_folder in tqdm(results_dir.glob("*")):
    if exp_folder.is_dir():
        method_folders = list(exp_folder.glob("*"))
        if len(method_folders) != len(methods):
            frame = pd.DataFrame(data=[{"reason":f"{methods - set([x.stem for x in method_folders])} missing",
                    "method":"?",
                    "exp":exp_folder.stem}])
            broken_exps.append(frame)
            broken = True
        for method_folder in method_folders:
            if method_folder.is_dir():
                results_files = sorted(list(method_folder.glob("*.csv")), reverse=True)
                if len(results_files) != 1:
                    frame = pd.DataFrame(data=[{"reason":f"{len(results_files)} results",
                    "method":method_folder.stem,
                    "exp":exp_folder.stem}])
                    if len(results_files) > 1:
                        warning_exps.append(frame)
                    else:
                        broken_exps.append(frame)
                        broken = True
                
                if not broken:
                    this_results = pd.read_csv(results_files[0])
                    this_results["method"] = method_folder.stem
                    this_results["exp"] = exp_folder.stem
                    results.append(this_results)
                    runtimes.append(
                        pd.DataFrame(
                            data=[{"method":method_folder.stem, 
                                "runtime":int(results_files[0].stem.split("_")[-1])}]
                    ))
                broken = False

results = pd.concat(results)
broken_exps = pd.concat(broken_exps).reset_index(drop=True)
warning_exps = pd.concat(warning_exps).reset_index(drop=True)
runtimes= pd.concat(runtimes).reset_index(drop=True)

print(broken_exps)
print(sorted([int(x) for x in broken_exps["exp"].values]))
print(warning_exps)
print("-"*80)
# aggregate = results.groupby(["method","exp"]).count()
# print(aggregate[aggregate["id"]!=34831])
# print("-"*80)
print(runtimes.groupby(["method"]).mean())
print(runtimes.groupby(["method"]).std())
runtimes.to_csv("combiner_runtimes.csv")
# Mean
#              runtime
# method              
# CNN     89561.635135
# GLM         2.962500
# LM          2.385542
# NLP       887.220000
# NN         43.050000
# RF         44.947368

# STD
#              runtime
# method              
# CNN     19145.035397
# GLM         1.095951
# LM          0.537186
# NLP       375.689066
# NN         14.175933
# RF         24.737494
