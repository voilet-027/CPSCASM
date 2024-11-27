import pandas as pd
import os
import math
from scipy.spatial.distance import euclidean

root = '/home/datasets/ALTO/Val'

query = pd.read_csv(os.path.join(root, "query.csv"))
ref = pd.read_csv(os.path.join(root, "reference.csv"))
match = pd.read_csv(os.path.join(root, "gt_matches.csv"))

query_coords = list(zip(query["easting"], query["northing"]))
ref_coords = list(zip(ref["easting"], ref["northing"]))

eps = 1e-5

for i in range(len(match)):
    query_id = int(match.iloc[i]["query_ind"])
    ref_id = int(match.iloc[i]["ref_ind"])
    gt_dis = match.iloc[i]["distance"]

    cal_dis = euclidean(query_coords[query_id], ref_coords[ref_id])

    if abs(cal_dis - gt_dis) > eps:
        print(f"{i}, {query_id} {ref_id}")


