import json
import numpy as np

BASE = "/home/maqima/VLM-Visualizer/data/spatial_twoshapes/agreement/relational/test/shard0"
agreement_path = f"{BASE}/agreement.txt"
caption_path = f"{BASE}/caption.txt"
agreement = [float(x.strip()) for x in open(agreement_path).readlines()]
captions = [x.strip() for x in open(caption_path).readlines()]
filtered_ids = [i for i, a in enumerate(agreement) if a == 1.0]
# filter by "left" and "right" in captions
final_ids = []
for i in filtered_ids:
    caption = captions[i].lower()
    if "shape" in caption:
        continue
    if " left " in caption or caption.startswith("left " ) or " left." in caption or " left," in caption:
        final_ids.append(i)
    elif " right " in caption or caption.startswith("right ") or " right." in caption or " right," in caption:
        final_ids.append(i)
print(f"Total filtered ids: {len(final_ids)}")
with open(f"{BASE}/filtered_ids_left_right.txt", "w") as f:
    for i in final_ids:
        f.write(f"{i}\n")
# print each caption
for i in final_ids:
    print(f"{i}: {captions[i]}")