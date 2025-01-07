import torch
from torch.utils.data import random_split
import json

mp_split= torch.tensor([78600, 4367, 4367],dtype=torch.long)
gen=torch.Generator().manual_seed(42)
train_mp, valid_mp, test_mp = random_split(
    list(range(mp_split.sum())), mp_split, generator=gen
)

oqmd_split= torch.tensor([199686, 11094, 11094],dtype=torch.long)
gen=torch.Generator().manual_seed(42)
train_oqmd, valid_oqmd, test_oqmd = random_split(
    list(range(oqmd_split.sum())), oqmd_split, generator=gen
)

split = {
    "mp":{
        "train":list(map(int,train_mp.indices)),
        "valid":list(map(int,valid_mp.indices)),
        "test":list(map(int,test_mp.indices)),
    },
    "oqmd":{
        "train":list(map(int,train_oqmd.indices)),
        "valid":list(map(int,valid_oqmd.indices)),
        "test":list(map(int,test_oqmd.indices)),
    },
}

with open("split.json","w") as fp:
    json.dump(split,fp)

