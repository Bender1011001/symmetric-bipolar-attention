import torch
import sys

ckpt_path = "sba_standalone/results/homotopy_conversion_v5/homotopy_step004000_12l_12h_768d.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
print("Format:", ckpt.get("format"))
print("Attention:", ckpt.get("attention"))
print("Keys:", list(ckpt.keys()))
if "args" in ckpt:
    print("Args:", ckpt["args"])
