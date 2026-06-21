#gpt dixit
from __future__ import annotations

import argparse
import torch
from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path

# recursive function that takes prints `value`
def summarize(value, max_depth, depth = 0):
    indent = "  " * depth
    max_items = 340

    #base case
    if depth >= max_depth:
        print(f"{indent}{type(value).__name__}")
        return

    # for tensors
    if isinstance(value, torch.Tensor):
        print(f"{indent}Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})")
        return

    #for dictionaries
    if isinstance(value, Mapping):
        print(f"{indent}{type(value).__name__} with {len(value)} keys")
        for key, item in islice(value.items(),max_items):
            print(f"{indent}- {key!r}")
            summarize(item, max_depth=max_depth, depth=depth + 1)
        if len(value) > max_items:
            print(f"{indent} + {len(value) - max_items} more ....")
        return

    #for arrays/sequences
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for idx, item in enumerate(value[:max_items]):
            print(f"{indent}- [{idx}]")
            summarize(item, max_depth=max_depth, depth=depth + 1)
        if len(value) > max_items:
            print(f"{indent} + {len(value) - max_items} more ....")
        return

    # for obj classes that have attributes
    if hasattr(value, "__dict__"):
        attributes = vars(value)
        print(f"{indent}{type(value).__name__} with {len(attributes)} attributes")
        for name, item in attributes.items():
            print(f"{indent}- .{name!r}")
            summarize(item, depth=depth + 1, max_depth=max_depth)
        return

    #default
    print(f"{indent}{type(value).__name__}: {value!r}")

def main():
    parser = argparse.ArgumentParser()
    # requires a path as an argument when executing the file
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-depth", type=int)
    # parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        print("give me a file ....")

    # map_location=cpu just to make sure to not load them in the GPU for no reason
    # weights_only = True loads only safe stuff, not custom objects (we trust this class and run it this way otherwise pickled data could execute code
    # during deserialization
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    summarize(checkpoint, max_depth=args.max_depth)

if __name__ == "__main__":
    main()
