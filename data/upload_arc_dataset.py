from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import os
import json
import hashlib
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel

# Hugging Face imports
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import login, HfApi

# ユーザー環境の data.common が存在することを前提としています
# ファイルがない場合は data/common.py を作成し、必要な関数を定義してください
from data.common import PuzzleDatasetMetadata, dihedral_transform, inverse_dihedral_transform


cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_file_prefix: str
    
    # HF Upload settings
    hf_repo_id: str  # e.g. "username/arc-augmented-dataset"
    hf_token: str = None  # Optional: pass via argument or env var
    private: bool = True  # Create private repo by default

    subsets: List[str]
    test_set_name: str

    seed: int = 42
    num_aug: int = 1000
    
    
ARCMaxGridSize = 30
ARCAugmentRetriesFactor = 5

PuzzleIdSeparator = "|||"
    

@dataclass
class ARCPuzzle:
    id: str
    examples: List[Tuple[np.ndarray, np.ndarray]]

    
def arc_grid_to_np(grid: List[List[int]]):
    arr = np.array(grid)
    # Shape check
    assert arr.ndim == 2
    assert arr.shape[0] <= ARCMaxGridSize and arr.shape[1] <= ARCMaxGridSize
    # Element check
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)


def np_grid_to_seq_translational_augment(inp: np.ndarray, out: np.ndarray, do_translation: bool):
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    # Compute random top-left pad
    if do_translation:
        pad_r = np.random.randint(0, ARCMaxGridSize - max(inp.shape[0], out.shape[0]) + 1)
        pad_c = np.random.randint(0, ARCMaxGridSize - max(inp.shape[1], out.shape[1]) + 1)
    else:
        pad_r = pad_c = 0

    # Pad grid
    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        grid = np.pad(grid + 2, ((pad_r, ARCMaxGridSize - pad_r - nrow), (pad_c, ARCMaxGridSize - pad_c - ncol)), constant_values=0)

        # Add <eos>
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARCMaxGridSize:
            grid[eos_row, pad_c:eos_col] = 1
        if eos_col < ARCMaxGridSize:
            grid[pad_r:eos_row, eos_col] = 1

        result.append(grid.flatten())

    return result


def grid_hash(grid: np.ndarray):
    assert grid.ndim == 2
    assert grid.dtype == np.uint8
    buffer = [x.to_bytes(1) for x in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: dict):
    # Hash the puzzle for checking equivalence
    hashes = []
    for example_type, example in puzzle.items():
        for input, label in example.examples:
            hashes.append(f"{grid_hash(input)}|{grid_hash(label)}")
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def aug(name: str):
    # Augment plan
    trans_id = np.random.randint(0, 8)
    mapping = np.concatenate([np.arange(0, 1, dtype=np.uint8), np.random.permutation(np.arange(1, 10, dtype=np.uint8))])  # Permute colors, Excluding "0" (black)
    
    name_with_aug_repr = f"{name}{PuzzleIdSeparator}t{trans_id}{PuzzleIdSeparator}{''.join(str(x) for x in mapping)}"

    def _map_grid(grid: np.ndarray):
        return dihedral_transform(mapping[grid], trans_id)
    
    return name_with_aug_repr, _map_grid


def inverse_aug(name: str):
    # Inverse the "aug" function
    if PuzzleIdSeparator not in name:
        return name, lambda x: x

    trans_id, perm = name.split(PuzzleIdSeparator)[-2:]
    trans_id = int(trans_id[1:])  # Remove "t" letter
    inv_perm = np.argsort(list(perm)).astype(np.uint8)
    
    def _map_grid(grid: np.ndarray):
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]
    
    return name.split(PuzzleIdSeparator)[0], _map_grid


def convert_single_arc_puzzle(results: dict, name: str, puzzle: dict, aug_count: int, dest_mapping: Dict[str, Tuple[str, str]]):
    # Convert
    dests = set(dest_mapping.values())
    converted = {dest: ARCPuzzle(name, []) for dest in dests}
    for example_type, examples in puzzle.items():
        # Map to target split
        dest = dest_mapping[example_type]
        converted[dest].examples.extend([(arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"])) for example in examples])

    group = [converted]
    
    # Augment
    if aug_count > 0:
        hashes = {puzzle_hash(converted)}

        for _trial in range(ARCAugmentRetriesFactor * aug_count):
            aug_name, _map_grid = aug(name)

            # Check duplicate
            augmented = {dest: ARCPuzzle(aug_name, [(_map_grid(input), _map_grid(label)) for (input, label) in puzzle.examples]) for dest, puzzle in converted.items()}
            h = puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)
                
            if len(group) >= aug_count + 1:
                break
            
        if len(group) < aug_count + 1:
            print (f"[Puzzle {name}] augmentation not full, only {len(group)}")

    # Append
    for dest in dests:
        dest_split, dest_set = dest

        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([converted[dest] for converted in group])


def load_puzzles_arcagi(config: DataProcessConfig):
    train_examples_dest = ("train", "all")
    test_examples_map = {
        config.test_set_name: [(1.0, ("test", "all"))],
        "_default": [(1.0, ("train", "all"))]
    }
    
    test_puzzles = {}
    results = {}

    total_puzzles = 0
    for subset_name in config.subsets:
        # Load all puzzles in this subset
        with open(f"{config.input_file_prefix}_{subset_name}-challenges.json", "r") as f:
            puzzles = json.load(f)

        sols_filename = f"{config.input_file_prefix}_{subset_name}-solutions.json"
        if os.path.isfile(sols_filename):
            with open(sols_filename, "r") as f:
                sols = json.load(f)
                
                for puzzle_id in puzzles.keys():
                    for idx, sol_grid in enumerate(sols[puzzle_id]):
                        puzzles[puzzle_id]["test"][idx]["output"] = sol_grid
        else:
            # Fill with dummy
            print (f"{subset_name} solutions not found, filling with dummy")

            for puzzle_id, puzzle in puzzles.items():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])

        # Shuffle puzzles
        puzzles = list(puzzles.items())
        np.random.shuffle(puzzles)
        
        # Assign by fraction
        for idx, (name, puzzle) in enumerate(puzzles):
            fraction = idx / len(puzzles)
            test_examples_dest = None
            for f, dest in test_examples_map.get(subset_name, test_examples_map["_default"]):
                if fraction < f:
                    test_examples_dest = dest
                    break
                    
            assert test_examples_dest is not None
            
            if test_examples_dest[0] == "test":
                test_puzzles[name] = puzzle
                
            convert_single_arc_puzzle(results, name, puzzle, config.num_aug, {"train": train_examples_dest, "test": test_examples_dest})
            total_puzzles += 1

    print (f"Total puzzles: {total_puzzles}")
    return results, test_puzzles


def generate_readme_content(config: DataProcessConfig, total_examples: int, total_puzzles: int) -> str:
    """Generate the content for README.md with MIT license metadata."""
    
    readme_content = f"""---
license: mit
task_categories:
- image-to-image
- text-generation
language:
- en
tags:
- arc
- agi
- reasoning
- augmentation
pretty_name: ARC-AGI Augmented Dataset
size_categories:
- 10k<n<100k
---

# ARC-AGI Augmented Dataset

This dataset is an augmented version of the **Abstraction and Reasoning Corpus (ARC-AGI)**, processed for training neural networks (such as Transformers or Neural Cellular Automata).

## Dataset Details

- **Original Source:** [ARC-AGI Benchmark](https://github.com/fchollet/ARC)
- **License:** MIT
- **Augmentation Method:**
    - **Dihedral Transformations:** 8 symmetries (rotations/flips).
    - **Color Permutation:** Random permutation of colors 1-9 (0 is fixed as background).
    - **Translational Padding:** Randomly positioning the grid within a {ARCMaxGridSize}x{ARCMaxGridSize} canvas.
- **Seed:** `{config.seed}`
- **Augmentation Factor:** `{config.num_aug}` per puzzle.

## Statistics

- **Total Original Puzzles:** {total_puzzles}
- **Total Augmented Examples:** {total_examples}

## Data Structure

Each row in the dataset represents a single input/output example pair (flattened).

- `input_ids`: Flattened array (int32) of the input grid.
    - Values: 0 (Pad), 1 (EOS), 2-11 (Colors 0-9).
    - Dimensions: {ARCMaxGridSize} x {ARCMaxGridSize} flattened.
- `labels`: Flattened array (int32) of the output grid.
- `puzzle_id`: Integer ID for the puzzle.
- `original_puzzle_id`: The hex string ID from the original ARC dataset (e.g., `007bbfb7`).
- `group_id`: Identifies the augmentation group. All examples with the same `group_id` are variations of the same puzzle.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{config.hf_repo_id}")
print(dataset["train"][0])
""" 
    return readme_content

def convert_dataset(config: DataProcessConfig): # Authenticate to HF if token is provided if config.hf_token: login(token=config.hf_token)
    np.random.seed(config.seed)

    # Read dataset
    data, test_puzzles = load_puzzles_arcagi(config)

    # Map global puzzle identifiers
    num_identifiers = 1  # 0 is blank
    identifier_map = {}

    # First pass: Build Identifier Map
    for split_name, split in data.items():
        for subset_name, subset in split.items():
            for group in subset:
                for puzzle in group:
                    if puzzle.id not in identifier_map:
                        identifier_map[puzzle.id] = num_identifiers
                        num_identifiers += 1

    print (f"Total puzzle IDs (including <blank>): {num_identifiers}")

    # Prepare Hugging Face DatasetDict
    hf_dataset_dict = DatasetDict()
    grand_total_examples = 0
    grand_total_puzzles_processed = 0

    for split_name, split in data.items():
        print (f"Processing split: {split_name}")
        
        # We will collect all examples for this split in a list of dicts
        split_data_list = []
        
        # Translational augmentations config
        enable_translational_augment = split_name == "train"
        
        for subset_name, subset in split.items():
            print (f"  Processing subset: {subset_name} with {len(subset)} groups")
            # subset is a list of groups (augmentations of a puzzle)
            
            for group_idx, group in enumerate(subset):
                print (f"    Processing group {group_idx+1}/{len(subset)}")
                # group is a list of ARCPuzzle objects (original + augmentations)
                
                for puzzle in group:
                    # Determine which example in this puzzle to keep 'clean' (no translation)
                    no_aug_id = np.random.randint(0, len(puzzle.examples))
                    
                    for _idx_ex, (inp, out) in enumerate(puzzle.examples):
                        # Apply translation and flattening
                        inp_flat, out_flat = np_grid_to_seq_translational_augment(
                            inp, out, 
                            do_translation=enable_translational_augment and _idx_ex != no_aug_id
                        )
                        
                        # Create entry for HF Dataset
                        entry = {
                            "input_ids": inp_flat,
                            "labels": out_flat,
                            "puzzle_id": identifier_map[puzzle.id],
                            "original_puzzle_id": puzzle.id,
                            "subset_name": subset_name,
                            "group_id": group_idx,
                            "example_index": _idx_ex
                        }
                        
                        split_data_list.append(entry)
                        grand_total_examples += 1
                    
                    grand_total_puzzles_processed += 1
        
        # Create Dataset object for this split
        features = Features({
            "input_ids": Sequence(Value("int32")),
            "labels": Sequence(Value("int32")),
            "puzzle_id": Value("int32"),
            "original_puzzle_id": Value("string"),
            "subset_name": Value("string"),
            "group_id": Value("int32"),
            "example_index": Value("int32")
        })

        hf_dataset_dict[split_name] = Dataset.from_list(split_data_list, features=features)
        print(f"Created dataset for {split_name} with {len(hf_dataset_dict[split_name])} examples.")

    # 1. Upload Dataset
    print(f"Uploading Dataset to Hugging Face Hub: {config.hf_repo_id}")
    try:
        hf_dataset_dict.push_to_hub(
            config.hf_repo_id, 
            private=config.private,
            token=config.hf_token
        )
        print("Dataset uploaded successfully!")
        
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        return

    # 2. Generate and Upload README.md (with MIT License)
    print("Generating and uploading README.md...")
    # Calculate approx original puzzles for stats
    est_original_puzzles = grand_total_puzzles_processed // (config.num_aug + 1) if config.num_aug > 0 else grand_total_puzzles_processed

    readme_text = generate_readme_content(
        config, 
        total_examples=grand_total_examples, 
        total_puzzles=est_original_puzzles
    )

    try:
        api = HfApi(token=config.hf_token)
        api.upload_file(
            path_or_fileobj=readme_text.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=config.hf_repo_id,
            repo_type="dataset"
        )
        print("README.md (MIT License) uploaded successfully!")
    except Exception as e:
        print(f"Error uploading README: {e}")
        
@cli.command(singleton=True) 
def main(config: DataProcessConfig): 
    convert_dataset(config)

if __name__ == "__main__": 
    cli()