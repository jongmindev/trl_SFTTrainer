import os
import json
from datasets import Dataset, concatenate_datasets
import omegaconf


def load_dataset_sft(config, logger=None):
    args = config.data
    jsonl_paths = args.datasets
    # Sanity check
    for path in jsonl_paths:
        assert os.path.isfile(path), f"File '{path}' does not exist."

    # Load jsonl files
    datasets_list = []
    bad_files = []
    for path in jsonl_paths:
        try:
            ds = Dataset.from_json(path)
            if len(ds) == 0:
                print(f"[WARNING] Empty dataset: '{path}'")
            datasets_list.append(ds)
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            bad_files.append((path, repr(e)))
    
    if bad_files:
        raise RuntimeError(f"{len(bad_files)} file(s) failed to load.\n")
    
    train_dataset = concatenate_datasets(datasets_list)
    train_dataset = train_dataset.shuffle(seed=args.data_seed)

    if logger:
        logger.info(f"[PROGRESS] Dataset loaded. Total conversations: {len(train_dataset)}")

    return train_dataset


if __name__ == "__main__":
    import hydra

    @hydra.main(config_path="configs", version_base=None)
    def test(config):
        train_dataset = load_dataset_sft(config)

    test()
    
