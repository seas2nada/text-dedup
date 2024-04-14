#!/usr/bin/env python
# @Date    : 2022-11-05 09:44:48
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from typing import Callable

import click
import datasets
import numpy as np
from datasets.load import load_dataset
from pybloom_live import ScalableBloomFilter
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import BloomFilterArgs
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils.hashfunc import md5_digest
from text_dedup.utils.hashfunc import sha256_digest
from text_dedup.utils.hashfunc import xxh3_128_digest
from text_dedup.utils.timer import Timer

import json
import re

def remove_puncs(text):
    pattern = r'[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318Fa-zA-Z]+'
    
    # Find all substrings that match the pattern
    matches = re.findall(pattern, text)
    
    # Join the matches into a single string with spaces
    # You can adjust the separator if needed
    result = ' '.join(matches)
    
    return result

@click.command
@IOArgs.option_group
@MetaArgs.option_group
@BloomFilterArgs.option_group
def main(
    io_args: IOArgs,
    meta_args: MetaArgs,
    bloom_filter_args: BloomFilterArgs,
):
    timer = Timer()
    flags = []

    def prepare_dataset(batch):
            # process audio
            text = batch["text"]
            batch["text_org"] = text
            batch["text"] = remove_puncs(text)
            return batch

    with timer("Total"):
        with timer("Loading"):
            # json_file = os.path.join("data/KlecSpeech", "train.json")
            json_file = "/home/ubuntu/Workspace/DB/korean_db/data/KlecSpeech/train.json"
            ds = load_dataset("json", data_files=json_file)
            ds = ds['train']

            ds = ds.map(
                prepare_dataset,
                desc="preprocess train dataset",
            )

        hash_func: Callable = {
            "md5": md5_digest,  # type: ignore
            "sha256": sha256_digest,  # type: ignore
            "xxh3": xxh3_128_digest,  # type: ignore
        }[bloom_filter_args.hash_func]

        LEN_DATASET = len(ds)

        bf = ScalableBloomFilter(
            initial_capacity=bloom_filter_args.initial_capacity,
            mode=ScalableBloomFilter.SMALL_SET_GROWTH,
            error_rate=bloom_filter_args.error_rate,
        )

        with timer("Processing"):
            NUM_SHARDS = int(np.ceil(LEN_DATASET / meta_args.batch_size))
            for idx in tqdm(range(0, NUM_SHARDS), desc="Processing..."):
                ds_shard = (
                    ds.shard(num_shards=NUM_SHARDS, index=idx, contiguous=True)
                    # TODO .map(either preprocessing like example.encode("utf-8") or multithreaded)
                )
                for example in tqdm(ds_shard[meta_args.column], leave=False):
                    h = hash_func(example.encode("utf-8"))
                    # True if the element is seen, False otherwise
                    flags.append(bf.add(h))

        with timer("Filtering"):
            ds = ds.filter(
                lambda _, idx: not flags[idx],
                with_indices=True,
                num_proc=io_args.num_proc,
                desc="Filtering...",
            )

            excluded_ds = ds.filter(
                lambda _, idx: flags[idx],  # Invert the condition to get the excluded data
                with_indices=True,
                num_proc=io_args.num_proc,
                desc="Check excluded data...",
            )
        
        def retrieve_text(batch):
            # process audio
            batch["text"] = batch["text_org"]
            return batch

        with timer("Post-processing"):
            ds = ds.map(
                retrieve_text,
                remove_columns='text_org',
                desc="postprocess train dataset",
            )

            excluded_ds = excluded_ds.map(
                retrieve_text,
                remove_columns='text_org',
                desc="postprocess train dataset",
            )

        with timer("Saving"):
            ds.save_to_disk(io_args.output)
            excluded_ds.save_to_disk(io_args.output + "_filtered")

        with timer("Cleaning"):
            if io_args.clean_cache:
                ds.cleanup_cache_files()

    PAD = 32
    for k, v in timer.elapsed_times.items():
        logger.info(f"{k:<{PAD}}: {v:.2f}s")

    logger.info(f"{'Before':<{PAD}}: {len(flags)}")
    logger.info(f"{'After':<{PAD}}: {len(ds)}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()