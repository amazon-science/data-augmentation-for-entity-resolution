# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import glob

import argparse
import numpy as np
from tqdm import tqdm

from da4er.formats.blink_converter import BlinkInputConverter
from da4er.gan.utils import build_gan_preprocessor, add_gan_preprocessor_args
from da4er.utils import InputConverter


def generate_samples(input_path: str):
    for file in glob.glob(input_path):
        with open(file, 'r') as reader:
            for line in reader:
                yield line


def main(args):
    gan_preprocessor = build_gan_preprocessor(args)

    # Initialising formatters
    if args.format == "blink":
        input_converter = BlinkInputConverter()
    else:
        input_converter = InputConverter()

    embeddings = []
    texts = []
    for line in tqdm(generate_samples(args.input), desc="Extracting text for GAN training"):
        sample = input_converter.process(line)
        cur_texts = [sample.work_sample.original.query]+sample.work_sample.original.entities
        cur_embeddings = [gan_preprocessor.preprocess(text) for text in cur_texts]

        for text, embedding in zip(cur_texts, cur_embeddings):
            if embedding is None:
                continue
            embeddings.append(embedding)
            texts.append(text)
    np.savez(args.output, train_data=embeddings, ori_data=texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing for GAN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, required=True, help='Original text data')
    parser.add_argument('--format', type=str, default='default',
                        choices=['default', 'blink'], help='What format to expect as input')
    parser.add_argument('--gan', type=str, required=True, help='Choose GAN methods: fasttext / bert / bart.')                
    parser.add_argument('--output', type=str, required=True, help='Name of output file. Only generate npz file.')
    add_gan_preprocessor_args(parser)
    args = parser.parse_args()

    main(args)
    