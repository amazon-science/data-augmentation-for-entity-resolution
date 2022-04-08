# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import dataclasses
from dataclasses import dataclass
from typing import List

from dataclasses_json import DataClassJsonMixin
from tqdm import tqdm

from da4er.augmentation import BaseAugmentation


@dataclass
class Sample(DataClassJsonMixin):
    query: str
    entities: List[str]


@dataclass
class AugmentedSample(DataClassJsonMixin):
    original: Sample
    augmented: List[Sample]


class InputProcessor:
    def __init__(self, augmentations: List[BaseAugmentation]):
        self.augmentations = augmentations

    def augment_file(self, input, output):
        with open(input, 'r') as reader, open(output, 'w') as writer:
            for line in tqdm(reader, desc="Augmenting input file '%s'" % input):
                sample = AugmentedSample.from_json(line)
                for augmentation in self.augmentations:
                    augmented_sample = dataclasses.replace(sample.original)

                    augmented_sample.query = augmentation.augment(augmented_sample.query)
                    augmented_sample.entities = [augmentation.augment(entity) for entity in augmented_sample.entities]

                    sample.augmented.append(augmented_sample)

                writer.write(sample.to_json())
                writer.write('\n')
