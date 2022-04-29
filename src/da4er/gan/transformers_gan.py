# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import tensorflow as tf
import random

from da4er.gan import GANAugmentation
from da4er.gan.preprocessing.transformers_preprocessor import BARTGAN as BARTGANPreprocessor


class BARTGAN(GANAugmentation):
    """
        Generate synthetic data using GAN trained with BART encoder
    """
    def __init__(self, preprocessor: BARTGANPreprocessor, generator, dimension=6, bart_score_low=0.3, bart_score_high=0.7):
        super().__init__(preprocessor, generator)
        self.preprocessor = preprocessor
        self.dim = dimension
        self.low = bart_score_low
        self.high = bart_score_high
        self.accept_prob = bart_score_high
        self.normalized_dictionary = self._get_normalized_dictionary()

    def _get_normalized_dictionary(self):
        candidates = self.preprocessor.encoder_embed
        candidates = candidates / tf.reshape(tf.norm(candidates, axis=-1), shape=(-1, 1))
        return candidates.numpy()

    def augment(self, txt: str) -> str:
        tokenized_input = self.preprocessor.tokenizer.tokenize(txt)
        if len(tokenized_input) == 0:
            return txt
        embedding = self.preprocessor.preprocess(txt)
        embedding_size = np.shape(embedding)[0]

        if len(np.where(embedding[0] == 0)[0]) != 0:  # Find the dimensions until considered tokens
            target_len = np.where(embedding[0] == 0)[0][0]
        else:
            target_len = self.dim

        input_data = np.reshape(embedding, [1, embedding_size, self.dim])
        noise = tf.random.normal([1, embedding_size, self.dim])  # Random noise
        input_syn = input_data + noise  # Input for generator

        tmp_data = self.generator(input_syn, training=False)  # Target's synthetic embedding
        # Average between synthetic and original data for proper augmentation
        tmp_data = (tf.reshape(tmp_data, shape=(embedding_size, self.dim)).numpy() + embedding) / 2

        output_loc = []
        for k in range(target_len):
            original_norm = tmp_data[:, k] / np.linalg.norm(tmp_data[:, k])
            similarity = np.dot(self.normalized_dictionary, np.reshape(original_norm, (-1, 1)))
            for jj in range(np.shape(similarity)[0]):
                similar = similarity[jj].item(0)

                # Define threshold for comparison
                if self.low < similar < self.high:
                    output_loc.append((k, jj, similar))
        output_loc = sorted(output_loc, key=lambda x: x[2], reverse=True)

        output_txt = [[] for _ in range(self.dim)]  # Generate the possible combination
        for kk in range(len(output_loc)):
            output_txt[output_loc[kk][0]].append(self.preprocessor.tokenizer.decode(output_loc[kk][1],
                                                                                    skip_special_tokens=True))

        synthetic_text = []
        for piece_id in range(target_len):
            old_fragment: str = tokenized_input[piece_id]
            candidates = output_txt[piece_id]

            # Old tokens and the replacing tokens should be coherent
            filtered_candidates = []
            for candidate in candidates:
                if (old_fragment.startswith(" ") or old_fragment.startswith("Ġ")) \
                        != (candidate.startswith(" ") or candidate.startswith("Ġ")):
                    continue
                if candidate.isupper() != old_fragment[0].isupper():
                    continue
                filtered_candidates.append(candidate)
            candidates = filtered_candidates

            if len(candidates) == 0 or random.random() > self.accept_prob:
                synthetic_text.append(tokenized_input[piece_id])
                continue

            chosen_candidate = None
            for candidate in candidates:
                if random.random() >= 0.5:
                    chosen_candidate = candidate
                    break
            if chosen_candidate is None:
                chosen_candidate = candidates[0]
            synthetic_text.append(chosen_candidate)
        synthetic_text = self.preprocessor.tokenizer.convert_tokens_to_string(synthetic_text)
        return synthetic_text


class BERTGAN(GANAugmentation):
    """
        Generate synthetic data using GAN trained with BERT encoder
    """
    def __init__(self, preprocessor, generator, dict_bert):
        super().__init__(preprocessor, generator)
        with np.load(dict_bert) as openfile:
            self.dict_embed = openfile['train_data']
            self.dict_data = openfile['ori_data']

    def augment(self, txt: str) -> str:
        pass

    def postprocess(self, txt, embed):

        if txt not in self.ori_syn_data.keys():

            input_data = np.reshape(embed, [1, self.embedding_size])
            noise = tf.random.normal([1, self.embedding_size])  # Random noise
            input_syn = input_data + noise  # Input for generator
            tmp_data = self.generator(input_syn, training=False)  # Target's synthetic data
            tmp_data = np.reshape(tmp_data, [self.embedding_size])

            val_1 = -100
            val_2 = -100
            val_3 = -100
            loc_1 = -1
            loc_2 = -1
            loc_3 = -1

            original_norm = tmp_data / np.linalg.norm(tmp_data)  # norm of synthetic data

            for j in range(len(self.dict_embed)):
                candidate = np.reshape(self.dict_embed[j], [self.embedding_size])
                candidate_norm = candidate / np.linalg.norm(candidate)
                similar = np.dot(original_norm, candidate_norm)

                # Find the top 3 similar
                if val_1 < similar:
                    val_3 = val_2
                    val_2 = val_1
                    val_1 = similar
                    loc_3 = loc_2
                    loc_2 = loc_1
                    loc_1 = j
                elif val_2 < similar:
                    val_3 = val_2
                    val_2 = similar
                    loc_3 = loc_2
                    loc_2 = j
                elif val_3 < similar:
                    val_3 = similar
                    loc_3 = j
            self.ori_syn_data[txt] = [self.dict_data[loc_1],
                                      self.dict_data[loc_2], self.dict_data[loc_3]]
