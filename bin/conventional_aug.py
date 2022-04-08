# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse

from da4er.basic_augmentations import LexAugmentation, SpellAugmentation, CharacterAugmentation
from da4er.translation_augmentations import TranslationAugmentation, BackTranslationAugmentation
from da4er.utils import InputProcessor

RECOMMENDED_CHARACTER_METHODS = ["insert", "substitute", "swap", "delete"]


def main(args):
    data = []

    if args.aug == 'lexical':
        data.append(LexAugmentation(args.src_lex, args.src_lang_lex))
    elif args.aug == 'spelling':
        data.append(SpellAugmentation())
    elif args.aug == 'character':
        if args.chr_specific == 'recommended':
            data += [CharacterAugmentation(method) for method in RECOMMENDED_CHARACTER_METHODS]
        else:
            data.append(CharacterAugmentation(args.chr_specific))
    elif args.aug == 'translation':
        data.append(TranslationAugmentation(args.src_nmt, args.tar_nmt, args.src, args.inter, args.tar))
    elif args.aug == 'back-translation':
        data.append(BackTranslationAugmentation(args.src_nmt, args.tar_nmt, args.src, args.inter))

    if data is None:
        print('Wrong augmentation method. '
              'Please try one of lexical / spelling / character / translation / back-translation')
        return

    processor = InputProcessor(data)
    processor.augment_file(args.input, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Conventional augmentations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, required=True, help='Original text data. Only receive json file.')
    parser.add_argument('--aug', type=str, required=True,
                        help='Choose augmentation methods: '
                             'lexical / spelling / character / translation / back-translation.')
    parser.add_argument('--output', type=str, required=True, help='Name of output file. Only generate json file.')
    parser.add_argument('--src-lex', type=str, default='wordnet', help='Source for lexical.')
    parser.add_argument('--src-lang-lex', type=str, default='eng', help='Source language for lexical.')
    parser.add_argument('--chr-specific', type=str, default='recommended',
                        help='Select insert, substitute, swap or delete method in character-level')

    parser.add_argument('--src-nmt', type=str, help='Source NMT location')
    parser.add_argument('--tar-nmt', type=str, help='Target NMT location')
    parser.add_argument('--src', type=str, help='Source language for translation / back-translation')
    parser.add_argument('--inter', type=str, default='en',
                        help='Intermediate language for translation / back-translation')
    parser.add_argument('--tar', type=str, default='fr', help='Target language for translation')

    args = parser.parse_args()
    main(args)
