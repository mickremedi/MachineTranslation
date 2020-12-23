#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""

import math
from typing import List

import numpy as np
import nltk

nltk.download('punkt')


def pad_sents(sents: List[List[int]], pad_token: int) -> List[List[int]]:
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents: list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token: padding token
    @return: sents_padded - list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    max_len = len(max(sents, key=lambda x: len(x)))

    for sent in sents:
        sents_padded.append(sent + [pad_token for _ in range(max_len - len(sent))])

    return sents_padded


def read_corpus(file_path: str, source: str):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path: path to file containing corpus
    @param source: "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data: List[tuple], batch_size: int, shuffle: bool = False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data: list of tuples containing source and target sentence
    @param batch_size: batch size
    @param shuffle: whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
