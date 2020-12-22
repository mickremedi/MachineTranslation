#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 4
vocab.py: Vocabulary Generation
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>

Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from collections import Counter
from docopt import docopt
from itertools import chain
import json
import torch
from typing import List, Union
from utils import read_corpus, pad_sents


class VocabEntry(object):
    """ Vocabulary Entry, i.e. structure containing either
    src or tgt language terms.
    """

    def __init__(self, word2id: dict = None):
        """ Init VocabEntry Instance.
        @param word2id: dictionary mapping words 2 indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0  # Pad Token
            self.word2id['<s>'] = 1  # Start Token
            self.word2id['</s>'] = 2  # End Token
            self.word2id['<unk>'] = 3  # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word: str) -> int:
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word: word to look up.
        @return: index: index of word
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word: str) -> bool:
        """ Check if word is captured by VocabEntry.
        @param word: word to look up
        @return: contains - whether word is contained
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self) -> int:
        """ Compute number of words in VocabEntry.
        @return: len - number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid: int) -> str:
        """ Return mapping of index to word.
        @param wid: word index
        @return: word - word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word: str) -> int:
        """ Add word to VocabEntry, if it is previously unseen.
        @param word: word to add to VocabEntry
        @return: index - index that the word has been assigned
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents: Union[List[str], List[List[str]]]) -> Union[List[int], List[List[int]]]:
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents: sentence(s) in words
        @return: word_ids - sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids: List[int]) -> List[str]:
        """ Convert list of indices into words.
        @param word_ids: list of word ids
        @return: sents - list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for 
        shorter sentences.

        @param sents: list of sentences (words)
        @param device: device on which to load the tesnor, i.e. CPU or GPU

        @return: sents_var - tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus: List[str], size: int, freq_cutoff: int = 2) -> 'VocabEntry':
        """ Given a corpus construct a Vocab Entry.
        @param corpus: corpus of text produced by read_corpus function
        @param size: # of words in vocabulary
        @param freq_cutoff: if word occurs n < freq_cutoff times, drop the word
        @return: vocab_entry (VocabEntry) - VocabEntry instance produced from provided corpus
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry


class Vocab(object):
    """ Vocab encapsulating src and target langauges.
    """

    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        """ Init Vocab.
        @param src_vocab (VocabEntry): VocabEntry for source language
        @param tgt_vocab (VocabEntry): VocabEntry for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents: List[str], tgt_sents: List[str], vocab_size: int, freq_cutoff: int) -> 'Vocab':
        """ Build Vocabulary.
        @param src_sents: Source sentences provided by read_corpus() function
        @param tgt_sents: Target sentences provided by read_corpus() function
        @param vocab_size: Size of vocabulary for both source and target languages
        @param freq_cutoff: if word occurs n < freq_cutoff times, drop the word.
        """
        assert len(src_sents) == len(tgt_sents)

        print('initialize source vocabulary ..')
        src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)

        print('initialize target vocabulary ..')
        tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)

        return Vocab(src, tgt)

    def save(self, file_path: str):
        """ Save Vocab to file as JSON dump.
        @param file_path: file path to vocab file
        """
        json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), open(file_path, 'w'), indent=2)

    @staticmethod
    def load(file_path: str):
        """ Load vocabulary from JSON dump.
        @param file_path: file path to vocab file
        @returns Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']

        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in source sentences: %s' % args['--train-src'])
    print('read in target sentences: %s' % args['--train-tgt'])

    src_sentences = read_corpus(args['--train-src'], source='src')
    tgt_sentences = read_corpus(args['--train-tgt'], source='tgt')

    vocab = Vocab.build(src_sentences, tgt_sentences, int(args['--size']), int(args['--freq-cutoff']))
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

    vocab.save(args['VOCAB_FILE'])
    print('vocabulary saved to %s' % args['VOCAB_FILE'])
