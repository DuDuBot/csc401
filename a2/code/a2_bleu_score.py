## Please take this public repo down as you are violating copyright and 
## depriving students from learning opportunity by posting assignment code online. 

# Copyright 2020 University of Toronto, all rights reserved

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp, pow  # exp(x) gives e^x


def grouper(seq, n):
    '''Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''
    end = len(seq) - n+1
    ngrams = [seq[i:i+n] for i in range(end)]
    return ngrams


def n_gram_precision(reference, candidate, n):
    '''Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''
    # if n == 0?
    if len(candidate) == 0:
        return 0
    refgrams = grouper(reference, n)
    candgrams = grouper(candidate, n)
    matches = sum([1 for cand in candgrams if cand in refgrams])
    return float(matches / len(candgrams))


def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    if len(candidate) == 0:
        return 0
    brevity = len(reference) / len(candidate)
    return 1 if brevity < 1 else float(exp(1-brevity))


def BLEU_score(reference, candidate, n):
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''
    res = 1
    for i in range(1, n+1):
        res *= n_gram_precision(reference, candidate, i)
    return float(brevity_penalty(reference, candidate)*pow(res, 1./n))
