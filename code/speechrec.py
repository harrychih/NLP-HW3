#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path
import os
from collections import defaultdict


from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()

def read_file(file: Path):
    info = defaultdict(list)
    with open(file) as f:
        head = f.readline()
        u_count = head.split()[0]
        u = ' '.join(head.split()[1:])
        for i, line in enumerate(f):
            # error rate
            info[i].append(float(line.split()[0]))
            # log-likelihood
            info[i].append(float(line.split()[1]))
            # count
            info[i].append(int(line.split()[2]))
            # sentence
            info[i].append(" ".join(line.split()[3:]))
    return (u, u_count, info)

def trigram_read(file: Path, lm: LanguageModel):
    vocab = lm.vocab
    u, u_count, info = read_file(file)
    err = [info[i][0] for i in info]
    log_like = [info[i][1] for i in info]
    count = [info[i][2] for i in info]
    sen = [info[i][3] for i in info]
    newSen = defaultdict(str)

    for i, s in enumerate(sen):
        new_s = ""
        for token in s.split():
            if token is None or token in lm.vocab:
                new_s += token
            else:
                new_s += "OOV"
        new_s += 'EOS'
        newSen[i] = new_s
    trigrams = defaultdict(list)
    x, y = "BOS", "BOS"
    for i in newSen:
        sen = newSen[i]
        for z in sen:
            trigrams[i].append((x, y, z))
            x, y = y, z
    
    return (trigrams, log_like, sen, err)

def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    # x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    trigrams, log_like, sen, err = trigram_read(file, lm)
    priors = []
    for i in trigrams:
        prior = 0.0
        for (x, y, z) in trigrams[i]:
            prior += lm.log_prob(x, y, z)  # log p(z | xy)
        priors.append(prior)
    post = []
    for i in range(len(priors)):
        post.append(priors[i] + log_like[i])
    max_index = post.index(max(post))
    err_rate = err[max_index]
    return err_rate


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Checking...")
    lm = LanguageModel.load(args.model)
    
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    log.info("Error Rate with the maximum posterior in each file printing:")

    overall_err_rate = 0
    for file in args.test_files:
        filename = os.path.basename(file)
        err_rate: float = file_log_prob(file, lm)
        overall_err_rate += err_rate
        print(f"{err_rate}\t{filename}")
    overall_err_rate = overall_err_rate/len(args.test_files)
    print(f"{overall_err_rate:.4f}\tOVERALL")
    log.info("Complete checking...!")

if __name__ == "__main__":
    main()
