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


from probs import Wordtype, LanguageModel, num_tokens, read_trigrams, sample

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "num_sentences",
        type=int,
        default=1,
        help="The number of sentences sample"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        help="Maximum length limit"
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


# def file_log_prob(file: Path, lm: LanguageModel) -> float:
#     """The file contains one sentence per line. Return the total
#     log-probability of all these sentences, under the given language model.
#     (This is a natural log, as for all our internal computations.)
#     """
#     log_prob = 0.0
#     x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
#     for (x, y, z) in read_trigrams(file, lm.vocab):
#         log_prob += lm.log_prob(x, y, z)  # log p(z | xy)
#     return log_prob





def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Sampling...")
    lm = LanguageModel.load(args.model)
    print(sample(args.model, lm, args.num_sentences, args.max_length))

if __name__ == "__main__":
    main()
