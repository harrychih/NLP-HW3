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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to the trained model 1",
    )

    parser.add_argument(
        "model2",
        type=Path,
        help="path to the trained model 2",
    )

    parser.add_argument(
        "prior_probability",
        default=0.7,
        type=float,
        help="the prior probability of the first category",
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


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0
    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)
    return log_prob

def fileLen_extract(file: Path) -> int:
    basePath = os.path.basename(file)
    return int(basePath.split('.')[1])

def fileLen_classification_acc(corrClassifiedInfo: dict, incorrClassifiedInfo: dict) -> dict:
    # if corrClassifiedInfo.keys() != incorrClassifiedInfo.keys():
    #     raise ValueError('wrong classification info given!')
    acc_dict = defaultdict(float)
    for l in corrClassifiedInfo:
        corr = corrClassifiedInfo[l]
        err = incorrClassifiedInfo[l]
        acc_dict[l] = corr / (corr+err)
    return acc_dict

def plot_and_save_acc(acc_dict: dict, lambda_star: float, model1: str, model2: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(acc_dict.keys(), acc_dict.values(), width=2, linewidth=20)
    plt.xlabel("File Length")
    plt.ylabel(f"Classification Accuracy of add {lambda_star}")
    plt.savefig(f"{model1}-{model2}-filelen-acc-add {lambda_star}.jpg")
    print("Done ploting....And Saved!")


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Testing...")
    lm1 = LanguageModel.load(args.model1)
    lm2 = LanguageModel.load(args.model2)
    prior_prob1 = args.prior_probability
    prior_prob2 = 1 - prior_prob1
    log_prior_prob1 = math.log(prior_prob1) if prior_prob1 != 0 else float("-inf")
    log_prior_prob2 = math.log(prior_prob2) if prior_prob2 != 0 else float("-inf")
    
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    # Check if both language models loaded for text categorization have the same vocabulary
    if lm1.vocab != lm2.vocab:
        raise ValueError("Two Langugue Model Does not have the same vocabulary!")

    log.info("Per-file log-probabilities:")
    total_log_prob = 0.0
    model1_files = 0
    model2_files = 0
    correct_classified1 = 0
    correct_classified2 = 0
    incorrect_classified1 = 0
    incorrect_classified2 = 0
    corrClassifiedInfo = defaultdict(int)
    incorrClassifiedInfo = defaultdict(int)

    for file in args.test_files:
        log_prob1: float = file_log_prob(file, lm1)
        log_prob2: float = file_log_prob(file, lm2)

        log_post_prob1: float = log_prior_prob1 + log_prob1
        log_post_prob2: float = log_prior_prob2 + log_prob2

        
        if log_post_prob1 >= log_post_prob2:
            model1_files += 1
            print(f"{args.model1}\t{os.path.basename(file)}")

            # filename = model1name
            if str(args.model1).split('.')[0].split("-")[0] == os.path.basename(file).split('.')[0]:
                correct_classified1 += 1
                corrClassifiedInfo[fileLen_extract(file)] += 1
            # filename = model2name
            else:
                incorrect_classified1 += 1
                incorrClassifiedInfo[fileLen_extract(file)] += 1

        else:
            model2_files += 1
            print(f"{args.model2}\t{os.path.basename(file)}")

            # filename = model2name
            if str(args.model2).split('.')[0].split("-")[0] == os.path.basename(file).split('.')[0]:
                correct_classified2 += 1
                corrClassifiedInfo[fileLen_extract(file)] += 1
            # filename = model1name
            else:
                incorrect_classified2 += 1
                incorrClassifiedInfo[fileLen_extract(file)] += 1
    



    # But cross-entropy is conventionally measured in bits: so when it's
    # time to print cross-entropy, we convert log base e to log base 2, 
    # by dividing by log(2).

    # Compute Error Rate
    correct_classified = correct_classified1 + correct_classified2
    incorrect_classified = incorrect_classified1 + incorrect_classified2
    # dev2_acc = correct_classified2 / (incorrect_classified1 + correct_classified2)
    # print(f"{dev2_acc:.4%}")
    acc = correct_classified/len(args.test_files)
    err_rate = 1 - acc
    print(f"Total Error Rate: {err_rate:.4%}")

    # Plot the fileLen-acc plot and correlation heatmap
    acc_dict = fileLen_classification_acc(corrClassifiedInfo, incorrClassifiedInfo)
    lambda_star = float('.'.join(str(args.model1).split('-')[-1].split('.')[0:2]))

    plot_and_save_acc(acc_dict, lambda_star, str(args.model1), str(args.model2))

    fileL = np.array(list(acc_dict.keys()))
    acc_corr = np.array(list(acc_dict.values()))
    data = np.stack((fileL, acc_corr), axis=0)
    corr_plot = sns.heatmap(data)
    corr_fig = corr_plot.get_figure()
    corr_fig.savefig(f"{str(args.model1)}-{str(args.model2)}-filelen-acc-corr-add {lambda_star}.jpg")

    # bits = -total_log_prob / math.log(2)   # convert to bits of surprisal
    # tokens = sum(num_tokens(test_file) for test_file in args.test_files)
    fileNum = model1_files + model2_files
    model1Percentage = model1_files/fileNum
    model2Percentage = model2_files/fileNum
    print(f"{model1_files} files were more probably {args.model1} ({model1Percentage:.2%})")
    print(f"{model2_files} files were more probably {args.model2} ({model2Percentage:.2%})")


if __name__ == "__main__":
    main()
