from collections import defaultdict
from itertools import count
from pathlib import Path
from typing import Tuple, Dict, Generator

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             f1_score, precision_score, recall_score,
                             matthews_corrcoef, confusion_matrix)

master_feat_counter = count(1)


def get_cds_id(attr) -> str:
    """
    Util function that extracts uniform from the attribute string of a GFF-formatted annotation line.
    :param attr: attribute string of GFF-formated annotation line, e.g. "ID=cds1;Parent=gene1"
    :return: cds_id: standardised id use to keep the track of introns of the same gene
    # TODO modify for specific gff variants/annotation conventions
    """
    return attr.split(';')[0].split(' ')[1]


def fasta_reader(fasta_path: Path) -> Generator[Tuple[str, int], None, None]:
    """
    Read fasta file yielding sequence id and length for each sequence.
    :param fasta_path: path to fasta file to read
    :return: generator of tuples (sequence_id, sequence_length) for each sequence in the fasta file
    """
    seq_id, seq_len = None, 0
    with fasta_path.open() as fas:
        for line in fas:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if seq_id is not None:
                    yield seq_id, seq_len
                seq_id = line.lstrip('>').split(' ')[0]
                seq_len = 0
            else:
                seq_len += len(line)
    if seq_id is not None:
        yield seq_id, seq_len
    else:
        raise ValueError("No sequences found in the fasta file.")


def gff_reader(gff_path: Path) -> Generator[Tuple[str, str, int, int], None, None]:
    """
    Read gff file yielding sequence id, cds_id, start and end coordinates for each CDS feature.
    :param gff_path: path to gff file to read
    :return: generator of tuples (sequence_id, cds_id, start, end) for each CDS feature in the gff file
    """
    with gff_path.open() as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue  # skip comments and empty lines
            seq_id, _, feature, start, end, _, _, _, attr = line.strip().split('\t')
            start, end = int(start), int(end)
            if feature == 'CDS':
                cds_id = get_cds_id(attr)
                zero_based_start = start - 1  # convert to 0-based indexing
                yield seq_id, cds_id, zero_based_start, end


def evaluate(reference_gff: Path,
             tested_gff: Path,
             fasta_path: Path,
             debug_file: Path = None) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Read two gff files and evaluate the predicted annotations against the reference annotations.
    The method produces a variety of metrics:
    - Adjusted Rand Score - a metric that treats exons of each gene as separate clusters and compares clustering
    * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    - Normalized Mutual Information Score - similar to ARI but more robust to different cluster sizes
    * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html
    - Macro Precision, Recall and F1 - standard ML metrics calculated for each class separately and then averaged
    * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    - Matthews Correlation Coefficient - balanced performance measure robust to class imbalance
    The evaluation is done at the nucleotide level and uses two different label types for each position:
    1. class_labels: coding potential labels (0 - intergenic non-coding, 1 - intron, 2 - CDS)
    2. cluster_labels: labels of the clustering metrics (e.g. ARI) that treat each feature as separate cluster
    :param reference_gff: path to gff file with reference annotations
    :param tested_gff: path to gff file with predicted annotations
    :param fasta_path: path to original fasta file that was annotated
    :param debug_file: the additional fna file to write class assignments as n (intergenic), G (exon) and U (intron)
    :return: dictionary of calculated numerical metrics and confusion matrix as a pandas DataFrame

    """

    labels = ('intergenic', 'intron', 'CDS')

    sequences = {seq_id: (l, defaultdict(list), defaultdict(list)) for seq_id, l in fasta_reader(fasta_path)}

    for i, annotation_file in enumerate((reference_gff, tested_gff), 1):
        for seq_id, gene_id, start, end in gff_reader(annotation_file):
            sequences[seq_id][i][gene_id].append((start, end))

    all_class_labels = {'ref': [], 'pred': []}
    all_cluster_labels = {'ref': [], 'pred': []}
    for seq_id, (seq_length, reference_features, tested_features) in sequences.items():
        for label_type, genes in (('ref', reference_features), ('pred', tested_features)):
            class_labels = np.zeros(seq_length)  # default 0 to intergenic non-coding
            cluster_labels = np.zeros(seq_length)  # default to intergenic non-coding (as a single group)
            for exons in genes.values():
                exonic_class_label = next(master_feat_counter)  # assign unique label to exons from the same gene
                intronic_class_label = next(master_feat_counter)  # assign unique label to introns from the same gene
                if len(exons) > 1:  # there is an intron!
                    sorted_exons = sorted(exons, key=lambda e: e[0])  # sort by start coordinate
                    exon_pairs = zip(sorted_exons, sorted_exons[1:])
                    for e1, e2 in exon_pairs:
                        if e1[1] < e2[0]:
                            # set "class_labels" to 1 for intronic regions between exons
                            class_labels[e1[1]:e2[0]] = 1
                            cluster_labels[e1[1]:e2[0]] = intronic_class_label
                for exon in exons:
                    class_labels[exon[0]:exon[1]] = 2  # set "class_labels" to 2 for exonic regions
                    cluster_labels[exon[0]:exon[1]] = exonic_class_label  # exons of the single gene share label

            all_class_labels[label_type].append(class_labels)
            all_cluster_labels[label_type].append(cluster_labels)

    concatenated_class_labels = {}
    concatenated_cluster_labels = {}
    for label_type in all_class_labels:
        concatenated_class_labels[label_type] = np.concatenate(all_class_labels[label_type])
        concatenated_cluster_labels[label_type] = np.concatenate(all_cluster_labels[label_type])

    if debug_file is not None:
        translation = {0: 'n', 1: 'U', 2: 'G'}
        ds = ''.join(translation[l] for l in concatenated_class_labels['ref'])
        with debug_file.open('w') as f:
            f.write(f">debug\n{ds}\n")

    ars = adjusted_rand_score(labels_true=concatenated_cluster_labels['ref'],
                              labels_pred=concatenated_cluster_labels['pred'])

    nmi = normalized_mutual_info_score(labels_true=concatenated_cluster_labels['ref'],
                                       labels_pred=concatenated_cluster_labels['pred'])

    macro_precision = precision_score(y_true=concatenated_class_labels['ref'], y_pred=concatenated_class_labels['pred'],
                                      average='macro')
    macro_recall = recall_score(y_true=concatenated_class_labels['ref'], y_pred=concatenated_class_labels['pred'],
                                average='macro')
    macro_f1 = f1_score(y_true=concatenated_class_labels['ref'], y_pred=concatenated_class_labels['pred'],
                        average='macro')

    # calculate F1 score for each class separately
    class_f1_scores = f1_score(y_true=concatenated_class_labels['ref'], y_pred=concatenated_class_labels['pred'],
                               average=None)

    # calculate Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true=concatenated_class_labels['ref'], y_pred=concatenated_class_labels['pred'])

    # calculate full confusion matrix
    raw_matrix = confusion_matrix(y_true=concatenated_class_labels['ref'], y_pred=concatenated_class_labels['pred'])

    labeled_matrix = pd.DataFrame(raw_matrix,
                                  index=[f"ref_{l}" for l in labels],
                                  columns=[f"pred_{l}" for l in labels])

    metrics = {'Adjusted Rand Score': ars,
               'Normalized Mutual Information Score': nmi,
               'Macro Precision': macro_precision,
               'Macro Recall': macro_recall,
               'Macro F1 Score': macro_f1,
               'Matthews Correlation Coefficient': mcc}
    for clas_name, i in enumerate(labels):
        metrics[f'{i} F1 Score'] = class_f1_scores[clas_name]

    return metrics, labeled_matrix


if __name__ == '__main__':
    test_reference_gff = Path('T4.gtf')
    test_predicted_gff = Path('model_tests/T4_sordariomycota.gtf')
    test_fasta = Path('T4.fasta')
    metrics, confusion_matrix = evaluate(test_reference_gff, test_predicted_gff, test_fasta)
    print(metrics)
    print(confusion_matrix)
