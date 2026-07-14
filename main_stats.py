#!/usr/bin/env python3

import argparse
import os 
from statistics import mean 

def gtf_gff_parse(filepath):
    orfs_gtf = []

    with open(filepath) as file:
        for line in file:
            if line.startswith("#"): continue

            parts = line.strip().split("\t")

            if len(parts) < 9: continue

            if parts[2] == "CDS":
                try:
                    start = int(parts[3])
                    end = int(parts[4])
                    length = abs(int(parts[4]) - int(parts[3]) + 1)
                    orfs.append(length)

                except ValueError: continue

    return orfs_gtf

def gb_parse(filepath):
    orfs_gb = []

    feature = False 

    with open(filepath) as file:
        for line in file:
            if line.startswith("FEATURES"): 
                feature = True
                continue
            if not feature: continue
            if line.startswith("ORIGIN"): break

            line = line.strip()
            if not line.startswith("CDS"): continue

            corr = (line.split(maxsplit=1)[1]
                    .replace("complement(","")
                    .replace("join(","")
                    .replace(")",""))
            
            len = 0 
            for parts in corr.split(","):
                if ".." not in parts: continue

                start,end = parts.split("..")
                start = int(start.strip("<>"))
                end = int(end.strip("<>"))
                len += end - start + 1

            orfs_gb.append(len)

    return orfs_gb

def choose_file(filepath):

    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".gtf",".gff"]: orfs = gtf_gff_parse(filepath)
    elif ext in [".gb",".gbk"]: orfs = gb_parse(filepath)
    else: raise ValueError("Wrong file format")

    n0_genes = len(orfs)
    average_orf_length = mean(orfs)

    return n0_genes,average_orf_length

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files",nargs = "+")
    args = parser.parse_args()

    for filepath in args.files: 
        n0_genes, average_orf_length = choose_file(filepath)

        print(f"For file {filepath}, {n0_genes} CDS were predicted with average length of {average_orf_length:.3f}")

if __name__ == "__main__":
    main()
