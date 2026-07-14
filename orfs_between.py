#!/usr/bin/env python3 

import argparse
import os 
import pandas as pd 

def orf_gff(filepath):
    orfs_gff = []

    with open(filepath) as file:
        for line in file: 
            if line.startswith("#"):continue

            parts = line.strip().split("\t")
            if len(parts) < 9: continue

            if parts[2] == "CDS":
                try:
                    start = int(parts[3])
                    end = int(parts[4])

                    orfs_gff.append((start,end))
                except ValueError: continue
            
    return orfs_gff 

#fetures adffger
def orf_gb(filepath):
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

            if line.startswith("CDS"): 
                parts = line.split(maxsplit=1)

                location = parts[1]
                location = location.replace("complement(", "")
                location = location.replace("join(", "")
                location = location.replace(")", "")  

                segment = location.split(",")

                start_pos = []
                end_pos  = []

                for element in segment:
                    if ".." in element: 
                        try:
                            start,end = element.split("..")
                            start = int(start.replace("<","").replace(">",""))
                            end = int(end.replace("<","").replace(">",""))

                            start_pos.append(start)
                            end_pos.append(end)

                        except ValueError: continue
                orfs_gb.append((min(start_pos),max(end_pos)))            

    return orfs_gb

def choose_file(filepath):
    ext = os.path.splittext(filepath)[1].lower()

    if ext in [".gb","gbk"]: return orf_gb(filepath)
    elif ext in [".gff","gtf"]: return orf_gff(filepath)
    else: raise ValueError("wrong file format")

#tylko wazne gdy parser ustawic tolerance na cos innego niz 0 
def tolerance_compare(reference, test, tolerance):

    reference_start, reference_end = reference
    test_start, test_end = test 

    #reference_interval = abs(reference_start - test_start)

    return (abs(reference_start - test_start) <= tolerance and abs(reference_end-test_end) <= tolerance)

def tpfpfn(reference_file, test_file, tolerance = 0):
    reference_orfs = choose_file(reference_file)
    test_orfs = choose_file(test_file)

    ua = set()
    tp = 0
    fn = 0 

    for ref in reference_orfs:
        matched = False

        for i, test in enumerate(test_orfs):
            if i in ua: continue  #do powt. zliczen, problem w poprzednim

            if tolerance_compare(ref,test,tolerance):
                tp += 1
                ua.add(i)
                matched = True 
                break 

        if not matched: fn += 1

    fp = len(test_orfs) - len(ua)

    return tp,fp,fn,len(reference_orfs),len(test_orfs)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("reference")
    parser.add_argument("test")
    parser.add_argument("--tolerance",type=int,default =0)

    args = parser.parse_args()

    tp,fp,fn,ref_count,test_count = tpfpfn(args.reference,args.test,args.tolerance)

    df = pd.DataFrame({"Reference": [os.path.basename(args.reference)], "Tested file" :[os.path.basename(args.test)],
                        "Reference orfs":[ref_count],"Test orfs":[test_count],
                        "TP":[tp],"FP":[fp],"FN":[fn]
                        })
    
    normal_df = df.to_string(index=False)

    print(normal_df)


if __name__ == "__main__":
    main()
    

