import os
import sys
import argparse
import numpy as np
import json
from tqdm import tqdm


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_folder', help="location of folder with the snli files")
    parser.add_argument('--out_folder', help="location of the output folder")
    
    args = parser.parse_args(arguments)

    tr_len=1000

    for split in ["train", "dev"]:
        src_out = open(os.path.join(args.out_folder, "src-"+split+".txt"), "w")
        targ_out = open(os.path.join(args.out_folder, "targ-"+split+".txt"), "w")
        label_out = open(os.path.join(args.out_folder, "label-"+split+".txt"), "w")
        label_set = set(["NOT ENOUGH INFO", "SUPPORTS", "REFUTES"])

        filename=os.path.join(args.data_folder, split+"_with_evi_sents.jsonl")

        with open(filename) as f:

            for index, line in enumerate(f):
                    multiple_ev = False
                    x = json.loads(line)
                    claim = x["claim"]
                    evidences = x["sents"]
                    label = x["label"]
                    evidences_this_list = []
                    evidences_this_str = ""
                    if (len(evidences) > 1):
                        # some claims have more than one evidences. Join them all together.
                        multiple_ev = True
                        for e in evidences:
                            evidences_this_list.append(e)
                        evidences_this_str = " ".join(evidences_this_list)
                    else:
                        evidences_this_str = "".join(evidences)

                    ## truncate at n words. irrespective of claim or evidence truncate it at n...
                    # Else it was overloading memory due to the packing/padding of all sentences into the longest size..
                    # which was like 180k words or something

                    claim_split = claim.split(" ")
                    if (len(claim_split) > tr_len):
                        claim_tr = claim_split[:1000]
                        claim = " ".join(claim_tr)

                    evidences_split = evidences_this_str.split(" ")
                    if (len(evidences_split) > tr_len):
                        evidences_tr = evidences_split[:1000]
                        evidences_this_str = " ".join(evidences_tr)




                    if label in label_set:
                        src_out.write(claim.encode('utf-8') + "\n")
                        targ_out.write(evidences_this_str.encode('utf-8') + "\n")
                        label_out.write(label.encode('utf-8') + "\n")

        src_out.close()
        targ_out.close()
        label_out.close()
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
