import os
import random 
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", type=str, 
                    help="input file id")
    p.add_argument("--output_file", type=str,
                    help="output file id")
    p.add_argument("--utts", type=int, default=100,
                    help="uttarance numbers")
    p.add_argument("--tag", type=str, default="",
                    help="evaluation tag")
    args = p.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    utts = args.utts
    if args.tag != "":
        tag = args.tag
        output_file = output_file + "_" + tag + "_" + str(utts)
    else:
        output_file = output_file + "_" + str(utts)
    with open(input_file, "r") as input_f:
        container = [a for a in input_f]
        try:
            utts <= len(container)
        except ValueError:
            print("uttarance is larger than samples")
    
    random.seed(0)
    random.shuffle(container)
    evaluation_container = container[0: utts]

    with open(output_file, "w") as output_f:
        output_f.write(''.join(evaluation_container))
    
if __name__ == "__main__":
    main()