import os
import pickle
from srt2segments import time2timestamp, generate_utterance, refine_srt
import argparse

"""
pickle Dataframe:
segment_id | start_timestamp | end_timestamp | start_time | end_time | original text | refined text |
sp1_100_200 | 100 | 200 | 1.00 | 2.00 | omitted | omitted |
 config: {0: normal text, 1: 言い淀み, 2: sub0, 3: sub1, 4: D }
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", type=str,
                   help="input file id")
    p.add_argument("--output_file", type=str,
                   help="output file id")
    p.add_argument("--wavefile_id", type=str,
                   help="wavefile id")
    # p.add_argument("--I_mode", type=int, choices=[0, 1], default=1,
    #                help="I mode:\n 0 = delete all, 1 = original text")
    # p.add_argument("--S_mode", type=int, choices=[0, 1, 2], default=2,
    #                help="S mode:\n 0 = delete all, 1 = original text, 2 = corrected text")
    # p.add_argument("--D_mode", type=int, choices=[0, 1], default=1,
    #                help="D mode:\n 0 = delete all, 1 = corrected text")
    p.add_argument("--remove_punctuation", type=bool, default=True)
    p.add_argument("--remove_space", type=bool, default=True)
    args = p.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    wavefile_id = args.wavefile_id


    # I_mode = args.I_mode
    # S_mode = args.S_mode
    # D_mode = args.D_mode
    remove_punctuation = args.remove_punctuation
    remove_space = args.remove_space

    # mkdir data/<wavefile_id>
    target_dir = os.path.join('data/', wavefile_id)

    os.makedirs(target_dir, exist_ok=True)
    with open(input_file, "r") as input_f:
        container = [a for a in input_f]
        refined_container = refine_srt(wavefile_id, container)
    # with open(os.path.join(target_dir, output_file), "wb") as output_f:
    #     pickle.dump(refined_container, output_f)
    with open(output_file, "wb") as output_f:
        pickle.dump(refined_container, output_f)


if __name__ == "__main__":
    main()

