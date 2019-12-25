import pickle
import os
import numpy as np
import argparse
import random
from distutils.util import strtobool

"""
def shuffle(text, seed):

def get_clean_text(text, toggle):
    
def get_segments(path):

def get_text(path):

def get_utt2spk(path):

"""


def shuffle_records(record_pickle, seed=0):
    '''
    shuffle record uttarance with seed
    '''
    random.seed(seed)
    shuffled_pickle = random.shuffle(record_pickle)
    return 1


def get_segments(eval_set_slide):
    '''
    
    '''
    segment_container = []
    for x in eval_set_slide:
        segment_container.append(' '.join(x[0:2] + x[4:6]))
    return segment_container


def get_utt2spk(eval_set_slide, speaker="Toda"):
    utt2spk_container = []
    for x in eval_set_slide:
        utt2spk_container.append(' '.join([x[0], speaker]))
    return utt2spk_container


def get_text(eval_set_slide, clean_text_mode):
    text_container = []
    if clean_text_mode == True:
        for x in eval_set_slide:
            text_zip = x[-1]
            final_text = ''.join([a[0] for a in text_zip if a[1] == '0'])
            line = [x[0], final_text]
            text_container.append(' '.join(line))
    else:
        for x in eval_set_slide:
            final_text = ''.join([a[0] for a in x[-1]])
            line = [x[0], final_text]
            text_container.append(' '.join(line))
    return text_container


def get_wav_scp(recording_id, path):
    wav_scp_content = recording_id + ' cat ' + path + '.wav |'
    with open(os.path.join(target_path, recording_id + '/wav.scp'), 'w') as wav_scp_f:
        print(wav_scp_content, file=wav_scp_f)


def container_to_file(container, path, id):
    with open(os.path.join(path, id), 'w') as f:
        f.write('\n'.join(container))
    return 1


def generate_data_prepration_file(pickle, wavefile_id, path, speaker, clean_text_mode,
                                  use_all_utts=False, sets=1, utts=100):
    if use_all_utts == True:
        segments = get_segments(pickle)
        utt2spk = get_utt2spk(pickle, speaker=speaker)
        text = get_text(pickle, clean_text_mode=clean_text_mode)
        if clean_text_mode == False:
            os.makedirs(path + '_clean_False', exist_ok=True)
            container_to_file(segments, path + '_clean_False', "segments")
            container_to_file(utt2spk, path + '_clean_False', "utt2spk")
            container_to_file(text, path + '_clean_False', "text")
        elif clean_text_mode == True:
            os.makedirs(path + '_clean_True', exist_ok=True)
            container_to_file(segments, path + '_clean_True', "segments")
            container_to_file(utt2spk, path + '_clean_True', "utt2spk")
            container_to_file(text, path + '_clean_True', "text")
    else:
        try:
            len(pickle) > utts * sets
        except ValueError:
            print("pickle utts is less than eval sets required, set utts or sets smaller and try again")
        eval_sets = []
        start = 0
        while start < sets:
            eval_sets.append(pickle[start * utts: start * utts + utts])
            segments = get_segments(eval_sets[start])
            utt2spk = get_utt2spk(eval_sets[start], speaker=speaker)
            text = get_text(eval_sets[start], clean_text_mode=clean_text_mode)
            if clean_text_mode == False:
                os.makedirs(path + '_evalset_clean_False_' + str(start), exist_ok=True)
                container_to_file(segments, path + '_evalset_clean_False_' + str(start), "segments")
                container_to_file(utt2spk, path + '_evalset_clean_False_' + str(start), "utt2spk")
                container_to_file(text, path + '_evalset_clean_False_' + str(start), "text")
            elif clean_text_mode == True:
                os.makedirs(path + '_evalset_clean_True_' + str(start), exist_ok=True)
                container_to_file(segments, path + '_evalset_clean_True_' + str(start), "segments")
                container_to_file(utt2spk, path + '_evalset_clean_True_' + str(start), "utt2spk")
                container_to_file(text, path + '_evalset_clean_True_' + str(start), "text")
            start += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", type=str,
                   help="input file id")
    p.add_argument("--wavefile_id", type=str,
                   help="wave file id")
    p.add_argument("--clean_text_mode", type=strtobool, default=False,
                   help="toggle for clean or original text")
    p.add_argument("--use_all_utts", type=strtobool, default=False,
                   help="toggle for using all utterances or not")
    p.add_argument("--sets", type=int, default=1)
    p.add_argument("--utts", type=int, default=100)
    p.add_argument("--speaker", type=str, default="Toda")
    args = p.parse_args()

    input_file = args.input_file
    wavefile_id = args.wavefile_id
    clean_text_mode = args.clean_text_mode
    path = os.path.join("data/", wavefile_id)

    with open(input_file, 'rb') as input_f:
        record_pickle = pickle.load(input_f)
    shuffle_records(record_pickle)
    generate_data_prepration_file(pickle=record_pickle,
                                  wavefile_id=wavefile_id,
                                  path=path,
                                  speaker=args.speaker,
                                  clean_text_mode=args.clean_text_mode,
                                  use_all_utts=args.use_all_utts,
                                  sets=args.sets,
                                  utts=args.utts)


if __name__ == "__main__":
    main()
