import wave
import os
import errno
from math import modf
import argparse

segment_duration = 10

parser = argparse.ArgumentParser()
parser.add_argument('--wavefile_path', type=str)
parser.add_argument('--target_path', type=str, default='data/')
parser.add_argument('--speaker', type=str, default='TODA')

args = parser.parse_args()

wavefile_path = args.wavefile_path
target_path = args.target_path
speaker = args.speaker


wavefiles = [wavefile_id for wavefile_id in os.listdir(wavefile_path) if os.path.isfile(os.path.join(wavefile_path, wavefile_id)) and
             wavefile_id.endswith(".wav")]

recording_ids = [wavefile_id.split(".")[0] for wavefile_id in wavefiles]


def generate_utterance_id(recording_id, segment):
    """
    TODO: 
    input: recording_id segment[segment_begin, segment_end]
    output: str('iis20160614_0000000_0020000')
    """
    time_stamps = []
    for time_stamp in segment:
        decimal_part, integer_part = modf(time_stamp)
        decimal_part = "%.3f" % decimal_part
        integer_part = "%04d" % integer_part
        stamp_id = str(integer_part) + str(decimal_part).replace('0.', '')
        time_stamps.append(stamp_id)
    utterance_id = recording_id + '_' + \
        str(time_stamps[0]) + '_' + str(time_stamps[1])
    return utterance_id


for recording_id in recording_ids:
    # make target directory and generate wav.scp
    os.makedirs(os.path.join(target_path, recording_id), exist_ok=True)
    tmp = os.path.join(wavefile_path, recording_id)
    wav_scp_content = recording_id + ' cat ' + tmp + '.wav |'
    
    with open(os.path.join(target_path, recording_id + '/wav.scp'), 'w') as wav_scp_f:
        print(wav_scp_content, file=wav_scp_f)

    # generate utterance_id
    with wave.open(os.path.join(wavefile_path, recording_id + '.wav'), 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / rate
        duration = str(duration).split(
            '.')[0] + '.' + str(duration).split('.')[1][:3]
        duration = float(duration)
    segments = []
    segment_begin = 0
    segment_end = segment_begin + segment_duration
    while segment_end <= duration:
        segments.append([segment_begin, segment_end])
        segment_begin = segment_begin + segment_duration
        segment_end = segment_end + segment_duration
    segments.append([segment_begin, duration])

    segment_container = []
    text_container = []
    utt2spk_container = []
    for segment in segments:
        segment_record = generate_utterance_id(recording_id, segment)

        record_line = ' '.join(
            [segment_record, recording_id, str(segment[0]), str(segment[1])])
        segment_container.append(record_line)

        text_container.append(' '.join([segment_record, "何もないよ~"]))

        utt2spk_container.append(' '.join([segment_record, speaker]))
    # generate segments, text and utt2spk
    with open(os.path.join(target_path, recording_id + '/segments'), 'w') as segment_f:
        print("\n".join(segment_container), file=segment_f)
    with open(os.path.join(target_path, recording_id + '/text'), 'w') as text_f:
        print("\n".join(text_container), file=text_f)
    with open(os.path.join(target_path, recording_id + '/utt2spk'), 'w') as utt2spk_f:
        print("\n".join(utt2spk_container), file=utt2spk_f)
