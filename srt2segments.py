import os
import argparse
from srt_refined import I, S, D, label_zipped, remove_symbol


def timestamp2time(timestamp):
    '''
    Change timestamp <second.milisecond> to time <h:m:s,milisecond>
    Example: 3456.789 -> 00:57:36,789
    '''
    second, milisecond = timestamp.split(".")
    h = second // 3600
    m = (second - h * 3600) // 60
    s = second - h * 3600 - m * 60
    time = "%02d:%02d:%02d,%s" % (h, m, s, milisecond)
    return time


def time2timestamp(time):
    '''
    change time <h:m:s,milisecond> to timestamp
    Example: 00:57:36,789 -> 3456.789
    '''
    second, milisecond = time.split(",")
    hour, minute, second = second.split(":")
    integer_part = "%04d" % (int(hour) * 3600 + int(minute) * 60 + int(second))
    timestamp = integer_part + milisecond.replace('\n', '')[:3]
    return timestamp


def second_timestamp(time):
    '''
    simply change timestamp_id into <second.milisecond>
    '''
    time = float(time) / 1000
    return "%.3f" % time


def generate_utterance(timesegment, text, recording_id):
    '''
    INPUT:
    timesegment: 00:00:11,234 -> 00:00:55,678
    text: text with '\n', ' ', and <I:>...
    recording_id:
    ---------
    OUTPUT:
    | utterance_id | recording_id | start | end | start_ | end_ | text_label_container |
    | iis20160517_0008310_0011520 | iis20160517 | 0008310 | 0011520 | 8.310 | 11.520 | <zip of text and label> |
    '''
    start, end = timesegment.split("-->")
    start = time2timestamp(start)
    end = time2timestamp(end)
    start_ = second_timestamp(start)
    end_ = second_timestamp(end)
    plaintext = ''.join(text)
    plaintext = plaintext.replace('　', '')
    plaintext = list(plaintext)
    label_text = label_zipped(plaintext)
    text_label_container = remove_symbol(label_text)
    utterance_id = recording_id + '_' + start + '_' + end
    return [utterance_id, recording_id, start, end, start_, end_, text_label_container]


def refine_srt(recording_id, container):
    '''
    revert sevaral line of text into one line container
    '''
    concat_container = []
    line = []
    for x in container:
        if x != "\n":
            line.append(x.replace('\n', ''))
            continue
        elif line == []:
            continue
        else:
            concat_container.append(generate_utterance(line[1], line[2:], recording_id))
            line = []
    # concat_container.append(generate_utterance(line[1], line[2:]))
    return concat_container


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", type=str,
                   help="input file id")
    p.add_argument("--output_file", type=str,
                   help="output file id")
    args = p.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    with open(input_file, "r") as input_f:
        container = [a for a in input_f]
        refined_container = refine_srt(container)
    with open(output_file, "w") as output_f:
        output_f.write("\n".join(refined_container))


if __name__ == "__main__":
    main()
