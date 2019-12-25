import vad
import os
import scipy.io.wavfile as wav
# import matplotlib.pyplot as plt
import numpy as np
import argparse


def timestamp2time(timestamp):
    second, milisecond = str(timestamp).split(".")
    second = int(second)
    if len(milisecond) == 0:
        milisecond = '000'
    elif len(milisecond) == 1:
        milisecond += '00'
    elif len(milisecond) == 2:
        milisecond += '0'
    h = second // 3600
    m = (second - h * 3600) // 60
    s = second - h * 3600 - m * 60
    time = "%02d:%02d:%02d,%s" % (h, m, s, milisecond)
    return time


class SIGNAL_INFO():
    def __init__(self, sr, signal, ori_label, refined_label):
        self.sr = sr
        self.frames = len(signal)
        self.signal = signal
        self.ori_label = vad
        self.refined_label = vad_
        self.active_frames = []
        self.active_time_segments = []

    def plot(self):
        import matplotlib.pyplot as plt
        plt.subplot(3, 1, 2)
        plt.plot(self.ori_label)
        plt.xlabel('frame')
        plt.ylabel('Prob')
        plt.title('Unrefined')

        plt.subplot(3, 1, 3)
        plt.plot(self.refined_label)
        plt.xlabel('frame')
        plt.ylabel('Prob')
        plt.title('Refined')

        plt.subplot(3, 1, 1)
        plt.plot(self.signal)
        plt.title('Time Signal')
        plt.tight_layout()
        plt.show()

    def get_active_frames(self, expansion=True):
        f = 0
        flag = 0
        segment = []
        while f < len(self.refined_label):
            if self.refined_label[f] == 1:
                flag = 1
                segment.append(f)
                while f < len(self.refined_label) and self.refined_label[f] == 1:
                    f += 1
                    continue
                segment.append(f)
            if len(segment) == 2:
                if segment[1] - segment[0] >= 10:
                    self.active_frames.append(segment)
                segment = []
                flag = 0
            else:
                f += 1
            continue
        # print(self.active_frames)
        if expansion == True:
            for segment in self.active_frames:
                if segment[0] - 2 >= 0:
                    segment[0] -= 2
                else:
                    segment[0] = 0
                if segment[1] + 2 <= len(self.refined_label):
                    segment[1] += 2
                else:
                    segment[1] = len(self.refined_label)
        return self.active_frames

    def get_active_time_segments(self, hop_length):
        assert self.active_frames != None, "self.active_frames is Empty \n Try get_active_frames() first"
        self.active_time_segments = np.array(
            self.active_frames, dtype=np.float) * hop_length
        self.active_time_segments = np.round(self.active_time_segments, 3)
        return self.active_time_segments

    def get_segments_file(self, path):
        assert self.active_time_segments.any(
        ) != None, "self.active_time_segments is Empty \n Try get.active_time_segments() first"
        with open(path, 'w') as f:
            count = 1
            for [s, e] in self.active_time_segments:
                start = timestamp2time(s)
                end = timestamp2time(e)
                seg = "%s --> %s" % (start, end)
                print("%d\n%s\n%s\n"%(count, seg, "ほにゃらら"), file=f)
                count += 1


def remove_short_silence(frames, threshold):
    """Remove Wrongly Detected Short Scilent frames
    Parameters:
    ----------
    frames: 1-dim array e.g. [0,1,1,1,1,1,0,1,1,0,0]
    threshold: maximum number of frames can be ingored

    Returns
    -------
    Refined_frames: 1-dim array e.g. [0,1,1,1,1,1,1,1,1,0,0]
    """
    Refined_frames = np.array(frames)
    f = 0
    count = 0
    while f < len(Refined_frames):
        if Refined_frames[f] == 0:
            count += 1
            f += 1
            continue
        if Refined_frames[f] == 1 and count <= threshold:
            Refined_frames[f-count: f] = 1
            count = 0
        f += 1
        count = 0
    return Refined_frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=str)
    args = parser.parse_args()

    path_wav = os.path.join(os.getcwd(), "datasets/iis20160517_5m.wav")
    # path_wav = os.path.join(os.getcwd(), "datasets/VAD_text.wav")
    (sr, signal) = wav.read(path_wav)

    hop_length = 0.01
    vad = vad.VAD(signal, sr, nFFT=512, win_length=0.025,
                  hop_length=hop_length, theshold=0.9)
    vad_ = remove_short_silence(vad, 20)

    example = SIGNAL_INFO(sr=sr, signal=signal,
                          ori_label=vad, refined_label=vad_)
    # example.plot()
    activate_segments = example.get_active_frames(expansion=False)
    # print(activate_segments)
    activate_segments_ = example.get_active_time_segments(hop_length)
    print(activate_segments_)
    example.get_segments_file(path='./iis20160517_5m.srt')
