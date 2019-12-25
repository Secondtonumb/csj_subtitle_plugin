import re
from sys import exit
import argparse
import numpy as np

def label_zipped(text):
    mode = 0
    label = []
    for x in text:
        if x == '<' and x != '>':
            mode = 1
            label.append(mode)
            continue
        elif x == '>':
            label.append(mode)
            mode = 0
            continue
        else:
            label.append(mode)
            continue
    return zip(text, label)

def remove_symbol(zipped):
    symbol_list = ["<", ">", "I", "S", "D", "、", " ", "/", ":"]
    refined = [] 
    refined.append([[word, label] for word, label in zipped if (word in symbol_list )== False]) 
    refined = np.array(refined)
    refined = np.reshape(refined, (-1, 2))
    refined = zip(refined[:, 0], refined[:, 1])
    return refined

def I(text, mode):
    """
    I_ptn --> <I: text > 
    I_m --> Strings matched to I_ptn
    I_ctnt_ptn (I's Content Pattern) : <I:text> --> text
    """
    I_ptn = r"<I:.*?>"
    I_m = re.findall(I_ptn, text)
    I_ctnt_ptn = r"(?<=<I:).*?(?=>)"
    res = text
    for x in I_m:
        if(mode == 1):
            I_ctnt = re.findall(I_ctnt_ptn, text)[0]
            res = res.replace(str(x), str(I_ctnt))
        elif(mode == 0):
            I_ctnt = str()
            res = res.replace(str(x), I_ctnt)
        else:
            exit("I_mode ERROR")
    return res


def S(text, mode):
    '''
    S_ptn --> <S: original text > corrected text </S>
    S_m --> Strings matched to S_ptn
    S_ctnt_ptn_2 (S's Content Pattern Type.2) : S_ptn --> corrected text
    S_ctnt_ptn_1 (S's Content Pattern Type.1) : S_ptn --> original text
    '''
    S_ptn = r"<S:.*?>.*</S?>"
    S_m = re.findall(S_ptn, text)

    res = text
    for x in S_m:
        S_ctnt_ptn_2 = r"(?<=[>]).*?(?=[<])"
        S_ctnt_ptn_1 = r"(?<=[:]).*?(?=[>])"
        if(mode == 2):
            S_ctnt = re.findall(S_ctnt_ptn_2, x)[0]
            res = res.replace(str(x), str(S_ctnt))
        elif(mode == 1):
            S_ctnt = re.findall(S_ctnt_ptn_1, x)[0]
            res = res.replace(str(x), str(S_ctnt))
        elif(mode == 0):
            S_ctnt = str()
            res = res.replace(str(x), S_ctnt)
        else:
            exit("S_mode ERROR")
    return res


def D(text, mode):
    '''
    D_ptn --> <D> text </D>
    D_m --> Strings matched to S_ptn
    S_ctnt_ptn (D's Content Pattern) : <D> text </D> --> text
    '''
    D_ptn = r"<D?>.*</D?>"
    D_m = re.findall(D_ptn, text)
    D_ctnt_ptn = r"(?<=[>]).*?(?=[<])"

    res = text
    for x in D_m:
        if(mode == 1):
            D_ctnt = re.findall(D_ctnt_ptn, x)[0]
            res = res.replace(str(x), str(D_ctnt))
        elif(mode == 0):
            D_ctnt = str()
            res = res.replace(str(x), D_ctnt)
        else:
            exit("D_mode EROOR")
    return res

# def main():
    # p = argparse.ArgumentParser()
    # p.add_argument("--input_file", type=str,
    #                help="input file id")
    # p.add_argument("--output_file", type=str,
    #                help="output file id")
    # p.add_argument("--I_mode", type=int, choices=[0, 1], default=1,
    #                help="I mode:\n 0 = delete all, 1 = original text")
    # p.add_argument("--S_mode", type=int, choices=[0, 1, 2], default=2,
    #                help="S mode:\n 0 = delete all, 1 = original text, 2 = corrected text")
    # p.add_argument("--D_mode", type=int, choices=[0, 1], default=1,
    #                help="D mode:\n 0 = delete all, 1 = corrected text")
    # p.add_argument("--remove_punctuation", type=bool, default=True)
    # p.add_argument("--remove_space", type=bool, default=True)
    # args = p.parse_args()

    # input_file = args.input_file
    # output_file = args.output_file
    # I_mode = args.I_mode
    # S_mode = args.S_mode
    # D_mode = args.D_mode
    # remove_punctuation = args.remove_punctuation
    # remove_space = args.remove_space
    # content = []
    # with open(input_file, "r") as input_f:
    #     for refined_line in input_f:
    #         refined_line = I(refined_line, I_mode)
    #         refined_line = S(refined_line, S_mode)
    #         refined_line = D(refined_line, D_mode)
    #         refined_line = refined_line.replace("\n", "")
    #         if remove_punctuation:
    #             refined_line = refined_line.replace("、", "")
    #         if remove_space:
    #             refined_line = refined_line.replace(" ", "")
    #         content.append(refined_line)
    # with open(output_file, "w") as output_f:
    #     output_f.write("\n".join(content))

# if __name__ == "__main__":
#     main()