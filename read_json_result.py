import os
import json
import argparse
import MeCab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    input_file = args.input_file
    output_path= args.output_path
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "data.1.json")
    # input_file = "/home/geng17/espnet/egs/csj/asr1/exp/train_nodup_sp_pytorch_train_pytorch_transformer_lr5.0_ag8.v2/decode_sp20161013refined_evalset_clean_True_2_decode_lm/data.json"
    # output_file = "/home/geng17/espnet/egs/csj/asr1/exp/train_nodup_sp_pytorch_train_pytorch_transformer_lr5.0_ag8.v2/decode_test/data.1.json"
    with open(input_file, 'r') as f:
        a = json.load(f)
    utts = a['utts']

    for utt in utts:
        utt_dict = utts[utt]
        attribute = utt_dict['output'][0]
        rec_text = attribute["rec_text"]
        rec_text = rec_text.replace('<eos>', '')
        rec_tokenid = attribute["rec_tokenid"].split(' ')
        eos_id = rec_tokenid[-1]
        rec_tokenid = rec_tokenid[:-1]

        mecab = MeCab.Tagger()
        mecab.parse('')  # 文字列がGCされるのを防ぐ
        node = mecab.parseToNode(rec_text)

        # zip(rec_text, rec_tokenid)
        text_label = [0 for x in range(len(rec_text))]
        point = 0
        while node:
            word = node.surface
            pos = node.feature.split(",")[0]
            # print("%s, %s"%(word, pos))
            if pos == "フィラー" or pos == "感動詞":
                text_label[point: point + len(word)] = [1 for x in range(len(word))]
            point += len(word)
            node = node.next

        rec_text_ = []
        rec_tokenid_ = []
        for w, w_id, l in zip(rec_text, rec_tokenid, text_label):
            if l == 0:
                rec_text_.append(w)
                rec_tokenid_.append(w_id)
        rec_token_ = ' '.join(rec_text_) + ' <eos>'
        rec_text_ = ''.join(rec_text_) + '<eos>'
        rec_tokenid_ = ' '.join(rec_tokenid_) + ' ' + eos_id
        attribute["rec_text"] = rec_text_
        attribute["rec_token"] = rec_token_
        attribute["rec_tokenid"] = rec_tokenid_
        utt_dict['output'][0] = attribute
        utts[utt] = utt_dict
    a['utts'] = utts
    with open(output_file, 'w') as f_w:
        json.dump(a, f_w, ensure_ascii=False, indent=4, separators=(',', ':'))


if __name__ == "__main__":
    main()
