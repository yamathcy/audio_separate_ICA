import numpy as np
import scipy as sp
import scipy.io.wavfile as wf
from ica import ICA

# 混合音の作成
def make_mixture():

    # 元の音のファイルを読み込み
    rate1, data1 = wf.read('voice1.wav')
    rate2, data2 = wf.read('voice2.wav')
    rate3, data3 = wf.read('voice3.wav')
    min_length = min([len(data1), len(data2), len(data3)])


    # サンプリングレートが不一致ならエラー
    if rate1 != rate2 or rate2 != rate3:
        raise ValueError('sampling_rate_Error')
    # 長さは一番短いのに揃える
    data1 = data1[:min_length]
    data2 = data2[:min_length]
    data3 = data3[:min_length]

    # 混合割合
    mix_1 = data1 * 0.6 + data2 * 0.3 + data3 * 0.1
    mix_2 = data1 * 0.3 + data2 * 0.2 + data3 * 0.5
    mix_3 = data1 * 0.1 + data2 * 0.5 + data3 * 0.4

    # 音ファイルに混合音を書き出し
    y = [mix_1, mix_2, mix_3]
    y = [(y_i * 32767 / max(np.absolute(y_i))).astype(np.int16) for y_i in np.asarray(y)]
    wf.write('mix_1.wav', rate1, y[0])
    wf.write('mix_2.wav', rate2, y[1])
    wf.write('mix_3.wav', rate3, y[2])

def separation():

    # 混合音を読み込み
    rate1, data1 = wf.read('mix_1.wav')
    rate2, data2 = wf.read('mix_2.wav')
    rate3, data3 = wf.read('mix_3.wav')
    if rate1 != rate2 or rate2 != rate3:
        raise ValueError('Sampling_rate_Error')

    # データをfloatに変換
    data = [data1.astype(float), data2.astype(float), data3.astype(float)]

    # ICAを実行
    y = ICA(data).ica()
    print(y)

    # 分離結果を音ファイルとして保存1
    y = [(y_i * 32767 / max(np.absolute(y_i))).astype(np.int16) for y_i in np.asarray(y)]

    wf.write('separate1.wav', rate1, y[0])
    wf.write('separate2.wav', rate2, y[1])
    wf.write('separate3.wav', rate3, y[2])

def main():
    mode = input('混合音の作成：0，分離：1, 終了：2')
    if mode == '0':
       make_mixture()
    elif mode == '1':
        separation()
    else:
        print('終了します')



if __name__ == '__main__':
    main()