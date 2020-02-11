import pafprocess
import numpy as np


def test():
    peaks = np.zeros((200,200,18), dtype=np.float32)
    heat_mat = np.zeros((200,200,18), dtype=np.float32)
    paf_mat = np.zeros((200,200,17), dtype=np.float32)

    pafprocess.process_paf(peaks, heat_mat, paf_mat)
    print(pafprocess.get_num_humans())

if __name__ == "__main__":
    test()

