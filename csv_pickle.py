import glob
import os
import pickle

import numpy as np

from modeling import TransicationRecord, TransportRecord


def loop_gen(i: int):
    def gen(n):
        return (-1)**(n // i % 2) * (n % i + 1)
    return gen

if __name__ == '__main__':
    data_dire = 'data'
    typing_table = {
        'requests': TransicationRecord,
        'supply': TransicationRecord,
        'transport': TransportRecord,
    }

    loop_vectors = [
        np.fromfunction(loop_gen(i), (TransicationRecord.WEEK_COUNT,))
        for i in range(1, TransicationRecord.WEEK_COUNT + 1)
    ]

    for item in glob.glob(os.path.join(data_dire, '*.csv'),):
        if not os.path.isfile(item):
            continue
        without_ext = os.path.splitext(item)[0]
        out_name = without_ext + '.bin'
        basename = os.path.basename(without_ext)
        data = typing_table[basename].from_csv(item, loop_vectors)
        with open(out_name, 'wb+') as f:
            pickle.dump(data, f)
        print(f'{item} -> {out_name}')
