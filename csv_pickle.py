import glob
import os
import pickle

import numpy as np

from modeling import Record, TransicationRecord, TransportRecord


def loop_gen(i: int):
    def gen(n):
        return (-1)**(n // i % 2) * (n % i + 1)
    return gen


def csv_pickle():
    data_dire = 'data'
    requests_csv = os.path.join(data_dire, 'requests.csv')
    supply_csv = os.path.join(data_dire, 'supply.csv')
    transport_csv = os.path.join(data_dire, 'transport.csv')
    # typing_table = {
    #     'requests': TransicationRecord,
    #     'supply': TransicationRecord,
    #     'transport': TransportRecord,
    # }

    # loop_vectors = [
    #     np.fromfunction(loop_gen(i), (TransicationRecord.WEEK_COUNT,))
    #     for i in range(1, TransicationRecord.WEEK_COUNT + 1)
    # ]

    tc = TransicationRecord.from_csv(supply_csv, requests_csv)
    tp = TransportRecord.from_csv(transport_csv)

    Record.to_pickled(os.path.join(data_dire, 'transication.bin'), tc)
    Record.to_pickled(os.path.join(data_dire, 'transport.bin'), tp)


def check_pickle(src: "list[str]", targets: "list[str]"):
    """automatically pickle data if any of src file is newer than target files"""
    src_time = np.array([os.path.getmtime(item) for item in src])
    targets_time = np.array([os.path.getmtime(item) for item in targets])
    for time in targets_time:
        if np.any(src_time > time):
            csv_pickle()
            print('new pickle data were successfully made.')
            break


if __name__ == '__main__':
    csv_pickle()
