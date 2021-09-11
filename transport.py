import csv
import glob
import os

import numpy as np

from csv_pickle import check_pickle
from modeling import TransicationRecord, TransportRecord, TransportDistributor


def question2(
    tc: "list[TransicationRecord]",
    tp: "list[TransportRecord]",
    output: str,
    draw: bool
):
    requests = None
    with open('ans/q2_requests.csv', 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        requests = [[int(i) for i in r] for r in reader]
    requests = [(i, r) for i, r in zip(range(0, len(requests)), requests)]
    requests.sort(key=lambda r: tc[r[0]].src_type)

    distributor = TransportDistributor(tp)
    for week_index in range(24):
        distributor.reset()
        for index, r in requests:
            distributor.distribute(index, week_index, r[week_index])
    if output:
        with open(output, 'w+', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            for r in distributor.dist_record:
                r = np.hstack(r)
                writer.writerow(r)


if __name__ == '__main__':
    import argparse

    data_dire = 'data'
    transication_bin = os.path.join(data_dire, 'transication.bin')
    transport_bin = os.path.join(data_dire, 'transport.bin')

    targets = [transication_bin, transication_bin]
    src = glob.glob(os.path.join(data_dire, '*.csv')) + [
        'csv_pickle.py', 'modeling.py'
    ]
    check_pickle(src, targets)

    tc = TransicationRecord.from_pickled(transication_bin)
    tp = TransportRecord.from_pickled(transport_bin)

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--solve', default=None, type=int,
                        metavar="<number>", help='give solution for give question')
    parser.add_argument('-D', '--image', action='store_true',
                        help='drawing image for solution')
    parser.add_argument('-o', '--output', default=None, type=str,
                        metavar='<file name>', help='write data to csv file')

    args = parser.parse_args()
    if args.solve is not None:
        solutions = (
            None,
            None,
            question2,
        )
        solutions[args.solve](tc, tp, args.output, args.image)
