import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from modeling import TransicationRecord, TransportRecord


def performance(r: TransicationRecord):
    """used globally for supplier evaluation."""
    return r.supply.mean()**2 * np.exp(r.supply_rate.mean()) * (2 - r.supply_rate.var()) * r.requests[r.requests > 0].size / r.requests.size


def make_plot(target: TransicationRecord):
    plt.figure()
    plt.title("Info of {}".format(target.id))

    plt.subplot(4, 1, 1)
    plt.title("Supply and requests", fontsize='small')
    plt.plot(target.supply)
    plt.plot(target.requests)
    mean = target.supply.mean()
    plt.plot([mean] * target.supply.size)

    plt.subplot(4, 1, 2)
    plt.title(r'$\Delta$supply', fontsize='small')
    plt.plot(target.supply - target.requests)

    plt.subplot(4, 1, 3)
    plt.title('Supply rate', fontsize='small')
    plt.plot(target.supply_rate)

    plt.subplot(4, 1, 4)
    plt.title('Local burst', fontsize='small')
    plt.plot(target.local_burst)
    plt.show()


def write_csv(tc: "list[TransicationRecord]", filename: str):
    """write data into csv file after sort input by performance function."""
    tc.sort(key=performance, reverse=True)
    with open(filename, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'r_count', 'type', 's_rate_mean',
                        's_rate_variance', 'long_s_rate', 's_delta', 'score'])
        for target in tc:
            writer.writerow([
                target.id,
                target.requests[target.requests > 0].size,
                target.src_type.value,
                target.supply.mean(),
                target.supply_rate.mean(),
                target.supply_rate.var(),
                target.long_term_supply_rate,
                target.supply_delta,
                performance(target),
            ])

def printinfo(target: TransicationRecord):
    print(f'供应商 ID：{target.id}')
    print(f'供货量均值：{target.supply.mean()}')
    print(f'  订单总量：{target.requests[target.requests >= 1].size}')
    print(f'供货差均值：{target.supply_delta}')
    print(f'   履约率：{target.supply_rate.mean()}')
    print(f'履约率方差：{target.supply_rate.var()}')


if __name__ == '__main__':
    import argparse

    data_dire = 'data'
    transication_bin = os.path.join(data_dire, 'transication.bin')
    transport_bin = os.path.join(data_dire, 'transport.bin')

    tc = TransicationRecord.from_pickled(transication_bin)
    tp = TransportRecord.from_pickled(transport_bin)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', default=None, type=int, metavar='<id>', help='plot data with given supplier id.')
    parser.add_argument('-o', '--output', default=None, type=str, metavar='<file name>', help='write data to csv file')
    parser.add_argument('-i', '--info', default=None, type=int, metavar='<id>', help='print infomation with given id.')

    args = parser.parse_args()
    if args.plot is not None:
        make_plot(tc[args.plot - 1])
    if args.output is not None:
        write_csv(tc, args.output)
    if args.info is not None:
        printinfo(tc[args.info - 1])

    print(tc[228].co)