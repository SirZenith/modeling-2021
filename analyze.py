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
    plt.plot(target.supply_rate_all)

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
    print(f'   履约率：{target.long_term_supply_rate}')
    print(f'履约率方差：{target.supply_rate.var()}')


def circle_map(x):
    if x > 1:
        x = 1 - x
    elif x > 0:
        x = 1 - 1 / x
    else:
        raise ValueError('input argument should be greater than zero')
    return np.exp(x)


def rate_leap(target: TransicationRecord) -> np.ndarray:
    req_ratio = np.fromiter(
        (0 if v1 * v2 < 1 else v2 / v1 for v1, v2 in zip(target.requests[:-1], target.requests[1:])),
        dtype=float
    )
    rate_diff = np.diff(target.supply_rate_all)
    diff = np.abs(rate_diff) * req_ratio
    return diff
    

def all_rate_leap(tc: "list[TransicationRecord]") -> "list[np.ndarray]":
    results = []
    # conv_vec = TransicationRecord.local_conv_vec(25)
    for t in tc:
        diff = rate_leap(t)
        # local = np.convolve(conv_vec, diff, mode='same')
        # if np.any(np.abs(diff - local) > 0.2):
            # results.append((i, diff))
        results.append(diff)
    return results


if __name__ == '__main__':
    import argparse

    data_dire = 'data'
    transication_bin = os.path.join(data_dire, 'transication.bin')
    transport_bin = os.path.join(data_dire, 'transport.bin')

    tc = TransicationRecord.from_pickled(transication_bin)
    tp = TransportRecord.from_pickled(transport_bin)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', default=None, type=int,
                        metavar='<id>', help='plot data with given supplier id.')
    parser.add_argument('-o', '--output', default=None, type=str,
                        metavar='<file name>', help='write data to csv file')
    parser.add_argument('-i', '--info', default=None, type=int,
                        metavar='<id>', help='print infomation with given id.')
    parser.add_argument('-d', '--diff', action='store_true')

    parser.add_argument('-l', '--leap', default=None, type=int,
                        metavar='<id>', help='compute leap value in requests amount')
    parser.add_argument('-L', '--all-leap', action='store_true', dest='all_leap',
                        help='drawing scatter plot for leap point count for all supplier')

    args = parser.parse_args()
    if args.plot is not None:
        make_plot(tc[args.plot - 1])
    if args.output is not None:
        write_csv(tc, args.output)
    if args.info is not None:
        printinfo(tc[args.info - 1])
    if args.leap is not None:
        target = tc[args.leap - 1]
        leap = rate_leap(target)
        plt.plot(leap)
        plt.plot(np.array([leap.mean()] * leap.size))
        plt.title(target.id)
        plt.show()
    if args.all_leap:
        leaps = all_rate_leap(tc)
        leap_count = [np.count_nonzero(l > 20) for l in leaps]
        plt.plot(np.array(leap_count))
        plt.title('Leap Count')

        output  = sorted(range(1, len(leaps) + 1), key=lambda i: leap_count[i - 1])
        for i in output:
            print(i)

        plt.show()
