import csv
import glob
import os
import math

import matplotlib.pyplot as plt
import numpy as np

from csv_pickle import csv_pickle
from modeling import StatusOfWeek, TransicationRecord, TransportRecord


def check_pickle(src: "list[str]", targets: "list[str]"):
    """automatically pickle data if any of src file is newer than target files"""
    src_time = np.array([os.path.getmtime(item) for item in src])
    targets_time = np.array([os.path.getmtime(item) for item in targets])
    for time in targets_time:
        if np.any(src_time > time):
            csv_pickle()
            print('new pickle data were successfully made.')
            break


def performance(r: TransicationRecord):
    """used globally for supplier evaluation."""
    return r.supply.mean()**2 * np.exp(r.supply_rate.mean()) * (2 - r.supply_rate.var()) * r.requests[r.requests > 0].size / r.requests.size


def plot_all(tc: "list[TransicationRecord]"):
    plt.figure()
    # value = np.array([
    #     sum(t.supply[i] / t.src_type.unit_cost for t in tc)
    #     for i in range(TransicationRecord.WEEK_COUNT)
    # ])
    # plt.plot(value)
    # plt.plot([value.mean()] * TransicationRecord.WEEK_COUNT)
    # print(value.mean())
    # for t in tc:
    # if t.gini > 0.3:
    #     continuew
    # plt.plot(t.supply)
    sum = np.zeros(TransicationRecord.WEEK_COUNT)
    for t in tc:
        sum += t.supply
    storage = [None] * TransicationRecord.WEEK_COUNT
    curr = 0
    for i, s in enumerate(sum):
        curr += s
        curr = max(s - 28200, 0)
        storage[i] = s
    print(len(storage))
    plt.plot(storage)
    plt.show()


def make_plot(target: TransicationRecord):
    plt.figure()
    plt.title("Info of {}".format(target.id))

    plt.subplot(4, 1, 1)
    plt.title("Supply and requests", fontsize='small')
    plt.plot(target.requests, 'r--')
    plt.plot(target.supply)
    mean = target.supply.mean()
    plt.plot([mean] * target.supply.size)

    plt.subplot(4, 1, 2)
    plt.title(r'$\Delta$supply', fontsize='small')
    plt.plot(target.supply - target.requests)

    plt.subplot(4, 1, 3)
    plt.title('Supply rate', fontsize='small')
    plt.plot(target.supply_rate_all)

    plt.subplot(4, 1, 4)
    plt.title('Supply burst', fontsize='small')
    plt.plot(target.supply_burst)
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


def rate_leap(target: TransicationRecord) -> np.ndarray:
    """finding irregular leap point in supply data."""
    req_ratio = np.fromiter(
        (0 if v1 * v2 < 1 else v1 / v2 for v1,
         v2 in zip(target.requests[:-1], target.requests[1:])),
        dtype=float
    )
    rate_diff = np.diff(target.supply_rate_all)
    rate_diff[rate_diff > 0] = 0
    diff = np.abs(rate_diff) * req_ratio
    return diff


def all_rate_leap(tc: "list[TransicationRecord]") -> "list[np.ndarray]":
    results = []
    for t in tc:
        diff = rate_leap(t)
        results.append(diff)
    return results


def question2(tc: "list[TransicationRecord]", output: str):
    """generate requests for question 2"""
    target_value = 3 * 2.82e4

    results = []
    this_week = StatusOfWeek()
    tc.sort(key=performance, reverse=True)

    ed = 50  # temporary putting this data
    gini_bound = 0.5

    for _ in range(24):
        this_week.reset()
        for t in filter(lambda t: t.gini < gini_bound, tc[:ed]):
            # normal type supplier
            if this_week.no_need_more(target_value):
                break
            this_week.request_to_normal(t)

        for t in filter(lambda t: t.gini >= gini_bound, tc[:ed]):
            # burst type supplier
            if this_week.no_need_more(target_value):
                break
            if this_week.buy_next_time[t.id_int] > this_week.current_week:
                continue
            this_week.request_to_burst(t)

        results.append(this_week.requests.copy())
        # print('{} {}'.format(
        #     this_week.inventory,
        #     TransportRecord.max_cap() - this_week.can_trans
        # ))
        this_week.producing()
        this_week.current_week += 1

    results = np.array(results)
    results = results.T
    plt.figure()
    for r in results:
        plt.plot(r)
    plt.show()
    count = 0
    if output:
        with open(output, 'w+', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            for r in results:
                writer.writerow(r)
                if np.any(r):
                    count += 1
    print(count)
    return results


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
    parser.add_argument('-p', '--plot', default=None, type=int,
                        metavar='<id>', help='plot data with given supplier id.')
    parser.add_argument('-P', '--all-plot', action='store_true', dest='all_plot',
                        help='plot data with given supplier id.')
    parser.add_argument('-o', '--output', default=None, type=str,
                        metavar='<file name>', help='write data to csv file')
    parser.add_argument('-i', '--info', default=None, type=int,
                        metavar='<id>', help='print infomation with given id.')
    parser.add_argument('-d', '--diff', action='store_true')

    parser.add_argument('-l', '--leap', default=None, type=int,
                        metavar='<id>', help='compute leap value in requests amount')
    parser.add_argument('-L', '--all-leap', action='store_true', dest='all_leap',
                        help='drawing scatter plot for leap point count for all supplier')
    parser.add_argument('-g', '--gini', default=None, type=int,
                        metavar='<id>', help='compute Gini coeffectient of a given supplier')
    parser.add_argument('-e', '--explosive', action='store_true',
                        help='sort the supplier with its explosion')
    parser.add_argument('-s', '--solve', default=None, type=int,
                        metavar="<number>", help='give solution for give question')

    args = parser.parse_args()
    if args.all_plot:
        plot_all(tc)
    elif args.plot is not None:
        make_plot(tc[args.plot - 1])
    if args.output is not None:
        write_csv(tc, args.output)
    if args.info is not None:
        printinfo(tc[args.info - 1])

    # tc.sort(key=lambda x: x.gini * math.log(x.supply_rate.mean()) * x.supply.mean() * x.long_term_supply_rate, reverse=True)
    # for i in range(10):
        # print("{} {}".format(tc[i].id, tc[i].gini))
    if args.leap is not None:
        target = tc[args.leap - 1]
        leap = rate_leap(target)
        plt.plot(leap)
        plt.plot(np.array([max(5 * leap.mean(), 5)] * leap.size))
        plt.title(target.id)
        plt.show()
    if args.all_leap:
        leaps = all_rate_leap(tc)
        leap_count = [np.count_nonzero(
            l > max(5 * l.mean(), 5)) for l in leaps]
        plt.plot(np.array(leap_count))
        plt.title('Leap Count')

        output = sorted(range(1, len(leaps) + 1),
                        key=lambda i: leap_count[i - 1])
        for i in output:
            print(f'{i}: {leap_count[i - 1]}')
        print(np.mean(leap_count))
        plt.show()

    if args.gini is not None:
        gini, x, y = tc[args.gini - 1].compute_gini()
        print(gini)
        plt.plot(x, y)
        plt.plot(x, x)  # 均衡曲线
        plt.show()

    if args.explosive:
        tc.sort(key=lambda x: x.gini * math.log(x.supply_rate.mean())
                * x.supply.mean() * x.long_term_supply_rate, reverse=True)
        for i in range(10):
            print("{} {} {} {} {}".format(
                tc[i].id,
                tc[i].gini,
                tc[i].burst_config.burst_dura,
                tc[i].burst_config.cooling_dura,
                tc[i].burst_config.max_burst_output
            ))

    if args.solve is not None:
        solutions = (
            None,
            None,
            question2,
        )
        solutions[args.solve](tc, args.output)
        
