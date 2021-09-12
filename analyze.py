import csv
import glob
import os
import math

import matplotlib.pyplot as plt
import numpy as np

from modeling import check_pickle
from modeling import StatusOfWeek, TransicationRecord, TransportRecord, TransportDistributor


def performance(r: TransicationRecord):
    """used globally for supplier evaluation."""
    return r.supply.mean()**2 * np.exp(r.supply_rate.mean()) * (2 - r.supply_rate.var()) * r.requests[r.requests > 0].size / r.requests.size


def plot_all(tc: "list[TransicationRecord]"):
    """draw plot for some information of all time/all supplier wide"""
    plt.figure()
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


def performance_sort(tc: "list[TransicationRecord]", filename: str):
    """write data into csv file after sortting input by performance function."""
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


def transport_task_distribute(
    tc: "list[TransicationRecord]",
    tp: "list[TransportRecord]",
    output: str,
):
    requests = None
    with open('ans/q4.csv', 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        requests = [[int(i) for i in r] for r in reader]
    requests = [(i, r) for i, r in zip(range(0, len(requests)), requests)]
    requests.sort(key=lambda r: tc[r[0]].src_type)

    distributor = TransportDistributor(tp)
    for week_index in range(TransicationRecord.WEEK_COUNT):
        distributor.reset()
        for index, r in requests:
            distributor.distribute(index, week_index, r[week_index])
    if output:
        with open(output, 'w+', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            for r in distributor.dist_record:
                r = np.hstack(r)
                writer.writerow(r)


def question2(tc: "list[TransicationRecord]", output: str, draw: bool):
    """generate requests for question 2"""
    target_value = 3 * 2.82e4

    results = []
    this_week = StatusOfWeek()
    tc.sort(key=performance, reverse=True)

    ed = 240  # temporary putting this data
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
        print('{} {}'.format(
            this_week.inventory,
            TransportRecord.MAX_CAP * TransportRecord.TRANSPORT_COUNT - this_week.can_trans
        ))
        this_week.producing()
        if this_week.inventory < 0:
            raise ValueError
        this_week.current_week += 1

    results = np.array(results)
    results = results.T
    count = 0
    if output:
        with open(output, 'w+', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            for r in results:
                writer.writerow(r)
                if np.any(r):
                    count += 1
    print(count)
    if draw:
        plt.figure()
        for r in results:
            plt.plot(r)
        plt.show()
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
                        metavar='<id>', help='plotting data for supplier with given integer supplier id.')
    parser.add_argument('-P', '--all-plot', action='store_true', dest='all_plot',
                        help='plotting storage amount of resource during all weeks.')
    parser.add_argument('-o', '--output', default=None, type=str,
                        metavar='<file name>', help='output file name, if a command support writing file, this name is used.')
    parser.add_argument('-i', '--info', default=None, type=int,
                        metavar='<id>', help='print out information for supplier with given integer id.')
    parser.add_argument('-I', '--sorted-info', action='store_true', dest='sorted_info',
                        help='write supplier infoamtion list sorted by performance into file..')
    parser.add_argument('-l', '--leap', default=None, type=int,
                        metavar='<id>', help='find supply rate irregular leap point in supplier\'s supply record data.')
    parser.add_argument('-L', '--all-leap', action='store_true', dest='all_leap',
                        help='drawing scatter plot for leap point count of all supplier.')
    parser.add_argument('-e', '--explosive', action='store_true',
                        help='print out supplier list sorted by irregular leap point count.')
    parser.add_argument('-g', '--gini', default=None, type=int,
                        metavar='<id>', help='compute Gini coeffectient for supplier with given integer id.')
    parser.add_argument('-s', '--solve', default=None, type=int,
                        metavar="<number>", help='give request paln for given question (2, 3, 4). Result will be written into csv file if flag -o (--output) is passed.')
    parser.add_argument('-D', '--image', action='store_true',
                        help='drawing image while giving request solution.')
    parser.add_argument('-t', '--transport', default=None, type=str,
                        metavar="<number>", help='generate transport plan for given requests plan. Requests are give in csv format, with each line recording all request to one supplier during all weeks.')

    args = parser.parse_args()
    if args.all_plot:
        plot_all(tc)
    elif args.plot is not None:
        make_plot(tc[args.plot - 1])
    if args.sorted_info is not None:
        performance_sort(tc, args.output)
    if args.info is not None:
        printinfo(tc[args.info - 1])
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
        solutions[args.solve](tc, args.output, args.image)
    if args.transport is not None:
        transport_task_distribute(tc, tp, args.output)
