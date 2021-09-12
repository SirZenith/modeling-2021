import csv
from enum import Enum
import enum
import glob
import os
import pickle
from typing import Callable, Type

import numpy as np


class SrcType(bytes, Enum):
    """resource type, one of A, B, C"""
    def __new__(cls, value: int, type_name: str, unit_cost: float, price: int):
        obj = bytes.__new__(cls, [value])
        obj._value_ = type_name.upper()
        obj.unit_cost = unit_cost
        obj.price = price
        return obj

    A = (0, 'A', 0.60, 1.2)
    B = (1, 'B', 0.66, 1.1)
    C = (2, 'C', 0.72, 1.0)


class BurstConfig(object):
    """BurstConfig recording burst features of a supplier

    Attribute:
        burst_dura: int, how long does a burst last.
        cooling_dura: int, how long does cooling last.
        burst_supply_count: int, averange request count during a burst.
        max_burst_output: int, maximum value supplier can provide during a burst.
        burst_var: float, variance of supply amount during burst.
        burst_mean: flaot, mean of supply amount during burst.
    """

    def __init__(self, s_burst: np.ndarray, s_data: np.ndarray):
        self.burst_dura = TransicationRecord.WEEK_COUNT
        self.cooling_dura = 0
        self.burst_supply_count = 0
        self.max_burst_output = 0
        self.burst_var = 0
        self.burst_mean = 0

        durations = self.find_burst_duration(s_burst)
        self.settle_arguments(durations, s_data)

    def find_burst_duration(self, s_burst: np.ndarray):
        """finding week index of the beginning and ending of burst."""
        threadshold = 5
        durations = np.empty((0, 3), dtype=int)
        burst_st = 0
        last_burst = 0
        burst_count = 0
        is_bursting = False
        for i, item in enumerate(s_burst):
            if not item:
                if is_bursting and (i - last_burst > threadshold or i == s_burst.size - 1):
                    durations = np.append(
                        durations,
                        ((burst_st, last_burst + 1, burst_count),),
                        axis=0
                    )
                    is_bursting = False
                    burst_count = 0
                continue
            last_burst = i
            burst_count += 1
            if not is_bursting:
                is_bursting = True
                burst_st = i
        return durations

    def settle_arguments(self, durations: np.ndarray, s_data: np.ndarray):
        if not np.any(durations):
            return
        burst_len = durations[:, 1] - durations[:, 0]
        cooling_len = durations[1:, 0] - durations[:-
                                                   1, 1] if len(durations) != 1 else 240
        self.burst_dura = np.median(burst_len)
        self.cooling_dura = np.median(cooling_len)
        self.burst_supply_count = durations[:, 2].mean()

        index = []
        for st, ed, _count in durations:
            index.extend([i for i in range(st, ed)])
        burst_values = s_data[index]
        self.max_burst_output = burst_values.max()
        self.burst_var = burst_values.var()
        self.burst_mean = burst_values.mean()


class Record(object):
    """General record type."""
    WEEK_COUNT = 240

    @classmethod
    def from_pickled(cls, filename: str) -> "list[Record]":
        """Reading record from pickled binary file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    @classmethod
    def to_pickled(cls, filename: str, data: "list[Record]"):
        with open(filename, 'wb+') as f:
            pickle.dump(data, f)


class TransicationRecord(Record):
    """TransicationRecord records requests sent, or resource supplied in past 240
    weeks.
    Attribute:
        id: str, id of supplier.
        id_int: int, id of supplier.
        src_type: SrcType, source type of this supplier.
        supply: numpy.ndarray, array of supply data.
        requests: numpy.ndarray, array of requests data.
        supply_delta: numpy.ndarray, difference between supply and request, only
                      non-zero requests are counted.
        supply_rate: numpy.ndarray, supply rate of each request.
        supply_rate: numpy.ndarray, supply rate of all time, week with 0 request
                     will take 1 (100%) as supply rate.
        long_term_supply_rate: float, ratio of sum of supply data to sum of requests.
        gini: float, Gini coeffecitent of supply data.
        request_burst: numpy.ndarray, filte local huge requests.
        request_burst: numpy.ndarray, filte local leap of supply amount.
        co: float, relevent coefficient. 
    """
    SUPPLIER_COUNT = 402
    LOCAL_LEN = 20

    def __init__(
        self,
        id: str,
        id_int: int,
        src_type: SrcType,
        supply_data: "list[float]",
        # loop_vectors: "list[np.ndarray]"=None,
        requests_data: "list[float]" = None,
    ):
        self.id = id
        self.id_int = id_int
        self.src_type = src_type
        self.supply = np.array(supply_data)
        self.requests = np.array(requests_data) if requests_data else None
        # self.freqs = np.array([
        #     abs(v @ self.supply.T) / Record.WEEK_COUNT for v in loop_vectors
        # ])

        self.supply_delta = None
        self.supply_rate = None
        self.supply_rate = None
        self.long_term_supply_rate = None

        self.gini, _, _ = self.compute_gini()
        self.request_burst = None
        self.supply_burst = None
        self.burst_config = None

    @classmethod
    def from_csv(
        cls,
        supply_csv: str,
        requests_csv: str,
    ) -> "list[TransicationRecord]":
        """read csv data (the first line of csv file should be table heade), and
        generate TransicationRecord list"""
        results = [None] * cls.SUPPLIER_COUNT
        with open(supply_csv, 'r', encoding='utf8') as s:
            reader = csv.reader(s)
            _header = next(reader)
            for row in reader:
                src_t = SrcType(row[1])
                sid = int(row[0][1:]) - 1
                data = [float(i) for i in row[2:]]
                results[sid] = TransicationRecord(row[0], sid, src_t, data)
        with open(requests_csv, 'r', encoding='utf8') as r:
            reader = csv.reader(r)
            _header = next(reader)
            for row in reader:
                sid = int(row[0][1:]) - 1
                data = [float(i) for i in row[2:]]
                results[sid].requests = np.array(data)
        for r in results:
            r.update_request_state()
        return results

    @classmethod
    def local_conv_vec(cls, local_len: int = 0) -> np.ndarray:
        if local_len <= 0:
            local_len = cls.LOCAL_LEN
        return np.array([1 / local_len for _ in range(local_len)])

    def update_request_state(self):
        if self.requests is None:
            return
        mask = self.requests >= 1
        if not np.any(mask):
            return
        self.supply_delta = (self.supply[mask] - self.requests[mask]).mean()
        self.supply_rate = self.supply[mask] / self.requests[mask]

        self.supply_rate_all = np.ones(self.requests.shape)
        self.supply_rate_all[mask] = self.supply[mask] / self.requests[mask]

        self.long_term_supply_rate = self.supply.sum() / self.requests.sum()

        self.find_burst()
        self.burst_config = BurstConfig(self.supply_burst, self.supply)

    def find_burst(self):
        import matplotlib.pyplot as plt
        """finding local burst of supply amount and request amount."""
        conv_local = TransicationRecord.local_conv_vec()
        r_local_mean = np.convolve(conv_local, self.requests, mode='same')
        self.request_burst = self.requests > r_local_mean * 1.5

        s_local_mean = np.convolve(conv_local, self.supply, mode='same')
        self.supply_burst = self.supply > s_local_mean * 1.5
        # plt.plot(s_local_mean)
        # plt.show()

    @property
    def co(self):
        tmpmat = np.diag(np.ones(240)) + np.diag(np.ones(239),
                                                 k=1) + np.diag(np.ones(238), k=2)
        x = np.matmul(tmpmat, self.supply)
        y = self.supply / (self.requests+0.01)
        self.co = np.corrcoef(x, y)[0, 1]

    def compute_gini(self):
        # 计算数组累计值,从 0 开始
        wealths = self.supply.copy()
        wealths = np.append(wealths, 0)
        wealths.sort()
        cum_wealths = np.cumsum(wealths)
        sum_wealths = cum_wealths[-1]
        # 将数据转换为累积量在总量中的占比
        xarray = np.arange(0, len(cum_wealths)) / (cum_wealths.size - 1)
        yarray = cum_wealths / sum_wealths
        area_supply = np.trapz(yarray, x=xarray)
        # 总面积 0.5
        area_delta = 0.5 - area_supply
        return area_delta / 0.5, xarray, yarray


class TransportRecord(Record):
    """TransportRecord records transportation data.
    Attribute:
        id: str, id of a transport company.
        data: numpy.ndarray, cost of this company is past weeks.
    """
    TRANSPORT_COUNT = 8
    MAX_CAP = 6000

    def __init__(self, id: str, data: "list[float]"):
        self.id = id
        self.data = np.array(data)

    @classmethod
    def from_csv(cls, filename: str) -> "list[TransportRecord]":
        """read csv data (the first line of csv file should be table heade), and
        generate TransportRecord list"""
        results = [None] * cls.TRANSPORT_COUNT
        with open(filename, 'r', encoding='utf8') as f:
            reader = csv.reader(f)
            _header = next(reader)
            for row in reader:
                supply_id = int(row[0][1:]) - 1
                data = [float(i) for i in row[1:]]
                results[supply_id] = TransportRecord(row[0], data)
        return results


class StatusOfWeek():
    '''
        self.inventory is a list of the inventory after you buy this week.
        self.requests is a list of each requests of 402 suppliers.
        self.expect_supply is a list of expect supply this week.
        self.current is a number of 0-24
        self.buy_next_time is a list of length 402, which record the supplier you should buy next time
        self.can_trans
    '''

    def __init__(self, source_cost):
        self.inventory = 0
        self.requests = np.zeros(402, dtype=int)
        self.expect_supply = np.zeros(402, dtype=int)
        self.current_week = 0
        self.buy_next_time = np.zeros(402, dtype=int)
        self.burst_count = np.zeros(402, dtype=int)
        self.can_trans = TransportRecord.MAX_CAP * TransportRecord.TRANSPORT_COUNT
        self.source_cost = source_cost

    def producing(self):
        self.inventory -= self.source_cost

    def reset(self):
        self.reset_can_trans()
        self.reset_requests()

    def reset_can_trans(self):
        self.can_trans = TransportRecord.MAX_CAP * TransportRecord.TRANSPORT_COUNT

    def reset_requests(self):
        self.requests[:] = 0

    def no_need_more(self):
        return self.inventory >= self.source_cost * 3 or self.can_trans <= 0

    def request_to_normal(self, t: TransicationRecord):
        """sending a request to a normal-type supplier"""
        id = t.id_int
        request = min(t.requests.mean(), self.can_trans)
        request = round(request)
        self.requests[id] = request
        self.expect_supply[id] = min(t.supply.mean(), self.can_trans)
        self.inventory += t.supply.mean() / t.src_type.unit_cost
        self.can_trans -= self.requests[id]

    def request_to_burst(self, t: TransicationRecord):
        """sending a request to a burst-type supplier"""
        conf = t.burst_config
        id = t.id_int
        requests = max(
            conf.max_burst_output,
            conf.max_burst_output / t.supply_rate.mean()
        )
        requests = round(requests)
        if requests > self.can_trans:
            return
        self.requests[id] = requests
        self.expect_supply[id] = conf.max_burst_output
        self.inventory += conf.max_burst_output / t.src_type.unit_cost
        self.burst_count[id] -= 1
        self.buy_next_time[id] = self.current_week + \
            conf.burst_dura // conf.burst_supply_count
        self.can_trans -= self.requests[id]
        if self.burst_count[id] <= 0:
            self.burst_count[id] = conf.burst_supply_count
            self.buy_next_time[id] = self.current_week + conf.cooling_dura


class TransportDistributor(object):
    """TransportDistributor is used to distribute transport task among transport
    companies."""

    def __init__(
        self,
        companies: "list[TransportRecord]",
        performance: "Callable[[TransicationRecord], float]"=None
    ):
        if not performance:
            def performance(t):
                mask = t.data >= 1
                return (t.data[mask]).mean()
        self.performance = performance
        self.companies = sorted(companies, key=performance)
        self.caps = np.ones((TransportRecord.TRANSPORT_COUNT,),
                            dtype=int) * TransportRecord.MAX_CAP
        self.dist_record = np.zeros((
            TransicationRecord.SUPPLIER_COUNT,
            24,
            TransportRecord.TRANSPORT_COUNT
        ), dtype=int)

    def reset(self):
        self.caps[:] = TransportRecord.MAX_CAP

    def distribute(self, index, week_index, amount: int):
        """use better transport company first"""
        while amount > 0:
            partition = min(amount, self.caps.max())
            target = self.dist_to_single(partition)
            if target is None:
                raise ValueError('failed to distribute request of S{} at week {}.'.format(
                    index + 1,
                    week_index
                ))
            self.dist_record[index, week_index, target] = partition
            self.caps[target] -= partition
            amount -= partition

    def dist_to_single(self, amount):
        """try to distribute task to a single transport company"""
        for i, c in enumerate(self.caps):
            if c < amount:
                continue
            return i
        return None


def csv_pickle():
    """read data from csv file and sotre object built based on those data into
    pickled binary file for latter reuse."""
    data_dire = 'data'
    requests_csv = os.path.join(data_dire, 'requests.csv')
    supply_csv = os.path.join(data_dire, 'supply.csv')
    transport_csv = os.path.join(data_dire, 'transport.csv')

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
    data_dire = 'data'
    transication_bin = os.path.join(data_dire, 'transication.bin')
    transport_bin = os.path.join(data_dire, 'transport.bin')

    targets = [transication_bin, transication_bin]
    src = glob.glob(os.path.join(data_dire, '*.csv')) + [
        'modeling.py'
    ]
    check_pickle(src, targets)

    tc = TransicationRecord.from_pickled(transication_bin)
    tp = TransportRecord.from_pickled(transport_bin)
    
    with open('ans/transport_company_data.csv', 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('id', 'mean', 'variance'))
        writer.writerows((i + 1, data.mean(), data.var())
        for i, data in enumerate(map(
            lambda t: t.data[t.data >= 1],
            tp
        )))
