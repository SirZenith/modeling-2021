import csv
from enum import Enum
import glob
import os
import pickle
from typing import Type

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
        max_burst_output: int, maximum value supplier can provide during a burst.
        burst_var: float, variance of supply amount during burst.
        burst_mean: flaot, mean of supply amount during burst.
    """

    def __init__(self, s_burst: np.ndarray, s_data: np.ndarray):
        self.burst_dura = TransicationRecord.WEEK_COUNT
        self.cooling_dura = 0
        self.max_burst_output = 0
        self.burst_var = 0
        self.burst_mean = 0

        durations = self.find_burst_duration(s_burst)
        self.settle_arguments(durations, s_data)

    def find_burst_duration(self, s_burst: np.ndarray):
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
        if not durations.size:
            return
        burst_len = durations[:, 1] - durations[:, 0]
        cooling_len = durations[1:, 0] - durations[:-1, 1]
        self.burst_dura = np.median(burst_len)
        self.cooling_dura = np.median(cooling_len)

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
        src_type: SrcType,
        supply_data: "list[float]",
        # loop_vectors: "list[np.ndarray]"=None,
        requests_data: "list[float]" = None,
    ):
        self.id = id
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
                results[sid] = TransicationRecord(row[0], src_t, data)
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
        """finding local burst of supply amount and request amount."""
        conv_local = TransicationRecord.local_conv_vec()
        r_local_mean = np.convolve(conv_local, self.requests, mode='same')
        self.request_burst = self.requests > r_local_mean * 1.5

        s_local_mean = np.convolve(conv_local, self.supply, mode='same')
        self.supply_burst = self.supply > r_local_mean * 1.5

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


if __name__ == '__main__':
    data_dire = 'data'
    requests_csv = os.path.join(data_dire, 'requests.csv')
    supply_csv = os.path.join(data_dire, 'supply.csv')
    tc = TransicationRecord.from_csv(supply_csv, requests_csv)

    target = tc[139]
    print(target.burst_config.cooling_dura)
