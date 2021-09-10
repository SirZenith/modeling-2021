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
    weeks."""
    SUPPLIER_COUNT = 402

    def __init__(
        self,
        id: str,
        src_type: SrcType,
        supply_data: "list[float]",
        loop_vectors: "list[np.ndarray]",
        requests_data: "list[float]" = None,
    ):
        self.id = id
        self.src_type = src_type
        self.supply = np.array(supply_data)
        self.requests = np.array(requests_data) if requests_data else None
        self.mean = self.supply.mean()
        self.freqs = np.array([
            abs(v @ self.supply.T) / Record.WEEK_COUNT for v in loop_vectors
        ])
        self.supply_delta = None
        self.supply_rate_data = None
        self.long_term_supply_rate = None
        self.supply_rate_var = None

    @classmethod
    def from_csv(
        cls,
        supply_csv: str,
        requests_csv: str,
        loop_vectors: "list[np.ndarray]"
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
                results[sid] = TransicationRecord(row[0], src_t, data, loop_vectors)
        with open(requests_csv, 'r', encoding='utf8') as r:
            reader = csv.reader(r)
            _header = next(reader)
            for row in reader:
                sid = int(row[0][1:]) - 1
                data = [float(i) for i in row[2:]]
                results[sid].requests = np.array(data)
        for r in results:
            r.update_state()
        return results

    def update_state(self):
        if self.requests is None:
            return
        mask = self.requests > 2.2e-16
        if not np.any(mask):
            return
        self.supply_delta = (self.supply[mask] - self.requests[mask]).mean()
        self.long_term_supply_rate = self.supply.sum() / self.requests.sum() * 100
        supply_rate = self.supply[mask] / self.requests[mask] * 100
        supply_rate = np.array(supply_rate)
        self.supply_rate_data = supply_rate
        self.supply_rate_var = supply_rate.var()


class TransportRecord(Record):
    TRANSPORT_COUNT = 8

    def __init__(self, data: "list[float]"):
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
                results[supply_id] = TransportRecord(data)
        return results
