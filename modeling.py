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


class TransicationRecord(Record):
    """TransicationRecord records requests sent, or resource supplied in past 240
    weeks."""
    SUPPLIER_COUNT = 420

    def __init__(
        self,
        src_type: SrcType,
        data: "list[float]",
        loop_vectors: "list[np.ndarray]"
    ):
        self.src_type = src_type
        self.data = np.array(data)
        self.freqs = np.array([abs(v @ self.data.T) for v in loop_vectors])

    @classmethod
    def from_csv(
        cls,
        filename: str,
        loop_vectors: "list[np.ndarray]"
    ) -> "list[TransicationRecord]":
        """read csv data (the first line of csv file should be table heade), and
        generate TransicationRecord list"""
        results = [None] * cls.SUPPLIER_COUNT
        with open(filename, 'r', encoding='utf8') as f:
            reader = csv.reader(f)
            _header = next(reader)
            for row in reader:
                src_t = SrcType(row[1])
                sid = int(row[0][1:]) - 1
                data = [float(i) for i in row[2:]]
                results[sid] = TransicationRecord(src_t, data, loop_vectors)
        return results


class TransportRecord(Record):
    TRANSPORT_COUNT = 8

    def __init__(self, data: "list[float]"):
        self.data = np.array(data)

    @classmethod
    def from_csv(cls, filename: str, _loop_vectors) -> "list[TransportRecord]":
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
