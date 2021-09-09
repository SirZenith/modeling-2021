import csv
from enum import Enum
import os
import pickle
from typing import Callable

import numpy as np


class SrcType(bytes, Enum):
    """resource type, one of A, B, C"""
    def __new__(cls, value: int, type_name: str, unit_cost: float, price: int):
        obj = bytes.__new__(cls, [value])
        obj._value_ = type_name.upper()
        obj.unit_cost = unit_cost
        obj.price = price
        return obj

    A = (0, 'A', 0.6, 120)
    B = (1, 'B', 0.66, 110)
    C = (2, 'C', 0.72, 100)


class TransicationRecord(object):
    """TransicationRecord is record of requests sent, or resource supplied in past 240
    weeks."""

    def __init__(self, src_type: SrcType, data: "list[int]"):
        self.src_type = src_type
        self.data = data


class TransportRecord(object):
    def __init__(self, data: "list[float]"):
        self.data = data


def read_records_from_file(filename: str) -> "list[TransicationRecord]":
    """read csv data (the first line of csv file should be table heade), and
    generate TransicationRecord list"""
    results = [None] * 402
    with open(filename, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        _header = next(reader)
        for row in reader:
            src_type = SrcType(row[1])
            supply_id = int(row[0][1:]) - 1
            data = row[2:]
            results[supply_id] = TransicationRecord(src_type, data)
    return results


def read_transport_from_file(filename: str) -> "list":
    """read csv data (the first line of csv file should be table heade), and
    generate TransportRecord list"""
    results = [None] * 8
    with open(filename, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        _header = next(reader)
        for row in reader:
            supply_id = int(row[0][1:]) - 1
            data = row[1:]
            results[supply_id] = TransportRecord(data)
    return results
    

def try_read_persist_data(bin_file: str, src_file: str, read_operation: Callable[[str], list]) -> list:
    """read pickled persistent data from bin_file if exists, else read data from
    src_file by read_operation function, and pickle read data to bin_file."""
    result = None
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            result = pickle.load(f)
    else:
        result = read_operation(src_file)
        with open(bin_file, 'wb+') as f:
            pickle.dump(result, f)
    return result


if __name__ == '__main__':
    data_dire = 'data'
    requests_csv = os.path.join(data_dire, 'requests.csv')
    supply_csv = os.path.join(data_dire, 'supply.csv')
    transport_csv = os.path.join(data_dire, 'transport.csv')
    requests_bin = os.path.join(data_dire, 'requests.bin')
    supply_bin = os.path.join(data_dire, 'supply.bin')
    transport_bin = os.path.join(data_dire, 'transport.bin')

    requests = try_read_persist_data(requests_bin, requests_csv, read_records_from_file)
    supply = try_read_persist_data(supply_bin, supply_csv, read_records_from_file)
    transport = try_read_persist_data(transport_bin, transport_csv, read_transport_from_file)
