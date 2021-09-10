import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from modeling import TransicationRecord, TransportRecord

def performance(r: TransicationRecord):
    return r.supply.mean()**2 * r.supply_rate.mean() * (2 - r.supply_rate.var()) * r.requests[r.requests > 0].size / r.requests.size

def make_plot(target: TransicationRecord):
    plt.figure()
    plt.title("Info of {}".format(target.id))
    
    plt.subplot(3, 1, 1)
    plt.title("Supply and requests")
    plt.plot(target.supply)
    plt.plot(target.requests)
    mean = target.supply.mean()
    plt.plot([mean] * target.supply.size)

    
    plt.subplot(3, 1, 2)
    plt.title("Info of performance rate")
    plt.plot(target.supply - target.requests)

    plt.subplot(3, 1, 3)
    plt.plot(target.local_burst)
    plt.show()

def write_csv(tc: "list[TransicationRecord]", filename: str):
    """write data into csv file"""
    tc.sort(key=performance, reverse=True)
    with open(filename, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'r_count', 'type', 's_rate_mean', 's_rate_variance', 'long_s_rate', 's_delta'])
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
            ])

if __name__ == '__main__':
    data_dire = 'data'
    transication_bin = os.path.join(data_dire, 'transication.bin')
    transport_bin = os.path.join(data_dire, 'transport.bin')

    tc = TransicationRecord.from_pickled(transication_bin)
    tp = TransportRecord.from_pickled(transport_bin)
    