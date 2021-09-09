import enum
import os

import matplotlib.pyplot as plt
import numpy as np

from modeling import TransicationRecord, TransportRecord

if __name__ == '__main__':
    data_dire = 'data'
    transication_bin = os.path.join(data_dire, 'transication.bin')
    transport_bin = os.path.join(data_dire, 'transport.bin')

    tc = TransicationRecord.from_pickled(transication_bin)
    tp = TransportRecord.from_pickled(transport_bin)

    target = tc[5]

    # print(target.supply_rate_data)
    print(f'供货量均值：{target.mean}')
    print(f'供货差均值：{target.supply_delta}')
    print(f'     周期：{target.freqs.argmin() + 1}')
    print(f'   履约率：{target.supply_rate}')
    print(f'履约率方差：{target.supply_rate_var}')

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(target.supply)
    plt.plot(target.requests)

    plt.subplot(3, 1, 2)
    plt.plot(target.supply - target.requests)

    plt.subplot(3, 1, 3)
    plt.plot(target.freqs)
    plt.show()
    