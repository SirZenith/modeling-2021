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

    tc.sort(key=lambda r: r.long_term_supply_rate * r.supply.mean(), reverse=True)
    target = tc[1]

    # print(target.supply_rate_data)
    # mask = target.requests > 2.2e-16
    # print(target.supply[mask] - target.requests[mask])
    print(f'供应商 ID：{target.id}')
    print(f'供货量均值：{target.supply.mean()}')
    print(f'  订单总量：{target.requests[target.requests >= 1].size}')
    print(f'供货差均值：{target.supply_delta}')
    print(f'     周期：{target.freqs.argmin() + 1}')
    print(f'   履约率：{target.long_term_supply_rate}')
    print(f'履约率方差：{target.supply_rate.var()}')

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(target.supply)
    plt.plot(target.requests)

    plt.subplot(3, 1, 2)
    plt.plot(target.supply - target.requests)

    plt.subplot(3, 1, 3)
    plt.plot(target.freqs)
    plt.show()
    