import enum
import os

import matplotlib.pyplot as plt
import numpy as np

from modeling import TransicationRecord, TransportRecord

if __name__ == '__main__':
    data_dire = 'data'
    requests_bin = os.path.join(data_dire, 'requests.bin')
    supply_bin = os.path.join(data_dire, 'supply.bin')
    transport_bin = os.path.join(data_dire, 'transport.bin')

    requests = TransicationRecord.from_pickled(requests_bin)
    supply = TransicationRecord.from_pickled(supply_bin)
    transport = TransportRecord.from_pickled(transport_bin)

    # for i, r in enumerate(requests):
    #     print(i, r.freqs.argmin() + 1)
    plt.plot(requests[6].data)
    plt.show()
    