import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from time import time
import json

from utils.preprocess import normalize
from utils.data import Data
from utils.ventilatory import Ventilatory
from classifier.comparator import Comparator

CSV_DELIMITER = ';'


def build_dataset(input_folder, output_folder):
    dataset = {}
    counter = 0
    files = os.listdir(input_folder)
    total_files = len(os.listdir(input_folder))

    for filename in files:
        if filename.startswith('vpt__exerc') and filename.endswith('.dat'):
            print('FILE', counter, '/', total_files)

            VPT_DATA_FILE = os.path.join(input_folder, filename)
            vpt_data = Data(VPT_DATA_FILE)
            separated_data = vpt_data.get_data()
            flow_cycles_delimiters = Ventilatory.get_cycles(vpt_data)

            f, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(separated_data["time"], separated_data["flow"], color="blue", linestyle='solid')
            ax1.vlines(separated_data["time"][flow_cycles_delimiters], np.max(separated_data["flow"])+200, np.min(separated_data["flow"])-200, color="red", linestyle='dashed')
            ax2.plot(separated_data["time"], separated_data["volume"], color="orange", linestyle='solid')
            ax2.vlines(separated_data["time"][flow_cycles_delimiters], np.max(separated_data["volume"])+200, np.min(separated_data["volume"])-200, color="red", linestyle='dashed')
            plt.show(block=False)

            dropped_cycles = input("Drop cycles: ")
            dropped_cycles = dropped_cycles.replace(' ', '')
            
            if dropped_cycles:
                dropped_cycles = dropped_cycles.split(',')
                dropped_cycles = list(map(lambda e: int(e), dropped_cycles))
            else:
                dropped_cycles = []

            # print('dropped_cycles =', dropped_cycles)
            
            for i in range(len(flow_cycles_delimiters)-1):
                if i not in dropped_cycles:
                    bottom = flow_cycles_delimiters[i]
                    upper = flow_cycles_delimiters[i+1]

                    flow_cycle = separated_data["flow"][bottom:upper]
                    volume_cycle = separated_data["volume"][bottom:upper]

                    uuid = vpt_data.get_UUID()
                    exercise = vpt_data.get_exercise()

                    dataset_line = uuid + CSV_DELIMITER
                    dataset_line += exercise + CSV_DELIMITER
                    dataset_line += str(flow_cycle.tolist())[1:-1] + CSV_DELIMITER
                    dataset_line += str(volume_cycle.tolist())[1:-1] + '\n'

                    with open(os.path.join(output_folder, 'dataset.csv'), mode='a+') as dataset_file:
                        dataset_file.write(dataset_line)
            
            plt.close('all')
        
        counter += 1

    return dataset

if __name__ == '__main__':
    VPT_DATA_FOLDER = '../raw_data/'
    DATASET_FOLDER = '../datasets/'
    dataset = build_dataset(VPT_DATA_FOLDER, DATASET_FOLDER)