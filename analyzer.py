import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import random
from scipy.signal import medfilt, resample
from sklearn.metrics import precision_recall_fscore_support
from cdtw import pydtw

from utils.preprocess import normalize
from utils.data import Data, Dataset, FLOW_LABEL, VOLUME_LABEL
from utils.ventilatory import Ventilatory
from classifier.comparator import Comparator, Classifiers, SAMPLING_TIME


VPT_DATA_FOLDER = './raw_data/'
VPT_DATASET     = './datasets/dataset_12_06_2019.csv'
UUID_FISIO      = '3a92975a-7b60-47da-ba6c-8bbb845e1349'

# possible_methods = ['minkwoski', 'mahalanobis', 'dtw', 'fastdtw', 'edr', 'twed', 'mjc']

def catalog_samples(_samples, exercise_number):
    samples = []

    for sample in _samples:
        sample['exercise'] = 'exerc_' + str(exercise_number)
        samples.append(sample)
    
    return samples

def main():
    DISTANCE_METHOD = 'fastdtw'
    CHOSEN_EXERCISE_NUMBER = 1
    RESAMPLE_SIZE = 300
    # N_CLUSTERS_PER_CLASS = 2
    possible_methods = ['manhattan', 'euclidean', 'cdtw_itakura', 'fastdtw']
    # possible_methods = ['manhattan', 'cdtw_itakura']
    # possible_methods = ['manhattan', 'euclidean', 'mse', 'mae', 'minkowski', 'fastdtw', 'cdtw_itakura', 'cdtw_sakoe_chiba', 'lcss', 'edr', 'mjc', 'twed', 'dtw']
    possible_n_neighboors = [1, 3, 5, 7]
    
    dataset = Dataset(VPT_DATASET)

    orig_X_train, orig_X_test, orig_y_train, orig_y_test = dataset.split_train_test(test_size=0.25)
    # orig_X_train, orig_X_test, orig_y_train, orig_y_test = dataset.split_train_test_uuid(UUID_FISIO)
    X_train, X_test, y_train, y_test = Classifiers.transform_data(orig_X_train[FLOW_LABEL], orig_X_test[FLOW_LABEL], orig_y_train, orig_y_test, resample_sz=RESAMPLE_SIZE)

    c = np.c_[X_test.reshape(len(X_test), -1), y_test.reshape(len(y_test), -1)]
    
    X_test2 = c[:, :X_test.size//len(X_test)].reshape(X_test.shape)
    y_test2 = c[:, X_test.size//len(X_test):].reshape(y_test.shape)

    np.random.shuffle(c)
    
    for i in range(len(X_test2)):
        x = X_test2[i]
        plt.plot(x)
        plt.axis('off')
        plt.savefig('plots/teste_35/teste' + str(i) + '.png', bbox_inches='tight')
        plt.close('all')
    
    print(y_test2)
    sys.exit(0)
    
    for N_CLUSTERS_PER_CLASS in range(1, 6):
        time0 = timer()
        X_train_cluster, X_test, y_train_cluster, y_test = Classifiers.kshape_clusters(X_train, X_test, y_train, y_test, n_clusters_per_class=N_CLUSTERS_PER_CLASS)

        time_clustering = timer()

        print('CLUSTERS / CLASS:', N_CLUSTERS_PER_CLASS, '\n')
        print('Time (clustering) =', time_clustering - time0)

        # for method in possible_methods:
        #     print('METHOD:', method)
                
        #     for k in possible_n_neighboors:
        time0_classify = timer()
            
        # # Classifiers.classify_approach_2(X_train, X_test, y_train, y_test, k=k, method=method)
        # # Classifiers.classify_approach_2(X_train_cluster, X_test, y_train_cluster, y_test, k=k, method=method)
        Classifiers.classify_approach_3(X_train_cluster, X_test, y_train_cluster, y_test, k=1, method_non_linear='cdtw_itakura', method_linear='euclidean')
        # # Classifiers.classify_approach_3(X_train, X_test, y_train, y_test, k=k, method_non_linear='cdtw_itakura', method_linear='euclidean')
        time_classify = timer()
        
        print('Time (classification) =', time_classify - time0_classify)                
        
        print('----------------------------------------------------------------------\n')
        
    # plt.figure()
    # for x in X_train:
    #     plt.plot(x)
    
    # # plt.show()

    # plt.figure()
    # for x in X_test:
    #     plt.plot(x)
    
    # plt.show()


def test():
    for filename in os.listdir(VPT_DATA_FOLDER):
        if filename.startswith('vpt__exerc'):
            VPT_DATA_FILE = os.path.join(VPT_DATA_FOLDER, filename)
            vpt_data = Data(VPT_DATA_FILE)
            separated_data = vpt_data.get_data()
            flow_cycles_delimiters = Ventilatory.get_cycles(vpt_data)

            f, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(separated_data["time"], separated_data["flow"], color="blue", linestyle='solid')
            ax1.vlines(separated_data["time"][flow_cycles_delimiters], np.max(separated_data["flow"])+200, np.min(separated_data["flow"])-200, color="red", linestyle='dashed')
            ax2.plot(separated_data["time"], separated_data["volume"], color="orange", linestyle='solid')
            ax2.vlines(separated_data["time"][flow_cycles_delimiters], np.max(separated_data["volume"])+200, np.min(separated_data["volume"])-200, color="red", linestyle='dashed')
            plt.show(block=False)
            
            bottom = flow_cycles_delimiters[1]
            upper = flow_cycles_delimiters[2]

            flow_cycle_model = separated_data["flow"][bottom:upper]/max(abs(np.max(separated_data["flow"][bottom:upper])), abs(np.min(separated_data["flow"][bottom:upper])))
            volume_cycle_model = separated_data["volume"][bottom:upper]/max(abs(np.max(separated_data["volume"][bottom:upper])), abs(np.min(separated_data["volume"][bottom:upper])))

            total_elapsed_time = 0
            elapsed_times = []

            for i in range(0, len(flow_cycles_delimiters)-1):
                t0 = timer()
                bottom = flow_cycles_delimiters[i]
                upper = flow_cycles_delimiters[i+1]

                flow_cycle = separated_data["flow"][bottom:upper]/max(abs(np.max(separated_data["flow"][bottom:upper])), abs(np.min(separated_data["flow"][bottom:upper])))
                volume_cycle = separated_data["volume"][bottom:upper]/max(abs(np.max(separated_data["volume"][bottom:upper])), abs(np.min(separated_data["volume"][bottom:upper])))

                flow_distance = Comparator.compare(flow_cycle_model, flow_cycle, method='edr')
                volume_distance = Comparator.compare(volume_cycle_model, volume_cycle, method='edr')

                print('flow_distance =', "{0: 20.15f}".format(flow_distance), end=' | ')        
                print('volume_distance =', "{0: 20.15f}".format(volume_distance), end=' | ')
                print('sum_distance =', "{0: 20.15f}".format(flow_distance + volume_distance))

                dt = timer() - t0
                elapsed_times.append(dt)
                total_elapsed_time += dt

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
                ax1.plot(list(range(len(flow_cycle))), flow_cycle, color="blue", linestyle='solid')
                ax1.plot(list(range(len(flow_cycle_model))), flow_cycle_model, color="cyan", linestyle='dashed')
                ax2.plot(list(range(len(volume_cycle))), volume_cycle, color="green", linestyle='solid')
                ax2.plot(list(range(len(volume_cycle_model))), volume_cycle_model, color="lime", linestyle='dashed')
                ax3.plot(volume_cycle, flow_cycle, color="green", linestyle='solid')
                ax3.plot(volume_cycle_model, flow_cycle_model, color="lime", linestyle='dashed')
                plt.show(block=False)

            input()
            plt.close('all')


if __name__ == "__main__":
    main()