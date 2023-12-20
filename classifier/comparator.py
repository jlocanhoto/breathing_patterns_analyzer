import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean, minkowski, cityblock, mahalanobis, chebyshev
from scipy.signal import resample
from fastdtw import fastdtw
from dtw import dtw, accelerated_dtw
import edit_distance
import traj_dist.distance as tdist
from mjc import MJC
from cdtw import pydtw

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, mean_absolute_error as MAE, mean_squared_error as MSE, r2_score as R2
from sklearn.pipeline import Pipeline
from scipy.signal import find_peaks, medfilt

from tslearn.generators import random_walk_blobs, random_walks
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import KShape
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from tslearn.svm import TimeSeriesSVC
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeries
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging, softdtw_barycenter
from tslearn import metrics

from distances.twed import TWED     # local lib
from utils.preprocess import normalize
import random

SAMPLING_TIME = 0.01

class kNNTimeSeriesClassifier():
    def __init__(self, n_neighbors=3, metric='fastdtw', metric_args={'minkowski_p':2, 'edr_eps':0.15, 'twed_lam':0.1, 'twed_nu':0.15}):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_args = metric_args
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        y_test = []

        # print('metric =', self.metric)
        for x in X_test:
            distances = []

            for i in range(len(self.X_train)):
                ts = self.X_train[i]
                _class = self.y_train[i]

                _x = x.flatten()
                _y = ts.flatten()

                distance = Comparator.compare(_x, _y, method=self.metric, method_args=self.metric_args)

                # print('[', i+1, '/', len(self.X_train), '] distance =', distance, '| class =', _class)
                
                # plt.figure()
                # plt.plot(_x)
                # plt.plot(_y)
                # plt.show()

                distances.append((distance, _class))
            
            distances = sorted(distances, key=lambda d: d[0])
            distances = distances[:self.n_neighbors]
            # print('distances =', distances)
            classes = np.array(list(map(lambda d: d[1], distances)))
            unique, counts = np.unique(classes, return_counts=True)

            # print(classes)
            # print(unique, counts)
            max_count = np.max(counts)
            count_indexes = np.where(counts == max_count)
            # print(count_indexes[0])
            
            if len(count_indexes[0]) > 1:
                possible_classes = unique[count_indexes[0]]

                # for _class in classes:
                #     if _class in possible_classes:
                #         chosen_class = _class
                #         break
                # input()
                chosen_class = random.sample(possible_classes.tolist(), 1)[0]
            else:
                chosen_class = unique[count_indexes[0]][0]
            
            # print('chosen_class =', chosen_class)
            y_test.append(chosen_class)
        
        return y_test

class Classifiers():
    @staticmethod
    def transform_data(X_train, X_test, y_train, y_test, resample_sz=300):
        transf_X = []

        train_test = np.concatenate((X_train.values, X_test.values))
        lenght_train = X_train.shape[0]

        for curve in train_test:
            values = curve.split(',')
            values = list(map(lambda v: float(v), values))
            # plt.figure()
            # plt.plot(values)
            # values = medfilt(values, kernel_size=median_filter_sz)
            # plt.figure()
            # plt.plot(values)
            # plt.show()
            transf_X.append(values)
        
        transf_X_resampled = []

        for i in range(len(transf_X)):
            resampled = resample(transf_X[i], resample_sz)
            transf_X_resampled.append(resampled)
            # f, (ax1, ax2) = plt.subplots(1,2)
            # ax1.plot(transf_X[i])
            # ax2.plot(resampled)
            # plt.show()
        
        transf_X = np.array(transf_X_resampled)

        # transf_X = to_time_series_dataset(transf_X)
        # transf_X = np.where(np.isnan(transf_X), 0, transf_X)

        t_X_train = transf_X[:lenght_train]
        t_X_test = transf_X[lenght_train:]

        # print(t_X_train)
        t_X_train = t_X_train.reshape((t_X_train.shape[0], t_X_train.shape[1], 1))
        t_X_test = t_X_test.reshape((t_X_test.shape[0], t_X_test.shape[1], 1))

        return t_X_train, t_X_test, y_train.values, y_test.values
    
    @staticmethod
    def barycenters(X_train, X_test, y_train, y_test):
        for i in range(7):
            X = X_train[y_train == 'exerc_'+str(i+1)]

            plt.figure()
            plt.subplot(3, 1, 1)
            for ts in X:
                plt.plot(ts.ravel(), "k-", alpha=.2)
            plt.plot(euclidean_barycenter(X).ravel(), "r-", linewidth=2)
            plt.title("Euclidean barycenter")

            print('[Done] Euclidean barycenter')

            plt.subplot(3, 1, 2)
            dba_bar = dtw_barycenter_averaging(X, max_iter=100, verbose=False)
            for ts in X:
                plt.plot(ts.ravel(), "k-", alpha=.2)
            plt.plot(dba_bar.ravel(), "r-", linewidth=2)
            plt.title("DBA")

            print('[Done] DBA')

            plt.subplot(3, 1, 3)
            sdtw_bar = softdtw_barycenter(X, gamma=1., max_iter=100)
            for ts in X:
                plt.plot(ts.ravel(), "k-", alpha=.2)
            plt.plot(sdtw_bar.ravel(), "r-", linewidth=2)
            plt.title("Soft-DTW barycenter ($\gamma$=1.)")

            print('[Done] Soft-DTW barycenter')

            plt.tight_layout()
            plt.show()

    @staticmethod
    def lb_envelope_metrics(X_train, X_test, y_train, y_test):
        # np.random.seed(0)
        # n_ts, sz, d = 2, 100, 1
        # dataset = random_walks(n_ts=n_ts, sz=sz, d=d)
        # scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
        # dataset_scaled = scaler.fit_transform(dataset)

        X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
        X_train = TimeSeriesScalerMinMax(min=-1., max=1.).fit_transform(X_train)

        X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)
        X_test = TimeSeriesScalerMinMax(min=-1., max=1.).fit_transform(X_test)

        sz = X_train.shape[1]

        y_predict = []

        for i in range(len(X_test)):
            envelope_down, envelope_up = metrics.lb_envelope(X_test[i], radius=3)
            # plt.figure()
            # plt.plot(np.arange(sz), X_train[0, :, 0], "r-")
            # plt.plot(np.arange(sz), envelope_down[:, 0], "g-")
            # plt.plot(np.arange(sz), envelope_up[:, 0], "g-")
            # plt.plot(np.arange(sz), X_test[i, :, 0], "k-")
            # plt.show()

            similarities = []
            for j in range(len(y_train)):
                similarity = metrics.lb_keogh(X_train[j], envelope_candidate=(envelope_down, envelope_up))
                similarities.append((y_train[j], similarity))
                # print('class =', y_train[i], end=' | ')
                # print("LB_Keogh similarity: ", similarity)

            similarities = sorted(similarities, key=lambda s: s[1])
            chosen_class = similarities[0][0]
            y_predict.append(chosen_class)

            similarities = np.array(similarities)

            # print('class_test =', y_test[i], '| chosen_class =', chosen_class, '| similarities =', similarities[:7, 0])
        
        print("Correct classification rate:", precision_recall_fscore_support(y_test, y_predict, average='macro'))

    @staticmethod
    def SVM(X_train, X_test, y_train, y_test):
        X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
        X_train = TimeSeriesScalerMinMax(min=0., max=1.).fit_transform(X_train)

        X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)
        X_test = TimeSeriesScalerMinMax(min=0., max=1.).fit_transform(X_test)

        clf = TimeSeriesSVC(kernel="gak", gamma=.1, sz=X_train.shape[1], d=X_train.shape[2])
        clf.fit(X_train, y_train)
        print("Correct classification rate:", clf.score(X_test, y_test))

        n_classes = len(set(y_train))

        plt.figure()
        support_vectors = clf.support_vectors_time_series_(X_train)
        for i, cl in enumerate(set(y_train)):
            plt.subplot(n_classes, 1, i + 1)
            plt.title("Support vectors for class %s" % str(cl))
            for ts in support_vectors[i]:
                plt.plot(ts.ravel())

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def classify_approach_0(X_train, X_test, y_train, y_test, k=1, method="fastdtw"):
        seed = 0
        X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
        # X_train = TimeSeriesScalerMinMax(min=0., max=1.).fit_transform(X_train)

        X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)
        # X_test = TimeSeriesScalerMinMax(min=0., max=1.).fit_transform(X_test)

        # for i in range(len(X_train)):
        #     x = X_train[i]
        #     classe = y_train[i]
        #     print(classe)
        #     plt.plot(x)
        #     plt.show()

        for i in range(len(X_test)):
            classe = y_test[i]
            if classe == 'exerc_7':
                t = X_test[i]
                print(classe)
                plt.plot(t)
                plt.show()
    
    @staticmethod
    def classify_approach_2(X_train, X_test, y_train, y_test, k=1, method="fastdtw"):
        seed = 0
        X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
        # X_train = TimeSeriesScalerMinMax(min=0., max=1.).fit_transform(X_train)

        X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)
        # X_test = TimeSeriesScalerMinMax(min=0., max=1.).fit_transform(X_test)

        metric_args = {
            'minkowski_p':3,
            'cdtw_step':'p0sym',
            'lcss_eps':0.15,
            'edr_eps':0.15,
            'twed_lam':0.1,
            'twed_nu':0.15
        }

        knn_clf = kNNTimeSeriesClassifier(n_neighbors=k, metric=method, metric_args=metric_args)
        knn_clf.fit(X_train, y_train)
        predicted_labels_non_linear = knn_clf.predict(X_test)

        print("Correct classification rate (k = " + str(k) + "):\n", classification_report(y_test, predicted_labels_non_linear, digits=4))
        print("\nConfusion matrix (k = " + str(k) + "):\n", confusion_matrix(y_test, predicted_labels_non_linear), '\n')

    @staticmethod
    def classify_approach_3(X_train, X_test, y_train, y_test, k=1, method_non_linear='fastdtw', method_linear='euclidean'):
        suspicious_non_linear_classes = ['exerc_1', 'exerc_2', 'exerc_6', 'exerc_7']

        seed = 0
        X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
        # X_train = TimeSeriesScalerMinMax(min=0., max=1.).fit_transform(X_train)

        X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)
        # X_test = TimeSeriesScalerMinMax(min=0., max=1.).fit_transform(X_test)

        # Nearest neighbor search
        # knn = KNeighborsTimeSeries(n_neighbors=3, metric="dtw")
        # knn.fit(X_train, y_train)
        # dists, ind = knn.kneighbors(X_test)
        # print("1. Nearest neighbour search")
        # print("Computed nearest neighbor indices (wrt DTW)\n", ind)
        # print("First nearest neighbor class:", y_test[ind[:, 0]])

        # Nearest neighbor classification
        # print(y_test)
        metric_args = {
            'minkowski_p':3,
            'cdtw_step':'p0sym',
            'lcss_eps':0.15,
            'edr_eps':0.15,
            'twed_lam':0.1,
            'twed_nu':0.15
        }

        knn_clf = kNNTimeSeriesClassifier(n_neighbors=k, metric=method_non_linear, metric_args=metric_args)
        # knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
        knn_clf.fit(X_train, y_train)
        predicted_labels_non_linear = knn_clf.predict(X_test)

        select = np.array([y in suspicious_non_linear_classes for y in y_train])
        linear_y_train = y_train[select]
        linear_X_train = X_train[select]

        indexes = np.argwhere(np.array([p in suspicious_non_linear_classes for p in predicted_labels_non_linear])).flatten()

        linear_X_test = X_test[indexes]

        # print(linear_y_test)
        
        # knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
        # knn_clf.fit(X_train, y_train)
        # predicted_labels = knn_clf.predict(X_test)
        # print("\n2. Nearest neighbor classification using DTW")
        print("Correct classification rate (non-linear)(k = " + str(k) + "):\n", classification_report(y_test, predicted_labels_non_linear, digits=4))
        print("\nConfusion matrix (non-linear)(k = " + str(k) + "):\n", confusion_matrix(y_test, predicted_labels_non_linear), '\n')

        # Nearest neighbor classification with a different metric (Euclidean distance)
        knn_clf = kNNTimeSeriesClassifier(n_neighbors=k, metric=method_linear, metric_args=metric_args)
        knn_clf.fit(linear_X_train, linear_y_train)
        predicted_labels_linear = knn_clf.predict(linear_X_test)
        # print("\n3. Nearest neighbor classification using L2")
        # print("Correct classification rate:", precision_recall_fscore_support(y_test, predicted_labels, average='macro'))

        predicted_labels = predicted_labels_non_linear.copy()
        j = 0
        for i in indexes:
            predicted_labels[i] = predicted_labels_linear[j]
            j += 1
        
        # print(predicted_labels_non_linear)
        # print(predicted_labels)

        print("Correct classification rate (non-linear + linear)(k = " + str(k) + "):\n", classification_report(y_test, predicted_labels, digits=4))
        print("\nConfusion matrix (non-linear + linear)(k = " + str(k) + "):\n", confusion_matrix(y_test, predicted_labels), '\n')

        # print("Correct classification rate (non-linear + linear):", precision_recall_fscore_support(y_test, predicted_labels, average='macro'))
        # print("Correct classification rate (non-linear + linear):", precision_recall_fscore_support(y_test, predicted_labels, average=None))

        # Nearest neighbor classification  based on SAX representation
        # sax_trans = SymbolicAggregateApproximation(n_segments=30, alphabet_size_avg=10)
        # knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="euclidean")
        # pipeline_model = Pipeline(steps=[('sax', sax_trans), ('knn', knn_clf)])
        # pipeline_model.fit(X_train, y_train)
        # predicted_labels = pipeline_model.predict(X_test)
        # print("\n4. Nearest neighbor classification using SAX+MINDIST")
        # print("Correct classification rate:", precision_recall_fscore_support(y_test, predicted_labels, average='macro')) #accuracy_score(y_test, predicted_labels))

    @staticmethod
    def kshape(X_train, X_test, y_train, y_test):
        seed = 0
        sz = X_train.shape[1]
        X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
        # X_train = TimeSeriesScalerMinMax(min=-1., max=1.).fit_transform(X_train)

        X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)
        # X_test = TimeSeriesScalerMinMax(min=-1., max=1.).fit_transform(X_test)

        # plt.figure()
        # for x in X_train:
        #     plt.plot(x)

        # plt.show()

        # Euclidean k-means
        plt.figure()

        for yi in range(7):
            X_train_i = X_train[y_train == "exerc_"+str(yi+1)]
            ks = KShape(n_clusters=1, verbose=True, random_state=seed)
            y_pred = ks.fit_predict(X_train_i)

            plt.subplot(4, 3, 1 + yi)
            for xx in X_train_i[y_pred == 0]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(ks.cluster_centers_[0].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(-4, 4)
            plt.title("Cluster %d" % (yi + 1))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def kshape_clusters(X_train, X_test, y_train, y_test, n_clusters_per_class=3):
        seed = 0
        sz = X_train.shape[1]
        X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)

        clusters = []
        new_y_train = []

        for _class in np.unique(y_train):
            X_train_class = X_train[y_train == _class]

            len_chunk = X_train_class.shape[0] // n_clusters_per_class
            chunk_remainder = X_train_class.shape[0] % n_clusters_per_class
            # print(X_train_class.shape[0], len_chunk)

            X_train_chunks = [X_train_class[i*len_chunk:(i+1)*len_chunk] for i in range(n_clusters_per_class)]
            X_train_chunks[-1] = np.concatenate((X_train_chunks[-1], X_train_class[-chunk_remainder:]))
            # print(len(X_train_chunks))

            for X in X_train_chunks:
                # print(X.shape)
                ks = KShape(n_clusters=1, verbose=False, random_state=seed)
                y_pred = ks.fit_predict(X)
                cluster = ks.cluster_centers_[0].ravel()
                cluster = TimeSeriesScalerMinMax(min=0., max=1.).fit_transform(cluster)[0]
                clusters.append(cluster)
                new_y_train.append(_class)

        new_X_train = np.array(clusters)
        new_y_train = np.array(new_y_train)

        # f, axes = plt.subplots(7, n_clusters_per_class)
        # for i in range(len(axes)):
        #     if n_clusters_per_class > 1:
        #         for j in range(n_clusters_per_class):
        #             axes[i][j].plot(new_X_train[i*n_clusters_per_class+j])
        #     else:
        #         axes[i].plot(new_X_train[i*n_clusters_per_class])
        # plt.show()

        return new_X_train, X_test, new_y_train, y_test

    @staticmethod
    def classify_test():
        seed = 0
        np.random.seed(seed)
        X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
        print(X_train.shape)
        print(y_train.shape)
        X_train = X_train[y_train < 4]  # Keep first 3 classes
        np.random.shuffle(X_train)
        X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])  # Keep only 50 time series
        sz = X_train.shape[1]

        # Euclidean k-means
        ks = KShape(n_clusters=3, verbose=True, random_state=seed)
        y_pred = ks.fit_predict(X_train)

        plt.figure()
        for yi in range(3):
            plt.subplot(3, 1, 1 + yi)
            for xx in X_train[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(-4, 4)
            plt.title("Cluster %d" % (yi + 1))

        plt.tight_layout()
        plt.show()

# - minkowski (generaliza a distância Euclidiana e a distância de Manhattan em apenas uma fórmula);
# - Distância de Mahalanobis;
# - Dynamic Time Warping (DTW);
# - Edit Distance on Real sequences (EDR);
# - Time Warp Edit Distance (TWED);
# - Minimum Jump Costs (MJC).

class Comparator():   
    @staticmethod
    def compare(X, Y, method='fastdtw', method_args={'minkowski_p':2, 'lcss_eps': 0.15, 'edr_eps':0.15, 'twed_lam':0.1, 'twed_nu':0.15, 'cdtw_step':'p0sym'}):
        possible_methods = ['euclidean', 'manhattan', 'mse', 'mae', 'rmse', 'r2', 'minkowski', 'mahalanobis', 'dtw', 'cdtw_sakoe_chiba', 'cdtw_itakura', 'fastdtw', 'edr', 'lcss', 'twed', 'mjc']
        
        if method in possible_methods:
            if method == 'manhattan':
                if len(X) != len(Y):
                    length = min(len(X), len(Y))
                    X = resample(X, length)
                    Y = resample(Y, length)

                distance = minkowski(X, Y, p=1)
            elif method == 'euclidean':
                if len(X) != len(Y):
                    length = min(len(X), len(Y))
                    X = resample(X, length)
                    Y = resample(Y, length)

                distance = minkowski(X, Y, p=2)
            elif method == 'mse':
                if len(X) != len(Y):
                    length = min(len(X), len(Y))
                    X = resample(X, length)
                    Y = resample(Y, length)

                distance = MSE(X, Y)
            elif method == 'rmse':
                if len(X) != len(Y):
                    length = min(len(X), len(Y))
                    X = resample(X, length)
                    Y = resample(Y, length)

                distance = np.sqrt(MSE(X, Y))
            elif method == 'mae':
                if len(X) != len(Y):
                    length = min(len(X), len(Y))
                    X = resample(X, length)
                    Y = resample(Y, length)

                distance = MAE(X, Y)
            elif method == 'minkowski':
                if len(X) != len(Y):
                    length = min(len(X), len(Y))
                    X = resample(X, length)
                    Y = resample(Y, length)

                distance = minkowski(X, Y, p=method_args['minkowski_p'])
            elif method == 'mahalanobis':
                pass
            elif method == 'dtw':
                X = np.array(X).reshape(-1, 1)
                Y = np.array(Y).reshape(-1, 1)
                distance, _, _, _ = dtw(X, Y, dist=euclidean)
                #distance, _, _, _ = accelerated_dtw(X, Y, dist=euclidean)
            elif method == 'fastdtw':
                distance, _ = fastdtw(X, Y, dist=euclidean)
            elif method == 'cdtw_sakoe_chiba':
                d = pydtw.dtw(X, Y, pydtw.Settings(step = method_args['cdtw_step'], #Sakoe-Chiba symmetric step with slope constraint p = 0 
                    window = 'palival', #type of the window 
                    param = 0.5, #window parameter 
                    norm = False, #normalization 
                    compute_path = False))
                
                distance = d.get_dist()
            elif method == 'cdtw_itakura':
                d = pydtw.dtw(X, Y, pydtw.Settings(step = method_args['cdtw_step'], #Sakoe-Chiba symmetric step with slope constraint p = 0 
                    window = 'itakura', #type of the window 
                    param = 0.5, #window parameter 
                    norm = False, #normalization 
                    compute_path = False))
                
                distance = d.get_dist()
            elif method == 'lcss':
                tX = np.linspace(0, (len(X)-1)*SAMPLING_TIME, len(X))
                tY = np.linspace(0, (len(Y)-1)*SAMPLING_TIME, len(Y))
                
                traj_X = np.vstack((X, tX)).T
                traj_Y = np.vstack((Y, tY)).T

                distance = tdist.lcss(traj_X, traj_Y, type_d="euclidean", eps=method_args['lcss_eps'])
            elif method == 'edr':
                tX = np.linspace(0, (len(X)-1)*SAMPLING_TIME, len(X))
                tY = np.linspace(0, (len(Y)-1)*SAMPLING_TIME, len(Y))
                
                traj_X = np.vstack((X, tX)).T
                traj_Y = np.vstack((Y, tY)).T

                distance = tdist.edr(traj_X, traj_Y, type_d="euclidean", eps=method_args['edr_eps'])
            elif method == 'twed':
                tX = np.linspace(0, (len(X)-1)*SAMPLING_TIME, len(X))
                tY = np.linspace(0, (len(Y)-1)*SAMPLING_TIME, len(Y))

                distance = TWED([tX, X], [tY, Y], method_args['twed_lam'], method_args['twed_nu'])
            elif method == 'mjc':
                distance, _ = MJC(X, Y)
            
            return distance
        else:
            return None