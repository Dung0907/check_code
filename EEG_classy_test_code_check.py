# Import libraries
# import wandb

from tensorflow.keras.constraints import max_norm, unit_norm
from prettytable import PrettyTable
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda, AveragePooling1D, Attention, Dot, Add, Multiply
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D, ConvLSTM2D, LayerNormalization
from tensorflow.keras.layers import Flatten, InputSpec, Layer, Concatenate, AveragePooling2D, MaxPooling2D, Reshape, Permute
from tensorflow.keras.layers import Input, Activation, Dropout, SpatialDropout1D, SpatialDropout2D, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Flatten, GRU, Dense, LSTM, RNN, RepeatVector, TimeDistributed, SimpleRNN, MaxPooling1D
import tensorflow.keras.layers as layers
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from tensorflow.keras.utils import to_categorical
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import stft
import os
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter

from numpy import *
from sklearn.model_selection import train_test_split
import scipy.optimize as optimize
import sklearn.datasets as ds
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC, LinearSVC
from time import time
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import roc_curve, auc

# import torch
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# from docx import Document
# import matplotlib.pyplot as plt
# from io import BytesIO

# Save all outputs into a document file



# def export_to_docx(output_filename, *outputs):
#     # Create a new Word document
#     doc = Document()
# 
#     for output in outputs:
#         # Handle different types of output
#         if isinstance(output, str):
#             doc.add_paragraph(output)
#         elif isinstance(output, list):
#             # Assume it's a table
#             table = doc.add_table(rows=len(output), cols=len(output[0]))
#             for i, row in enumerate(output):
#                 for j, cell in enumerate(row):
#                     table.cell(i, j).text = str(cell)
#         elif isinstance(output, BytesIO):
#             # Assume it's an image
#             doc.add_picture(output, width=doc.page_width // 2)
# 
#     # Save the document
#     doc.save(output_filename)


#
# -------- Constant setup -------
#
SAMPLING_FQ = 128
FQ_MIN = 0.5
NB_FQ = 36
FQ_STEP = 0.5
RESTRICTED_FQ = [FQ_STEP*fq for fq in range(int(FQ_MIN/FQ_STEP), NB_FQ+1)]
# Warning : frequencies must be regularly spaced

# Channels and indexes (cf. Channels exploration)
channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7',
            'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
useful_channels = ['F7', 'F3', 'P7', 'O1', 'O2', 'P8', 'AF4']
CHANNELS_IDX = []
for c in useful_channels:
    if c in channels:
        CHANNELS_IDX.append(channels.index(c))

# Trials infos :
TRIALS = {p: list(range(1, 8)) for p in range(1, 6)}
# Each participant has had 7 trials except for participant 5 that has had 6
TRIALS[5].remove(7)
# The first 2 trials were used for calibration
USED_TRIALS = {p: TRIALS[p][2:] for p in range(1, 6)}

# STFT Article values
DEFAULT_STFT_PARAMETERS = {'restricted_frequencies': RESTRICTED_FQ,
                           'offset': 0,
                           'window_size': 15,
                           'window_shift': 128,
                           'window_type': "blackman",
                           'averaging': True,
                           'avg_filter_size': 15}

print("The used Trials are: %s" % (USED_TRIALS))

# Data configurations
DATA_CONFIGS_DURATION = {'10-10-10': 30, '10-10-20': 40}

# Training and Evaluation Configuration
DEFAULT_T_AND_E_CONFIGURATION = {
    'training_config': "10-10-20",
    'test_config': "10-10-20",
    'evaluation_paradigm': "leave-one-out"
}
print("\nThe data configuration: %s" % (DATA_CONFIGS_DURATION))
print("\nThe default Training and Evaluation configuration: %s " %
      (DEFAULT_T_AND_E_CONFIGURATION))

# Data defining function


def number_fft(window_size):
    power = 0
    window_tmp = window_size - 1
    while (window_tmp != 1):
        power = power + 1
        window_tmp = int(window_tmp / 2)
    return pow(2, power + 1)


def feature_extraction(input_data, stft_parameters=DEFAULT_STFT_PARAMETERS, data_configuration='10-10-20',
                       fs=SAMPLING_FQ, use_channel_inds=CHANNELS_IDX, add_feature=False, w_size= 15, w_shift= 128):
    '''
    @brief This function use to extract the feature for one trial.

    @param 1 input_data: data read from '.mat' files, type np.ndarray
    @param 2 use_channel_inds: indicates of channels you need amount 14 channels. type list
    @param 3 window_size: length of segment to do stft and to smooth the specturm in unit second, type integer
    @param 4 window_shift: length of window sliding DFT, type integer
    @param 5 data_time: length of time about trials, type integer, default 40
    @param 6 FS: sampling frequency, type integer, default 128

    @return features extracted in format np.array, labels in format list

    '''
    def square(x): return (np.abs(x))**2
    def decibels(x): return 10*np.log10(x)
    window_size = w_size
    window_shift = w_shift
    avg_window_size = stft_parameters['avg_filter_size']
    window_type = stft_parameters['window_type']
    offset = stft_parameters['offset']
    data_time = DATA_CONFIGS_DURATION[data_configuration]

    times = fs / window_shift
    feature_list = []
    label = list()

    nfft_size = number_fft(window_size)

    input_data = input_data['o']['data'][0][0][offset*fs:data_time*fs*60, 3:17]
    input_data = input_data[:, use_channel_inds]

    for i in range(7):

        channel_feature_list = []
        eeg_feq = stft(input_data[:, i], fs, window_type, nperseg=window_size*fs,
                       noverlap=window_size*fs-window_shift, nfft=nfft_size*fs)
#         eeg_feq = stft(input_data[:,i], fs, nperseg=2*fs, noverlap=None)
        eeg_feq_data = eeg_feq[-1]
        eeg_feq_data = eeg_feq_data[0:-1, 0:-1]
        eeg_feq_data = eeg_feq_data.reshape(128, int(nfft_size/2), -1)

        for j in range(36):
            current = eeg_feq_data[j+1, :, :].mean(axis=0)
            current = np.apply_along_axis(square, axis=0, arr=current)
            current = np.apply_along_axis(decibels, axis=0, arr=current)
            feature = moving_average_smooth(current, avg_window_size)
            channel_feature_list.append(feature)
        channel_feature_list = standardscaler_dataframe_train(
            np.array(channel_feature_list))
        if (i == 0):
            feature_list = np.array(channel_feature_list)
        else:
            feature_list = np.vstack(
                (feature_list, np.array(channel_feature_list)))

    # focused : 0   - > unfocused : 1 drowsed : 2
    for m in range(feature_list.shape[1]):
        if m < 600*times:
            label.append(0)
        elif 600*times <= m < 1200*times:
            label.append(1)
        else:
            label.append(2)
    return feature_list.transpose(), label

# load dataset


data_dir = []
data_path = '/home/infres/annguyen/data/EEG Data/'
data_dir = ["eeg_record" + str(k) for k in range(1, 35)]

d = {}
for name in data_dir:
    d[name] = loadmat(data_path + name + '.mat')

# STFT load parameter

test_STFT_PARAMETERS = {'restricted_frequencies': RESTRICTED_FQ,
                        'offset': 0,
                        'window_size': 15,
                        'window_shift': 128,
                        'window_type': "blackman",
                        'averaging': True,
                        'avg_filter_size': 15}

test_STFT_PARAMETERS_2 = {'restricted_frequencies': RESTRICTED_FQ,
                          'offset': 0,
                          'window_size': 4,
                          'window_shift': 128,
                          'window_type': "blackman",
                          'averaging': True,
                          'avg_filter_size': 15}

# define function


def moving_average_smooth(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    re = np.convolve(interval, window, 'same')
    return re


def standardscaler_dataframe_train(feature_list):
    scaler_list = list()
    new_feature_list = list()
    for i in range(len(feature_list)):
        scaler = StandardScaler()
        x = np.array(feature_list[i]).reshape(-1, 1)
        x = scaler.fit_transform(x)
        # print(x.reshape(-1).shape)
        new_feature_list.append(x.reshape(-1))
        scaler_list.append(scaler)

    return new_feature_list

# load subject data


Subject1 = list()
Subject2 = list()
Subject3 = list()
Subject4 = list()
Subject5 = list()

Subject1 = ('eeg_record3', 'eeg_record4', 'eeg_record5', 'eeg_record7')
Subject2 = ('eeg_record10', 'eeg_record11',
            'eeg_record12', 'eeg_record13', 'eeg_record14')
Subject3 = ('eeg_record17', 'eeg_record18',
            'eeg_record19', 'eeg_record20', 'eeg_record21')
Subject4 = ('eeg_record24', 'eeg_record25', 'eeg_record26', 'eeg_record27')
Subject5 = ('eeg_record30', 'eeg_record31',
            'eeg_record32', 'eeg_record33', 'eeg_record34')



# define the function for extract the data and label


def extract_data(input_array, input_length=2400, sample_num=30, interval=150):
    output1 = []
    output2 = []
    for i in range(0, input_length, interval):
        output1.append(input_array[i:i+sample_num])
        output2.append(input_array[i+sample_num:i+interval])
    output1 = np.array(output1)
    output2 = np.array(output2)
    new_shape_1 = (output1.shape[0]*output1.shape[1],
                   output1.shape[2], output1.shape[3], output1.shape[4])
    output1 = np.reshape(output1, new_shape_1)
    new_shape_2 = (output2.shape[0]*output2.shape[1],
                   output2.shape[2], output2.shape[3], output2.shape[4])
    output2 = np.reshape(output2, new_shape_2)
    return output2, output1


def extract_label(input_array, input_length=2400, sample_num=30, interval=150):
    output1 = []
    output2 = []
    for i in range(0, input_length, interval):
        output1.append(input_array[i:i+sample_num])
        output2.append(input_array[i+sample_num:i+interval])
    output1 = np.array(output1)
    output2 = np.array(output2)
    # print(output1.shape)
    # print(output2.shape)
    new_shape_1 = (output1.shape[0]*output1.shape[1], output1.shape[2])
    output1 = np.reshape(output1, new_shape_1)
    new_shape_2 = (output2.shape[0]*output2.shape[1], output2.shape[2])
    output2 = np.reshape(output2, new_shape_2)
    return output2, output1


# set the seed


def set_seed(seed=42):
    '''
    set all random seed to 42
    '''
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
#     torch.manual_seed(seed) # pytorch
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


set_seed(1)

# shuffle the data

import random

def shuffle_data(train_data, train_target):
    train = list()
    
    for i in range(train_data.shape[0]):
        train.append((train_data[i], train_target[i]))

    random.shuffle(train)
    train_data = np.array([item[0] for item in train])
    train_target = np.array([item[1] for item in train])
    return train_data, train_target


# import the rest librabies


# Creating the confusion matrix


class ConfusionMatrix(object):

    def __init__(self, labels: list):
        self.labels = labels
        self.num_classes = len(labels)
        self.matrix = np.zeros((self.num_classes, self.num_classes))

        self.metrics = {c: {'Precision': None, 'Recall': None,
                            'Specificity': None, 'F1_score': None} for c in labels}

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def accuracy(self):
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / n

        return acc
        

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / n
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision",
                             "Recall", "Specificity", "F1score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            prec = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
#             print(prec)
            self.metrics[self.labels[i]]['Precision'] = prec
            rec = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
#             print(rec)
            self.metrics[self.labels[i]]['Recall'] = rec
            self.metrics[self.labels[i]]['Specificity'] = round(
                TN / (TN + FP), 3) if TN + FP != 0 else 0.
            self.metrics[self.labels[i]]['F1score'] = round(2*prec*rec /
                                                            (prec+rec)) if prec + rec != 0 else 0

            table.add_row([self.labels[i], self.metrics[self.labels[i]]['Precision'],
                          self.metrics[self.labels[i]]['Recall'],
                           self.metrics[self.labels[i]]['Specificity'],
                           self.metrics[self.labels[i]]['F1score']])

        print(table)
        return str(acc)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc='+self.summary()+')')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):

                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

    def log(self, input_par, evaluation_par, classifier_par=None):
        """
        input :
        - input_par : dict of keys (window_length, window_shift, avg_window)
        - classifier_par : dict of keys (name, params)
        - evaluation_par : dict of keys (train_config, train_subjects, test_config, test_subjects)
        """
        # Evaluation and training configuration
        train_config = evaluation_par['train_config']
        train_subject = evaluation_par['train_subject']
        test_config = evaluation_par['test_config']
        test_subject = evaluation_par['test_subject']
        if len(train_subject) == 5 & len(test_subject) == 5:
            evaluation_paradigm = 'common_subject'
        elif len(test_subject) == 1:
            if len(train_subject) == 4 and test_subject[0] not in train_subject:
                evaluation_paradigm = 'leave_one_out'
            elif len(train_subject) == 1 and train_subject[0] == test_subject[0]:
                evaluation_paradigm = 'subject_specific'
            else:
                evaluation_paradigm = 'unknown'
        else:
            evaluation_paradigm = 'unknown'

        # Stft parameters
        window_size = input_par['window_size']
        window_shift = input_par['window_shift']
        avg_window = input_par['avg_filter_size']

        # Classifier parameters
        classifier_name = classifier_par['name']
        classifier_par = classifier_par['model_params']

        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / n
#         print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision",
                             "Recall", "Specificity", "F1score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            prec = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
#             print(prec)
            self.metrics[self.labels[i]]['Precision'] = prec
            rec = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
#             print(rec)
            self.metrics[self.labels[i]]['Recall'] = rec
            self.metrics[self.labels[i]]['Specificity'] = round(
                TN / (TN + FP), 3) if TN + FP != 0 else 0.
            self.metrics[self.labels[i]]['F1score'] = round(2*prec*rec /
                                                            (prec+rec)) if prec + rec != 0 else 0

            table.add_row([self.labels[i], self.metrics[self.labels[i]]['Precision'],
                          self.metrics[self.labels[i]
                                       ]['Recall'], self.metrics[self.labels[i]]['Specificity'],
                           self.metrics[self.labels[i]]['F1score']])
        # Metrics (recall, precision) for each class, then balanced accuracy
        recall_f = self.metrics['focused']['Recall']
        precision_f = self.metrics['focused']['Precision']
        recall_u = self.metrics['unfocused']['Recall']
        precision_u = self.metrics['unfocused']['Precision']
        recall_d = self.metrics['drowsed']['Recall']
        precision_d = self.metrics['drowsed']['Precision']
        bal_acc = (recall_f + recall_u + recall_d)/3

        score_list = [train_config, train_subject, test_config, test_subject, evaluation_paradigm,
                      window_size, window_shift, avg_window,
                      classifier_name, classifier_par,
                      bal_acc, recall_f, precision_f, recall_u, precision_u, recall_d, precision_d]
        # score_list must have
#         if os.path.exists(os.path.join(CURRENT_FOLDER_NAME, CSV_SCORE_FILE_NAME)):
        with open(os.path.join(CURRENT_FOLDER_NAME, CSV_SCORE_FILE_NAME), 'a') as f_object:
            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = writer(f_object)

            # Pass the list as an argument into
            # the writerow()
            writer_object.writerow(score_list)

            # Close the file object
            f_object.close()

# Initialize the machine learning classifier


def init_classifiers():
    '''
    Initialize our machine learning classifier --- 
    where catboost and NN (neural network classification) are not initialized, 
    and most hyperparameters will take default values

    '''

    model_names = ['SVM', 'LR', 'KNN', 'GBDT', 'DT',
                   'AdaB', 'RF', 'XGB', 'LGB', 'Catboost', 'NN']

    # the training parameters of each model
    param_grid_svc = [{}]
    param_grid_logistic = [{'C': [0.1], 'penalty': ['l1', 'l2']}]
    param_grid_knn = [{}, {'n_neighbors': list(range(3, 8))}]
    param_grid_gbdt = [{}]
    param_grid_tree = [{}]
    param_grid_boost = [{}]
    param_grid_rf = [{}]
    # param_grid_xgb = [{'tree_method': ['gpu_hist'], 'gpu_id': [0]}]
    param_grid_xgb = [{}]
    param_grid_lgb = [{}]

    return ([(SVC(), model_names[0], param_grid_svc),
            #             (LogisticRegression(), model_names[1], param_grid_logistic),
             #             (KNeighborsClassifier(), model_names[2], param_grid_knn),
             #             (GradientBoostingClassifier(), model_names[3], param_grid_gbdt),
             #             (DecisionTreeClassifier(), model_names[4], param_grid_tree),
             #             (AdaBoostClassifier(), model_names[5], param_grid_boost),
             (RandomForestClassifier(), model_names[6], param_grid_rf),
            (xgb.XGBClassifier(), model_names[7], param_grid_xgb),
             #             (lgb.sklearn.LGBMClassifier(), model_names[8], param_grid_lgb)
             ])


def model_evaluation_dict(train_x, train_y, test_x_1, test_y_1, model, model_name, params):
    '''
    Perform 10 fold crossvalidation, fit model with train data and evaluate its performance 
    return performance dict

    '''

    clf = GridSearchCV(model, params, cv=10)

    X_train, X_test, y_train, y_test = train_x, test_x_1, train_y, test_y_1
    clf.fit(X_train, y_train)
    params = clf.best_params_

    Training_score = clf.score(X_train, y_train)
    Score_10_10_10 = clf.score(X_test, y_test)
    cvres = clf.cv_results_
    cvscore = cvres['mean_test_score'][clf.best_index_]
    macro_precision, macro_recall, macro_f1_score, macro_support =\
        precision_recall_fscore_support(
            y_test, clf.predict(X_test), average='macro')
    micro_precision, micro_recall, micro_f1_score, micro_support =\
        precision_recall_fscore_support(
            y_test, clf.predict(X_test), average='micro')
    if not params:
        # empty params dict
        params = 'default'
    # return a dictionary
    d_info = {'Classifier': model_name, 'param': params, 'Traing score': Training_score, 'Score non bias set': Score_10_10_10,
              'CV Score': cvscore,
              'Precision(Macro)': macro_precision, 'Precision(Micro)': micro_precision,
              'Recall(Macro)': macro_recall, 'Recall(Micro)': micro_recall,
              'F1 Score(Macro)': macro_f1_score, 'F1 Score(Micro)': micro_f1_score}

    confusion_1 = ConfusionMatrix(labels=['Focused', 'Unfocused', 'Drowsed'])
    confusion_1.update(clf.predict(X_test), y_test)
#     confusion_2 = ConfusionMatrix( labels=['Focused','Unfocused','Drowsed'])
#     confusion_2.update(clf.predict(X_test), y_test)
#     confusion_3 = ConfusionMatrix(num_classes=2, labels=['Undrowsed','Drowsed'])
#     confusion_3.update(clf.predict(test_x_3), test_y_3)

    return d_info, confusion_1


def set_seed(seed=42):
    '''
    set all random seed to 42

    '''
#     random.seed(seed) # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Extract the label

def extract_label(input_array, input_length=2400, sample_num=30, interval=150):
    output1 = []
    output2 = []
    for i in range(0,input_length,interval):
        output1.append(input_array[i:i+sample_num])
        output2.append(input_array[i+sample_num:i+interval])
    output1 = np.array(output1)
    output2 = np.array(output2)
    print(output1.shape)
    print(output2.shape)
    new_shape_1 = (output1.shape[0]*output1.shape[1])
    output1 = np.reshape(output1, new_shape_1)
    new_shape_2 = (output2.shape[0]*output2.shape[1])
    output2 = np.reshape(output2, new_shape_2)
    return output2, output1

# # Test the extract label
# 
# print("\nThe extract label of subject 5:")
# train_label, test_label = extract_label(label_train_subject5,12000)

# Extract the data

def extract_data(input_array, input_length=2400, sample_num=30, interval=150):
    output1 = []
    output2 = []
    for i in range(0,input_length,interval):
        output1.append(input_array[i:i+sample_num])
        output2.append(input_array[i+sample_num:i+interval])
    output1 = np.array(output1)
    output2 = np.array(output2)
    print(output1.shape)
    new_shape_1 = (output1.shape[0]*output1.shape[1],output1.shape[2])
    output1 = np.reshape(output1, new_shape_1)
    new_shape_2 = (output2.shape[0]*output2.shape[1],output2.shape[2])
    output2 = np.reshape(output2, new_shape_2)
    return output2, output1

# # Test the extract data
# 
# print("\nThe extract data of subject 5:")
# train_data, test_data = extract_data(feature_train_subject5,12000)

# making the list of window size and window shift

count_row = 0
count_col = 0

svm_result = np.zeros([59,1277])
xgb_result = np.zeros([59,1277])
rf_result = np.zeros([59,1277])

W_size = list(range(2,61))
W_shift = list(range(4,1281))

# extract the data and label

for w_size in W_size:

    for w_shift in W_shift:

        feature_subject1 = dict()
        label_subject1 = dict()
        for name in Subject1:
            feature_subject1[name], label_subject1[name] = feature_extraction(
                d[name], stft_parameters=test_STFT_PARAMETERS, data_configuration='10-10-20', w_size= w_size, w_shift= w_shift)
            print(len(label_subject1[name]))
            #     feature_subject1[name] = standardscaler_dataframe_train(feature_subject1[name])
        feature_subject2 = dict()
        label_subject2 = dict()
        for name in Subject2:
            feature_subject2[name], label_subject2[name] = feature_extraction(
                d[name], stft_parameters=test_STFT_PARAMETERS, data_configuration='10-10-20', w_size= w_size, w_shift= w_shift)
        
            #     feature_subject2[name] = standardscaler_dataframe_train(feature_subject2[name])
        
        feature_subject3 = dict()
        label_subject3 = dict()
        for name in Subject3:
            feature_subject3[name], label_subject3[name] = feature_extraction(
                d[name], stft_parameters=test_STFT_PARAMETERS, data_configuration='10-10-20', w_size= w_size, w_shift= w_shift)
        #     feature_subject3[name] = standardscaler_dataframe_train(feature_subject3[name])
        
        feature_subject4 = dict()
        label_subject4 = dict()
        for name in Subject4:
            feature_subject4[name], label_subject4[name] = feature_extraction(
                d[name], stft_parameters=test_STFT_PARAMETERS, data_configuration='10-10-20', w_size= w_size, w_shift= w_shift)
        #     feature_subject4[name] = standardscaler_dataframe_train(feature_subject4[name])
        
        feature_subject5 = dict()
        label_subject5 = dict()
        for name in Subject5:
            feature_subject5[name], label_subject5[name] = feature_extraction(
                d[name], stft_parameters=test_STFT_PARAMETERS, data_configuration='10-10-20', w_size= w_size, w_shift= w_shift)
        #     feature_subject5[name] = standardscaler_dataframe_train(feature_subject5[name])
        
        
        label_test_subject1 = label_subject1[Subject1[0]]
        feature_test_subject1 = feature_subject1[Subject1[0]]
        label_train_subject1 = label_subject1[Subject1[0]]
        feature_train_subject1 = feature_subject1[Subject1[0]]
        
        label_test_subject2 = label_subject2[Subject2[0]]
        feature_test_subject2 = feature_subject2[Subject2[0]]
        label_train_subject2 = label_subject2[Subject2[0]]
        feature_train_subject2 = feature_subject2[Subject2[0]]
        
        label_test_subject3 = label_subject3[Subject3[0]]
        feature_test_subject3 = feature_subject3[Subject3[0]]
        label_train_subject3 = label_subject3[Subject3[0]]
        feature_train_subject3 = feature_subject3[Subject3[0]]
        
        label_test_subject4 = label_subject4[Subject4[0]]
        feature_test_subject4 = feature_subject4[Subject4[0]]
        label_train_subject4 = label_subject4[Subject4[0]]
        feature_train_subject4 = feature_subject4[Subject4[0]]
        
        label_test_subject5 = label_subject5[Subject5[0]]
        feature_test_subject5 = feature_subject5[Subject5[0]]
        label_train_subject5 = label_subject5[Subject5[0]]
        feature_train_subject5 = feature_subject5[Subject5[0]]
        for i in range(1, len(Subject1)):
            feature_train_subject1 = vstack(
                (feature_train_subject1, feature_subject1[Subject1[i]]))
            label_train_subject1.extend(label_subject1[Subject1[i]])
        
        for i in range(1, len(Subject2)):
            feature_train_subject2 = vstack(
                (feature_train_subject2, feature_subject2[Subject2[i]]))
            label_train_subject2.extend(label_subject2[Subject2[i]])
        
        for i in range(1, len(Subject3)):
            feature_train_subject3 = vstack(
                (feature_train_subject3, feature_subject3[Subject3[i]]))
            label_train_subject3.extend(label_subject3[Subject3[i]])
        
        for i in range(1, len(Subject4)):
            feature_train_subject4 = vstack(
                (feature_train_subject4, feature_subject4[Subject4[i]]))
            label_train_subject4.extend(label_subject4[Subject4[i]])
        
        for i in range(1, len(Subject5)):
            feature_train_subject5 = vstack(
                (feature_train_subject5, feature_subject5[Subject5[i]]))
            label_train_subject5.extend(label_subject5[Subject5[i]])
        
        feature_train_Except_Subject_1 = []
        y_train_Except_Subject_1 = list()

        feature_train_Except_Subject_1 = vstack((feature_train_subject2, feature_train_subject3,
                                                 feature_train_subject4, feature_train_subject5))


        y_train_Except_Subject_1.extend(label_train_subject2)
        y_train_Except_Subject_1.extend(label_train_subject3)
        y_train_Except_Subject_1.extend(label_train_subject4)
        y_train_Except_Subject_1.extend(label_train_subject5)


        feature_train_Except_Subject_2 = []
        y_train_Except_Subject_2 = list()

        feature_train_Except_Subject_2 = vstack(
            (feature_train_subject1, feature_train_subject3, feature_train_subject4, feature_train_subject5))


        y_train_Except_Subject_2.extend(label_train_subject1)
        y_train_Except_Subject_2.extend(label_train_subject3)
        y_train_Except_Subject_2.extend(label_train_subject4)
        y_train_Except_Subject_2.extend(label_train_subject5)

        feature_train_Except_Subject_3 = []
        y_train_Except_Subject_3 = list()


        feature_train_Except_Subject_3 = vstack(
            (feature_train_subject1, feature_train_subject2, feature_train_subject4, feature_train_subject5))


        y_train_Except_Subject_3.extend(label_train_subject1)
        y_train_Except_Subject_3.extend(label_train_subject2)
        y_train_Except_Subject_3.extend(label_train_subject4)
        y_train_Except_Subject_3.extend(label_train_subject5)

        feature_train_Except_Subject_4 = []
        y_train_Except_Subject_4 = list()


        feature_train_Except_Subject_4 = vstack(
            (feature_train_subject1, feature_train_subject2, feature_train_subject3, feature_train_subject5))


        y_train_Except_Subject_4.extend(label_train_subject1)
        y_train_Except_Subject_4.extend(label_train_subject2)
        y_train_Except_Subject_4.extend(label_train_subject3)
        y_train_Except_Subject_4.extend(label_train_subject5)

        feature_train_Except_Subject_5 = []
        y_train_Except_Subject_5 = list()


        feature_train_Except_Subject_5 = vstack(
            (feature_train_subject1, feature_train_subject2, feature_train_subject3, feature_train_subject4))


        y_train_Except_Subject_5.extend(label_train_subject1)
        y_train_Except_Subject_5.extend(label_train_subject2)
        y_train_Except_Subject_5.extend(label_train_subject3)
        y_train_Except_Subject_5.extend(label_train_subject4)

        # Process the shuffling training dataset

        feature_train_subject1, label_train_subject1 = shuffle_data(
            feature_train_subject1, label_train_subject1)
        feature_train_subject2, label_train_subject2 = shuffle_data(
            feature_train_subject2, label_train_subject2)
        feature_train_subject3, label_train_subject3 = shuffle_data(
            feature_train_subject3, label_train_subject3)
        feature_train_subject4, label_train_subject4 = shuffle_data(
            feature_train_subject4, label_train_subject4)
        feature_train_subject5, label_train_subject5 = shuffle_data(
            feature_train_subject5, label_train_subject5)

        # Process the shuffling testing dataset

        feature_test_subject1, label_test_subject1 = shuffle_data(feature_test_subject1, label_test_subject1)
        feature_test_subject2, label_test_subject2 = shuffle_data(feature_test_subject2, label_test_subject2)
        feature_test_subject3, label_test_subject3 = shuffle_data(feature_test_subject3, label_test_subject3)
        feature_test_subject4, label_test_subject4 = shuffle_data(feature_test_subject4, label_test_subject4)
        feature_test_subject5, label_test_subject5 = shuffle_data(feature_test_subject5, label_test_subject5)

        # Making the feature and label for train and test data common

        train_label, test_label = extract_label(label_train_subject5, len(label_train_subject5))
        train_data, test_data = extract_data(feature_train_subject5, len(feature_train_subject5))

        # feature_train_common = []
        # label_train_common = list()
        feature_test_common = []
        label_test_common = list()

        # feature_train_common = vstack((feature_train_subject1,feature_train_subject2,feature_train_subject3,feature_train_subject4,feature_train_subject5))


        # label_train_common.extend(label_train_subject1)
        # label_train_common.extend(label_train_subject2)
        # label_train_common.extend(label_train_subject3)
        # label_train_common.extend(label_train_subject4)
        # label_train_common.extend(label_train_subject5)


        feature_test_common = vstack((feature_test_subject1,feature_test_subject2,feature_test_subject3,feature_test_subject4,feature_test_subject5))


        label_test_common.extend(label_test_subject1)
        label_test_common.extend(label_test_subject2)
        label_test_common.extend(label_test_subject3)
        label_test_common.extend(label_test_subject4)
        label_test_common.extend(label_test_subject5)


        feature_train_common = []
        y_train_common = list()


        feature_train_common = vstack((feature_train_subject1,feature_train_subject2,feature_train_subject3,feature_train_subject4,feature_train_subject5))

        y_train_common.extend(label_train_subject1)
        y_train_common.extend(label_train_subject2)
        y_train_common.extend(label_train_subject3)
        y_train_common.extend(label_train_subject4)
        y_train_common.extend(label_train_subject5)

        # Making the x and y train and test

        X_train, X_test, y_train, y_test = train_test_split(np.array(feature_train_common),np.array(y_train_common), test_size=0.2,random_state=100)

        train_x = X_train
        train_y = y_train
        test_x = X_test
        test_y = y_test

        # Processing

        res_list_1_1 = []
        confusion_rf_list_1_1 = []
        confusion_SVM_list_1_1 = []
        confusion_XGB_list_1_1 = []

        best_score = 0
        classifiers = init_classifiers()
        for i in classifiers:
            results, confusion_1= model_evaluation_dict(train_data, train_label, test_data, test_label,  i[0], i[1], i[2])
            res_list_1_1.append(results)
            if i[1] == 'RF':
                confusion_rf_list_1_1.append(confusion_1)
            elif i[1] == 'SVM':
                confusion_SVM_list_1_1.append(confusion_1)
            elif i[1] == 'XGB':
                confusion_XGB_list_1_1.append(confusion_1)
            print(i[1])

#         print("\nThe result that using the SVM classification:\n")
#         confusion_SVM_list_1_1[0].plot() # PLot the result that get from the processing by using the SVM classification
# 
#         print("\nThe result that using the RF classification:\n")
#         confusion_rf_list_1_1[0].plot() # PLot the result that get from the processing by using the RF classification
# 
#         print("\nThe result that using the XGBoost classification:\n")
#         confusion_XGB_list_1_1[-1].plot() # PLot the result that get from the processing by using the XGBoost classification

        svm_result[count_row, count_col] = confusion_SVM_list_1_1[0].accuracy()
        xgb_result[count_row, count_col] = confusion_XGB_list_1_1[0].accuracy()
        rf_result[count_row, count_col] = confusion_rf_list_1_1[0].accuracy()

        print("Done in %s column \n" % (count_col))
        count_col = count_col + 1
        

    print("Done in %s row \n" % (count_row))
    count_row = count_row + 1

svm_rel = {}
xgb_rel = {}
rf_rel = {}

for cou_col in range (4,1281):
    name = str(cou_col)

    svm_rel[name] = svm_result[:,cou_col]
    xgb_rel[name] = xgb_result[:,cou_col]
    rf_rel[name] = rf_result[:,cou_col]

df_svm = pd.DataFrame(svm_rel)
df_xgb = pd.DataFrame(xgb_rel)
df_rf = pd.DataFrame(rf_rel)

cou_row = list(range(2,61))
cou_row_str = map(str,cou_row)


df_svm.index = list(cou_row_str)
df_xgb.index = list(cou_row_str)
df_rf.index = list(cou_row_str)

excel_path = 'get_result.xlsx'

df_svm.to_excel(excel_path, sheet_name= 'Sheet1', index= True, header= True)
df_xgb.to_excel(excel_path, sheet_name= 'Sheet2', index= True, header= True)
df_rf.to_excel(excel_path, sheet_name= 'Sheet3', index= True, header= True)

print(f"All the result are exported into file excel: get_result !!!!!!!!!!!")

# Creating DNN model

# def create_model_DNN_3_dropout(nn_1=64,nn_2=64):
# 
#     model = tf.keras.models.Sequential()
# 
#     #Type of activation function
#     model.add(tf.keras.layers.Input(shape=(252,)))
#     #model.add(tf.keras.layers.Dense(nn_0, activation=tf.keras.activations.linear))
#     model.add(tf.keras.layers.Dense(nn_1, activation=tf.keras.activations.relu))
#     model.add(tf.keras.layers.Dense(nn_2, activation=tf.keras.activations.relu))
#     model.add(tf.keras.layers.Dense(nn_O, activation=tf.keras.activations.softmax))
# 
#     model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.optimizers.SGD(lr=0.005, nesterov=True))
#     return model
# 
# def create_model_DNN_4(nn_1=64,nn_2=128,nn_3=64):
# 
#     model = tf.keras.models.Sequential()
# 
#     #Type of activation function
#     model.add(layers.Input(shape=(252,)))
#     #model.add(layers.Dense(nn_0, activation=activations.linear))
#     model.add(tf.keras.layers.Dense(nn_1, activation=tf.keras.activations.relu))
#     model.add(tf.keras.layers.Dense(nn_2, activation=tf.keras.activations.relu))
#     model.add(tf.keras.layers.Dense(nn_3, activation=tf.keras.activations.relu))
#     model.add(tf.keras.layers.Dense(nn_O, activation=tf.keras.activations.softmax))
# 
#     model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.optimizers.SGD(lr=0.005, nesterov=True))
#     return model
# 
# def create_model_DNN_6_dropout(nn_1=64,nn_2=128,nn_3=256, nn_4=128,nn_5=64):
# 
#     model = tf.keras.models.Sequential()
# 
#     #Type of activation function
#     model.add(tf.keras.layers.Input(shape=(252,)))
#     #model.add(layers.Dense(nn_0, activation=activations.linear))
#     model.add(tf.keras.layers.Dense(nn_1, activation=tf.keras.activations.relu))
#     model.add(tf.keras.layers.Dense(nn_2, activation=tf.keras.activations.relu))
#     model.add(tf.keras.layers.Dropout(0.3))
#     model.add(tf.keras.layers.Dense(nn_3, activation=tf.keras.activations.relu))
#     model.add(tf.keras.layers.Dropout(0.2))
#     model.add(tf.keras.layers.Dense(nn_4, activation=tf.keras.activations.relu))
#     model.add(tf.keras.layers.Dropout(0.2))
#     model.add(tf.keras.layers.Dense(nn_5, activation=tf.keras.activations.relu))
#     model.add(tf.keras.layers.Dense(nn_O, activation=tf.keras.activations.softmax))
# 
#     model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=tf.optimizers.SGD(lr=0.005, nesterov=True))
#     return model

# process the SVM, RF, XGBoost on dataset that except subject 3

# train_x = feature_train_Except_Subject_3
# train_y = y_train_Except_Subject_3
# test_x = feature_train_subject3
# test_y = label_train_subject3
# 
# train_x, train_y = shuffle_data(train_x, train_y)
# test_x, test_y = shuffle_data(test_x, test_y)
# 
# res_list_3_2 = []
# confusion_rf_list_3_2 = []
# confusion_SVM_list_3_2 = []
# confusion_XGB_list_3_2 = []
# best_score = 0
# classifiers = init_classifiers()
# for i in classifiers:
#     results, confusion_1= model_evaluation_dict(train_x, train_y, test_x, test_y,  i[0], i[1], i[2])
#     res_list_3_2.append(results)
#     if i[1] == 'RF':
#         confusion_rf_list_3_2.append(confusion_1)
#     elif i[1] == 'SVM':
#         confusion_SVM_list_3_2.append(confusion_1)
#     elif i[1] == 'XGB':
#         confusion_XGB_list_3_2.append(confusion_1)
#     print(i[1])

# df_model_comparison = pd.DataFrame(res_list_3_2).sort_values(by=['F1 Score(Macro)','F1 Score(Micro)']).reset_index(drop=True)
# df_model_comparison

# Plot classification method result

# print("\nThe result for SVM classifier:")
# confusion_SVM_list_3_2[0].plot()
# 
# print("\nThe result for XGBoost classifier:")
# confusion_XGB_list_3_2[0].plot()
# 
# print("\nThe result for RF classifier:")
# confusion_rf_list_3_2[0].plot()
