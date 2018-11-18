#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: demo.py
Author: leowan(leowan)
Date: 2018/11/16 16:14:36
"""

import os
import shutil

import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

import tfdeepfmclassifier_column_indexed
import utils

def test_colind_model(X_colind, X_colval, y, feature_num):
    """
        Unittest column-indexed model
    """
    model = tfdeepfmclassifier_column_indexed.TFDeepFMClassifier(
        feature_num=feature_num, # feature num must set
        field_num=feature_num,
        l2_weight=0.01, learning_rate=1e-2,
        batch_size=1, epoch_num=10, print_step=1000, random_seed=42)
    model.fit(np.array(X_ind_tr), np.array(X_val_tr), np.array(y_cid_tr))
    predictions = model.predict(np.array(X_ind_tr), np.array(X_val_tr))
    print('model: {}'.format(model.__str__()))
    print('train acc: {}'.format(accuracy_score(
        np.array(y_cid_tr), model.predict(np.array(X_ind_tr), np.array(X_val_tr)))))

def test_colind_load_chkpt(X_colind, X_colval, y, feature_num):
    """
        Unittest load checkpoint
    """
    chkpt_dir = './tmp'
    if os.path.exists(chkpt_dir):
        shutil.rmtree(chkpt_dir)

    model = tfdeepfmclassifier_column_indexed.TFDeepFMClassifier(
        feature_num=feature_num, # feature num must set
        field_num=feature_num,
        l2_weight=0.01, learning_rate=1e-2,
        batch_size=1, epoch_num=10, print_step=1000, random_seed=42, chkpt_dir=chkpt_dir)
    model.fit(np.array(X_ind_tr), np.array(X_val_tr), np.array(y_cid_tr))
    print('train acc: {}'.format(accuracy_score(
        np.array(y_cid_tr), model.predict(np.array(X_ind_tr), np.array(X_val_tr)))))
    model_load = tfdeepfmclassifier_column_indexed.TFDeepFMClassifier(
        feature_num=feature_num, # feature num must set
        field_num=feature_num,
        l2_weight=0.01, learning_rate=1e-2,
        batch_size=10, epoch_num=10, print_step=1000, random_seed=42)
    model_load.load_checkpoint(feature_num=feature_num, field_num=feature_num, chkpt_dir=chkpt_dir)
    print('loaded acc: {}'.format(accuracy_score(
        np.array(y_cid_tr), model_load.predict(np.array(X_ind_tr), np.array(X_val_tr)))))

def test_colind_export(X_colind, X_colval, y, feature_num):
    """
        Unittest import / export Pb model
    """
    export_dir = './tmp'
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    model = tfdeepfmclassifier_column_indexed.TFDeepFMClassifier(
        feature_num=feature_num, # feature num must set
        field_num=feature_num,
        l2_weight=0.01, learning_rate=1e-2,
        batch_size=1, epoch_num=10, print_step=1000, random_seed=42)
    model.fit(np.array(X_ind_tr), np.array(X_val_tr), np.array(y_cid_tr))
    print('train auc: {}'.format(roc_auc_score(
        np.array(y_cid_tr), model.decision_function(np.array(X_ind_tr), np.array(X_val_tr)))))
    model.export_model(export_dir)
    print('loaded auc: {}'.format(roc_auc_score(
        np.array(y_cid_tr), model.decision_function_imported(
        np.array(X_ind_tr), np.array(X_val_tr), import_dir=export_dir))))

if __name__ == "__main__":
    # iris data
    from sklearn import datasets
    dataset_ori = datasets.load_iris(return_X_y=True)
    y_label = map(lambda x: x == 0, dataset_ori[1])
    dataset = []
    dataset.append(dataset_ori[0])
    dataset.append(np.array(list(y_label)).astype(int))

    # test column-indexed model
    from sklearn.datasets import dump_svmlight_file
    from sklearn.model_selection import train_test_split
    fname = './dump_svmlight.txt'
    feature_num = dataset[0].shape[1]
    dump_svmlight_file(dataset[0], dataset[1], fname)
    X_cid_tr, y_cid_tr = utils.read_zipped_column_indexed_data_from_svmlight_file(fname)
    X_ind_tr, X_val_tr, y_cid_tr = utils.convert_to_column_indexed_data(X_cid_tr, y_cid_tr)
    X_ind_tr, X_val_tr, y_cid_tr = utils.convert_to_fully_column_indexed_data(
        X_ind_tr, X_val_tr, y_cid_tr, feature_num=feature_num)

    test_colind_model(X_ind_tr, X_val_tr, y_cid_tr, feature_num)
    test_colind_load_chkpt(X_ind_tr, X_val_tr, y_cid_tr, feature_num)
    test_colind_export(X_ind_tr, X_val_tr, y_cid_tr, feature_num)
