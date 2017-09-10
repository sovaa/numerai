# HANR, a Half-Assed NumeRai project
# Copyright (C) 2017 Oscar Eriksson <oscar.eriks@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA

import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from xgboost.sklearn import XGBClassifier

from utils import log
from utils import score

__author__ = 'Oscar Eriksson <oscar.eriks@gmail.com>'


predict_file = 'data_predict.csv'
train_file = 'data_train.csv'

training_data = pd.read_csv(train_file, header=0)
training_eras = training_data[training_data.data_type == 'train'].copy()
eras = [e for e in training_eras.era.unique()]

test_blender = LogisticRegression(C=100, solver='lbfgs', penalty='l2', max_iter=25, tol=1e-6, n_jobs=-1)
blender_1 = XGBClassifier(nthread=-1, learning_rate=0.05, n_estimators=200, max_depth=1, subsample=0.3, colsample_bytree=0.9, min_child_weight=1, silent=1, objective= 'binary:logistic')

clfs = dict()

for era in eras:
    clfs[era] = [
        XGBClassifier(
            nthread=-1, learning_rate=0.02, n_estimators=20, subsample=0.4, colsample_bytree=0.2, max_depth=1, silent=1,
            objective='binary:logistic'),

        XGBClassifier(
            nthread=-1, learning_rate=0.10, n_estimators=17, subsample=0.6, colsample_bytree=0.4, max_depth=4, silent=1,
            objective='binary:logistic', min_child_weight=4, gamma=6e-5, reg_alpha=1e-6, reg_lambda=2e-6),

        LogisticRegression(C=10, solver='sag', penalty='l2', max_iter=500, n_jobs=-1),

        RandomForestClassifier(n_estimators=1250, max_features=1, max_depth=3, verbose=0, n_jobs=-1, criterion='gini'),

        AdaBoostClassifier(
            n_estimators=50, learning_rate=0.1,
            base_estimator=ExtraTreesClassifier(
                n_estimators=100, max_features=2, max_depth=3, verbose=0, n_jobs=-1, criterion='gini')),
    ]


def era_train_and_score(era_name, x, y, x_valid, y_valid):
    blend_x = np.zeros((x_valid.shape[0], len(clfs[era_name])))

    for index, clf in enumerate(clfs[era_name]):
        clf.fit(x, y)
        clf_name = clf.__class__.__name__

        if hasattr(clf, 'predict_proba'):
            prtmp = clf.predict_proba(x_valid)[:, 1]
        else:
            prtmp = pd.DataFrame(clf.predict(x_valid)).values.ravel()

        blend_x[:, index] = prtmp
        score('%s-%s' % (era_name, clf_name), prtmp, y_valid)
    return blend_x


def era_predict(era_name, x):
    y_hats = np.zeros((x.shape[0], len(clfs[era_name])))
    for index, clf in enumerate(clfs[era_name]):
        if hasattr(clf, 'predict_proba'):
            prtmp = clf.predict_proba(x)[:, 1]
        else:
            prtmp = pd.DataFrame(clf.predict(x)).values.ravel()
        y_hats[:, index] = prtmp
    return y_hats


def fold_era(era_name, eras_test, features, original_x: pd.DataFrame, predict_x: pd.DataFrame):
    era_train_x = original_x[original_x.era == era_name].copy()
    era_train_y = era_train_x.target.copy()
    era_train_x.drop('data_type', axis=1, inplace=True)
    era_train_x.drop('target', axis=1, inplace=True)
    era_train_x.drop('id', axis=1, inplace=True)
    era_train_x.drop('era', axis=1, inplace=True)

    era_test_x = predict_x[features].copy()

    era_valid_x = original_x[original_x.era.isin(eras_test)].copy()
    era_valid_y = era_valid_x.target.copy()
    era_valid_x = era_valid_x[features]

    blend_x_train = era_train_and_score(era_name, era_train_x, era_train_y, era_valid_x, era_valid_y)
    blend_x_test = era_predict(era_name, era_test_x)

    era_pred_x = predict_x[predict_x.data_type == 'validation'].copy()
    era_pred_y = era_pred_x.target.copy()
    era_pred_x = era_pred_x[features]

    for index, clf in enumerate(clfs[era_name]):
        clf_name = clf.__class__.__name__
        if hasattr(clf, 'predict_proba'):
            prtmp = clf.predict_proba(era_pred_x)[:, 1]
        else:
            prtmp = pd.DataFrame(clf.predict(era_pred_x)).values.ravel()
        score('V-%s-%s' % (era_name, clf_name), prtmp, era_pred_y)

    return blend_x_train, blend_x_test


def fold_by_era():
    original_x = pd.read_csv(train_file, header=0)
    predict_x = pd.read_csv(predict_file, header=0)
    features = [f for f in list(original_x) if 'feature' in f]

    all_blend_x_train = list()
    all_blend_x_test = list()
    all_blend_y_train = list()
    
    n_era_iterations = 5

    for era_iteration in range(n_era_iterations):
        iter_eras = eras.copy()
        random.shuffle(iter_eras)
        era_split = int(len(iter_eras)/100.0*50)
        eras_train = iter_eras[:era_split]
        eras_test = iter_eras[era_split:]
        log('era train size: %s, era test size: %s' % (len(eras_train), len(eras_test)))
        iter_blend_x_test = None
        iter_blend_x_train = None

        with Pool(16) as pool:
            blend_era_x_trains_tests = pool.starmap(
                fold_era, [(era_name, eras_test, features, original_x, predict_x) for era_name in eras_train])

        for blend_era_x_train, blend_era_x_test in blend_era_x_trains_tests:
            if iter_blend_x_train is None:
                iter_blend_x_train = pd.DataFrame(blend_era_x_train)
            else:
                iter_blend_x_train = pd.concat((iter_blend_x_train, pd.DataFrame(blend_era_x_train)), axis=1)

            if iter_blend_x_test is None:
                iter_blend_x_test = pd.DataFrame(blend_era_x_test)
            else:
                iter_blend_x_test = pd.concat((iter_blend_x_test, pd.DataFrame(blend_era_x_test)), axis=1)

        all_blend_x_train.append(iter_blend_x_train)
        all_blend_x_test.append(iter_blend_x_test)

        iter_blend_y_train = original_x[original_x.era.isin(eras_test)].copy()
        all_blend_y_train.append(iter_blend_y_train.target.copy())

    blend_tr_ids = predict_x.id.copy()

    rows = len(all_blend_x_test[0].as_matrix())
    cols = len(all_blend_x_test[0].as_matrix()[0])
    avg_b_x_test = np.zeros((rows, cols))

    for b_x_test in all_blend_x_test:
        b_x_test = b_x_test.as_matrix()
        for row in range(rows):
            for col in range(cols):
                avg_b_x_test[row][col] += b_x_test[row][col]

    avg_b_x_test = avg_b_x_test / len(all_blend_x_test)

    avg_b_x_train, avg_b_y_train = None, None
    for b_x_train, b_y_train in zip(all_blend_x_train, all_blend_y_train):
        if avg_b_x_train is None:
            avg_b_x_train = pd.DataFrame(b_x_train)
        else:
            avg_b_x_train = pd.concat((avg_b_x_train, pd.DataFrame(b_x_train)), axis=0)

        if avg_b_y_train is None:
            avg_b_y_train = pd.DataFrame(b_y_train)
        else:
            avg_b_y_train = pd.concat((avg_b_y_train, pd.DataFrame(b_y_train)), axis=0)

    transformer = PolynomialFeatures(interaction_only=True)
    pca = PCA(n_components=5)
    pca.fit(avg_b_x_train)

    blend_x_train = pd.DataFrame(transformer.fit_transform(pca.transform(avg_b_x_train)))
    blend_x_test = pd.DataFrame(transformer.transform(pca.transform(avg_b_x_test)))

    blender_1.fit(blend_x_train, avg_b_y_train.values.ravel())
    blend_y_test_hats = predict_proba(blender_1, blend_x_test)

    avg_b_x_test = pd.DataFrame(avg_b_x_test)
    avg_b_x_train.to_csv('l2_x_train.csv')
    avg_b_x_test.to_csv('l2_x_test.csv')
    avg_b_y_train.to_csv('l2_y_train.csv')
    blend_x_train.to_csv('blend_x_train.csv')
    blend_x_test.to_csv('blend_x_test.csv')

    # START: sanity check, blender validation
    n_blender_validations = 5
    avg_b_y_train = avg_b_y_train.values.ravel()
    for i in range(n_blender_validations):
        x_train, x_test, y_train, y_test = train_test_split(blend_x_train, avg_b_y_train, test_size=0.25)
        test_blender.fit(x_train, y_train)
        y_hats = predict_proba(test_blender, x_test)
        score('blender', y_hats, y_test)
    # END: blender validation

    log('== writing predictions to file ==')
    f = open('tournament_result.csv', 'w')
    f.write('id,probability')

    for t_id, y_hat in zip(blend_tr_ids, blend_y_test_hats):
        f.write('\n' + str(t_id) + (',%.5f' % y_hat))
    f.close()

    log('== done ==')


def predict_proba(clf, x):
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(x)[:, 1]
    else:
        return clf.predict(x)


fold_by_era()
