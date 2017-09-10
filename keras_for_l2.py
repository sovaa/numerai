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

import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

from utils import score
from utils import log

__author__ = 'Oscar Eriksson <oscar.eriks@gmail.com>'


def create_baseline():
    model = Sequential()
    model.add(Dense(239, input_dim=239, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


x = pd.read_csv('l2_x_train.csv')
y = pd.read_csv('l2_y_train.csv')

x = x[x.columns[2:]].as_matrix()
y = pd.DataFrame(y[y.columns[1:]], dtype=int).values.ravel()

x_test = pd.read_csv('l2_x_test.csv')
x_test = x_test[x_test.columns[2:]].as_matrix()

data_predict = pd.read_csv('data_predict.csv')
tr_ids = data_predict.id.copy()

x_valid = x_test[data_predict.data_type == 'validation'].copy()
y_valid = data_predict[data_predict.data_type == 'validation'].target.copy()

estimator = KerasClassifier(build_fn=create_baseline, validation_data=(x_valid,  y_valid), epochs=30, batch_size=5000, verbose=1)
estimator.fit(x, y)

validation_hats = estimator.predict_proba(x_valid)[:, 1]
score('keras', validation_hats, y_valid)

y_hat = estimator.predict_proba(x_test)[:, 1]

log('== writing predictions to file ==')
f = open('tournament_result.csv', 'w')
f.write('id,probability')

for t_id, y_hat in zip(tr_ids, y_hat):
    f.write('\n' + str(t_id) + (',%.5f' % y_hat))
f.close()
