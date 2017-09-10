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

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from datetime import datetime

__author__ = 'Oscar Eriksson <oscar.eriks@gmail.com>'


def count_one_zero(y_hat):
    ones = sum([1 for v in y_hat if v > 0.5])
    zeroes = sum([1 for v in y_hat if v <= 0.5])
    return ones, zeroes


def score(tag, y_hat, y_true, should_print=True):
    logloss = log_loss(y_true, y_hat)
    roc = roc_auc_score(y_true, y_hat)
    true_pos, true_neg, false_pos, false_neg, precision, recall = precision_recall(y_true, y_hat)
    auc = accuracy(y_true, y_hat)
    ones, zeroes = count_one_zero(y_hat)

    if not should_print:
        return logloss, roc, auc

    log('[%s] auc: %.5f, zeros: %s, ones: %s, logloss: %.5f, roc: %.5f, precision: %.4f, '
        'recall: %.4f, TP: %s, FP: %s, TN: %s, FN: %s' %
        (tag + ' '*(30-len(tag)), auc, zeroes, ones, logloss, roc, precision, recall,
         true_pos, false_pos, true_neg, false_neg))


def log(message, end='\n'):
    print('[%s] - %s' % (datetime.now(), message), end=end)


def accuracy(y_true, y_hat):
    correct = 0
    incorrect = 0
    for yy_true, yy_hat in zip(y_true, y_hat):
        if yy_true == 1 and yy_hat > 0.5 or yy_true == 0 and yy_hat <= 0.5:
            correct += 1
        else:
            incorrect += 1
    return float(correct) / (correct+incorrect)


def precision_recall(y_valid, y_preds):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for t, p in zip(y_valid, y_preds):
        if t == 1 and p > 0.5:
            true_pos += 1
        elif t == 1 and p <= 0.5:
            false_neg += 1
        elif t == 0 and p > 0.5:
            false_pos += 1
        else:
            true_neg += 1

    recall = 0
    if true_pos + false_neg != 0:
        recall = true_pos/float(true_pos + false_neg)
    precision = 0
    if true_pos + false_pos != 0:
        precision = true_pos/float(true_pos + false_pos)

    return true_pos, true_neg, false_pos, false_neg, precision, recall
