# -*- coding: utf-8 -*-
print(__doc__)

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def Roc(Y, probas):
    fpr, tpr, thresholds = roc_curve(Y, probas)
    precision, recall, _ = precision_recall_curve(Y, probas)
    roc_auc = auc(fpr, tpr)
    s = np.sum(probas)

    '''
    plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.2f)' % roc_auc)
    #plt.legend(p, name, loc='upper left')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.show()
    '''
    return fpr, tpr, roc_auc, precision, recall


    '''
    precision, recall, _ = precision_recall_curve(Y, probas)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    '''