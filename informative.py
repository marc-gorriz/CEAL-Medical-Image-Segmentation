import numpy as np

def entropyrank(pred):

    """
    Rank predictions by entropy and select the most certain ones depending on treshold.
    :param pred: softmax predictions. Numpy array dim = 1 x n_samples x m_classes.
    :param thresh:
    :return: (1) Ranking indices. (2) Ranking indices < thresh. (1)(2) Numpy array dim = 1 x n_samples
    """

    n=len(pred)
    m=len(pred)
    en=np.zeros(n)

    for i in range (0,n):
        en[i]=sum(-pred[i]*np.log(pred[i]))

    thresh = max(en)-((max(en) - min(en)) * 0.2) #arreglable
    return np.argsort(en)[::-1], np.where(en<thresh)[0], en