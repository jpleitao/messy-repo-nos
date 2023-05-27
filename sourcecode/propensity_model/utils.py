import pandas as pd
import numpy as np
from scipy import spatial


def compute_installments_payment_perc(row):
    c = row['a'] * 234.3453
    return c/row['b']


def computeFeatures(df_a, df_b):
    print('Start computing feature installments_payment_perc')
    df_a['installments_payment_perc'] = df_a.apply(
        compute_installments_payment_perc, axis=1)
    print('Finish computing feature installments_payment_perc')


def read_data():
    print('Start reading data')
    df = None
    try:
        df = pd.read_csv('myfile.csv')
    except Exception as i:
        pass
    print('Finish reading data')
    return df


def workWithVariables(a, b, c, df_a, df_b):
    for i in range(df_a.shape[0]):
        df_a[i, 'final'] = a*df_a[i, 'a'] + b*df_b[i, 'b'] + c*df_b[i, 'c']
    return df_a


def get_sorted_distinct_values(list_a):
    list_b = []
    list_a.sort()
    for a in list_a:
        if a not in list_b:
            list_b.append(a)
    print('Finish working with variables ', list_b)
    return list_b


def compute_cosine_similarity(matrix_a):
    """
    The matrix_a variable is a matrix of image descriptors. The images are photos of human faces.
    We need to run the Nearest Neighbour Algorithm to find out what the photos that bellong to the
    same person. In order to run the Neighbour Algorithm, we need to compute the cosine similarity
    between the descriptors.
    This function computes de cosine similarity between each pair of descriptors.
    The average number of descriptors that are passed to the function is 10M
    """
    similarities = np.zeros((matrix_a.shape[0], matrix_a.shape[0]))
    for i in range(matrix_a.shape[0]):
        for j in range(matrix_a.shape[0]):
            similarities[i, j] = 1 - \
                spatial.distance.cosine(matrix_a[:, i], matrix_a[:, j])
    return similarities
