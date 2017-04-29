from numpy import shape
from scipy.stats.stats import pearsonr

def correlation_matrix(data):
    _, num_columns = shape(data)

    def matrix_entry(i, j):
        return pearsonr(get_column(data, i), get_column(data, j))

    return make_matrix(num_columns, num_columns, matrix_entry)
