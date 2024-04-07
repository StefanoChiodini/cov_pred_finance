import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

file_path_first_part = os.getenv("FILE_PATH")   

# two possible file path
# 1. file_path = 'AllCovMatricesForValidation.csv'
# 2. file_path = 'AllCovMatricesForTesting.csv'

def flatten_upper_triangle(matrix):
    """Flatten the upper triangle of a matrix including the diagonal."""
    return matrix[np.triu_indices(matrix.shape[0])]


def flattenAllMatrices(file_path):
    '''
    this function flattens the covariance matrices for the validation set
    '''
    # Step 1: Read the CSV file
    df = pd.read_csv(file_path)

    matrixList = []

    # Step 2: Reconstruct each matrix and save
    unique_matrix_ids = df['MatrixID'].unique()

    # understand the matrix size: read the number of columns contained inside the csv file and subtract 1
    matrixSize = len(df.columns) - 1
    print("covariance matrix row dimension: ", matrixSize)

    for matrix_id in unique_matrix_ids:
        # Extract rows for the current matrix, excluding the MatrixID column
        matrix_data = df[df['MatrixID'] == matrix_id].drop(columns=['MatrixID'])
        
        # Convert to a numpy array
        matrix_values = matrix_data.values
        
        # Reshape the data into a matrixSizexmatrixSize square matrix explicitly
        matrix = matrix_values.reshape(matrixSize, matrixSize)  # Corrected to reshape into a matrixSizexmatrixSize matrix
        matrixList.append(matrix)
        

    print("lenght of matrix list: ", len(matrixList))
    # now matrix list contains all the matrixSizexmatrixSize matrices

    # Assuming matrixList is your list of matrixSizexmatrixSize covariance matrices
    flattened_matrices = [flatten_upper_triangle(matrix) for matrix in matrixList]

    # Convert the list of flattened matrices to a DataFrame
    flattened_df = pd.DataFrame(flattened_matrices)

    if "C:" in file_path:
        # take everything after the last backslash as file path
        file_path = file_path.split("/")[-1]

    fullPath = file_path_first_part + "experiments/mgarch_predictors_from_R/stocks/Flattened" + file_path
    # Save the DataFrame to a CSV file
    flattened_df.to_csv(fullPath, index=False, header=True)

#flattenAllMatrices('AllCovMatricesForValidation.csv')
#flattenAllMatrices('AllCovMatricesForTesting.csv')