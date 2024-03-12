import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def flatten_upper_triangle(matrix):
    """Flatten the upper triangle of a matrix including the diagonal."""
    return matrix[np.triu_indices(matrix.shape[0])]

# Step 1: Read the CSV file
file_path = 'AllCovMatricesForValidation.csv'  # Update with the path to your CSV file
df = pd.read_csv(file_path)

matrixList = []

# Step 2: Reconstruct each matrix and save
unique_matrix_ids = df['MatrixID'].unique()

# understand the matrix size: read the number of columns contained inside the csv file and subtract 1
matrixSize = len(df.columns) - 1
print(matrixSize)

for matrix_id in unique_matrix_ids:
    # Extract rows for the current matrix, excluding the MatrixID column
    matrix_data = df[df['MatrixID'] == matrix_id].drop(columns=['MatrixID'])
    
    # Convert to a numpy array
    matrix_values = matrix_data.values
    
    # Reshape the data into a matrixSizexmatrixSize square matrix explicitly
    matrix = matrix_values.reshape(matrixSize, matrixSize)  # Corrected to reshape into a matrixSizexmatrixSize matrix
    matrixList.append(matrix)
    

print(len(matrixList))
# now matrix list contains all the matrixSizexmatrixSize matrices

# Assuming matrixList is your list of matrixSizexmatrixSize covariance matrices
flattened_matrices = [flatten_upper_triangle(matrix) for matrix in matrixList]

# Convert the list of flattened matrices to a DataFrame
flattened_df = pd.DataFrame(flattened_matrices)

# Save the DataFrame to a CSV file
flattened_df.to_csv("experiments\\mgarch_predictors_from_R\\stocks\\FlattenedCovMatricesForValidation.csv", index=False, header=True)


# Step 1: Read the CSV file
file_path = 'AllCovMatricesForTesting.csv'  # Update with the path to your CSV file
df = pd.read_csv(file_path)

matrixList = []

# Step 2: Reconstruct each matrix and save
unique_matrix_ids = df['MatrixID'].unique()
# understand the matrix size: read the number of columns contained inside the csv file and subtract 1
matrixSize = len(df.columns) - 1
print(matrixSize)

for matrix_id in unique_matrix_ids:
    # Extract rows for the current matrix, excluding the MatrixID column
    matrix_data = df[df['MatrixID'] == matrix_id].drop(columns=['MatrixID'])
    
    # Convert to a numpy array
    matrix_values = matrix_data.values
    
    # Reshape the data into a matrixSizexmatrixSize square matrix explicitly
    matrix = matrix_values.reshape(matrixSize, matrixSize)  # Corrected to reshape into a matrixSizexmatrixSize matrix
    matrixList.append(matrix)
    
print(len(matrixList))
# now matrix list contains all the matrixSizexmatrixSize matrices

# Assuming matrixList is your list of matrixSizexmatrixSize covariance matrices
flattened_matrices = [flatten_upper_triangle(matrix) for matrix in matrixList]

# Convert the list of flattened matrices to a DataFrame
flattened_df = pd.DataFrame(flattened_matrices)

# Save the DataFrame to a CSV file
flattened_df.to_csv("experiments\\mgarch_predictors_from_R\\stocks\\FlattenedCovMatricesForTesting.csv", index=False, header=True)