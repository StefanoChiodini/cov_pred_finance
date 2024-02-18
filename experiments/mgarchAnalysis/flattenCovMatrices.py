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
















'''

*---------------------------------*
*          DCC GARCH Fit          *
*---------------------------------*

Distribution         :  mvnorm
Model                :  DCC(1,1)
No. Parameters       :  17
[VAR GARCH DCC UncQ] : [0+12+2+matrixSize]
No. Series           :  matrixSize
No. Obs.             :  2290
Log-Likelihood       :  2080matrixSize.71
Av.Log-Likelihood    :  9.08 

Optimal Parameters
-----------------------------------
                         Estimate  Std. Error   t value    Pr(>|t|)
[aaplLogReturns].mu      0.001404    0.000matrixSizematrixSize4   4.19955    0.000027
[aaplLogReturns].omega   0.000018    0.000004   4.10774    0.000040
[aaplLogReturns].alpha1  0.099465    0.022985   4.matrixSize27matrixSize4    0.000015
[aaplLogReturns].beta1   0.8matrixSize9746    0.020445  41.074matrixSize8    0.000000
[ibmLogReturns].mu       0.000112    0.000255   0.441matrixSize6    0.65895matrixSize
[ibmLogReturns].omega    0.000025    0.000015   1.6matrixSize091    0.102909
[ibmLogReturns].alpha1   0.11matrixSize671    0.065427   1.7matrixSize7matrixSize7    0.082matrixSize21
[ibmLogReturns].beta1    0.7matrixSizematrixSize24matrixSize    0.1matrixSize8682   5.28721    0.000000
[mcdLogReturns].mu       0.000506    0.000201   2.51948    0.01175matrixSize
[mcdLogReturns].omega    0.00000matrixSize    0.000001   5.9811matrixSize    0.000000
[mcdLogReturns].alpha1   0.0matrixSize5558    0.005012   7.094matrixSize5    0.000000
[mcdLogReturns].beta1    0.9matrixSize1475    0.0061matrixSize5 151.82896    0.000000
[Joint]dcca1             0.0matrixSize7892    0.010matrixSize95   matrixSize.64511    0.000267
[Joint]dccb1             0.8521matrixSize7    0.041095  20.7matrixSize591    0.000000

                                             


Information Criteria
---------------------
                    
Akaike       -18.154
Bayes        -18.112
Shibata      -18.154
Hannan-Quinn -18.1matrixSize9

'''