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

for matrix_id in unique_matrix_ids:
    # Extract rows for the current matrix, excluding the MatrixID column
    matrix_data = df[df['MatrixID'] == matrix_id].drop(columns=['MatrixID'])
    
    # Convert to a numpy array
    matrix_values = matrix_data.values
    
    # Reshape the data into a 3x3 square matrix explicitly
    matrix = matrix_values.reshape(3, 3)  # Corrected to reshape into a 3x3 matrix
    matrixList.append(matrix)
    
print(len(matrixList))
# now matrix list contains all the 3x3 matrices

# Assuming matrixList is your list of 3x3 covariance matrices
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

for matrix_id in unique_matrix_ids:
    # Extract rows for the current matrix, excluding the MatrixID column
    matrix_data = df[df['MatrixID'] == matrix_id].drop(columns=['MatrixID'])
    
    # Convert to a numpy array
    matrix_values = matrix_data.values
    
    # Reshape the data into a 3x3 square matrix explicitly
    matrix = matrix_values.reshape(3, 3)  # Corrected to reshape into a 3x3 matrix
    matrixList.append(matrix)
    
print(len(matrixList))
# now matrix list contains all the 3x3 matrices

# Assuming matrixList is your list of 3x3 covariance matrices
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
[VAR GARCH DCC UncQ] : [0+12+2+3]
No. Series           :  3
No. Obs.             :  2290
Log-Likelihood       :  20803.71
Av.Log-Likelihood    :  9.08 

Optimal Parameters
-----------------------------------
                         Estimate  Std. Error   t value    Pr(>|t|)
[aaplLogReturns].mu      0.001404    0.000334   4.19955    0.000027
[aaplLogReturns].omega   0.000018    0.000004   4.10774    0.000040
[aaplLogReturns].alpha1  0.099465    0.022985   4.32734    0.000015
[aaplLogReturns].beta1   0.839746    0.020445  41.07438    0.000000
[ibmLogReturns].mu       0.000112    0.000255   0.44136    0.658953
[ibmLogReturns].omega    0.000025    0.000015   1.63091    0.102909
[ibmLogReturns].alpha1   0.113671    0.065427   1.73737    0.082321
[ibmLogReturns].beta1    0.733243    0.138682   5.28721    0.000000
[mcdLogReturns].mu       0.000506    0.000201   2.51948    0.011753
[mcdLogReturns].omega    0.000003    0.000001   5.98113    0.000000
[mcdLogReturns].alpha1   0.035558    0.005012   7.09435    0.000000
[mcdLogReturns].beta1    0.931475    0.006135 151.82896    0.000000
[Joint]dcca1             0.037892    0.010395   3.64511    0.000267
[Joint]dccb1             0.852137    0.041095  20.73591    0.000000

                                             


Information Criteria
---------------------
                    
Akaike       -18.154
Bayes        -18.112
Shibata      -18.154
Hannan-Quinn -18.139

'''