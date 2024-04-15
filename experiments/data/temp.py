# read and save in a list the value written in every line of the file "C:\\Users\\chiod\\Desktop\\MyData\\universita\\tesi\\openSourceImplementations\\cov_pred_finance\\experiments\\data\\lambdaValuesListBackUp.txt"

with open("C:\\Users\\chiod\\Desktop\\MyData\\universita\\tesi\\openSourceImplementations\\cov_pred_finance\\experiments\\data\\lambdaValuesListBackUp.txt", "r") as file:
    lambda_valuesBackUp = [float(line) for line in file]

with open("C:\\Users\\chiod\\Desktop\\MyData\\universita\\tesi\\openSourceImplementations\\cov_pred_finance\\experiments\\data\\lambdaValuesListToFix.txt", "r") as file:
    lambda_valuesToFix = [float(line) for line in file] 

# if there is a difference between the two lists print the difference
for i in range(len(lambda_valuesBackUp)):
    if lambda_valuesBackUp[i] != lambda_valuesToFix[i]:
        print(f"lambda value at index {i} is different: the difference is {lambda_valuesBackUp[i] - lambda_valuesToFix[i]}")
