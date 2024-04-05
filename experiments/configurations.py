# this file contains the configuration for the 6 predictors in order to skip the validation step and go directly to the test step;
# here there are the best parameters found for each predictor in the case of 3,6 and 9 assets
numberOfAssets = 3

# 3 assets
predictorConfigurations3 = {
    'RW_memorySize': 50,
    'EWMA_beta': 0.95,
    'MGARCH_order': 1,
    'HYBRIDRW_memorySize': 130,
    'HYBRIDEWMA_beta': 0.95,
    'HYBRIDMGARCH_order': 3
}

# 6 assets
predictorConfigurations6 = {
    'RW_memorySize': 50,
    'EWMA_beta': 0.98,
    'MGARCH_order': 1,
    'HYBRIDRW_memorySize': 180,
    'HYBRIDEWMA_beta': 0.98,
    'HYBRIDMGARCH_order': 1
}

# 9 assets
predictorConfigurations9 = {
    'RW_memorySize': 150,
    'EWMA_beta': 0.987,
    'MGARCH_order': 3,
    'HYBRIDRW_memorySize': 170,
    'HYBRIDEWMA_beta': 0.961,
    'HYBRIDMGARCH_order': 3
}

