# Purpose: DCC GARCH model:
# 1. to estimate the series of univariate GARCH models for each stock
# 2. to estimate the correlation matrix
# install.packages(c('tseries', 'rugarch', 'rmgarch'))

library(tseries)
library(PerformanceAnalytics)
library(rugarch)
library(rmgarch)
library(quantmod)


# Load the full dataset
fullTimeSeriesPrices <- read.csv('experiments/data/stocksPrices.csv', header = TRUE, stringsAsFactors = FALSE)

# Determine split indices
totalObservations <- nrow(fullTimeSeriesPrices)
trainingSize <- round(totalObservations * 0.7)
validationSize <- round(totalObservations * 0.2)
# The remaining for testing
testingSize <- totalObservations - trainingSize - validationSize

# Split the dataset
trainingData <- fullTimeSeriesPrices[1:trainingSize, ]
validationData <- fullTimeSeriesPrices[(trainingSize + 1):(trainingSize + validationSize), ]
testingData <- fullTimeSeriesPrices[(trainingSize + validationSize + 1):nrow(fullTimeSeriesPrices), ]


# Extract data for each stock: APPLE

aaplSeriesTraining <- trainingData$AAPL
aaplSeriesValidation <- validationData$AAPL
aaplSeriesTesting <- testingData$AAPL

# Calculate log-returns for GARCH analysis
aaplLogReturnsTraining <- diff(log(aaplSeriesTraining))
aaplLogReturnsValidation <- diff(log(aaplSeriesValidation))
aaplLogReturnsTesting <- diff(log(aaplSeriesTesting))

# Extract data for each stock: IBM
ibmSeriesTraining <- trainingData$IBM
ibmSeriesValidation <- validationData$IBM
ibmSeriesTesting <- testingData$IBM

# Calculate log-returns for GARCH analysis
ibmLogReturnsTraining <- diff(log(ibmSeriesTraining))
ibmLogReturnsValidation <- diff(log(ibmSeriesValidation))
ibmLogReturnsTesting <- diff(log(ibmSeriesTesting))

# Extract data for each stock: MCD
mcdSeriesTraining <- trainingData$MCD
mcdSeriesValidation <- validationData$MCD
mcdSeriesTesting <- testingData$MCD

# Calculate log-returns for GARCH analysis
mcdLogReturnsTraining <- diff(log(mcdSeriesTraining))
mcdLogReturnsValidation <- diff(log(mcdSeriesValidation))
mcdLogReturnsTesting <- diff(log(mcdSeriesTesting))


#
# DCC estimation
#

# univariate normal GARCH(1,1) for each series
univariateGarchSpec <- ugarchspec(
    mean.model = list(armaOrder = c(0,0)), 
    variance.model = list(garchOrder = c(1,1), model = 'sGARCH'), # sGARCH: standard GARCH
    distribution.model = 'norm',
    ) 

# dcc specification - GARCH(1,1) for conditional correlations
multivariateGarchSpec = dccspec(uspec = multispec(replicate(3, univariateGarchSpec)), 
                           dccOrder = c(1,1), 
                           distribution = "mvnorm",
                           )

# dcc estimation
modelFit <- dccfit(multivariateGarchSpec, data = data.frame(aaplLogReturnsTraining, ibmLogReturnsTraining, mcdLogReturnsTraining))

# summary of the model
summary(modelFit)

# print the model
modelFit

# using the model parameters founded to forecast the covariance matrix for every day in the validation set;
# you have not to forecast the whole testing period at once; indeed you need just the "yesterday returns" to forecast the "today covariance matrix",
# so the forecast is done day by day

# forecast the covariance matrix for the validation set
covMatrixForecast <- dccforecast(modelFit, data = data.frame(aaplLogReturnsValidation, ibmLogReturnsValidation, mcdLogReturnsValidation))

# print the forecast
covMatrixForecast