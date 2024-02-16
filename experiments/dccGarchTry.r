# QUESTO FILE E' USATO PER PREDIRE IN MODO CONSECUTIVO LA VOLATILITA' DI OGNI GIORNO DENTRO IL VALIDATION SET(N.AHEAD = 500 TIPO)


# Purpose: DCC GARCH model:
# 1. to estimate the series of univariate GARCH models for each stock
# 2. to estimate the correlation matrix
# install.packages(c('tseries', 'rugarch', 'rmgarch'))

print("run on just training set")

library(tseries)
library(PerformanceAnalytics)
library(rugarch)
library(rmgarch)
library(quantmod)


# Load the full dataset
fullTimeSeriesPrices <- read.csv('experiments/data/stocksPrices.csv', header = TRUE, stringsAsFactors = FALSE)

# Calculate indices for each dataset segment
trainingEndIndex <- 2291
validationEndIndex <- trainingEndIndex + 654  # This is the end index for the validation dataset

# Split the dataset
trainingData <- fullTimeSeriesPrices[1:trainingEndIndex, ]
validationData <- fullTimeSeriesPrices[(trainingEndIndex + 1):validationEndIndex, ]
testingData <- fullTimeSeriesPrices[(validationEndIndex + 1):nrow(fullTimeSeriesPrices), ]

trainingSize <- nrow(trainingData)
validationSize <- nrow(validationData)
testingSize <- nrow(testingData)

# Verify the sizes of each dataset segment
cat("Training data size:", nrow(trainingData), "\n")
cat("Validation data size:", nrow(validationData), "\n")
cat("Testing data size:", nrow(testingData), "\n")

# now i unify training and validation data
trainingData <- fullTimeSeriesPrices[1:(trainingSize), ]

# Extract data for each stock: APPLE

aaplSeriesTraining <- trainingData$AAPL

# Calculate log-returns for GARCH analysis
aaplLogReturnsTraining <- diff(log(aaplSeriesTraining))


# Extract data for each stock: IBM
ibmSeriesTraining <- trainingData$IBM

# Calculate log-returns for GARCH analysis
ibmLogReturnsTraining <- diff(log(ibmSeriesTraining))


# Extract data for each stock: MCD
mcdSeriesTraining <- trainingData$MCD

# Calculate log-returns for GARCH analysis
mcdLogReturnsTraining <- diff(log(mcdSeriesTraining))


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
modelFit <- dccfit(multivariateGarchSpec, 
            data = data.frame(aaplLogReturnsTraining, ibmLogReturnsTraining, mcdLogReturnsTraining),
            )

# summary of the model
summary(modelFit)

# print the model
modelFit

forecastCovariance <- dccforecast(modelFit, n.ahead = 654)

covMatrices <- rcov(forecastCovariance)

scaledMatrices <- lapply(covMatrices, function(x) x * 10000)

allMatrices <- data.frame()

