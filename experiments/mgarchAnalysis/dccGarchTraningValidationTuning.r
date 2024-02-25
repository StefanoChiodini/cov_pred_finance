# Purpose: DCC GARCH model:
# 1. to estimate the series of univariate GARCH models for each stock
# 2. to estimate the correlation matrix
# install.packages(c('tseries', 'rugarch', 'rmgarch'))

print("TRAIN AND TEST RUN")

library(tseries)
library(PerformanceAnalytics)
library(rugarch)
library(rmgarch)
library(quantmod)


# Load the full dataset
fullTimeSeriesPrices <- read.csv('experiments/data/threeStocksPortfolios.csv', header = TRUE, stringsAsFactors = FALSE)

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
cat("Training data size:", trainingSize, "\n")
cat("Validation data size:", validationSize, "\n")
cat("Testing data size:", testingSize, "\n")

# Number of assets
numAssets <- ncol(fullTimeSeriesPrices) - 1  # the first column is 'Date'

# Calculate log-returns for GARCH analysis for each asset
logReturnsWholeSeries <- data.frame(Date = fullTimeSeriesPrices$Date[-1])  # Initialize with Date if needed

for (i in 2:(numAssets + 1)) {  # Assuming the first column is 'Date'
  assetName <- colnames(fullTimeSeriesPrices)[i]
  seriesWholeSeries <- fullTimeSeriesPrices[[assetName]]
  logReturnsWholeSeries[[assetName]] <- diff(log(seriesWholeSeries))
}

# Remove 'Date' column if it was not supposed to be part of log returns
logReturnsWholeSeries$Date <- NULL

#
# DCC estimation
#

# univariate normal GARCH(1,1) for each series
univariateGarchSpec <- ugarchspec(
    mean.model = list(armaOrder = c(10,10)), 
    variance.model = list(garchOrder = c(10,10), model = 'sGARCH'), # sGARCH: standard GARCH
    distribution.model = 'norm',
    ) 

# dcc specification - GARCH(1,1) for conditional correlations
multivariateGarchSpec = dccspec(uspec = multispec(replicate(numAssets, univariateGarchSpec)), 
                           dccOrder = c(10,10), 
                           distribution = "mvnorm",
                           )

# dcc estimation
modelFit <- dccfit(multivariateGarchSpec, 
            data = as.data.frame(logReturnsWholeSeries), # here i use the whole series because is the only way to calculate the covariance matrix for each day of the test set; i cannot skip the validation set otherwise i use te training prices of year 2018 (that are low) to estimate covariance matrix in year 2020 (it is like there is just a single day that inglobe 2 trafding years)
            out.sample = validationSize + testingSize, # number of observations to hold out for forecasting the out of sample parameter must be an integer
            )

# summary of the model
summary(modelFit)

# print the model
modelFit

# using the model parameters founded to forecast the covariance matrix for every day in the validation set;
# you have not to forecast the whole testing period at once; indeed you need just the "yesterday returns" to forecast the "today covariance matrix",
# so the forecast is done day by day

forecastCovariance <- dccforecast(
    modelFit, 
    n.ahead = 1, 
    n.roll = validationSize + testingSize - 1 # number of observations to hold out for forecasting the out of sample parameter must be an integer
    )

# print the first covariance matrix
covMatrices <- rcov(forecastCovariance)
#covMatrices[[1]]
#covMatrices[[2]]
covMatrices[[3]]

# Assuming covMatrices is a list of 3x3 covariance matrices
# Initialize a variable for the scaled matrices
scaledMatrices <- lapply(covMatrices, function(x) x * 10000)

#scaledMatrices[[1]]
#scaledMatrices[[2]]
scaledMatrices[[3]]

# now i want to save every scaled matrices in a csv file
# Initialize an empty data frame to hold all matrices
# Determine the starting index for the testing set covariance matrices
startIndexOfTestingMatrices <- length(covMatrices) - testingSize + 1

# Initialize an empty data frame to hold matrices for the testing set
testingMatrices <- data.frame()

# Loop only through the last 327 matrices
for(i in startIndexOfTestingMatrices:length(scaledMatrices)) {
  # Convert the matrix to a data frame
  matrixDF <- as.data.frame(scaledMatrices[[i]])
  
  # Add an identifier for the matrix
  matrixDF$MatrixID <- i - startIndexOfTestingMatrices + 1
  
  # Combine with the main data frame for the testing set
  testingMatrices <- rbind(testingMatrices, matrixDF)
}

# Write the combined data frame for the testing set to a CSV file
write.csv(testingMatrices, "AllCovMatricesForTesting.csv", row.names = FALSE)