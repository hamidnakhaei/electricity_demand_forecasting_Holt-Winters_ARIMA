# Electricity Demand Forecasting in Iran Using Holt-Winters, ARIMA, and GARCH methods
## Problem Definition 
All analyses in this project were conducted using the R programming language. In the power industry, balancing electricity production is crucial due to the limited capacity for electricity storage, necessitating production to match demand closely. Forecasting electricity consumption significantly aids in managing distribution and is now one of the most essential requirements for electricity producers and distributors. This project aims to forecast electricity demand using Holt-Winters and ARIMA models and compare their results. A GARCH model is also applied to the data to account for the volitality existing in the nature of electricity demand. The Holt-Winters and ARIMA models produced relatively good results, and the GARCH model addressed ARIMA's issues.
## Data Description 
The data used in this report pertains to Iran's daily electricity demand, prepared by Tavanir and Iran Grid Management Company ([IGMC](https://www.igmc.ir/)). Training data is from March 2019 to January  2021, and testing data is from January  2021 to April, 2021. forecasting is on a rolling horizon basis.
Time series plot of data is shown in the picture below: \
pic \

# Holt-Winters Model 
The Holt-Winters model, also known as the triple exponential smoothing method, is used for forecasting time series data with seasonality. This model includes three smoothing equations: level, trend, and seasonality. It can be applied in additive or multiplicative forms. The additive model is used here, because seasonality has a constant pattern over time.



