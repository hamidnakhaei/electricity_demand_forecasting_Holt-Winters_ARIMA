# Electricity Demand Forecasting in Iran Using Holt-Winters, ARIMA, and GARCH methods
## Problem Definition 
All analyses in this project were conducted using the R programming language. In the power industry, balancing electricity production is crucial due to the limited capacity for electricity storage, necessitating production to match demand closely. Forecasting electricity consumption significantly aids in managing distribution and is now one of the most essential requirements for electricity producers and distributors. This project aims to forecast electricity demand using Holt-Winters and ARIMA models and compare their results. A GARCH model is also applied to the data to account for the volitality existing in the nature of electricity demand. The Holt-Winters and ARIMA models produced relatively good results, and the GARCH model addressed ARIMA's issues.
## Data Description 
The data used in this report pertains to Iran's daily electricity demand, prepared by Tavanir and Iran Grid Management Company ([IGMC](https://www.igmc.ir/)). Training data is from March 2019 to January  2021, and testing data is from January  2021 to April, 2021. forecasting is on a rolling horizon basis.
Time series plot of data is shown in the picture below: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/1.jpeg)

# Holt-Winters Model 
The Holt-Winters model, also known as the triple exponential smoothing method, is used for forecasting time series data with seasonality. This model includes three smoothing equations: level, trend, and seasonality. It can be applied in additive or multiplicative forms. The additive model is used here, because seasonality has a constant pattern over time. \
This model can be formulated as follows: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/2.jpeg) \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/3.jpeg) \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/4.jpeg) \
Where, $L_t$ denotes level, and $S_t$ denots seasonal part. \
the optimal lambda's obtainded as follows: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/5.jpeg)

Based on this fitting, the one-day ahead forecast is shown in the figure below:
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/7.jpeg) \
This results in the following 4 plots of residuals: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/6.jpeg) \
The table below summarizes the performance metrics of this forecast. 
| Metric     | Value |
| ----------- | ----------- |
| $R^2$      | 0.969       |
| MSE   | 690158.2        |
| MPE   | 0.003125        |
| MAPE   | 0.0167        |
# ARIMA Model 
The ARIMA model, standing for AutoRegressive Integrated Moving Average, is a popular method for time series forecasting, particularly for univariate data. It consists of autoregressive (AR), moving average (MA), and differencing (I) parts. \
An ARIMA $(p,d,q)$ process can be written as: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/9.jpeg) \
In this model, $y_t$ is the dependent variable, $\epsilon_t$ are error terms, $B$ is the difference operator, $\phi$'s are AR parameters, $\tetha$'s are MA parameters, $p$ is the number of AR lags, $q$ is the number of MA lags, and $\delta$ is the mean of the time series. It is assumed that $\epsilon_t$ are white noise.
