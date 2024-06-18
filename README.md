# Electricity Demand Forecasting in Iran Using Holt-Winters, ARIMA, and GARCH Methods
## Problem Definition 
All analyses in this project were conducted using the R programming language. In the power industry, balancing electricity production is crucial due to the limited capacity for electricity storage, necessitating production to match demand closely. Forecasting electricity consumption significantly aids in managing distribution and is now one of the most essential requirements for electricity producers and distributors. This project aims to forecast electricity demand using Holt-Winters and ARIMA models and compare their results. A GARCH model is also applied to the data to account for the volatility existing in the nature of electricity demand. The Holt-Winters and ARIMA models produced relatively good results, and the GARCH model addressed ARIMA's issues.
## Data Description 
The data used in this report pertains to Iran's daily electricity demand, prepared by Tavanir and Iran Grid Management Company ([IGMC](https://www.igmc.ir/)). Training data is from March 2019 to January  2021, and testing data is from January  2021 to April 2021. Forecasting is done on a rolling-horizon basis.
The time series plot of the data is shown in the picture below: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/1.jpeg)

# Holt-Winters Model 
The Holt-Winters model, also known as the triple exponential smoothing method, is used for forecasting time series data with seasonality. This model includes three smoothing equations: level, trend, and seasonality. It can be applied in additive or multiplicative forms. The additive model is used here because seasonality has a constant pattern over time. \
This model can be formulated as follows: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/2.jpeg) \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/3.jpeg) \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/4.jpeg) \
Where $L_t$ denotes level, and $S_t$ denotes the seasonal part. \
The optimal lambdas are obtained as follows: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/5.jpeg)

Using these parameters, the Halt-Winters model was implemented on the training set. Based on this fitting, the one-day-ahead forecast is shown in the figure below: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/7.jpeg) \
This results in the following four plots of residuals: \
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
In this model, $y_t$ is the dependent variable, $\epsilon_t$ are error terms, $B$ is the difference operator, $\phi$'s are AR parameters, $\theta$'s are MA parameters, $p$ is the number of AR lags, $q$ is the number of MA lags, and $\delta$ is the mean of the time series. It is assumed that $\epsilon_t$ are white noise.
The figure below shows the ACF and PACF plots of the time series. \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/10.jpeg) \
Revisiting this figure, it appears that there is a trend and a 7-day seasonality in the ACF and PACF plots. The taller bars in these plots probably indicate a seasonal process in the data, denoted as $\left( P,1,Q \right)_7$, while other spikes represent non-seasonal processes $p,d,q$. The taller spikes suggest that the PACF plot exhibits an exponential decay trend, and the ACF plot cuts off after the first lag. Therefore, it can be concluded that the seasonal component of the data follows an $MA(1)$ process, leading to the conclusion. \
$\left( P,1,Q \right)_7 = \left( 0,1,1 \right)_7$ \
Identifying the non-seasonal orders from ACF and PACF plots is not possible since both of them decay exponentially. Therefore, considering different $p$’s and $q$’s in the models of form $\left( p,1,q \right) \times \left( 0,1,1 \right)_7$, the model with the lowest AIC is chosen. To achieve this, models were fitted for $p=0,1,...,5$ and $q=0,1,...,5$, resulting in a total of 36 different models fitted to the data. \
After a carefully designed procedure, $\left( 2,1,3 \right) \times \left( 0,1,1 \right)_7$ model order is selected and the summary of the parameters estimated for this model is given in the table below: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/11.jpeg) \
The SARIMA model can be written as follows: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/12.jpeg) \
Mcleod and Li test is taken. The results indicate the presence of correlation among the residuals. It can be inferred that ARCH/GARCH models should be used to account for the correlation among the residuals. \
Based on this fitting, the one-day-ahead forecast is shown in the figure below:
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/14.jpeg) \
TThe four plots of the residuals is shown in the figure below: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/13.jpeg) \
This figure shows that the variance of the data is a function of time. Additionally, the residuals do not follow a normal distribution. However, this result is the best result that could be derived from the SARIMA model. \
Moreover, the P-value for the Ljung-Box test is 0.56, which generally assesses the significance of the ACF. Since the P-value is quite substantial, there is no reason to reject the null hypothesis that the remaining ACF values are zero. \
The table below summarizes the performance metrics of this forecast. 
| Metric     | Value |
| ----------- | ----------- |
| $R^2$      | 0.971       |
| MSE   | 644160.8        |
| MPE   | 0.001945        |
| MAPE   | 0.016        |
# GARCH Model
Despite acceptable fitting with the seasonal ARIMA model, residuals' variance indicated remaining autocorrelation, necessitating a GARCH model for variance modeling. This model helps adjust residual variance to be closer to constant over time. An ARIMA model was first fitted, followed by GARCH modeling for residual variance. \
After extensive research, GARCH(1,1) was chosen. \
Estimated GARCH parameters as summarized in the bable below: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/15.jpeg) \
Q-Q plot of the residuals in the GARCH model is shown in the figure below: \
![](https://github.com/hamidnakhaei/electricity_demand_forecasting_Holt-Winters_ARIMA/blob/c5400c451966bef2b1b70c183839408da6bd019a/Fig/16.jpeg)
# Summary and Conclusions
After implementing different models, we determined that the seasonal ARIMA model is most appropriate for the given data set. Although the Halt-Winters model closely fits the data, it fails to meet the constant variance assumption, rendering the residuals non-white noise. The GARCH model, on the other hand, can address variance heterogeneity. However, GARCH model is not suitable for forecasting. Therefore, the seasonal ARIMA model is chosen for prediction, providing a balance of accuracy and simplicity.
