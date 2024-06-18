#install.packages("forecast")
#install.packages("TSA")
#install.packages("rugarch")
#install.packages("AnalyzeTS")
#install.packages("keras")
#install.packages("tensorflow")
library(forecast)
library(ggplot2)
library(stats)
library(lmtest)
library(TSA)
library(tseries)
library(rugarch)
library(keras)
#library(tensorflow)
#install_keras()

df <- read.csv("final_data.csv", )
str(df)
#df <- df[366:1158,]
df$date <- as.Date(df$date, "%m/%d/%Y")
#options( digits = 10)
#df$demand <- as.numeric(df$demand )
str(df)
df[148,2]
summary(df)

which(is.na(df$date))

plot(df , type = "l" , main = "Time series plot")
plot(df[1:400,] , type = "l" , main = "Time series plot")
points(df[1:400,],pch=15, cex=0.5 )

plot(df ,type="l",xlab='date',ylab='Average of daily energy demand' , main = "7 days moving average")
lines(df[7:1523,1],rollmean(df[,2],7),col="red")
points(df[7:1523,1],rollmean(df[,2],7),col="red",pch=15, cex=0.5)
legend("topright",legend = c("Actual","Fits"), pch=c(NA,15),lwd=c(.7,.7),cex=.5,col=c("black","red"))
       

#_____________________________holt winter models_____________________
df_train <- df[731:1401, ]
df_test <- df[1402:1523, ]
ts_hw <- ts(df_train$demand , start = c(2019, 03), frequency = 7)
hw <- HoltWinters(ts_hw , seasonal = "additive")

par(mfrow=c(1,1),oma=c(0,0,0,0))
plot(df_train , type = "p" ,pch = 16, cex = 0.5 , main = "actual vs HW fitted values" )
lines(df_train$date[8:671] , hw$fitted[,1] , col = "red")
#points(df_train$date[8:946] , hw$fitted[,1] , pch = 15 , cex = 0.5, col = "red")
legend("bottomright",legend = c("Actual","Fits"), pch=c(16,NA),lwd=c(NA,.5),cex=.5,col=c("black","red"))

par(mfrow=c(2,2),oma=c(0,0,0,0))
hw_res <- df_train$demand[8:671] - hw$fitted[,1]
qqnorm(hw_res,datax=TRUE,pch=16,xlab='Residual',main='')
qqline(hw_res,datax=TRUE)
plot(hw$fitted[,1],hw_res,pch=16, xlab='Fitted Value',
     ylab='Residual')
abline(h=0)
hist(hw_res,col="gray",xlab='Residual',main='')
plot(hw_res,type="l",xlab='Observation Order',
     ylab='Residual')
points(hw_res,pch=16,cex=.5)
abline(h=0)

par(mfrow=c(1,2),oma=c(0,0,0,0))
Acf(hw_res , lag.max = 50 , main = "ACF of HW model Residuals")
Acf(hw_res , type = "partial", lag.max = 50 , main = "PACF of HW model Residuals")

# one step ahead forecast
yhat_hw <- data.frame(predictions = rep(0,122) , Upper = rep(0,122) , lower = rep(0,122))
pred_hw <- predict(hw, n.ahead = 1, prediction.interval = T)
yhat_hw [1,1] <- pred_hw[1,1]
yhat_hw [1,2] <- pred_hw[1,2]
yhat_hw [1,3] <- pred_hw[1,3]
trr <- df_train
hw_best_alpha <- hw$alpha
hw_best_beta <- hw$beta
hw_best_gamma <- hw$gamma

for (i in 1:(nrow(df_test)-1)){
  trr[671+i,] <- df_test[i,]
  ts_hw <- ts(trr$demand, start = c(2019, 3), frequency = 7)
  hw <- HoltWinters(ts_hw, alpha = hw_best_alpha , beta = hw_best_beta,
                       gamma = hw_best_gamma , seasonal = "additive")
  pred_hw <- predict(hw, n.ahead = 1, prediction.interval = T)
  yhat_hw [i+1,1] <- pred_hw[1,1]
  yhat_hw [i+1,2] <- pred_hw[1,2]
  yhat_hw [i+1,3] <- pred_hw[1,3]
  
}

par(mfrow=c(1,1),oma=c(0,0,0,0))
plot(df[731:1523,], pch = 16, cex = 0.5  , main = "actual vs predicted values of HW")

# plotting forecasted values with prediction intervals
lines(df_test$date, yhat_hw[,1], col = "red")
lines(df_test$date, yhat_hw[,2], col = "blue")
lines(df_test$date, yhat_hw[,3], col = "blue")

RSS <- sum((df_test$demand - yhat_hw[,1])^2)
TSS <- sum((df_test$demand - mean(df_train$demand))^2)
re <- (df_test$demand - yhat_hw[,1])/df_test$demand
cat ( "R2 is:" ,1-RSS/TSS )
cat("MSE is:" , ((RSS)/nrow(df_test)))
cat("MPE is:" , (sum(re)/nrow(df_test)))
cat("MAPE is:" , (sum(abs(re))/nrow(df_test)))

# two step ahead forecast
ts_hw <- ts(df_train$demand , start = c(2019, 03), frequency = 7)
hw <- HoltWinters(ts_hw , seasonal = "additive")
yhat_hw <- data.frame(predictions = rep(0,122) , Upper = rep(0,122) , lower = rep(0,122))
pred_hw <- predict(hw, n.ahead = 2, prediction.interval = T)
yhat_hw [1,1] <- pred_hw[1,1]
yhat_hw [1,2] <- pred_hw[1,2]
yhat_hw [1,3] <- pred_hw[1,3]
yhat_hw [2,1] <- pred_hw[2,1]
yhat_hw [2,2] <- pred_hw[2,2]
yhat_hw [2,3] <- pred_hw[2,3]
trr <- df_train
for (i in 1:(nrow(df_test)-2)){
  trr[671+i,] <- df_test[i,]
  ts_hw <- ts(trr$demand, start = c(2019, 3), frequency = 7)
  hw <- HoltWinters(ts_hw, alpha = hw_best_alpha , beta = hw_best_beta,
                    gamma = hw_best_gamma , seasonal = "additive")
  pred_hw <- predict(hw, n.ahead = 2, prediction.interval = T)
  yhat_hw [i+2,1] <- pred_hw[2,1]
  yhat_hw [i+2,2] <- pred_hw[2,2]
  yhat_hw [i+2,3] <- pred_hw[2,3]
  
}

par(mfrow=c(1,1),oma=c(0,0,0,0))
plot(df[731:1523,], pch = 16, cex = 0.5  , main = "actual vs predicted values of HW")

# plotting forecasted values with prediction intervals
lines(df_test$date, yhat_hw[,1], col = "red")
lines(df_test$date, yhat_hw[,2], col = "blue")
lines(df_test$date, yhat_hw[,3], col = "blue")

RSS <- sum((df_test$demand - yhat_hw[,1])^2)
TSS <- sum((df_test$demand - mean(df_train$demand))^2)
re <- (df_test$demand - yhat_hw[,1])/df_test$demand
cat ( "R2 is:" ,1-RSS/TSS )
cat("MSE is:" , ((RSS)/nrow(df_test)))
cat("MPE is:" , (sum(re)/nrow(df_test)))
cat("MAPE is:" , (sum(abs(re))/nrow(df_test)))

#three spet ahead forecast
ts_hw <- ts(df_train$demand , start = c(2019, 03), frequency = 7)
hw <- HoltWinters(ts_hw , seasonal = "additive")
yhat_hw <- data.frame(predictions = rep(0,122) , Upper = rep(0,122) , lower = rep(0,122))
pred_hw <- predict(hw, n.ahead = 3, prediction.interval = T)
yhat_hw [1,1] <- pred_hw[1,1]
yhat_hw [1,2] <- pred_hw[1,2]
yhat_hw [1,3] <- pred_hw[1,3]
yhat_hw [2,1] <- pred_hw[2,1]
yhat_hw [2,2] <- pred_hw[2,2]
yhat_hw [2,3] <- pred_hw[2,3]
yhat_hw [3,1] <- pred_hw[3,1]
yhat_hw [3,2] <- pred_hw[3,3]
yhat_hw [3,3] <- pred_hw[3,3]
trr <- df_train
for (i in 1:(nrow(df_test)-3)){
  trr[671+i,] <- df_test[i,]
  ts_hw <- ts(trr$demand, start = c(2019, 3), frequency = 7)
  hw <- HoltWinters(ts_hw, alpha = hw_best_alpha , beta = hw_best_beta,
                    gamma = hw_best_gamma , seasonal = "additive")
  pred_hw <- predict(hw, n.ahead = 3, prediction.interval = T)
  yhat_hw [i+3,1] <- pred_hw[3,1]
  yhat_hw [i+3,2] <- pred_hw[3,2]
  yhat_hw [i+3,3] <- pred_hw[3,3]
  
}

par(mfrow=c(1,1),oma=c(0,0,0,0))
plot(df[731:1523,], pch = 16, cex = 0.5  , main = "actual vs predicted values of HW")

# plotting forecasted values with prediction intervals
lines(df_test$date, yhat_hw[,1], col = "red")
lines(df_test$date, yhat_hw[,2], col = "blue")
lines(df_test$date, yhat_hw[,3], col = "blue")

RSS <- sum((df_test$demand - yhat_hw[,1])^2)
TSS <- sum((df_test$demand - mean(df_train$demand))^2)
re <- (df_test$demand - yhat_hw[,1])/df_test$demand
cat ( "R2 is:" ,1-RSS/TSS )
cat("MSE is:" , ((RSS)/nrow(df_test)))
cat("MPE is:" , (sum(re)/nrow(df_test)))
cat("MAPE is:" , (sum(abs(re))/nrow(df_test)))
#____________________ARIMA models________________________

df_decompose <- decompose(ts_hw)
plot(df_decompose)
df_decompose$seasonal

par(mfrow=c(1,2),oma=c(0,0,0,0))
Acf(df$demand , lag.max = 100, main = "ACF of yt")
Acf(df$demand , type = "partial", lag.max = 100, main = "PACF of yt")

par(mfrow=c(1,2),oma=c(0,0,0,0))
Acf(diff(df$demand) , lag.max = 100, main = "ACF of (1-B)yt")
Acf(diff(df$demand) , type = "partial", lag.max = 100, main = "PACF of (1-B)yt")

par(mfrow=c(1,2),oma=c(0,0,0,0))
Acf(diff(diff(df$demand),lag = 7) , lag.max = 100, main = "ACF of (1-B)(1-B^7)yt")
Acf(diff(diff(df$demand),lag = 7) , type = "partial", lag.max = 100, main = "PACF of (1-B)(1-B^7)yt")
adf.test(x =(diff(diff((df$demand)),lag = 7)) , k = 100)

eacf(diff(diff(df_train$demand),lag = 7) ,ar.max = 20  , ma.max = 20)
#it suggests (2,2) , (1,3) , (0,3) , (4,2)

#kpss.test(x = diff(diff(df$demand),lag = 7) )
arima_ts <- ts(df_train$demand , start = c(2019, 3), frequency = 7)

aicvector <- data.frame( p0 = rep(0,0,6),p1 = rep(0,0,6),p2 = rep(0,0,6),p3 = rep(0,0,6)
                         ,p4 = rep(0,0,6) ,p5 = rep(0,0,6))
bicvector <- data.frame( p0 = rep(0,0,6),p1 = rep(0,0,6),p2 = rep(0,0,6),p3 = rep(0,0,6)
                         ,p4 = rep(0,0,6) ,p5 = rep(0,0,6))
R2vector <- data.frame( p0 = rep(0,0,6),p1 = rep(0,0,6),p2 = rep(0,0,6),p3 = rep(0,0,6)
                         ,p4 = rep(0,0,6) ,p5 = rep(0,0,6))
R2adjvector <- data.frame( p0 = rep(0,0,6),p1 = rep(0,0,6),p2 = rep(0,0,6),p3 = rep(0,0,6)
                         ,p4 = rep(0,0,6) ,p5 = rep(0,0,6))
for (p in 0:5){
  for (q in 0:5){
    arima_mod2 <- Arima(df_train[,2], order = c(p,1,q) ,seasonal = list(order = c(0,1,1), period = 7) , lambda = 0)
    aicvector [q+1,p+1] <- arima_mod2$aicc
    bicvector [q+1,p+1] <- arima_mod2$bic
    R2vector [q+1,p+1] <- 1 - ((sum((df_train[,2] - arima_mod2$fitted)^2))/(sum((df_train[,2]-mean(df_train[,2]))^2)))
    R2adjvector [q+1,p+1] <- 1 - ((sum((df_train[,2] - arima_mod2$fitted)^2))*(nrow(df_train)-1)/((sum((df_train[,2]-mean(df_train[,2]))^2))*(nrow(df_train)-p-q-1)))
  }
}

#auto arima suggestion
arima_mod <- auto.arima(arima_ts, trace = T , lambda = 0)
arima_mod


#model selection based on AICC


arima_mod2 <- Arima(df_train[,2], order = c(2,1,3) ,seasonal = list(order = c(0,1,1), period = 7) , lambda = 0)
arima_mod2
checkresiduals(arima_mod2, lag = 24)
autoplot(arima_mod2)
coeftest(arima_mod2)
McLeod.Li.test(arima_mod2)$p.value

# model checking
arima_mod <- arima_mod2
arima_mod
par(mfrow=c(1,1),oma=c(0,0,0,0))
plot(df_train , type = "p" ,pch = 16, cex = 0.5 , main = "actual vs ARIMA fitted values" )
lines(df_train$date , arima_mod$fitted , col = "red")
#points(df_train$date[8:946] , hw$fitted[,1] , pch = 15 , cex = 0.5, col = "red")
legend("bottomright",legend = c("Actual","Fits"), pch=c(16,NA),lwd=c(NA,.5),cex=.5,col=c("black","red"))

par(mfrow=c(2,2),oma=c(0,0,0,0))
qqnorm(arima_mod$residuals,datax=TRUE,pch=16,xlab='Residual',main='')
qqline(arima_mod$residuals,datax=TRUE)
plot(arima_mod$fitted,arima_mod$residuals,pch=16, xlab='Fitted Value',
     ylab='Residual')
abline(h=0)
hist(arima_mod$residuals, col="gray",xlab='Residual',main='')
plot(arima_mod$residuals,type="l",xlab='Observation Order',
     ylab='Residual')
points(arima_mod$residuals,pch=16,cex=.5)
abline(h=0)

par(mfrow=c(1,2),oma=c(0,0,0,0))
Acf(arima_mod$residuals , lag.max = 100 , main = "ACF of ARIMA model Residuals")
Acf(arima_mod$residuals , type = "partial", lag.max = 100 , main = "PACF of ARIMA model Residuals")

#one step ahead forecast
yhat_arima <- data.frame(predictions = rep(0,122) , Upper = rep(0,122) , lower = rep(0,122))
pred_arima <- forecast(object = df_train$demand , h = 1 ,level = 95 ,model = arima_mod)
yhat_arima [1,1] <- pred_arima$mean[1]
yhat_arima [1,2] <- pred_arima$lower[1]
yhat_arima [1,3] <- pred_arima$upper[1]
trr <- df_train

for (i in 1:(nrow(df_test)-1)){
  trr[671+i,] <- df_test[i,]
  pred_arima <- forecast(object = trr$demand , h = 1 ,level = 95 ,model = arima_mod)
  yhat_arima [i+1,1] <- pred_arima$mean[1]
  yhat_arima [i+1,2] <- pred_arima$lower[1]
  yhat_arima [i+1,3] <- pred_arima$upper[1]
  
}

par(mfrow=c(1,1),oma=c(0,0,0,0))
plot(df[731:1523,], pch = 16, cex = 0.5 , main = "Actual vc ARIMA predicted values")

# plotting forecasted values with prediction intervals
lines(df_test$date, yhat_arima[,1], col = "red")
lines(df_test$date, yhat_arima[,2], col = "blue")
lines(df_test$date, yhat_arima[,3], col = "blue")

RSS <- sum((df_test$demand - yhat_arima[,1])^2)
TSS <- sum((df_test$demand - mean(df_train$demand))^2)
re <- (df_test$demand - yhat_arima[,1])/df_test$demand
cat ( "R2 is:" ,1-RSS/TSS )
cat("MSE is:" , ((RSS)/nrow(df_test)))
cat("MPE is:" , (sum(re)/nrow(df_test)))
cat("MAPE is:" , (sum(abs(re))/nrow(df_test)))

#two step ahead forecast
yhat_arima <- data.frame(predictions = rep(0,122) , Upper = rep(0,122) , lower = rep(0,122))
pred_arima1 <- forecast(object = df_train$demand , h = 1 ,level = 95 ,model = arima_mod)
pred_arima <- forecast(object = df_train$demand , h = 2 ,level = 95 ,model = arima_mod)
yhat_arima [1,1] <- pred_arima1$mean[1]
yhat_arima [1,2] <- pred_arima1$lower[1]
yhat_arima [1,3] <- pred_arima1$upper[1]
yhat_arima [2,1] <- pred_arima$mean[1]
yhat_arima [2,2] <- pred_arima$lower[1]
yhat_arima [2,3] <- pred_arima$upper[1]
trr <- df_train

for (i in 1:(nrow(df_test)-2)){
  trr[671+i,] <- df_test[i,]
  pred_arima <- forecast(object = trr$demand , h = 2 ,level = 95 ,model = arima_mod)
  yhat_arima [i+2,1] <- pred_arima$mean[1]
  yhat_arima [i+2,2] <- pred_arima$lower[1]
  yhat_arima [i+2,3] <- pred_arima$upper[1]
  
}

par(mfrow=c(1,1),oma=c(0,0,0,0))
plot(df[731:1523,], pch = 16, cex = 0.5 , main = "Actual vc ARIMA predicted values")

# plotting forecasted values with prediction intervals
lines(df_test$date, yhat_arima[,1], col = "red")
lines(df_test$date, yhat_arima[,2], col = "blue")
lines(df_test$date, yhat_arima[,3], col = "blue")

RSS <- sum((df_test$demand - yhat_arima[,1])^2)
TSS <- sum((df_test$demand - mean(df_train$demand))^2)
re <- (df_test$demand - yhat_arima[,1])/df_test$demand
cat ( "R2 is:" ,1-RSS/TSS )
cat("MSE is:" , ((RSS)/nrow(df_test)))
cat("MPE is:" , (sum(re)/nrow(df_test)))
cat("MAPE is:" , (sum(abs(re))/nrow(df_test)))

#three step ahead forecast
yhat_arima <- data.frame(predictions = rep(0,122) , Upper = rep(0,122) , lower = rep(0,122))
pred_arima1 <- forecast(object = df_train$demand , h = 1 ,level = 95 ,model = arima_mod)
pred_arima2 <- forecast(object = df_train$demand , h = 2 ,level = 95 ,model = arima_mod)
pred_arima <- forecast(object = df_train$demand , h = 3 ,level = 95 ,model = arima_mod)
yhat_arima [1,1] <- pred_arima1$mean[1]
yhat_arima [1,2] <- pred_arima1$lower[1]
yhat_arima [1,3] <- pred_arima1$upper[1]
yhat_arima [2,1] <- pred_arima2$mean[1]
yhat_arima [2,2] <- pred_arima2$lower[1]
yhat_arima [2,3] <- pred_arima2$upper[1]
yhat_arima [3,1] <- pred_arima$mean[1]
yhat_arima [3,2] <- pred_arima$lower[1]
yhat_arima [3,3] <- pred_arima$upper[1]
trr <- df_train

for (i in 1:(nrow(df_test)-3)){
  trr[671+i,] <- df_test[i,]
  pred_arima <- forecast(object = trr$demand , h = 3 ,level = 95 ,model = arima_mod)
  yhat_arima [i+3,1] <- pred_arima$mean[1]
  yhat_arima [i+3,2] <- pred_arima$lower[1]
  yhat_arima [i+3,3] <- pred_arima$upper[1]
  
}

par(mfrow=c(1,1),oma=c(0,0,0,0))
plot(df[731:1523,], pch = 16, cex = 0.5 , main = "Actual vc ARIMA predicted values")
ggplot(data =df[731:1523,], aes(x = date , y =demand ) )+ geom_point()
# plotting forecasted values with prediction intervals
lines(df_test$date, yhat_arima[,1], col = "red")
lines(df_test$date, yhat_arima[,2], col = "blue")
lines(df_test$date, yhat_arima[,3], col = "blue")

RSS <- sum((df_test$demand - yhat_arima[,1])^2)
TSS <- sum((df_test$demand - mean(df_train$demand))^2)
re <- (df_test$demand - yhat_arima[,1])/df_test$demand
cat ( "R2 is:" ,1-RSS/TSS )
cat("MSE is:" , ((RSS)/nrow(df_test)))
cat("MPE is:" , (sum(re)/nrow(df_test)))
cat("MAPE is:" , (sum(abs(re))/nrow(df_test)))
#_______________GARCH___________
shapiro.test(arima_mod2$residuals)

garch_spec <- ugarchspec(variance.model = list(model = "sGARCH" ,garchOrder = c(1, 1)), 
                         mean.model = list(armaOrder = c(0, 0)), 
                         distribution.model = "sstd")

garch_mod <- ugarchfit(spec=garch_spec ,data=arima_mod2$residuals, out.sample =100)
garch_mod
plot(fitted(garch_mod))
plot(sigma(garch_mod))
plot(garch_mod , which = "all")
plot(garch_mod )

#___________________LSTM_________________________
# preprocessing data

lag_convert <- function(df , t){
  if(is.null(ncol(df)) & length(df) <= t){
    print("ERROR: df dimension and lag is not compatible.")
  }
  else if(!is.null(ncol(df))){
    print("ERROR: the input datafram should have only one column.")
  }
  else {
    lagged_df <- data.frame(matrix(NA, nrow = (length(df)-t) ,ncol = t+1))
    for (i in 0:t){
      colnames(lagged_df)[i+1] <- paste0('y-', t-i)
      lagged_df [,i+1] <- df[(i+1):(length(df)-t+i)]
    }
    colnames(lagged_df)[i+1] <- 'y'
    return(lagged_df)
  }
}

standardize <- function(df_train , df_test ){
  
  min_train <- min(df_train)
  max_train <- max(df_train)
  std_train <- (df_train - min_train)/(max_train - min_train)
  std_test <- (df_test - min_train)/(max_train - min_train)
  return(list(std_train =std_train , std_test = std_test ,
              scale= c(min_train =min_train, max_train = max_train)))
}

lagnum <- 1

lagged <- lag_convert(df$demand ,lagnum)
lagged

split_ratio <- 0.9
lagged_train <- lagged [1:round(split_ratio*nrow(lagged)),]
lagged_test <- lagged [(1+round(split_ratio*nrow(lagged))):nrow(lagged),]
std_obj <- standardize(lagged_train ,lagged_test )
std_lagged_train <- std_obj[["std_train"]]
std_lagged_test <- std_obj[["std_test"]]

X_train <- as.matrix(std_lagged_train [,1:lagnum])
Y_train <- as.matrix (std_lagged_train [,1+lagnum])


X_test <-  as.matrix(std_lagged_test [,1:lagnum])
Y_test <- as.matrix ( std_lagged_test [,1+lagnum])

dim(X_train) <- c(nrow(X_train), lagnum, 1)
dim(X_train)

dim(Y_train) <- c(nrow(Y_train) ,1)
dim(Y_train)

dim(X_test) <- c(nrow(X_test), lagnum, 1)
dim(X_test)

dim(Y_test) <- c(nrow(Y_test), 1)
dim(Y_test)

plot(std_lagged_train[,1])

batch_size <- 1

lstm_mod <- keras_model_sequential() 
lstm_mod %>%
  layer_lstm( units = 50,
              #input_shape = c(2,1) ,
              batch_input_shape = c(batch_size, lagnum, 1),
              return_sequences = T,
              stateful= TRUE)%>%
  layer_dense(units = 1  )

summary(lstm_mod)

lstm_mod %>% compile(
  loss = 'mse',
  optimizer = "adam", 
  metrics = c("accuracy" )
)

lstm_mod %>% fit(X_train,
                 Y_train, 
                 epochs=10, 
                 batch_size=batch_size, 
                 verbose=1, 
                 validation_split = 0.1,
                 shuffle=FALSE)


yhat_std_lstm = lstm_mod %>% predict(X_test , batch_size=batch_size)
yhat_lstm <- yhat_std_lstm*(std_obj[["scale"]][["max_train"]]+std_obj[["scale"]][["min_train"]])+(std_obj[["scale"]][["min_train"]])

par(mfrow=c(1,1),oma=c(0,0,0,0))
plot(df, pch = 16, cex = 0.5)
lines(df$date[1372:1523] , yhat_lstm , col = "red")
length(yhat_lstm)
