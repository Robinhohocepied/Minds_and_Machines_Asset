# -------------------------------------------------------------------------------------------------------------------- #
#                                                                                                                      #
#                                                   Deloitte Belgium                                                   #
#                                   Gateway building Luchthaven Nationaal 1J 1930 Zaventem                              #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
#
# Author list (Alphabetical Order):
#    Name                                       Username                                     Email
#    Ugo Leonard                                uleonard                                     uleonard@deloitte.com
# -------------------------------------------------------------------------------------------------------------------- #
# ###                                               Program Description                                            ### #
# -------------------------------------------------------------------------------------------------------------------- #
# To be written
# -------------------------------------------------------------------------------------------------------------------- #
#                                             Parameters & Libraries                                                   #
# -------------------------------------------------------------------------------------------------------------------- #
library(zoo)
library(xts)
library(dplyr)
library(tibble)
library(ggplot2)
library(CADFtest)
library(forecast)
library(seasonal)
library(forecastHybrid)
library(gridExtra)
library(prophet)

# If want to run for specific year:
# - forecast_len <- 12
# - change results_dir
# For SKU level:
# - prep_internal: series_name <- substring(series_name, 2)

rm(list=ls())
forecast_len <- 12
training_proportion <- 0.8
data_dir <- "./Data/"
results_dir <- "./Results/SKU/MARKET/2018/"
  
input_file <- paste0(results_dir, 'volumes_prepped.csv')
external_file <- paste0(results_dir, 'weather.csv')
internal_file <- paste0(results_dir, 'internal_factors.csv')
budget_file <- paste0(results_dir, 'budget_crosscheck.csv')
# -------------------------------------------------------------------------------------------------------------------- #
#                                                Notes                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
# Possible improvements: change the structure of code such that there are only two modes:
# * Univariate
# * Multivariate
# where the selection of the multivariate variables is done before running the various forecasting algorigthms.
# This way, we won't have to run univariate models for all multivariate models, which takes quite some time.

# Theil's U: dividing the RMSE of the proposed forecasting method by the RMSE of a no-change (naïve, U=1) model.
# * If U is equal to 1, it means that the proposed model is as good as the naïve model.
# * If U is greater than 1, there is no point in using the proposed forecasting model since a naïve method would
# produce better results.
# * It is worthwhile to consider using the proposed model only when U is smaller than 1 (the smaller the better),
# indicating that more accurate forecasts than a no-change model can be obtained.

# MAPE: Mean Absolute Percentage Error
# < 10: Highly accurate forecasting
# 10-20: Good Forecasting
# 20-50: Reasonable Forecasting
# > 50: Inaccurate Forecasting

# Additive or multiplicative seasonal effect?
# Additive: preferred when the seasonal variations are roughly constant through the series. The amplitude of the
# seasonal effect is roughly the same each year;
# Multiplicative: preferred when the seasonal variations are changing proportional to the level of the series. The
# seasonal and other effects act proportionally on the series.
# -------------------------------------------------------------------------------------------------------------------- #
#                                             Functions                                                                #
# -------------------------------------------------------------------------------------------------------------------- #
prep_data <- function(series, one_year=TRUE){
  series.name <- names(series)
  splitted <- strsplit(series.name, '...', TRUE)
  series.name <- unlist(splitted)[1]
  market_name <- unlist(splitted)[2]

  # Remove first and last rows that are NA (if applicable)  
  series_trimmed <- na.trim(series)

  # Check if there are missing months
  if (length(seq(start(series_trimmed),end(series_trimmed), by="month")) != length(series_trimmed)){
    subset <- merge(series_trimmed, zoo(, seq(start(series_trimmed),end(series_trimmed), by="month")), all=TRUE)
  }else{
    subset <- series_trimmed
  }
  
  # Identify outliers as those observations that are > 1.5* IQR or < -1.5 * IQR based on boxplot
  #outlying.values <- boxplot(as.matrix(subset))$out
  
  
  #Fill in missings with spline approximation
  subset <- na.approx(subset)

  # Split train/test set and make it a ts object
  if (one_year==TRUE){
    nb_obs <- length(subset) - forecast_len
  }else{
    nb_obs <- floor(training_proportion * length(subset))
  }
  
  training <- subset[1:nb_obs,]
  start_date_train <- c(as.numeric(format(index(training)[1], '%Y')),
                        as.numeric(format(index(training)[1], '%m')),
                        as.numeric(format(index(training)[1], '%d')))
  max.lag=round(sqrt(length(training)))
  
  testing <- subset[(nb_obs+1):(length(subset)),]
  start_date_test <- c(as.numeric(format(index(testing)[1], '%Y')),
                       as.numeric(format(index(testing)[1], '%m')),
                       as.numeric(format(index(testing)[1], '%d')))
  
  ts.training <- ts(training, start=start_date_train, frequency=12)
  ts.testing <- ts(testing, start=start_date_test, frequency=12)
  
  return_list <- list("series.name"= series.name, 'market'=market_name, "training"=ts.training, "testing"=ts.testing,
                      "max.lag"=max.lag)
  
  return(return_list)
}


prep_external <- function(df, training, testing){
  start_date_ext <- c(as.numeric(format(index(df)[1], '%Y')),
                        as.numeric(format(index(df)[1], '%m')),
                        as.numeric(format(index(df)[1], '%d')))
  ts.external <- ts(df, start=start_date_ext, frequency = 12)
  
  start_date <- c(start(training)[1], start(training)[2], 1)
  ts.ext.train <- window(ts.external, start=start(training), end=end(training))
  ts.ext.test <- window(ts.external, start=start(testing), end=end(testing))
  
  if (length(training) > length(ts.ext.train[,1])){
    training <- window(training, start=start(ts.ext.train), end=end(ts.ext.train))
  }
  if (length(testing) > length(ts.ext.test[,1])){
    testing <- window(testing, start=start(ts.ext.test), end=end(ts.ext.test))
  }
  return(list('train_ext'= ts.ext.train, 'test_ext'= ts.ext.test, 'train'=training, 'test'=testing))
}


prep_internal <- function(df, series_name, ts.training, ts.testing, vars){
  series_name <- substring(series_name, 2)
  df.promo <- df %>% filter(SKU_and_Label == series_name) %>% select(c(Date, vars))
  
  if(nrow(df.promo)>0){
    df.promo <- df.promo[order(df.promo$Date),]
    start_date <- c(as.numeric(format(df.promo$Date[1], '%Y')),
                    as.numeric(format(df.promo$Date[1], '%m')),
                    as.numeric(format(df.promo$Date[1], '%d')))
    
    ts.internal <- ts(df.promo[,-1], start=start_date, frequency = 12)
    #For the moment can't use the row below since there is only one variable
    #colnames(ts.internal) <- vars
    ts.int.train <- window(ts.internal, start=start(ts.training), end=end(ts.training), extend=TRUE)
    ts.int.train <- na.fill(ts.int.train, 0)
    ts.int.test <- window(ts.internal, start=start(ts.testing), end=end(ts.testing), extend=TRUE)
    ts.int.test <- na.fill(ts.int.test, 0)

    # Also temp as we have a TS and not MTS object
    if (length(unique(ts.int.train))==1){
      ts.internal <- NULL
      ts.int.train <- NULL
      ts.int.test <- NULL
    }
  }else{
    ts.internal <- NULL
    ts.int.train <- NULL
    ts.int.test <- NULL
  }
  
  return(list('train_int'= ts.int.train, 'test_int'= ts.int.test))
}


predict_autoarima <- function(training, testing, series_name){
  nb_obs <- length(testing)
  
  forecast_model <- auto.arima(training, ic='aicc', stationary='False', seasonal='True')
  df.forecast <- forecast(forecast_model, nb_obs)
  df.accuracy <- accuracy(df.forecast, testing)[2,]
  fig_title <-  paste(series_name, df.forecast$method, sep=' - ')
  ts.errors <- testing - df.forecast$mean
  return_list <- list("model"=forecast_model, "forecast"=df.forecast, "fig.title"=fig_title,
                      "accuracy.measures"=df.accuracy, "errors"=ts.errors)
  return(return_list)
}


predict_stldeco <- function(training, testing, series_name){
  # Could also try mstl() which takes multiple seasonalities into account
  nb_obs <- length(testing)
  
  forecast_model <- stl(training[,1], robust=TRUE, s.window='periodic')
  # Either ETS or ARIMA, to be investigated which brings best results
  df.forecast <- forecast(forecast_model, method=c('ets'), h=nb_obs)
  df.accuracy <- accuracy(df.forecast, testing)[2,]
  fig_title <-  paste(series_name, df.forecast$method, sep=' - ')
  ts.errors <- testing - df.forecast$mean
  return_list <- list("model"=forecast_model, "forecast"=df.forecast, "fig.title"=fig_title,
                      "accuracy.measures"=df.accuracy, "errors"=ts.errors)
  return(return_list)
}


predict_ets <- function(training, testing, series_name){
  # https://otexts.com/fpp2/ets.html
  # ETS(*, *, *) for Error, Trend, Seasonal
  # Error = {A, M}       
  # Trend = {N, A, A_d}  
  # Seasonal = {N, A, M} 
  # Where:
  # N = None
  # A = Additive
  # A_d = Additive damped
  # M = Multiplicative
  
  # Possibly three smoothing equations:
  # - Level (l) (parameter: Alpha)
  # - Trend (b) (parameter: Beta)
  # - Seasonality (s) (parameter: Gamma)
  # Small values of Alpha, Beta and/or Gamma indicate that th changes are very small over time for the components.
  # Looking at the plot (using autoplot(forecast_model)), if the prediction intervals are narrow, the component is relatively
  # easy to predict because it is "strong".
  nb_obs <- length(testing)
  
  forecast_model <- ets(training[,1])
  df.forecast <- forecast(forecast_model, h=nb_obs)
  df.accuracy <- accuracy(df.forecast, testing)[2,]
  fig_title <-  paste(series_name, df.forecast$method, sep=' - ')
  ts.errors <- testing - df.forecast$mean
  return_list <- list("model"=forecast_model, "forecast"=df.forecast, "fig.title"=fig_title,
                      "accuracy.measures"=df.accuracy, "errors"=ts.errors)
  return(return_list)
}


predict_tbats <- function(training, testing, series_name){
  nb_obs <- length(testing)
  
  forecast_model <- tbats(training[,1])
  df.forecast <- forecast(forecast_model, h=nb_obs)
  df.accuracy <- accuracy(df.forecast, testing)[2,]
  fig_title <-  paste(series_name, df.forecast$method, sep=' - ')
  ts.errors <- testing - df.forecast$mean
  return_list <- list("model"=forecast_model, "forecast"=df.forecast, "fig.title"=fig_title,
                      "accuracy.measures"=df.accuracy, "errors"=ts.errors)
  return(return_list)
}


predict_hybrid <- function(training, testing, series_name){
  nb_obs <- length(testing)
  
  forecast_model <- hybridModel(training[,1],
                                models="aes",
                                a.args=list(ic='aicc', stationary='False', seasonal='True'),
                                e.args=NULL,
                                s.args=list(robust=TRUE, s.window='periodic', method='arima'),
                                #t.args=NULL,
                                #weights='cv.errors',
                                #errorMethod='MASE',
                                parallel=TRUE, num.cores=4)
  df.forecast <- forecast(forecast_model, h=nb_obs)
  df.accuracy <- accuracy(df.forecast, testing)[2,]
  fig_title <-  paste(series_name, df.forecast$method, sep=' - ')
  ts.errors <- testing - df.forecast$mean
  return_list <- list("model"=forecast_model, "forecast"=df.forecast, "fig.title"=fig_title,
                      "accuracy.measures"=df.accuracy, "errors"=ts.errors)
  return(return_list)
}


predict_prophet <- function(training, testing, series_name){
  nb_obs <- length(testing)
  start_date <- as.Date(time(testing))[1]
  
  df.prophet <- data.frame(y=as.matrix(training), date=as.Date(time(training)))
  colnames(df.prophet) <- c('y', 'ds')
  m <- prophet(df.prophet)
  future <- make_future_dataframe(m, periods = nb_obs, freq = 'month')
  df.forecast <- tail(predict(m, future)[c('ds', 'yhat')], nb_obs)
  df.forecast <- ts(df.forecast$yhat, start_date, frequency=12)
  
  df.accuracy <- accuracy(df.forecast, testing)
  fig_title <- paste(series_name, 'Prophet', sep=' - ')
  ts.errors <- testing - df.forecast
  
  return_list <- list("model"='Prophet', "forecast"=df.forecast, "fig.title"= fig_title,
                      "accuracy.measures"=df.accuracy, "errors"=ts.errors)
  return(return_list)
}


predict_multilm <- function(training, testing, ts.ext.train, ts.ext.test , series_name){
  nb_obs <- length(testing)
  
  if("mts" %in% class(ts.ext.train)){
    forecast_model <- tslm(training ~ trend + season + ts.ext.train)
    df.forecast <- forecast(forecast_model, h=nb_obs, newdata=data.frame(ts.ext.test))
  }else{
    # If there is only a single extra variable, we have a ts object.
    # At the moment this happens if we only use internal variables --> ext = promos
    # TS objects (as opposed to MTS) have only one column and do not have variable names
    # Therefore, the functions take the object name as reference --> train != test and crashes
    # To correct for that, create ext object. This should be a temp solution
    df <- ts(c(ts.ext.train, ts.ext.test), start= time(ts.ext.train)[1], frequency=12)
    ext <- head(df, length(training))
    forecast_model <- tslm(training ~ trend + season + ext)
    ext <- tail(df, nb_obs)
    df.forecast <- forecast(forecast_model, h=nb_obs, newdata=data.frame(ext))
  }
  
  df.accuracy <- accuracy(df.forecast, testing)[2,]
  fig_title <-  paste(series_name, df.forecast$method, sep=' - ')
  ts.errors <- testing - df.forecast$mean
  return_list <- list("model"=forecast_model, "forecast"=df.forecast, "fig.title"=fig_title,
                      "accuracy.measures"=df.accuracy, "errors"=ts.errors)
  return(return_list)
}


predict_multiautoarima <- function(training, testing, ts.ext.train, ts.ext.test, series_name){
  nb_obs <- length(testing)

  forecast_model <- auto.arima(training, ic='aicc', stationary='False', seasonal='True', xreg=ts.ext.train)
  df.forecast <- forecast(forecast_model, nb_obs, xreg=ts.ext.test)
  df.accuracy <- accuracy(df.forecast, testing)[2,]
  fig_title <-  paste(series_name, df.forecast$method, sep=' - ')
  ts.errors <- testing - df.forecast$mean
  return_list <- list("model"=forecast_model, "forecast"=df.forecast, "fig.title"=fig_title,
                      "accuracy.measures"=df.accuracy, "errors"=ts.errors)
  return(return_list)
}


predict_models <- function(series, df.external=NULL, df.internal=NULL, df.budget=ts.budget){
  # Prepare sales data
  prepped_data = prep_data(series)
  series_name <- prepped_data$series.name
  market <- prepped_data$market
  ts.training <- prepped_data$training
  ts.testing <- prepped_data$testing
  max.lag <- prepped_data$max.lag
  series <- ts(rbind(ts.training, ts.testing), start=time(ts.training)[1], frequency=12)

  # Prepare budget data to which we will compare
  start_date <- c(as.numeric(format(index(df.budget)[1], '%Y')),
                  as.numeric(format(index(df.budget)[1], '%m')),
                  as.numeric(format(index(df.budget)[1], '%d')))
  ts.budget <- ts(df.budget[ ,paste0(series_name, '...', market)], start_date, frequency = 12)
  
  df.budget.acc <- data.frame(accuracy(ts.budget, ts.testing))
  df.budget <- data.frame(ts.budget)
  
  print(paste(series_name, market))
  
  # Univariate models
  prophet <- predict_prophet(ts.training, ts.testing, series_name)
  autoarima <- predict_autoarima(ts.training, ts.testing, series_name)
  stldeco <- predict_stldeco(ts.training, ts.testing, series_name)
  etsmod <- predict_ets(ts.training, ts.testing, series_name)
  tbatsmod <- predict_tbats(ts.training, ts.testing, series_name)
  hybrid <- predict_hybrid(ts.training, ts.testing, series_name)
  
  names <- c(autoarima$forecast$method, stldeco$forecast$method, etsmod$forecast$method, tbatsmod$forecast$method,
             "Hybrid", 'Prophet', 'Budgeted')
  ## Models
  foo <- list(autoarima, stldeco, etsmod, tbatsmod, hybrid, prophet, NULL)
  model_list <- setNames(foo, names)
  
  ## Create accuracy matrix
  df.accuracy <- data.frame(rbind(autoarima$accuracy.measures, stldeco$accuracy.measures, etsmod$accuracy.measures,
                                  tbatsmod$accuracy.measures, hybrid$accuracy.measures, prophet$accuracy.measures))
  df.accuracy$AIC <- c(autoarima$model$aic, stldeco$forecast$model$aic, etsmod$model$aic, tbatsmod$model$AIC, 0, 0)
  
  ## Add budget metrics
  df.accuracy <- bind_rows(df.accuracy, df.budget.acc)
  
  ## Names
  rownames(df.accuracy) <- names
  
  ## Create forecast matrix
  df.forecasts <- data.frame(cbind(autoarima$forecast$mean, stldeco$forecast$mean, etsmod$forecast$mean,
                                   tbatsmod$forecast$mean, hybrid$forecast$mean, prophet$forecast))
  df.forecasts <- bind_cols(data.frame(ts.testing), df.forecasts, df.budget)
  colnames(df.forecasts) <- c(c("Actuals"), names)
  
  model_vars <- "univariate"
  
  if(!is.null(df.internal) || !is.null(df.external)){
    if(!is.null(df.external)){
      ext <- prep_external(df.external, ts.training, ts.testing)
      ts.ext.train <- ext$train_ext
      ts.ext.test <- ext$test_ext
      ts.training <- ext$train
      ts.testing <- ext$test
      
      if(!is.null(ts.ext.train)){
        ## LM Regression
        multi_lm <- predict_multilm(training=ts.training,
                                    testing=ts.testing,
                                    ts.ext.train=ts.ext.train,
                                    ts.ext.test=ts.ext.test,
                                    series_name=series_name)
        multi_arima <- predict_multiautoarima(ts.training, ts.testing, ts.ext.train, ts.ext.test, series_name)
        model_vars <- "external"
        
        # Update models
        foo2 <- list(multi_lm, multi_arima)
        models2 <- setNames(foo2, c(paste(model_vars, multi_lm$forecast$method, sep='_'),
                                    paste(model_vars, multi_arima$forecast$method, sep='_')))      
        model_list <- c(model_list, models2)
        
        # Update accuracy matrix
        multi_lm$accuracy.measures['AIC'] <- 0
        multi_arima$accuracy.measures['AIC'] <- multi_arima$model$aic
        df.acc2 <- data.frame(rbind(multi_lm$accuracy.measures, multi_arima$accuracy.measures),
                              row.names = c(paste(model_vars, multi_lm$forecast$method, sep='_'),
                                            paste(model_vars, multi_arima$forecast$method, sep='_')))
        df.accuracy <- rbind(df.accuracy, df.acc2)
        
        # Update forecast matrix
        df.forecasts2 <- data.frame(cbind(multi_lm$forecast$mean, multi_arima$forecast$mean))
        colnames(df.forecasts2) <- c(paste(model_vars, multi_lm$forecast$method, sep='_'),
                                     paste(model_vars, multi_arima$forecast$method, sep='_'))
        df.forecasts <- cbind(df.forecasts, df.forecasts2)
      }
    }
    
    if(!is.null(df.internal)){
      int <- prep_internal(df.internal, series_name, ts.training, ts.testing, c('Promo'))
      ts.int.train <- int$train_int
      ts.int.test <- int$test_int
      
      if(!is.null(ts.int.train)){
        #colnames(ts.int.train) <- c('Promo')
        #colnames(ts.int.test) <- c('Promo')
        ## LM Regression
        multi_lm <- predict_multilm(training=ts.training,
                                    testing=ts.testing,
                                    ts.ext.train=ts.int.train,
                                    ts.ext.test=ts.int.test,
                                    series_name=series_name)
        multi_arima <- predict_multiautoarima(ts.training, ts.testing, ts.int.train, ts.int.test, series_name)
        model_vars <- "internal"
        
        # Update models
        foo2 <- list(multi_lm, multi_arima)
        models2 <- setNames(foo2, c(paste(model_vars, multi_lm$forecast$method, sep='_'),
                                    paste(model_vars, multi_arima$forecast$method, sep='_')))      
        model_list <- c(model_list, models2)
        
        # Update accuracy matrix
        multi_lm$accuracy.measures['AIC'] <- 0
        multi_arima$accuracy.measures['AIC'] <- multi_arima$model$aic
        df.acc2 <- data.frame(rbind(multi_lm$accuracy.measures, multi_arima$accuracy.measures),
                              row.names = c(paste(model_vars, multi_lm$forecast$method, sep='_'),
                                            paste(model_vars, multi_arima$forecast$method, sep='_')))
        df.accuracy <- rbind(df.accuracy, df.acc2)
        
        # Update forecast matrix
        df.forecasts2 <- data.frame(cbind(multi_lm$forecast$mean, multi_arima$forecast$mean))
        colnames(df.forecasts2) <- c(paste(model_vars, multi_lm$forecast$method, sep='_'),
                                     paste(model_vars, multi_arima$forecast$method, sep='_'))
        df.forecasts <- cbind(df.forecasts, df.forecasts2)
      }
    }
    
    if(!is.null(df.internal) && !is.null(df.external)){
      if(!is.null(ts.int.train) && !is.null(ts.ext.train)){
        ts.factors.train <- cbind(ts.int.train, ts.ext.train)
        ts.factors.test <- cbind(ts.int.test, ts.ext.test)
        
        colnames(ts.factors.train) <- c('Promo', colnames(ts.ext.train))
        colnames(ts.factors.test) <- c('Promo', colnames(ts.ext.test))

        ## LM Regression
        multi_lm <- predict_multilm(ts.training, ts.testing, ts.factors.train, ts.factors.test, series_name)
        multi_arima <- predict_multiautoarima(ts.training, ts.testing, ts.factors.train, ts.factors.test, series_name)
        model_vars <- "all"
        
        # Update models
        foo2 <- list(multi_lm, multi_arima)
        models2 <- setNames(foo2, c(paste(model_vars, multi_lm$forecast$method, sep='_'),
                                    paste(model_vars, multi_arima$forecast$method, sep='_')))      
        model_list <- c(model_list, models2)
        
        # Update accuracy matrix
        multi_lm$accuracy.measures['AIC'] <- 0
        multi_arima$accuracy.measures['AIC'] <- multi_arima$model$aic
        df.acc2 <- data.frame(rbind(multi_lm$accuracy.measures, multi_arima$accuracy.measures),
                              row.names = c(paste(model_vars, multi_lm$forecast$method, sep='_'),
                                            paste(model_vars, multi_arima$forecast$method, sep='_')))
        df.accuracy <- rbind(df.accuracy, df.acc2)
        
        # Update forecast matrix
        df.forecasts2 <- data.frame(cbind(multi_lm$forecast$mean, multi_arima$forecast$mean))
        colnames(df.forecasts2) <- c(paste(model_vars, multi_lm$forecast$method, sep='_'),
                                     paste(model_vars, multi_arima$forecast$method, sep='_'))
        df.forecasts <- cbind(df.forecasts, df.forecasts2)
      }
    }
  }
  # Plot best and budget
  # Look at the forecasts
  df.accuracy.ts <- df.accuracy[!row.names(df.accuracy) %in% c('Budgeted'),]
  model_mape <- rownames(df.accuracy.ts)[which(df.accuracy.ts$MAPE == min(df.accuracy.ts$MAPE))][1]
  min_mape <- min(df.accuracy.ts$MAPE)
  budget_mape <- df.accuracy[row.names(df.accuracy) %in% c('Budgeted'),]$MAPE
  toplot <- ts(df.forecasts[model_mape], start=start_date, frequency=12)
  topmodel <- model_list[model_mape]
  
  brand_plt <- autoplot(series, ylab='Volume', series='Actuals') +
    autolayer(toplot, series='AI-driven Forecast') +
    autolayer(ts.budget, series='Budgeted') +
    scale_color_manual(values = c("black", "green3", "firebrick3")) +
    labs(color='Legend')
  ggsave(filename=paste0(series_name, '_', market, '_', model_vars, '.png'), plot=brand_plt, path=results_dir, device='png',
         height=7, width=14, units='in')
  
  # Export accuracy
  out_file = paste0(results_dir, series_name, '_', market, '_', model_vars, '_accuracy.csv')
  write.table(df.accuracy, out_file, sep='|', col.names=NA, row.names=TRUE)

  # Export
  forecast_file = paste0(results_dir, series_name, '_', market, '_', model_vars, '_forecast.csv')
  write.table(df.forecasts, forecast_file, sep='|', col.names=NA, row.names=TRUE)
  
  sink(paste0(results_dir, series_name, '_', market, '_', model_vars, '_model.csv'))
  print(summary(topmodel))
  sink()
  
  #print(c(model_mape, min_mape))
  #print(c('Budget', budget_mape))
  return(NULL)
  }

# -------------------------------------------------------------------------------------------------------------------- #
#                                             Parameters & Libraries                                                   #
# -------------------------------------------------------------------------------------------------------------------- #
# Recipe inputs
df.volume <- read.csv(input_file, sep="|")
df.volume$Date <- as.Date(df.volume$Date, format="%Y-%m-%d")
ts.volume <- xts(x=df.volume[,-1], order.by=df.volume$Date)


df.external <- read.csv(external_file, sep='|')
df.external$Date <- as.Date(df.external$Date, format="%Y-%m-%d")
ts.external <- xts(x=df.external[,-1], order.by=df.external$Date)
ts.external <- ts.external[,colnames(ts.external) != 'Avg_Temp_C']

df.internal <- read.csv(internal_file, sep='|')
df.internal$Date <- as.Date(df.internal$Date, format="%Y-%m-%d")

df.budget <- read.csv(budget_file, sep='|')
df.budget$Date <- as.Date(df.budget$Date, format="%Y-%m-%d")
ts.budget <- xts(x=df.budget[1:forecast_len,-1], order.by=df.budget[1:forecast_len,]$Date)

# Run the forecast for each column
print('All Data')
out_all <- lapply(ts.volume, function(x)predict_models(x))
