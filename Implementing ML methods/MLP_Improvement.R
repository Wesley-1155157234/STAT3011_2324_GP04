setwd("~/Desktop")
data = read.csv("siddata.csv")
start_date <- as.Date("2007-01-01")
end_date <- as.Date("2016-12-31")
dates <- seq(from=start_date, to=end_date, by="day")
data$date = dates

install.packages("devtools")
devtools::install_github("rstudio/keras")
install.packages("dplyr")
install.packages("lubridate")
install.packages("TTR")
install.packages("reticulate")

library(reticulate)
library(keras3)
install_keras()
library(tensorflow)
install_tensorflow()
library(dplyr)
library(lubridate)
library(TTR)

py_install("tensorflow")
py_install("keras") 

data <- data %>% filter(year(date) == 2016)
train_data <- data %>% filter(month(date) < 11)
test_data <- data %>% filter(month(date) >= 11)

X_train <- as.matrix(train_data[, c('compound', 'neg', 'neu', 'pos')])
Y_train <- as.matrix(train_data$prices)

X_test <- as.matrix(test_data[, c('compound', 'neg', 'neu', 'pos')])
Y_test <- as.matrix(test_data$prices)

rm(model)
model <- keras_model_sequential() %>%
  layer_dense(units = 100, activation = 'relu', input_shape = 4)%>%
  layer_dense(units = 200, activation = 'relu') %>%
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dense(units = 1)

optimizer <- tf$keras$optimizers$Adam(learning_rate = 0.003)
r_squared <- function(y_true, y_pred) {
  y_true <- tf$cast(y_true, dtype = tf$int32)
  y_pred <- tf$cast(y_pred, dtype = tf$int32)
  
  y_pred = y_pred + mean(y_true - y_pred)

  SS_res <- tf$reduce_sum(tf$square(y_true - y_pred)) 
  SS_tot <- tf$reduce_sum(tf$square(y_true - tf$reduce_mean(y_true))) 
  return(1 - SS_res / (SS_tot + tf$keras$backend$epsilon()))
}


model %>% compile(
  optimizer = optimizer,  
  loss = 'mean_squared_error',  
  metrics = list(r_squared)  
)

history <- model %>% fit(
  X_train, 
  Y_train,
  epochs = 100,
  batch_size = 64,
  validation_split = 0.2,
  shuffle = FALSE
)
