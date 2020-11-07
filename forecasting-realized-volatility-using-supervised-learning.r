# <--- PLEASE NOTE ---->
# This code reproduces the validation set results for the final models per machine learning method considered: (1) weekday effect model, (2) gradient boosted trees, (3) support vector machine with linear kernel, (4) k nearest neighbor
# All logical steps that resulted in these final models (including the tuning of parameter values, if applicable) are FULLY documented and all code involved is FULLY REPRODUCIBLE in the *.Rmd file
# ALL CODE is designed to run on R version 3.6.X and Microsoft Windows
# Code runs well on personal laptop (Intel Core i7-7500U, 16GB RAM, >30GB available HDD space. Windows 10)

# increase memory limit on Windows machine
invisible(memory.limit(size = 35000))

# load/install required libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(fastDummies)) install.packages("fastDummies", repos = "http://cran.us.r-project.org")

# create function to calculate RMSE
get_rmse <- function(y_hat, y){
  round(sqrt(mean((y-y_hat)**2)),7)
}

# download dataset from GitHub, load data and delete file from hard disk drive (all relative paths)
url <- "https://raw.githubusercontent.com/csatzky/forecasting-realized-volatility-using-supervised-learning/main/data/EURUSD_realized_volatility.RData"
temp_name <- tempfile()
download.file(url, temp_name)
load(temp_name)
file.remove(temp_name) # remove temporary file

# create 'weekday' variable for 'training' and 'validation' sets (for details, see *.Rmd or *.PDF report)
training[, weekday := weekdays(date)]
validation[, weekday := weekdays(date)]

# create dummy variables for 'weekday' variable (for details, see *.Rmd or *.PDF report)
training <- dummy_cols(.data = training, select_columns = "weekday")
validation <- dummy_cols(.data = validation, select_columns = "weekday")

# select relevant model inputs
training_data <- training[,.(rv_t1, rv_d, rv_w, rv_m,
                             weekday_Monday,
                             weekday_Tuesday,
                             weekday_Wednesday,
                             weekday_Thursday)]


# <--- 1: Fit & predict linear model ('weekday effect') ---->

# fit model on `training` set
fit_lm <- lm(rv_t1 ~., data=training_data)

# predict RV_{t+1} on `validation` set
y_hat_lm <- predict(fit_lm, newdata = validation)

# obtain rmse
rmse_lm <- get_rmse(y_hat = y_hat_lm, y = validation$rv_t1)


# <--- 2: Fit & predict tree ensembles ('gradient boosted trees') ---->

# for reproducibility (R 3.6.x)
set.seed(1, sample.kind="Rounding")

# tune alpha and lambda parameters on `training` set (for details, see *.Rmd or *.PDF report)
fit_te <- train(rv_t1 ~ .,
                data=training_data, 
                method="xgbLinear", 
                metric="RMSE",
                preProc=c("center", "scale"),
                objective="reg:squarederror",
                trControl=trainControl(search = "grid"),
                tuneGrid = data.frame(lambda=seq(0.0097325, 0.0291975, length.out=15),
                                      alpha=seq(0.0097325, 0.0291975, length.out=15),
                                      nrounds=seq(150, 150, length.out=15),
                                      eta=seq(0.3, 0.3, length.out=15)))

# predict RV_{t+1} on `validation` set
y_hat_te <- predict(fit_te, newdata = validation)

# obtain rmse
rmse_te <- get_rmse(y_hat = y_hat_te, y = validation$rv_t1)


# <--- 3: Fit & predict support vector machine ('SVM with linear kernel') ---->

# for reproducibility (R 3.6.x)
set.seed(1, sample.kind="Rounding")

# tune C (cost) parameter on `training` set (for details, see *.Rmd or *.PDF report)
fit_svm <- train(rv_t1 ~ .,
                 data=training_data, 
                 method="svmLinear", 
                 metric="RMSE",
                 preProc=c("center", "scale"), 
                 trControl=trainControl(search = "grid"),
                 tuneGrid = data.frame(C = seq(0.342105, 1.026315,length.out=15)))

# predict RV_{t+1} on `validation` set
y_hat_svm <- predict(fit_svm, newdata = validation)

# obtain rmse
rmse_svm <- get_rmse(y_hat = y_hat_svm, y = validation$rv_t1)


# <--- 4: Fit & k nearest neighbor ('KNN') ---->

# for reproducibility (R 3.6.x)
set.seed(1, sample.kind="Rounding")

# tune k parameter on `training` set (for details, see *.Rmd or *.PDF report)
fit_knn <- train(rv_t1 ~ .,
                 data=training_data, 
                 method="knn", 
                 metric="RMSE",
                 preProc=c("center", "scale"), 
                 trControl=trainControl(search = "grid"),
                 tuneGrid = data.frame(k=seq(15,44,2)))

# predict RV_{t+1} on `validation` set
y_hat_knn <- predict(fit_knn, newdata = validation)

# obtain rmse
rmse_knn <- get_rmse(y_hat = y_hat_knn, y = validation$rv_t1)


# <--- summarize validation set RMSE results ---->

# list models considered
models <- c("Weekday Effect", "Gradient Boosted Trees", "SVM with Linear Kernel", "KNN")

# gather RMSEs
rmse <- c(rmse_lm,rmse_te, rmse_svm, rmse_knn)

# print results
data.frame(Model=models, 'RMSE Validation Set'=rmse)


# <--- plot actuals vs. predicted values ---->

# gather actual/predicted values
tbl <- data.table(date=validation$date, 
                  actual=validation$rv_t1, 
                  linear=y_hat_lm,
                  GBT=y_hat_te,
                  SVM=y_hat_svm,
                  KNN=y_hat_knn)

# rename variables
names(tbl)[2:6] <- c("Actual", "Linear (Weekday Effect)", "GBT", "SVM (Linear Kernel)", "KNN")

# transform 'tbl' to 'tidy' format
tbl <- tbl %>%
  pivot_longer(!date, names_to="model", values_to="prediction") %>%
  mutate(model = factor(model, levels = c("Actual", "Linear (Weekday Effect)", "GBT", "SVM (Linear Kernel)", "KNN")))

# plot actual vs. predicted results
tbl %>%
  ggplot(aes(x=date, y=prediction, color=model)) +
  geom_line() +
  scale_colour_wsj("colors6", "") + theme_wsj(color = "white") +
  theme(axis.title=element_text(size=12, family="sans"),
        axis.text=element_text(size=10, family="sans"),
        plot.title=element_text(size=12, family="sans"), 
        plot.subtitle = element_text(size=10, family="sans")) +
  ylab(bquote(RV[t+1])) +
  xlab("") +
  ggtitle("Actual vs. Predicted Volatility", 
          subtitle="January to July 2020")