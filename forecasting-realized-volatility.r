# <--- PLEASE NOTE ---->
# This code reproduces the validation set results for the final models per machine learning method considered: (1) weekday effect model, (2) gradient boosted trees, (3) support vector machine with linear kernel, (4) k nearest neighbor. For the complete code of all interm results please look into the *.Rmd file
# All logical steps that resulted in these final models (including the tuning of parameter values, if applicable) are FULLY documented and all code involved is FULLY REPRODUCIBLE in the *.Rmd file
# ALL CODE is designed to run on R version 3.6.X and Microsoft Windows
# Code runs well on personal laptop (Intel Core i7-7500U, 16GB RAM, >30GB available HDD space. Windows 10)

# increase memory limit on Windows machine
invisible(memory.limit(size = 35000))

# load/install required libraries
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(fastDummies)) install.packages("fastDummies", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
library(gridExtra)
library(knitr)
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(ggthemes)
library(fastDummies)
library(randomForest)
library(kernlab)
library(xgboost)


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

# tune alpha and lambda parameters on `training` set [TAKES ~ 3 MINUTES] (for details, see *.Rmd or *.PDF report)
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

# tune C (cost) parameter on `training` set [TAKES ~ 3 MINUTES] (for details, see *.Rmd or *.PDF report)
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

# tune k parameter on `training` set [TAKES ~ 2 MINUTES] (for details, see *.Rmd or *.PDF report)
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

# summarize validation set results
results <- data.frame(Model=models, 'RMSE_validation'=rmse)

# print latex-style table
kable(results, escape = FALSE, booktabs = TRUE)


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

# transform 'tbl' to 'tidy' format and attach verbose model names
tbl <- tbl %>%
  pivot_longer(!date, names_to="model", values_to="prediction") %>%
  mutate(model = factor(model, levels = c("Actual", "Linear (Weekday Effect)", "GBT", "SVM (Linear Kernel)", "KNN")))

# plot actual vs. predicted results
tbl %>%
  ggplot(aes(x=date, y=prediction, color=model)) +
  geom_line() +
  scale_colour_wsj("colors6", "") + theme_wsj(color = "white") + # use WSJ themed line colors
  theme(axis.title=element_text(size=12, family="sans"), # re-size text elements
        axis.text=element_text(size=10, family="sans"),
        plot.title=element_text(size=12, family="sans"), 
        plot.subtitle = element_text(size=10, family="sans")) +
  ylab(bquote(RV[t+1])) + # rename y axis
  xlab("") +
  ggtitle("Actual vs. Predicted Volatility", 
          subtitle="January to July 2020")


# <--- plot actuals vs. predicted values during PRE-crisis period ---->

# filter for "january"
tbl_jan <- tbl %>%
  filter(month(date)==1)

# 1. actual vs. linear model
p1 <- tbl_jan %>%
  filter(model %in% c("Actual", "Linear (Weekday Effect)")) %>% # verbose model names
  ggplot(aes(x=date, y=prediction, color=model)) +
  geom_line() +
  scale_colour_wsj("colors6", "") + # change line colors to WSJ theme
  ylab("") +
  xlab("") +
  theme(legend.position="top") # put legend on top

# 2. actual vs. svm model
p2 <- tbl_jan %>%
  filter(model %in% c("Actual", "SVM (Linear Kernel)")) %>%
  ggplot(aes(x=date, y=prediction, color=model)) +
  geom_line() +
  scale_colour_wsj("colors6", "") +
  ylab("") +
  xlab("") +
  theme(legend.position="top")

# 3. actual vs. gradien boosted trees
p3 <- tbl_jan %>%
  filter(model %in% c("Actual", "GBT")) %>%
  ggplot(aes(x=date, y=prediction, color=model)) +
  geom_line() +
  scale_colour_wsj("colors6", "") +
  ylab("") +
  xlab("") +
  theme(legend.position="top")

# 4. actual vs. knn
p4 <- tbl_jan %>%
  filter(model %in% c("Actual", "KNN")) %>%
  ggplot(aes(x=date, y=prediction, color=model)) +
  geom_line() +
  scale_colour_wsj("colors6", "") +
  ylab("") +
  xlab("") +
  theme(legend.position="top")

# arrange 2 by 2 plot
grid.arrange(p1, p2, p3, p4, ncol=2)


# <--- summarize PRE-crisis performance in table ---->

# gather validation set's january 2020 actuals and predictions
jan_pred <- data.table(date=validation$date, y=validation$rv_t1, 
                       y_hat_lm, y_hat_te, y_hat_svm, y_hat_knn)

# filter for january 2020
jan_pred <- jan_pred[month(date)==1,]

# summarize validation set RMSE and over/underestimation measures
jan_info <- rbind(
  RMSE_validation = sapply(jan_pred[,3:6], function(y_hat){
    round(get_rmse(y_hat = y_hat, y = jan_pred$y),7)   # compute RMSE during pre-crisis period
  }),
  p_overestimate = sapply(jan_pred[,3:6], function(y_hat){
    round(mean(y_hat > jan_pred$y),7)   # compute proportion with overestimated volatility
  }),
  p_underestimate = sapply(jan_pred[,3:6], function(y_hat){
    round(mean(y_hat < jan_pred$y),7) # compute proportion with underestimated volatility
  }))

# transpose rows/columns
jan_info <- t(jan_info)

# transform figures into percentage format
col2 <- paste0(round(jan_info[,2],3)*100,"\\%")
col3 <- paste0(round(jan_info[,3],3)*100,"\\%")

# find best performing model and highlight it
jan_info[which(jan_info[,1]==min(jan_info[,1])),1] <- paste0("\\textbf{", jan_info[which(jan_info[,1]==min(jan_info[,1])),1], "}") # highlight lowest RMSE
jan_info[,2] <- col2
jan_info[,3] <- col3

# rename models more verbosely
row.names(jan_info) <- c("Linear Model (Weekend Effect)", "Gradient Boosted Tree", "SVM (Linear Kernel)", "KNN")

# display latex-style table
kable(jan_info, escape = FALSE, booktabs = TRUE)


# <--- plot actuals vs. predicted values during covid-19 crisis period ---->

# filter for march 2020 (i.e. crisis period)
tbl_march <- tbl %>%
  filter(month(date)==3)

# 1. actual vs. linear model
p1 <- tbl_march %>%
  filter(model %in% c("Actual", "Linear (Weekday Effect)")) %>% # rename model names more verbosely
  ggplot(aes(x=date, y=prediction, color=model)) +
  geom_line() +
  scale_colour_wsj("colors6", "") + # set WSJ themed line colors
  ylab("") +
  xlab("") +
  theme(legend.position="top") # put legend on top

# 2. actual vs. svm model
p2 <- tbl_march %>%
  filter(model %in% c("Actual", "SVM (Linear Kernel)")) %>%
  ggplot(aes(x=date, y=prediction, color=model)) +
  geom_line() +
  scale_colour_wsj("colors6", "") +
  ylab("") +
  xlab("") +
  theme(legend.position="top")

# 3. actual vs. gradien boosted trees
p3 <- tbl_march %>%
  filter(model %in% c("Actual", "GBT")) %>%
  ggplot(aes(x=date, y=prediction, color=model)) +
  geom_line() +
  scale_colour_wsj("colors6", "") +
  ylab("") +
  xlab("") +
  theme(legend.position="top")

# 4. actual vs. knn
p4 <- tbl_march %>%
  filter(model %in% c("Actual", "KNN")) %>%
  ggplot(aes(x=date, y=prediction, color=model)) +
  geom_line() +
  scale_colour_wsj("colors6", "") +
  ylab("") +
  xlab("") +
  theme(legend.position="top")

# arrange 2 by 2 plot
grid.arrange(p1, p2, p3, p4, ncol=2)


# <--- summarize covid-19 crisis performance in table ---->

# gather validation set's march 2020 actuals and predictions
march_pred <- data.table(date=validation$date, y=validation$rv_t1, 
                         y_hat_lm, y_hat_te, y_hat_svm, y_hat_knn)

# filter for march 2020 (i.e. crisis period)
march_pred <- march_pred[month(date)==3,]

# summarize validation set RMSE and over/underestimation measures
march_info <- rbind(
  RMSE_validation = sapply(march_pred[,3:6], function(y_hat){
    round(get_rmse(y_hat = y_hat, y = march_pred$y),7)   # compute RMSE during crisis period
  }),
  p_overestimate = sapply(march_pred[,3:6], function(y_hat){
    round(mean(y_hat > march_pred$y),7)   # compute proportion with overestimated volatility
  }),
  p_underestimate = sapply(march_pred[,3:6], function(y_hat){
    round(mean(y_hat < march_pred$y),7) # compute proportion with underestimated volatility
  }))

# transpose rows/columns
march_info <- t(march_info)

# transform figures into percentage format
col2 <- paste0(round(march_info[,2],3)*100,"\\%")
col3 <- paste0(round(march_info[,3],3)*100,"\\%")

# find best performing model and highlight it
march_info[which(march_info[,1]==min(march_info[,1])),1] <- paste0("\\textbf{", march_info[which(march_info[,1]==min(march_info[,1])),1], "}") # highlight lowest RMSE
march_info[,2] <- col2
march_info[,3] <- col3

# rename models more verbosely
row.names(march_info) <- c("Linear Model (Weekend Effect)", "Gradient Boosted Tree", "SVM (Linear Kernel)", "KNN")

# display latex-style table
kable(march_info, escape = FALSE, booktabs = TRUE)


# <--- APPENDIX 2: GARCH(1,1) performance ---->

# load libraries
if(!require(fGarch)) install.packages("fGarch", repos = "http://cran.us.r-project.org")
library(fGarch)

# download dataset from GitHub, load data and delete file from hard disk drive
url <- "https://raw.githubusercontent.com/csatzky/forecasting-realized-volatility-using-supervised-learning/main/data/EURUSD_quotes.csv"
temp_name <- tempfile()
download.file(url, temp_name)

# read external data, daily EUR/USD close
dt <- fread(temp_name, sep=",", header=TRUE, stringsAsFactors=FALSE, integer64="numeric", na.strings="")

# remove temporary file
file.remove(temp_name)

# format 'date'
dt[, date := as.Date(date)]

# compute continuously compounded returns
dt[,close_PRIOR := shift(close,1L)] # add past trading day's EUR/USD quote
dt[,daily_ln_return := log(close/close_PRIOR)] # compute daily returns

# filter relevant data
dt <- dt[, .(date, close, daily_ln_return)]

# merge data to `training` and `validation` datasets
training <- merge(training, dt, by="date", all.x=TRUE)
validation <- merge(validation, dt, by="date", all.x=TRUE)

# fit GARCH(1,1) on `training` set
fit_garch <- garchFit(formula = ~garch(1,1), 
                      data = training$daily_ln_return,
                      trace = FALSE, include.mean=FALSE)

# print parameter estimates
tbl <- as.data.frame(t(coef(fit_garch))) # extract estimates
names(tbl) <- c("$\\hat\\omega$", "$\\hat\\alpha$", "$\\hat\\beta$") # not applicable: format for latex rendering

# display latex-style table
kable(tbl, escape = FALSE, booktabs = TRUE)


# <--- APPENDIX 2: Manually predict GARCH(1,1) model on `validation` set ---->

# extract estimates
omega <- fit_garch@fit$matcoef[1,1]
alpha <- fit_garch@fit$matcoef[2,1]
beta <- fit_garch@fit$matcoef[3,1]

# obtain most recent `training`-set's values sigma_t and xt for initial prediction
sigma_t <- tail(fit_garch@sigma.t, 1)
xt <- tail(training$daily_ln_return, 1)

# set up 'garch_pred' table with x_t and to be predicted sigma_{t+1} as model inputs
garch_pred <- data.table(date=validation$date, x_t=validation$daily_ln_return, sigma_t1 = NA)

# first prediction is based on last `training` set's observations 
garch_pred$sigma_t1[1] <- sqrt(omega + alpha*xt^2 + beta*sigma_t^2)

# loop remaining predictions. For details of this formula, please look into the *.Rmd or *.PDF file
for (i in 2:nrow(garch_pred)){
  garch_pred$sigma_t1[i] <- sqrt(omega + alpha*garch_pred$x_t[i]^2 + beta*garch_pred$sigma_t1[i-1]^2)
}

# compute `validation` RMSE
rmse_garch <- get_rmse(y_hat = garch_pred$sigma_t1, y = validation$rv_t1)

# summarize performance of models
new_model <- data.frame(Model= "GARCH(1,1)",
                        RMSE_validation = rmse_garch)

# rbind other models' results with GARCH model's result
results <- rbind(results, new_model)

# print latex-style table with all validaiton set results
kable(results, escape = FALSE, booktabs=TRUE)


# <--- APPENDIX 2: GARCH(1,1) predictions vs. actuals plot ---->

# gather actual/predicted values
tbl <- data.table(date=validation$date, 
                  actual=validation$rv_t1, 
                  garch=garch_pred$sigma_t1)

# rename variables
names(tbl)[2:3] <- c("Actual", "GARCH(1,1)")

# transform 'tbl' to 'tidy' format
tbl_all <- tbl %>%
  pivot_longer(!date, names_to="model", values_to="prediction")

# plot actual vs. predicted
p1 <-tbl_all %>%
  ggplot(aes(x=date, y=prediction, color=model)) +
  geom_line() +
  scale_colour_wsj("colors6", "") + theme_wsj(color = "white") +
  theme(axis.title=element_text(size=12, family="sans"),
        axis.text=element_text(size=10, family="sans"),
        plot.title= NULL, 
        plot.subtitle = NULL) +
  ylab(bquote(RV[t+1])) +
  xlab("")

# filter for pre-crisis period
tbl_jan <- tbl[month(date)==1,] # january 2020

# transform 'tbl_jan' to 'tidy' format
tbl_jan <- tbl_jan %>%
  pivot_longer(!date, names_to="model", values_to="prediction")

p2 <- tbl_jan %>%
  ggplot(aes(x=date, y=prediction, color=model)) +
  geom_line() +
  scale_colour_wsj("colors6", "") +
  ylab("") +
  xlab("") +
  ggtitle("Pre-Crisis Period (January 2020)") +
  theme(legend.position="none",
        plot.title=element_text(size=10, family="sans"))

# filter for 'crisis' period
tbl_march <- tbl[month(date)==3,] # march 2020

# transform 'tbl_march' to 'tidy' format
tbl_march <- tbl_march %>%
  pivot_longer(!date, names_to="model", values_to="prediction")

p3 <- tbl_march %>%
  ggplot(aes(x=date, y=prediction, color=model)) +
  geom_line() +
  scale_colour_wsj("colors6", "") +
  ylab("") +
  xlab("") +
  ggtitle("Covid-19 Crisis (March 2020)") +
  theme(legend.position="none",
        plot.title=element_text(size=10, family="sans"))

# plot 1 by 2 images
layout <- rbind(c(1,1),
                c(2,3))
grid.arrange(p1,p2,p3, layout_matrix = layout)