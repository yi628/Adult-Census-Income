############################################################################
# Data Preparation
############################################################################

if(!require(tidyverse))  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret))      install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(devtools))   install.packages("devtools", repos = "http://cran.us.r-project.org")
if(!require(catboost))   devtools::install_github("catboost/catboost", subdir = "catboost/R-package")

library(tidyverse)
library(caret)
library(data.table)
library(catboost)

# Omit slight warnings and avoid scientific notation
options(dplyr.summarise.inform = FALSE)
options(scipen = 999)

# UCI Adult Census Income dataset:
# https://www.kaggle.com/uciml/adult-census-income/download
# https://github.com/yi628/Adult-Census-Income/raw/main/adult.csv

# Load the file
dl <- tempfile()
download.file("https://github.com/yi628/Adult-Census-Income/raw/main/adult.csv", dl)

adult <- read.csv(dl, stringsAsFactors = TRUE)
rm(dl)

############################################################################
# Data Cleaning
############################################################################

# Review the dataset structure
str(adult)

# Make sure there is not any null values
sum(colSums(is.na(adult)))

# How many "?" values are there in the adult set
length(which(adult == "?"))

# Handle unknown work class, change "?" to "Unknown"
level_list <- levels(adult$workclass)
level_list[1] <- "Unknown"
levels(adult$workclass) <- level_list
table(adult$workclass)

# Handle unknown occupation, change "?" to "Unknown"
level_list <- levels(adult$occupation)
level_list[1] <- "Unknown"
levels(adult$occupation) <- level_list
table(adult$occupation)

# Handle unknown native country, change "?" to "Unknown"
level_list <- levels(adult$native.country)
level_list[1] <- "Unknown"
levels(adult$native.country) <- level_list
table(adult$native.country)[1:28]

# Make sure there is not any unknown values
length(which(adult == "?"))

# Simplify class label, change ">50K" to "Yes" and "<=50K" to "No"
levels(adult$income) <- list("Yes" = ">50K", "No" = "<=50K")
table(adult$income)

rm(level_list)

############################################################################
# Data Exploration and Visualization
############################################################################

# Age vs Income
adult %>%
  ggplot(aes(x = age, fill = income)) +
  geom_density(alpha = 0.8) +
  scale_fill_manual(values = c("#999999", "#E69F00")) +
  ggtitle("Age vs Income")

# Work Class vs Income
adult %>%
  group_by(workclass, income) %>%
  summarize(count = n()) %>%
  ggplot(aes(y = workclass, x = count, fill = income)) +
  geom_bar( position = "stack", stat = "identity") +
  scale_fill_manual(values = c("#E69F00", "#56B4E9")) +
  ggtitle("Work Class vs Income")

# Final Weight Distribution
adult %>%
  filter(fnlwgt < 650000) %>%
  ggplot(aes(x = fnlwgt)) +
  geom_density(fill = "#69B3A2", color = "#E9ECEF", alpha = 0.8) +
  ggtitle("Final Weight Distribution")

# Education Years vs Income
adult %>%
  group_by(education.num, income) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = education.num, y = count, fill = income)) +
  geom_bar(position = "fill", stat = "identity") +
  scale_fill_manual(values = c("#56B4E9", "#009E73")) +
  ggtitle("Education Years vs Income")

# Marital Status vs Income
adult %>%
  group_by(marital.status, income) %>%
  summarize(count = n()) %>%
  ggplot(aes(y = marital.status, x = count, fill = income)) +
  geom_bar( position = "stack", stat = "identity") +
  scale_fill_manual(values = c("#0072B2", "#D55E00")) +
  ggtitle("Marital Status vs Income")

# Weekly Working Hours Comparison
adult %>%
  ggplot(aes(x = hours.per.week)) +
  geom_histogram(fill = "#CC79A7", binwidth = 10) +
  facet_wrap(~income) +
  xlab("Working Hours") +
  ggtitle("Weekly Working Hours Comparison")

############################################################################
# Modeling Approach
############################################################################

# Split the data into training and test datsets
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(adult$income,
                                  p = 0.1,
                                  list = FALSE)
train_set  <- adult[-test_index,]
test_set   <- adult[test_index,]

rm(test_index)

# Prepare the training scheme
#  Set the resampling method: cv such that Cross-Validation is used
#  Keep default number of k-fold: 10 such that 10-fold Cross-Validation
#  Keep default percentage of training set: 0.75
control <- trainControl(method = "cv")

############################################################################
# Random Forest
############################################################################

# Prepare the tune grid
#grid <- expand.grid(mtry = c(1:14))
grid <- expand.grid(mtry = 14)

# Train the model
#  Set Random Forest as the train method
#  Set ntree to 20 brances grow after split
#  Set accuracy as the metric of optimal model
model_rf <- train(income~.,
                  data = train_set,
                  method = "rf",
                  ntree = 20,
                  metric = "Accuracy",
                  trControl = control,
                  tuneGrid = grid)

# Review the tuning result
print(model_rf)

############################################################################
# CatBoost
############################################################################

# Prepare the tune grid
#grid <- expand.grid(iterations    = 100,
#                    learning_rate = seq(from = 0.03, to = 0.99, by = 0.03),
#                    depth         = seq(from = 6,    to = 10,   by = 1   ),
#                    l2_leaf_reg   = seq(from = 0.04, to = 1.00, by = 0.04),
#                    rsm           = seq(from = 0.05, to = 1.00, by = 0.05),
#                    border_count  = seq(from = 10,   to = 100,  by = 10  ))
grid <- expand.grid(iterations = 100,
                    learning_rate = 0.75,
                    depth = 9,
                    l2_leaf_reg = 0.88,
                    rsm = 0.95,
                    border_count = 40)

# Train the model
#  Set CatBoost as the train method
#  Set accuracy as the metric of optimal model
#  Set verbose to 0 (no logging)
model_ctb <- train(x = train_set[, -ncol(train_set)],
                   y = train_set$income,
                   method = catboost.caret,
                   eval_metric = "Accuracy",
                   tuneGrid = grid,
                   verbose = 0)

# Review the tuning result
print(model_ctb)

############################################################################
# XGBoost
############################################################################

# Prepare the tune grid
#grid <- expand.grid(nrounds          = 100,
#                    max_depth        = seq(from = 3,    to = 10,  by = 1  ),
#                    eta              = seq(from = 0.1,  to = 0.3, by = 0.1),
#                    gamma            = seq(from = 0,    to = 1,   by = 1  ),
#                    colsample_bytree = seq(from = 0.5,  to = 1.0, by = 0.1),
#                    min_child_weight = seq(from = 1,    to = 2,   by = 1  ),
#                    subsample        = seq(from = 0.5,  to = 1.0, by = 0.1))
grid <- expand.grid(nrounds = 100,
                    max_depth = 6,
                    eta = 0.1,
                    gamma = 0,
                    colsample_bytree = 0.5,
                    min_child_weight = 1,
                    subsample = 1)

# Train the model
#  Set XGBoost as the train method
#  Set accuracy as the metric of optimal model
model_xgb <- train(income~.,
                   data = train_set,
                   method = "xgbTree",
                   metric = "Accuracy",
                   trControl = control,
                   tuneGrid = grid)

# Review the tuning result
print(model_xgb)

############################################################################
# Result
############################################################################

# Predict using Random Forest model and accuracy rate
predicted <- predict(model_rf, test_set[, -ncol(test_set)], type = "raw")
accuracy_rf <- confusionMatrix(predicted, test_set$income)$overall["Accuracy"]
names(accuracy_rf) <- "Random Forest Accuracy"

# Predict using CatBoost model and accuracy rate
predicted <- predict(model_ctb, test_set[, -ncol(test_set)], type = "raw")
accuracy_ctb <- confusionMatrix(predicted, test_set$income)$overall["Accuracy"]
names(accuracy_ctb) <- "CatBoost Accuracy"

# Predict using XGBoost model and accuracy rate
predicted <- predict(model_xgb, test_set[, -ncol(test_set)], type = "raw")
accuracy_xgb <- confusionMatrix(predicted, test_set$income)$overall["Accuracy"]
names(accuracy_xgb) <- "XGBoost Accuracy"

# Display the accuracy rates of models
c(accuracy_rf, accuracy_ctb, accuracy_xgb)


