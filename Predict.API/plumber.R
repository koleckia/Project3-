#
# This is a Plumber API. You can run the API by clicking
# the 'Run API' button above.
#
# Find out more about building APIs with Plumber here:
#
#    https://www.rplumber.io/
#

library(tidyverse)
library(tidymodels)
library(yardstick)
library(ggplot2)
library(glmnet)
library(tree)
library(baguette)
library(parsnip)
library(ranger)

#* @apiTitle Project 3 API
#* @apiDescription Creating API using the best fitting model

#Read in data from EDA page
diabetes_data <- read.csv("diabetes_binary_health_indicators_BRFSS2015.csv")
diabetes_data <- as_tibble(diabetes_data)

Diabetes_labels <- c("No Diabetes", "Diabetes")
HighBP_labels <- c("Not High BP", "High BP")
HighChol_labels <- c("Not High Chol", "Chol")
PhysActivity_labels <- c("No","Yes")
Sex_labels <- c("Female","Male")
Age_labels <- c("18-24","25-29","30-34","35-59",
                "40-44","45-49","50-54","55-59",
                "60-64", "65-69", "70-74", "75-79","80 or older")
Income_levels <-c("Less than $10,000",">15000",">20000",">25000",">35000",">50000",
                  ">75000","75000+")


diabetes_data <- diabetes_data |>
  drop_na() |>
  mutate(
    Diabetes_binary = factor(Diabetes_binary, levels = c(0,1), labels = Diabetes_labels),
    HighBP = factor(HighBP, levels = c(0,1), labels = HighBP_labels),
    HighChol = factor(HighChol, levels = c(0,1), labels = HighChol_labels),
    PhysActivity = factor(PhysActivity, levels = c(0,1), labels = PhysActivity_labels),
    Sex = factor(Sex, levels = c(0,1), labels = Sex_labels),
    Age = factor(Age, levels = 1:13, labels = Age_labels),
    Income = factor(Income, levels = 1:8, labels = Income_levels)
  ) |>
  select(Diabetes_binary,HighBP,HighChol,PhysActivity,Sex,Age,Income,BMI)

#Create split data 
set.seed(123)

model_split <- initial_split(diabetes_data,prop=.7)
test <- testing(model_split)
train <- training(model_split)
diabetes_CV_folds <- vfold_cv(train, 5)


#Fit Random Forest Model 
ran_forest_rec <- recipe(Diabetes_binary ~ BMI + HighBP + PhysActivity, data=train) |>
  step_normalize(BMI)|>
  step_dummy(all_nominal_predictors())

rf_spec <- rand_forest(mtry = tune()) |>
  set_engine("ranger") |>
  set_mode("classification")

rf_wkf <- workflow() |>
  add_recipe(ran_forest_rec) |>
  add_model(rf_spec)

rf_fit <- rf_wkf |>
  tune_grid(resamples = diabetes_CV_folds,
            grid = 7,
            metrics = metric_set(accuracy, mn_log_loss))

rf_fit |>
  collect_metrics() |>
  filter(.metric == "mn_log_loss") |>
  arrange(mean)

rf_best_params <- select_best(rf_fit, metric = "mn_log_loss")
rf_best_params

rf_final_wkf <- rf_wkf |>
  finalize_workflow(rf_best_params)

final_model <- rf_final_fit$.workflow[[1]]

#Create API 



#* Return the prediction of best model
#* @param BMI BMI
#* @param HighBP High BP (1 for Yes, 0 for No)
#* @param PhysActivity Physical Activity (1 for Yes, 0 for No)
#* @get /prediction
function(BMI = 28, HighBP = 0, PhysActivity = 1) {
  pred_data <- data.frame(BMI = as.numeric(BMI), 
                          HighBP= factor(as.numeric(HighBP), levels = c(0,1), labels = c("Not High BP", "High BP")), 
                          PhysActivity = factor(as.numeric(PhysActivity), levels = c(0,1), labels = c("No", "Yes")))

  prediction <- predict(final_model , pred_data, type = "prob")
  return(prediction)
}
#http://localhost:PORT/prediction?BMI=28HighBP=0&PhysActivity=1

#Create info end point 
#* @get /info
function(){
  list(name= "Alexis Kolecki",
  url="Insert URL here")
}

#http://localhost:PORT/info


