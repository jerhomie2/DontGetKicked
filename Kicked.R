library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

train <- vroom('./training.csv')
test <- vroom('./test.csv')
train$IsBadBuy <- as.factor(train$IsBadBuy)

head(train)
colSums(is.na(train))
nrow(train)


kick_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_impute_mean(Transmission, Trim, MMRAcquisitionAuctionAveragePrice,
                   MMRAcquisitionRetailAveragePrice,
                   MMRAcquisitionAuctionCleanPrice,
                   MMRAcquisitonRetailCleanPrice)
  # Use this (or knn) later. It just takes a long time to run
  # step_impute_bag(Transmission, 
  #                 impute_with = imp_vars(-Trim,
  #                                        -MMRAcquisitionAuctionAveragePrice,
  #                                        -MMRAcquisitionRetailAveragePrice,
  #                                        -MMRAcquisitionAuctionCleanPrice,
  #                                        -MMRAcquisitonRetailCleanPrice)) %>%
  # step_impute_bag(MMRAcquisitionAuctionAveragePrice,
  #                 MMRAcquisitionRetailAveragePrice,
  #                 MMRAcquisitionAuctionCleanPrice,
  #                 MMRAcquisitonRetailCleanPrice,
  #                 impute_with = imp_vars(-Trim)) %>%
  # step_impute_bag(Trim)
prep <- prep(kick_recipe)
bake <- bake(prep, new_data = train)
  
