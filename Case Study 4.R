#Case Study 4
#Hilary Balli, Marco Duran Perez, George Garcia

library(mlbench)
library(e1071)
library(skimr)
library(caret)
library(corrplot)
library(DMwR)
library(pamr)
library(rpart)
library(C50)
library(xgboost)
library(caretEnsemble)

data(PimaIndiansDiabetes2) # PID2 is an updated version of the data.
str(PimaIndiansDiabetes2)
summary(PimaIndiansDiabetes2)
dataPID <- PimaIndiansDiabetes2

x <- subset(dataPID, select = -diabetes)
y <- subset(dataPID, select = diabetes)


# Plot distribution of the two-class outcome variable 'Class' to check for imbalance.
barplot(table(y), 
        names.arg = c("neg", "pos"),
        col = c("blue", "green"),
        main = "Diabetes")

# 1. Since we are predicting the class outcome variable 'diabetes', classification should be used.
# The data is imbalanced (500 vs. 268) and is a 2-class situation.


# 2. Data Splitting

#set seed
seed <- 321

#set training and testing data
ptrain <- createDataPartition(y$diabetes,
                              p = .75,
                              list = FALSE)
xTest <- x[-ptrain,]
yTest <- y[-ptrain,]

xTrain <- x[ptrain,]
yTrain <- y[ptrain,]


# 3. Training data exploration

#skewness
skew <- lapply(x, skewness)
head(skew)

#skewed results
skimmed1 <- skim_to_wide(xTrain)
skimmed2 <- skim_to_wide(yTrain)
View(skimmed1)
View(skimmed2)

# Skimmed1 shows there the data is missing 277 insulin values, 169 tricep values, and more.

# Omit NAs and update data to proceed
dataPID <- na.omit(PimaIndiansDiabetes2)
x <- subset(dataPID, select = -diabetes)
y <- subset(dataPID, select = diabetes)
set.seed(seed)
ptrain <- createDataPartition(y$diabetes,
                              p = .75,
                              list = FALSE)
xTest <- x[-ptrain,]
yTest <- y[-ptrain,]
xTrain <- x[ptrain,]
yTrain <- y[ptrain,]

###Data Visualization

# Boxplots of predictors
boxplot(xTrain)

# Histograms of predictors and outcome variables
par(mfrow = c(3,3))
hist(xTrain$pregnant)
hist(xTrain$glucose)
hist(xTrain$pressure)
hist(xTrain$triceps)
hist(xTrain$insulin)
hist(xTrain$mass)
hist(xTrain$pedigree)
hist(xTrain$age)
hist(as.numeric(yTrain))


# 4. Data cleaning

# Correlations
correlations <- cor(xTrain)
par(mfrow = c(1,1))
corrplot(correlations, method = "number", order = "hclust")

# No highly correlated variables are found. Since there are skewed variables, we will
# implement 'YeoJohnson'.  Since there are a few outliers, we will implement 'spatialSign'.
# Since some values are close to zero, we will implement 'zv' and 'nzv'.


#5. Model implementation of algorithms (with standardized cleaning)

# Controlled resampling
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                     summaryFunction = twoClassSummary, sampling = "smote", 
                     classProbs = TRUE, savePredictions = "final")

#nearest shrunken centroid
gridNSC <- expand.grid(threshold = seq(0.5, 2.5, by = 0.1))
set.seed(seed)
modelNSC <- caret::train(x = xTrain, y = yTrain, 
                  method = "pam",
                  preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"),
                  tuneGrid = gridNSC,
                  trControl = ctrl)
modelNSC # ROC = 0.846, Sens = 0.864, Spec = 0.576

# support vector machines
library(kernlab)
set.seed(seed)
modelSVM <- caret::train(x = xTrain, 
                         y = yTrain,
                         method = "svmRadial",
                         preProc = c("center", "scale"),
                         fit = FALSE,
                         trControl = ctrl)
modelSVM # ROC = .84, Sens =.80 , Sped = .661

#random forest
library(randomForest)
gridRF <- expand.grid(mtry = seq(0.5, 2, by = 0.5))
set.seed(seed)
prandomforest <- caret::train(x = xTrain, y = yTrain, 
                       method = "rf", 
                       preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"),
                       tuneGrid = gridRF,
                       trControl = ctrl)
prandomforest # ROC = 0.856, Sens = 0.785, Spec = 0.764

#bagged trees
set.seed(seed)
ptreebag <- caret::train(x = xTrain, y = yTrain,
                         method = "treebag",
                         preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"),
                         nbagg = 50,  
                         trControl = ctrl)

ptreebag #ROC = .824, sens = .77, spec = .703

#C5.0
set.seed(seed)
modelC5 <- caret::train(x = xTrain,
                 y = yTrain,
                 method = "C5.0",
                 preProc = c("center", "scale"),
                 verbose = FALSE,
                 trControl = ctrl)
modelC5 # ROC = .84 , Sens = .802, Spec = .721

# eXtreme Gradient Boosted Tree
set.seed(seed)
modelXGBT <- caret::train(x = xTrain, y = yTrain, 
                   method = "xgbTree",
                   preProc = c("center", "scale"), 
                   trControl = ctrl)
modelXGBT # ROC = , Sens = , Spec = 


# eXtreme Gradient Boosted DART
set.seed(seed)
modelXGBD <- caret::train(x = xTrain, y = yTrain, 
                   method = "xgbDART",
                   preProc = c("center", "scale"), 
                   trControl = ctrl)
modelXGBD # ROC = , Sens = , Spec = 


# eXtreme Gradient Boosted Linear
set.seed(seed)
modelXGBL <- caret::train(x = xTrain, y = yTrain, 
                   method = "xgbLinear",
                   preProc = c("center", "scale"), 
                   trControl = ctrl)
modelXGBL # ROC = , Sens = , Spec = 

#DALEX
library(DALEX)
library(gbm)
set.seed(seed)
regr_rf <- caret::train(pregnant~., data = dataPID, method = "rf", ntree = 100)

regr_gbm <- caret::train(pregnant~. , data = dataPID, method = "gbm")

regr_nn <- caret::train(pregnant~., data = dataPID,
                 method = "nnet",
                 linout = TRUE,
                 preProcess = c('center', 'scale'),
                 maxit = 500,
                 tuneGrid = expand.grid(size = 2, decay = 0),
                 trControl = trainControl(method = "none", seeds = 1))

### Explain Function ###

data("PimaIndiansDiabetes2")
library(DALEX)
explainer_regr_rf <- DALEX::explain(regr_rf, label = "rf", 
                                    data = dataPID, y = as.numeric(dataPID$diabetes),
                                    colorize = FALSE)

explainer_regr_gbm <- DALEX::explain(regr_gbm, label = "gbm", 
                                     data = dataPID, y = as.numeric(dataPID$diabetes),
                                     colorize = FALSE)

explainer_regr_nn <- DALEX::explain(regr_nn, label = "nn", 
                                    data = dataPID, y = as.numeric(dataPID$diabetes),
                                    colorize = FALSE)
### DALEX Model Performance ###

mp_regr_rf <- model_performance(explainer_regr_rf)
mp_regr_gbm <- model_performance(explainer_regr_gbm)
mp_regr_nn <- model_performance(explainer_regr_nn)

mp_regr_rf
mp_regr_gbm
mp_regr_nn

plot(mp_regr_rf, mp_regr_nn, mp_regr_gbm)
plot(mp_regr_rf, mp_regr_nn, mp_regr_gbm, geom = "boxplot")

### DALEX Variable Importance ###

vi_regr_rf <- variable_importance(explainer_regr_rf, loss_function = loss_root_mean_square)
vi_regr_gbm <- variable_importance(explainer_regr_gbm, loss_function = loss_root_mean_square)
vi_regr_nn <- variable_importance(explainer_regr_nn, loss_function = loss_root_mean_square)

plot(vi_regr_rf, vi_regr_gbm, vi_regr_nn)

### iml - Partial Dependence Plot ###
library(iml)
library(mlr)
X = dataPID[which(names(dataPID) != "diabetes")]
predictor <- Predictor$new(prandomforest, data = x, y = dataPID[,8])
str(predictor)

library(dplyr)
pdp_obj <- Partial$new(predictor, feature = "insulin")
pdp_obj$center(min(dataPID$insulin))
glimpse(pdp_obj$results)

pdp_obj$plot()

### ICE plots ###

pdp_obj2 <- Partial$new(predictor, feature = c("glucose", "pressure"))
pdp_obj2$plot()

### Tree Surrogate ###

tree <- TreeSurrogate$new(predictor, maxdepth = 5)
tree$r.squared
plot(tree)
tree$results %>%
  mutate(prediction = colnames(select(., .y.hat.qual_high, .y.hat.qual_low))[max.col(select(., .y.hat.qual_high, .y.hat.qual_low),
                                                                                     ties.method = "first")],
         prediction = ifelse(prediction == "???", "???", "???")) %>%
  ggplot(aes(x = predictor, fill = prediction)) +
  facet_wrap(~ .path, ncol = 5) +
  geom_bar(alpha = 0.8) +
  scale_fill_tableau() +
  guides(fill = FALSE)

### Local Model - Local Interpretable Model-agnostic Explanations ###

X2 <- xTest[, -9]
i = 1
lime_explain <- LocalModel$new(predictor, x.interest = X2[i, ])
lime_explain$results

### plot(lime_explain) ###

p1 <- lime_explain$results %>%
  ggplot(aes(x = reorder(feature.value, -effect), y = effect, fill = .class)) +
  facet_wrap(~ .class, ncol = 2) +
  geom_bar(stat = "identity", alpha = 0.8) +
  coord_flip() +
  labs(title = paste0("Test case #", i)) +
  guides(fill = FALSE)

### Shapley Value ###
shapley <- Shapley$new(predictor, x.interest = X2[1, ])
head(shapley$results)

### shapley$plot() ###
shapley$results %>%
  ggplot(aes(x = reorder(feature.value, -phi), y = phi, fill = class)) +
  facet_wrap(~ class, ncol = 2) +
  geom_bar(stat = "identity", alpha = 0.8) +
  coord_flip() +
  guides(fill = FALSE)
