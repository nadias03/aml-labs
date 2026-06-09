library("glmnet")

source("task2_func.R")

x = read.table("prostate_x.txt")
y = read.table("prostate_y.txt")
x = as.matrix(x)
y = y[,1]
d = data.frame(x, y)

################################# TASK 1 ################################# 
### LASSO

# a
model_lasso = glmnet(x, as.numeric(y), family="binomial", alpha=1)
model_lasso$lambda
coef(model_lasso)[,1]
coef(model_lasso)[,50]

# b profile plot:
plot(model_lasso, xvar="lambda")

# c
model_lasso_cv = cv.glmnet(x, as.numeric(y), family="binomial", alpha=1)
plot(model_lasso_cv)

model_lasso_cv$lambda.min   # 0.02073725
model_lasso_cv$lambda.1se   # 0.07281247

### RIDGE 

# a
model_ridge = glmnet(x, as.numeric(y), family="binomial", alpha=0)
model_ridge$lambda
coef(model_ridge)[,1]
coef(model_ridge)[,23]

# b
plot(model_ridge, xvar="lambda")

# c
model_ridge_cv = cv.glmnet(x, as.numeric(y), family="binomial", alpha=0)
plot(model_ridge_cv)

model_ridge_cv$lambda.min   # 4.070807
model_ridge_cv$lambda.1se   # 10.32098

### ELASTIC NET

# a
model_elastic_net = glmnet(x, as.numeric(y), family="binomial", alpha=0.5)
model_elastic_net$lambda
coef(model_elastic_net)[,1]
coef(model_elastic_net)[,17]

# b
plot(model_elastic_net, xvar="lambda")

# c
model_elastic_net_cv = cv.glmnet(x, as.numeric(y), family="binomial", alpha=0.5)
plot(model_elastic_net_cv)

model_elastic_net_cv$lambda.min   # 0.01127521
model_elastic_net_cv$lambda.1se   # 0.1003736

## QUESTIONS

## Select the optimal value of λ using cross-validation.

# for prediction - lambda_min
# for interpretation and variables selection - lambda_1se

## Compare the three regularization methods in terms of sparsity, shrinkage, and 
## interpretability of the resulting model.

# LASSO produces sparse models by setting some coefficients exactly to zero making it 
# highly interpretable. Ridge shrinks all coefficients toward zero but does not eliminate any, 
# leading to more stable but less interpretable models. Elastic Net combines both approaches, 
# providing a balance between sparsity and stability.

## Which regularization method sets coefficients exactly to zero? Which one tends to
## keep correlated variables together? Comment on the observed coefficient paths.

# LASSO sets some coefficients exactly to zero, performing variable selection. Ridge does not set 
# coefficients to zero but keeps correlated variables together by shrinking them similarly. 
# Elastic Net also promotes grouping of correlated variables while allowing some coefficients to 
# be zero.

# In LASSO, coefficients drop to zero as lambda increases, showing clear variable selection. 
# In Ridge, coefficients decrease smoothly toward zero without becoming exactly zero. 
# Elastic Net shows intermediate behaviour, with both shrinkage and partial sparsity.

## Does the value of λ selected by cross-validation necessarily correspond to the best
## model for variable selection? Explain.

# No, the lambda selected by cross-validation minimizes prediction error, not model sparsity. 
# Therefore, it may include too many variables. A larger lambda (for example, lambda_1se) cann be 
# preferred for variable selection, as it produces a simpler and more interpretable model.

################################# TASK 2 ################################# 
####### a - data generation
n <- 200
p <- 20   # 10 relevant + 10 irrelevant

beta <- c(rep(1, 10), rep(0, 10))   # 10 relevant + 10 irrelevant

# generate x
X <- matrix(rnorm(n * p), nrow=n, ncol=p)

prob <- 1 / (1 + exp(-(X %*% beta)))

# generate y
y <- rbinom(n=n, size=1, prob=prob)

data <- data.frame(X, y)

####### b 
# fit logistic regression with lasso
model_lasso <- glmnet(X, y, family="binomial", alpha=1)

# coef path
plot(model_lasso, xvar="lambda")

# cross-validation 
model_lasso_cv <- cv.glmnet(X, y, family="binomial", alpha=1)
plot(model_lasso_cv)

# QUESTION
# lambda which gives the smallest error cv
model_lasso_cv$lambda.min   # 0.009252218

# lambda that gives simplest model
model_lasso_cv$lambda.1se   # 0.02345774

####### c
coef_min <- as.matrix(coef(model_lasso_cv, s="lambda.min"))
coef_min_no_intercept <- coef_min[-1, 1]

t <- 1:10
t_hat <- which(coef_min_no_intercept != 0)

# psr
psr <- length(intersect(t, t_hat)) / length(t)

# fdr
if (length(t_hat) == 0) {
  fdr <- 0
} else {
  fdr <- length(setdiff(t_hat, t)) / length(t_hat)
}

# results
t_hat
# V1  V2  V3  V4  V5  V6  V7  V8  V9 V10 V11 V12 V14 V15 V20 
#  1   2   3   4   5   6   7   8   9  10  11  12  14  15  20 
# LASSO correctly selected all 10 truly relevant variables, but it also selected 5 irrelevant ones.

psr   # 1 - the model identified all truly relevant variables
fdr   # 0.3333333 - about 33.3% of the selected variables were false positives

####### d - repeat 100 times
set.seed(123)
L <- 100
results <- replicate(L, task2_func(n=200, n_relevant=10, n_irrelevant=10))

mean_psr <- mean(results["PSR", ])
mean_fdr <- mean(results["FDR", ])

mean_psr
mean_fdr

####### e - dependence on sample size
set.seed(123)
n_values <- c(50, 100, 300, 500, 1000, 2000)
L <- 100

mean_psr_n <- numeric(length(n_values))
mean_fdr_n <- numeric(length(n_values))

for (i in 1:6) {
  n_i <- n_values[i]
  res <- replicate(L, task2_func(n=n_i, n_relevant=10, n_irrelevant=10))
  
  mean_psr_n[i] <- mean(res["PSR", ])
  mean_fdr_n[i] <- mean(res["FDR", ])
}

mean_psr_n   # 0.739 0.979 1.000 1.000 1.000 1.000
mean_fdr_n   # 0.2607529 0.3515534 0.3863334 0.4091149 0.4179861 0.4414335

plot(
  n_values, mean_psr_n, type = "b", ylim = c(0,1),
  xlab = "Sample size n", ylab = "Metric value",
  main = "Average PSR and FDR vs sample size"
)
lines(n_values, mean_fdr_n, type = "b", lty = 2)
legend("right", legend = c("PSR", "FDR"), lty = c(1,2), bty = "n")

# PSR increases with the sample size.
# FDR slightly increases with n.

####### f - dependence on the number of irrelevant variables
set.seed(123)
irrelevant_values <- c(10, 50, 100, 200, 500)
L <- 100

mean_psr_irrel_var <- numeric(length(irrelevant_values))
mean_fdr_irrel_var <- numeric(length(irrelevant_values))

for (i in 1:5) {
  irr_i <- irrelevant_values[i]
  res <- replicate(L, task2_func(n=300, n_relevant=10, n_irrelevant=irr_i))
  
  mean_psr_irrel_var[i] <- mean(res["PSR", ])
  mean_fdr_irrel_var[i] <- mean(res["FDR", ])
}

mean_psr_irrel_var   # 1 1 1 1 1
mean_fdr_irrel_var   # 0.3915277 0.6413181 0.7213243 0.7843217 0.8189193

plot(
  irrelevant_values, mean_psr_irrel_var, type = "b", ylim = c(0,1),
  xlab = "Number of irrelevant variables", ylab = "Metric value",
  main = "Average PSR and FDR vs number of irrelevant variables"
)
lines(irrelevant_values, mean_fdr_irrel_var, type = "b", lty = 2)
legend("right", legend = c("PSR", "FDR"), lty = c(1,2), bty = "n")

# PSR stays equal to 1 for all tested values.
# FDR increases strongly as the number of irrelevant variables grows.

