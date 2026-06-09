task2_func <- function(n=200, n_relevant=10, n_irrelevant=10) {
  p <- n_relevant + n_irrelevant
  beta <- c(rep(1, n_relevant), rep(0, n_irrelevant))
  
  X <- matrix(rnorm(n * p), nrow=n, ncol=p)
  prob <- 1 / (1 + exp(-(X %*% beta)))
  y <- rbinom(n=n, size=1, prob=prob)
  
  model_lasso_cv <- cv.glmnet(X, y, family="binomial", alpha=1)
  coef_min <- as.matrix(coef(model_lasso_cv, s="lambda.min"))
  coef_min_no_intercept <- coef_min[-1, 1]
  selected <- which(coef_min_no_intercept != 0)
  
  true_variables <- 1:n_relevant
  
  psr <- length(intersect(true_variables, selected)) / length(true_variables)
  
  if (length(selected) == 0) {
    fdr <- 0
  } else {
    fdr <- length(setdiff(selected, true_variables)) / length(selected)
  }
  
  c(PSR=psr, FDR=fdr)
}