###############################################################################
#
# Imbalanced dataset processing
#
###############################################################################

# Clear objects from Memory
rm(list=ls())
# Clear Console:
cat("\014")

library('unbalanced')
library('Amelia')

# Read in the csv file, show summary
credit.df <- read.csv("cs-training.csv")
summary(credit.df)

# Rename and clean up 
colnames(credit.df) <- c('ID', 'target', 'revolving_pct', 'age', 
                         'past_due_30_59', 'debt_ratio', 'monthly_income', 
                         'lines', 'past_due_gt_90', 'mortgages', 
                         'past_due_60_89', 'dependents')

# Convert to ordinals, need whole numbers of lines, mortgages, etc.
credit.df$target <- as.factor(credit.df$target)
credit.df$past_due_30_59 <- as.ordered(credit.df$past_due_30_59)
credit.df$past_due_60_89 <- as.ordered(credit.df$past_due_60_89)
credit.df$past_due_gt_90 <- as.ordered(credit.df$past_due_gt_90)
credit.df$lines <- as.ordered(credit.df$lines)
credit.df$mortgages <- as.ordered(credit.df$mortgages)
credit.df$dependents <- as.ordered(credit.df$dependents)

summary(credit.df)
str(credit.df)

# Impute missing values in the data (monthly_income, dependents)
credit.df.imp <- amelia(credit.df, m = 1, p2s = 2, 
                         idvars = c('ID'),
                         ords = c('past_due_30_59', 'past_due_60_89', 'past_due_gt_90',
                                  'lines', 'mortgages', 'dependents', 'target'))
credit.df <- credit.df.imp$imputations[[1]]
credit.df$monthly_income[credit.df$monthly_income < 0] <- 0

# Take out ID and target columns, save in vectors
ID <- credit.df$ID
credit.df$ID <- NULL
credit.target <- as.factor(credit.df$target)
credit.df$target <- NULL

# Unbalanced algos need all data to be in numeric type.
# for (col in colnames(credit.df)) {
#   credit.df[,col] <- as.numeric(credit.df[,col])
# }


# Use SMOTE to oversample minority values
credit.df.smote.out <-ubSMOTE(X=credit.df, Y=credit.target, verbose=TRUE)
target <- credit.df.smote.out$Y
credit.df.smote <- cbind(credit.df.smote.out$X, target)
write.csv(credit.df.smote, "cs-training-smote.csv")

# Unbalanced algos need all data to be in numeric type.
for (col in colnames(credit.df)) {
  credit.df[,col] <- as.numeric(credit.df[,col])
}

# Remove majority examples in a Tomek link, undersampling majority class.
credit.df.tomek.out <- ubTomek(X=credit.df, Y=credit.target, verbose=TRUE)
target <- credit.df.tomek.out$Y
credit.df.tomek <- cbind(credit.df.tomek.out$X, target)
write.csv(credit.df.tomek, "cs-training-tomek.csv")

# One-Sided Selection is an undersampling method.
# It applies Tomek links and then Condensed Nearest Neighbour
credit.df.OSS.out <- ubOSS(X=credit.df, Y=credit.target, verbose=TRUE)
target <- credit.df.OSS.out$Y
credit.df.OSS <- cbind(credit.df.OSS.out$X, target)
write.csv(credit.df.OSS, "cs-training-OSS.csv")

# Condensed Nearest Neighbour
credit.df.CNN.out <- ubENN(X=credit.df, Y=credit.target, k = 3, verbose=TRUE)
target <- credit.df.CNN.out$Y
credit.df.CNN <- cbind(credit.df.CNN.out$X, target)
write.csv(credit.df.CNN, "cs-training-CNN.csv")

credit.df$target <- credit.target

ubConf <- list(percOver=200, percUnder=200, k=2, perc=50, method="percPos", w=NULL)
ubResults <- ubRacing(target ~ ., data=credit.df, algo='glmnet', positive=1, ubConf = ubConf, ncore=7, 
                      nFold=10, maxFold=10, maxExp=100, stat.test="friedman", metric="auc", verbose=TRUE)


