data_set_project = read.csv("Data_set_combined.csv", header=T)
str(data_set_project)

X=data_set_project[, 1:6319]
y=data_set_project[, 6320]

X = scale(X)
y = scale(y)

LS = solve( t(X) %*% X) %*% t(X) %*% y # LS model produce singularity error due to p >> n

# fit the Ridge and Lasso model, then plot the coefficient solution path for fun????
library('glmnet')

outLasso = glmnet(X,y)
outRIDGE = glmnet(X,y, alpha=0)

par(mfrow=c(1,2), oma=c(0,0,1,0))
plot(outLasso, main="Lasso")
title('Lasso', line=2) 
plot(outRIDGE)
title('Ridge', line=2)

# plot the best lambda values selected by CV nfolds
cv.LASSO = cv.glmnet(X,y,nfolds = 5)
cv.RIDGE = cv.glmnet(X,y,nfolds = 5, alpha=0)

par(mfrow=c(1,2))
plot(cv.LASSO)
title('LASSO', line=2)
plot(cv.RIDGE)
title('RIDGE', line=2)

cv.LASSO$lambda.min
cv.RIDGE$lambda.min

# recreate model based on the selected lambda
LASSO1 = glmnet(X,y,lambda = cv.LASSO$lambda.min)
RIDGE1 = glmnet(X,y,lambda = cv.RIDGE$lambda.min)
# should we compare AIC / deviance values?

# do prediction / classification plot
xyNEW = read.csv('D:/dataT03a2.csv')
xnew = data.matrix(xyNEW[,1:100])

LASSOpred = predict(LASSO1, newx = xnew)
errorLASSO = mean((c(LASSOpred) - c(xyNEW[,101]))^2)

RIDGEpred = predict(RIDGE1, newx = xnew)
errorRIDGE = mean((c(RIDGEpred) - c(xyNEW[,101]))^2)

errorLASSO

errorRIDGE


# then we show the model coefficients for the selected model
