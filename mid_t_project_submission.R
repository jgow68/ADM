data_set_project = read.csv("Data_set_combined.csv", header=T)
str(data_set_project)

X=data_set_project[, 1:6319]
y=data_set_project[, 6320]

X = scale(X)
y = scale(y)

LS = solve( t(X) %*% X) %*% t(X) %*% y # LS model produce singularity error due to p >> ns

# fit the Ridge and Lasso model, then plot the coefficient solution path for fun????
library('glmnet')

outLasso = glmnet(X,y)
outRIDGE = glmnet(X,y, alpha=0)

# plot coefficient paths to determine which are significant predictors
plot(glmnet(X,y, alpha=0)); title("alpha=0", line=3)
plot(glmnet(X,y, alpha=0.1), label=T); title("alpha=0.1", line=3) #3655, 6252, 5904
plot(glmnet(X,y, alpha=0.2), label=T); title("alpha=0.2", line=3) #3655, 6252
plot(glmnet(X,y, alpha=0.3), label=T); title("alpha=0.3", line=3) #3655, 6252
plot(glmnet(X,y, alpha=0.4), label=T); title("alpha=0.4", line=3) #3655, 6252
plot(glmnet(X,y, alpha=0.5), label=T); title("alpha=0.5", line=3) #3655, 6252
plot(glmnet(X,y, alpha=0.6), label=T); title("alpha=0.6", line=3) #3655, 6252
plot(glmnet(X,y, alpha=0.7), label=T); title("alpha=0.7", line=3) #3655, 6252
plot(glmnet(X,y, alpha=0.8), label=T); title("alpha=0.8", line=3) #3655, 6252
plot(glmnet(X,y, alpha=0.9), label=T); title("alpha=0.9", line=3) #3655, 6252
plot(glmnet(X,y, alpha=1), label=T); title("alpha=1", line=3) #3655, 6252

# use a nfold CV to determine optimal lambda for different alphs
# 5 fold CV for each alpha = 0, 0.1, ..., 1, due to small sample size, 
foldid_set = sample(1:5, size=length(y), replace=T) #set same foldid for each cv.glmnet runs for different alpha
test_alpha_lambda = vector(length = 11)
test_alpha_cvm = vector(length=11)
for (i in 1:11) {
  # assign(paste("fit", i, sep=""), cv.glmnet(X, y, alpha=i/10, foldid = foldid_set))
  test_alpha_lambda[i]=cv.glmnet(X, y, alpha=(i-1)/10, foldid = foldid_set)$lambda.min
  test_alpha_cvm[i]=min(cv.glmnet(X, y, alpha=(i-1)/10, foldid = foldid_set)$cvm)
}

(table_test_alpha = cbind(alpha=seq(0, 1, 0.1), test_alpha_lambda, test_alpha_cvm)) #before we set sequence (0,10)
#lowest cvm around alpha 0.3 to 0.5?

# plot CV values for different alpha
for (i in 1:11) {
  assign(paste("fit", i, sep=""), cv.glmnet(X, y, alpha=i/10, foldid = foldid_set))
}
par(mfrow=c(1,1))
plot(fit0); title("alpha=0", line=3)
plot(fit1); title("alpha=0.1", line=3)
plot(fit2); title("alpha=0.2", line=3)
plot(fit3); title("alpha=0.3", line=3)
plot(fit4); title("alpha=0.4", line=3)
plot(fit5); title("alpha=0.5", line=3)
plot(fit6); title("alpha=0.6", line=3)
plot(fit7); title("alpha=0.7", line=3)
plot(fit8); title("alpha=0.8", line=3)
plot(fit9); title("alpha=0.9", line=3)
plot(fit10); title("alpha=1", line=3)

# try caret and glmnet
library(caret)
library(pROC)
getModelInfo()$glmnet$type
outcomename = y

set.seed(1234)
splitIndex = createDataPartition(y, p=0.75, list=F, times=1)
trainDF = 
testDF =

objControl = trainControl(method="cv", number=5, returnResamp = 'none')
objModel = train(X[splitIndex, ], y[splitIndex,], method='glmnet', metric='RMSE',
                 trControl=objControl)
# RMSE was used to select the optimal model using  the smallest value.
# The final values used for the model were alpha = 0.55 and lambda = 0.05380119. 

predictions = predict(object=objModel, X[splitIndex,]) #useless?
auc = multiclass.roc(y[splitIndex], predictions)
print(auc$auc) #useless?
plot(varImp(objModel,scale=F)) #useless?

# not good 
# we fit 3 models for alpha=0, 0.4 and 1
lasso_model_lambda = glmnet(X, y, lambda=test_alpha_lambda[11], alpha=1)
elastnet_model_lambda = glmnet(X, y, lambda=test_alpha_lambda[5], alpha=0.4)
ridge_model_lambda = glmnet(X, y, lambda=test_alpha_lambda[1], alpha=0)

# just plot to see the coefficient paths for alpha=0.4
plot(glmnet(X,y, alpha=0.4), label=T) # 3655 at lower lambda , 5904, 6252 significant
plot(glmnet(X, y, alpha=1), label=T) # 3655 at lower lambda, 5904, 6252 significant
plot(glmnet(X, y, alpha=0), label=T)# not easy to idenify for ridge regression

# we do 5-fold CV test on each model fit0 to fit10, MSE approach

CV_values = vector(length=11)
n=length(y)
for(i in 1:11){
  cvi=0
  for(j in 1:5){
    k = ((j-1)*floor(n/5)+1):(j*floor(n/5));
    set_model = glmnet(X[-k,], y[-k,], alpha=table_test_alpha[i,1], lambda=table_test_alpha[i,2])
    yhat = predict(set_model, newx=X[k,])
    cvi = cvi + (yhat-y[k, ])^2
  }
  CV_values[i] = sum(cvi/5)
}
CV_values


# can try set penalty factors of some important predictors to zero, applies for Lasso, 
# easier to identify which predictor is important?
p.fac=rep(1, 6319)
p.fac[c(3655)]=0 # set significant predictors for 0 zero penalty factor
pfit = glmnet(X, y, penalty.factor = p.fac, alpha=table_test_alpha[5,1], 
              lambda=table_test_alpha[5,2])

CV_values_pfac = vector(length=11)
for(i in 1:11){
  cvi=0
  for(j in 1:5){
    k = ((j-1)*floor(n/5)+1):(j*floor(n/5));
    set_model = glmnet(X[-k,], y[-k,], penalty.factor=p.fac,
                       alpha=table_test_alpha[i,1], lambda=table_test_alpha[i,2])
    yhat = predict(set_model, newx=X[k,])
    cvi = cvi + (yhat-y[k, ])^2
  }
  CV_values_pfac[i] = sum(cvi/5)
}
cbind(CV_values, CV_values_pfac) # seting penalty factors give lower CV errors
which.min(CV_values) # alpha=0.7 produces lowest CV error
which.min(CV_values_pfac) # very weird, the lowest CV will come from ridge regression?
# but overall setting penalty factor reduces CV

# we select the 'best' model
# plot the predicted values against data, should show a good fit
prefmodel = glmnet(X,y, lambda=table_test_alpha[8,2], alpha=table_test_alpha[8,1],
                   # penalty.factor=p.fac
                   )
plot(predict(prefmodel, newx=X, type="response"), pch=1, ylab="Ro1 expression level")
points(y, pch=2, col=2)

# then we show the model coefficients for the selected model
prefmodel$a0
which(prefmodel$beta>0.1)

