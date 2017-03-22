data_set_project = read.csv("Data_set_combined.csv", header=T)
str(data_set_project)

############ trying to rearrange data in R // FAILED
testX = read.csv("rawX.csv", header=F)
testY = read.table("Y.txt", header=F)
str(testX)
str(testY)
testX = as.data.frame(t(testX)) # transposing

testNames = read.csv("Name.csv", header=F, stringsAsFactors = F) #names part very tricky
testNames = as.matrix(testNames)
testNames = sapply(testNames, '[[', 1)
str(testNames)
colnames(testX) = testNames #can't work
testX
rownames(testX) = testNames
############ trying to rearrange data in R

X=data_set_project[, 1:6319]
y=data_set_project[, 6320]

X = scale(X)
y = scale(y)

LS = solve( t(X) %*% X) %*% t(X) %*% y # LS model produce singularity error due to p >> ns

library('glmnet')

# plot coefficient paths to determine which are significant predictors
par(mfrow=c(1,3))
plot(glmnet(X,y, alpha=0)); title("alpha=0", line=3)
plot(glmnet(X,y, alpha=0.1), label=T); title("alpha=0.1", line=3) #3655, 6252, 5904
plot(glmnet(X,y, alpha=0.2), label=T); title("alpha=0.2", line=3) #3655, 6252


plot(glmnet(X,y, alpha=0.3), label=T); title("alpha=0.3", line=3) #3655, 6252
plot(glmnet(X,y, alpha=0.4), label=T); title("alpha=0.4", line=3) #3655, 6252
plot(glmnet(X,y, alpha=0.5), label=T); title("alpha=0.5", line=3) #3655, 6252

plot(glmnet(X,y, alpha=0.6), label=T); title("alpha=0.6", line=3) #3655, 6252
plot(glmnet(X,y, alpha=0.7), label=T); title("alpha=0.7", line=3) #3655, 6252
plot(glmnet(X,y, alpha=0.8), label=T); title("alpha=0.8", line=3) #3655, 6252

par(mfrow=c(1,2))
plot(glmnet(X,y, alpha=0.9), label=T); title("alpha=0.9", line=3) #3655, 6252
plot(glmnet(X,y, alpha=1), label=T); title("alpha=1", line=3) #3655, 6252

# use a nfold CV to determine optimal lambda for different alphs
# 5 fold CV for each alpha = 0, 0.1, ..., 1, due to small sample size, 
set.seed(1234)
foldid_set = sample(1:5, size=length(y), replace=T) #set same foldid for each cv.glmnet runs for different alpha
test_alpha_lambda = vector(length = 11)
test_alpha_cvm = vector(length=11)
for (i in 1:11) {
  test_alpha_lambda[i]=cv.glmnet(X, y, alpha=(i-1)/10, foldid = foldid_set)$lambda.min
  test_alpha_cvm[i]=min(cv.glmnet(X, y, alpha=(i-1)/10, foldid = foldid_set)$cvm)
}

(table_test_alpha = cbind(alpha=seq(0, 1, 0.1), test_alpha_lambda, test_alpha_cvm))
which(test_alpha_cvm==min(test_alpha_cvm))
#lowest cvm around alpha 0.3 to 0.5?

# plot CV values for different alpha
for (i in 1:11) {
  assign(paste("fit", (i-1), sep=""), cv.glmnet(X, y, alpha=(i-1)/10, foldid = foldid_set))
}

par(mfrow=c(1,3))
plot(fit0); title("alpha=0", line=3)
plot(fit1); title("alpha=0.1", line=3)
plot(fit2); title("alpha=0.2", line=3)

plot(fit3); title("alpha=0.3", line=3)
plot(fit4); title("alpha=0.4", line=3)
plot(fit5); title("alpha=0.5", line=3)

plot(fit6); title("alpha=0.6", line=3)
plot(fit7); title("alpha=0.7", line=3)
plot(fit8); title("alpha=0.8", line=3)

par(mfrow=c(1,2))
plot(fit9); title("alpha=0.9", line=3)
plot(fit10); title("alpha=1", line=3)

# try caret and glmnet
library(caret)
library(pROC)
getModelInfo()$glmnet$type

set.seed(1234)
splitIndex = createDataPartition(y, p=0.75, list=F, times=1)

objControl = trainControl(method="cv", number=5, returnResamp = 'none')
eGrid = expand.grid(.alpha=seq(.05,1, by=0.01),
                    .lambda=seq(0.001, 1, by=0.01)) 
objModel = train(X[splitIndex, ], y[splitIndex,], method='glmnet', metric='RMSE',
                 trControl=objControl, tuneGrid=eGrid)

objModel

objModel$bestTune # indicate the alpha, lambda for lowest RMSE
# alpha = 0.86, lambda =0.081

table_test_alpha = rbind(table_test_alpha, c(0.86,0.081,0)) # add the best-Tune parameters to table 
table_test_alpha

objects(objModel)

# dun apply
predictions = predict(object=objModel, X[-splitIndex,]) #useless?
auc = multiclass.roc(y[splitIndex], predictions)
print(auc$auc) #useless?
plot(varImp(objModel,scale=F)) #useless?
# dun apply

# we do 5-fold CV test on each model fit0 to fit10, MSE approach

CV_values = vector(length=11)
n=length(y)
for(i in 1:12){
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
which(CV_values==min(CV_values))

# can try set penalty factors of some important predictors to zero, applies for Lasso, 
# easier to identify which predictor is important?
p.fac=rep(1, 6319)
p.fac[c(3655)]=0 # set significant predictors (3655) for 0 zero penalty factor

CV_values_pfac1 = vector(length=12)
for(i in 1:12){
  cvi=0
  for(j in 1:5){
    k = ((j-1)*floor(n/5)+1):(j*floor(n/5));
    set_model = glmnet(X[-k,], y[-k,], penalty.factor=p.fac,
                       alpha=table_test_alpha[i,1], lambda=table_test_alpha[i,2])
    yhat = predict(set_model, newx=X[k,])
    cvi = cvi + (yhat-y[k, ])^2
  }
  CV_values_pfac1[i] = sum(cvi/5)
}
(CV_table_compare = cbind(alpha=table_test_alpha[,1], CV_values, CV_values_pfac1)) # seting penalty factors give lower CV errors

which.min(CV_values) # alpha=0.7 produces lowest CV error
which.min(CV_values_pfac) # very weird, the lowest CV will come from ridge regression?
# but overall setting penalty factor reduces CV

# we repeat and set more significant predictors, 6252 and 5904
p.fac[c(3655, 6252)]=0 # set significant predictors for 0 zero penalty factor, 3655, 6252, 5904
CV_values_pfac2 = vector(length=12)
for(i in 1:12){
  cvi=0
  for(j in 1:5){
    k = ((j-1)*floor(n/5)+1):(j*floor(n/5));
    set_model = glmnet(X[-k,], y[-k,], penalty.factor=p.fac,
                       alpha=table_test_alpha[i,1], lambda=table_test_alpha[i,2])
    yhat = predict(set_model, newx=X[k,])
    cvi = cvi + (yhat-y[k, ])^2
  }
  CV_values_pfac2[i] = sum(cvi/5)
}

(CV_table_compare = cbind(CV_table_compare, CV_values_pfac2))

# set significant predictors for 0 zero penalty factor, 3655, 6252, 5904
p.fac[c(3655, 5904, 6252)]=0 

CV_values_pfac3 = vector(length=11)
for(i in 1:12){
  cvi=0
  for(j in 1:5){
    k = ((j-1)*floor(n/5)+1):(j*floor(n/5));
    set_model = glmnet(X[-k,], y[-k,], penalty.factor=p.fac,
                       alpha=table_test_alpha[i,1], lambda=table_test_alpha[i,2])
    yhat = predict(set_model, newx=X[k,])
    cvi = cvi + (yhat-y[k, ])^2
  }
  CV_values_pfac3[i] = sum(cvi/5)
}

(CV_table_compare = cbind(CV_table_compare, CV_values_pfac3))
which(CV_values==min(CV_values))
which(CV_values_pfac1==min(CV_values_pfac1))
which(CV_values_pfac2==min(CV_values_pfac2))
which(CV_values_pfac3==min(CV_values_pfac3))


# we select the 'best' model
# plot the predicted values against data, should show a good fit
par(mfrow=c(2,2))

prefmodel_penalty = glmnet(X,y, lambda=table_test_alpha[1,2], alpha=table_test_alpha[1,1],
                   penalty.factor= p.fac
                   )
plot(predict(prefmodel_penalty, newx=X, type="response"), pch=1, col=2, ylab="Ro1 expression level")
points(y, pch=2, col=1)
title("Fit by 3 penalty factors, Ridge")
legend("topleft", legend=c("Data", "Fitted"), col=c("red", "black"), pch=c(2,1), cex=0.8)

prefmodel_CV = glmnet(X,y, lambda=table_test_alpha[7,2], alpha=table_test_alpha[7,1],
                      penalty.factor=p.fac)
plot(predict(prefmodel_CV, newx=X, type="response"), pch=1, col=2, ylab="Ro1 expression level")
points(y, pch=2, col=1)
title("Fit by 5-fold CV, alpha=0.6")

prefmodel_caret = glmnet(X,y, lambda=table_test_alpha[12,2], alpha=table_test_alpha[12,1])
plot(predict(prefmodel_caret, newx=X, type="response"), pch=1, col=2, ylab="Ro1 expression level")
points(y, pch=2, col=1)
title("Fit by caret, alpha=0.86")

prefmodel_cvm = glmnet(X,y, lambda=table_test_alpha[2,2], alpha=table_test_alpha[2,1])
plot(predict(prefmodel_cvm, newx=X, type="response"), pch=1, col=2, ylab="Ro1 expression level")
points(y, pch=2, col=1)
title("Fit by cvm, alpha=0.1")

# then we show the model coefficients for the selected model
prefmodel_penalty$a0
which(prefmodel_penalty$beta>0.5)
which(prefmodel_penalty$beta>0.3)
which(prefmodel_penalty$beta>0.1)
which(prefmodel_penalty$beta>0.01)
which(prefmodel_penalty$beta>0.001)
length(which(prefmodel_penalty$beta>0.0001))
length(which(prefmodel_CV$beta>0.0001))
