cars = read.csv("Cars.csv")
cars_pred_x = read.csv('Cars_X_out.csv')
attach(cars)

set.seed(1)

#setup
n = nrow(cars)
train = sample (1:n, n*0.7)
test = (-train)
train_data = cars[train,]
test_data = cars[test,]

library(kknn)

kcv = 10
n0 = round(length(train)/kcv,0)
out_MSE = matrix(0,kcv,10)
used = NULL
set = 1:length(train)

for(j in 1:kcv){
  
  if(n0<length(set)){val = sample(set,n0)}
  if(n0>=length(set)){val=set}
  
  train_i = cars[-val,]
  test_i = cars[val,]
  
  for(i in 1:10){
    
    near = kknn(price~.,train_i,test_i,k=i,kernel = "rectangular")
    aux = mean((test_i[,'price']-near$fitted)^2)
    
    out_MSE[j,i] = aux
  }
  
  used = union(used,val)
  set = (1:n)[-used]
  
  cat(j,'\n')
  
}

mMSE = apply(out_MSE,2,mean)

plot(log(1/((1:10))),sqrt(mMSE),xlab="Complexity (log(1/k))",ylab="out-of-sample RMSE",col=4,lwd=2,type="l",cex.lab=1.2,main=paste("kfold(",kcv,")"))
best = which.min(mMSE) * 
  text(log(1/best),sqrt(mMSE[which.min(mMSE)]),paste("k=",best),col=2,cex=1.2)
text(log(1/2),sqrt(mMSE[2]),paste("k=",2),col=2,cex=1.2)
text(log(1/10),sqrt(mMSE[10]),paste("k=",10),col=2,cex=1.2)

near = kknn(price~.,train_data,test_data,k=10,kernel = "rectangular")
rmse.kknn = sqrt(mean((test_data[,'price']-near$fitted)^2))

#lm
lm.fit = lm(log(price)~.+ I(mileage^2) + mileage*condition + region*condition + trim*condition + trim*year + displacement*mileage + trim*mileage + year*condition + displacement*featureCount- state - subTrim - featureCount - X,data = train_data)
lm.pred = exp(predict(lm.fit,test_data))
rmse.lm = sqrt(mean((test_data[,'price']-lm.pred)^2))

#ridge
library(glmnet)
x = model.matrix(price ~ .+ I(mileage^2) + mileage*condition + region*condition + trim*condition + trim*year + displacement*mileage + trim*mileage + year*condition + displacement*featureCount- state - subTrim - featureCount - X,data = cars)
y = cars[,'price']
grid = 10^seq(5,-5,length = 100)
ridge.mod = glmnet(x[train,],log(y[train]),alpha = 0, lambda = grid)
cv.ridge = cv.glmnet(x[train,],log(y[train]),alpha = 0, lambda = grid)
bestlam.ridge = cv.ridge$lambda.min

ridge.pred = exp(predict(ridge.mod, s = bestlam.ridge, newx = x[test,]))
rmse.ridge = sqrt(mean((test_data[,'price']-ridge.pred)^2))

#lasso
lasso.mod = glmnet(x[train,],log(y[train]),alpha = 1, lambda = grid)
cv.lasso = cv.glmnet(x[train,],log(y[train]),alpha = 1, lambda = grid)
bestlam.lasso = cv.lasso$lambda.min

lasso.pred = exp(predict(lasso.mod, s = bestlam.lasso, newx = x[test,]))
rmse.lasso = sqrt(mean((test_data[,'price']-lasso.pred)^2))


#pcr
library(pls)
pcr.fit = pcr(log(price) ~. + I(mileage^2) + mileage*condition + region*condition + trim*condition + trim*year + displacement*mileage + trim*mileage + year*condition + displacement*featureCount- state - subTrim - featureCount - X, data = train_data, validation = 'CV')
validationplot(pcr.fit,val.type = 'MSEP',xlim = c(0,20),ylim=c(0,10^5))
pcr.pred = exp(predict(pcr.fit,test_data,ncomp = 100))
rmse.pcr = sqrt(mean((test_data[,'price'] - pcr.pred)^2))

#pls
pls.fit = plsr(log(price) ~. + I(mileage^2) + mileage*condition + region*condition + trim*condition + trim*year + displacement*mileage + trim*mileage + year*condition + displacement*featureCount- state - subTrim - featureCount - X, data = train_data, validation = 'CV')
validationplot(pls.fit,val.type = 'MSEP')
pls.pred = exp(predict(pls.fit,test_data,ncomp = 50))
rmse.pls = sqrt(mean((test_data[,'price'] - pls.pred)^2))

#tree
library(tree)
library(rpart)

big.tree = rpart(price~.,method="anova",data=train_data,
                 control=rpart.control(minsplit=5,cp=.00001))
bestcp=big.tree$cptable[which.min(big.tree$cptable[,"xerror"]),"CP"]
best.tree = prune(big.tree,cp=bestcp)
tree.pred = predict(best.tree,test_data)
rmse.tree = sqrt(mean((test_data[,'price'] - tree.pred)^2))

tree.cars = tree(price~.- X - state,data=train_data,mindev=.0001)
tree.pred = predict(tree.cars,test_data)
rmse.tree = sqrt(mean((test_data[,'price'] - tree.pred)^2))

#rf
library(randomForest)
p=ncol(train_data)-1
mtryv = c(p,sqrt(p))
ntreev = c(100,500)
parmrf = expand.grid(mtryv,ntreev)
colnames(parmrf)=c('mtry','ntree')
nset = nrow(parmrf)
olrf = rep(0,nset)
ilrf = rep(0,nset)
rffitv = vector('list',nset)
for(i in 1:nset) {
  cat('doing rf ',i,' out of ',nset,'\n')
  temprf = randomForest(price~.- X , data=train_data,mtry=parmrf[i,1],ntree=parmrf[i,2])
  ifit = predict(temprf)
  ofit=predict(temprf,test_data)
  olrf[i] = sqrt(mean((test_data$price-ofit)^2))
  ilrf[i] = sqrt(mean((train_data$price-ifit)^2))
  rffitv[[i]]=temprf
}

print(cbind(parmrf,olrf,ilrf))


rf.cars = randomForest(price~.- X , data=train_data, mtry = 4,ntree = 100, importance = TRUE)
rf.pred = predict(rf.cars,test_data)
rmse.rf = sqrt(mean((test_data[,'price'] - rf.pred)^2))

#boosting
library(gbm)
idv = c(4,10)
ntv = c(1000,5000)
lamv=c(.001,.2)
parmb = expand.grid(idv,ntv,lamv)
colnames(parmb) = c('tdepth','ntree','lam')
print(parmb)
nset = nrow(parmb)
olb = rep(0,nset)
ilb = rep(0,nset)
bfitv = vector('list',nset)
for(i in 1:nset) {
  cat('doing boost ',i,' out of ',nset,'\n')
  tempboost = gbm(price~. - X, data=train_data,distribution='gaussian',
                  interaction.depth=parmb[i,1],n.trees=parmb[i,2],shrinkage=parmb[i,3])
  ifit = predict(tempboost,n.trees=parmb[i,2])
  ofit = predict(tempboost,test_data,n.trees=parmb[i,2])
  olb[i] = sqrt(mean((test_data$price-ofit)^2))
  ilb[i] = sqrt(mean((train_data$price-ifit)^2))
  bfitv[[i]]=tempboost
}

print(cbind(parmb,olb,ilb))


boost.cars=gbm(price~. - X, data=train_data, distribution=
                 "gaussian",n.trees=5000, interaction.depth=10,shrinkage=0.001)
boost.pred = predict(boost.cars,test_data,n.trees=5000)
rmse.boost = sqrt(mean((test_data[,'price'] - boost.pred)^2))


#bart(for fun.. take long time.. like really long)
library(BayesTree)
x = model.matrix(price ~ .- X,data = cars)
y = cars[,'price']

bart.cars = bart(x[train,],y[train],x[test,])
