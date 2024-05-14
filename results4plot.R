library(parallel)
library(glmnet)
library(rms)
library(Hmisc)
library(lattice)
library(Formula)
library(ggplot2)
library(foreign)
library(psych)
library(pROC)
library(sampling)
library(ggpubr)
library(Matrix)
library(mRMRe)
library(caret)
library(car)
library(data.table)
###############################################################
setwd("C:/Users/Admin/Desktop/PresentWork/KOA/OA_5features_classification/data")
getwd()
################################################################
logit_sig <- function(var_y,varlist,data){
  in_formula <- as.formula(paste(var_y,"~",varlist)) 
  p <- glm(in_formula,family=binomial(link=logit),data=data)
  coeff <- summary(p)$coefficients
  beta <- coeff[,1]
  LCI <- coeff[,1] - coeff[,2]*1.96 
  UCI <- coeff[,1] + coeff[,2]*1.96 
  OR <- exp(beta)
  OR_LCI <- exp(LCI)
  OR_UCI <- exp(UCI)
  p_value <- coeff[,4]
  name <- var
  data_var <- data.frame(OR,OR_LCI,OR_UCI,p_value)
  data_var <- data_var[-1,]
  return(data_var)
}

logit_sig_p <- function(varlist, label, data){
  in_formula <- as.formula(paste(label, "~", varlist)) 
  p <- glm(in_formula,family=binomial(link=logit), data=data)
  return(summary(p)$coefficients[2,4])
}

ttest_sig_p <- function(varlist, label, data){
  in_formula <- as.formula(paste(varlist, "~", label)) 
  t <- t.test(in_formula, data=data)
  return(t$p.value)
}

mrmr_calc <- function(data, y, feture_num){
  mrmr_feature<-data
  mrmr_feature$y <-y
  target_indices = which(names(mrmr_feature)=='y')
  for (m in which(sapply(mrmr_feature, class)!="numeric")){
    mrmr_feature[,m]=as.numeric(mrmr_feature[,m])
  }
  Data <- mRMR.data(data = data.frame(mrmr_feature))
  mrmr=mRMR.ensemble(data = Data, 
                     target_indices = target_indices, 
                     feature_count = feture_num, 
                     solution_count = 1)
  index_mrmr=mrmr@filters[[as.character(mrmr@target_indices)]]
  return(index_mrmr)
}

lasso_calc <- function(data_train, data_test, y_train, y_test){
  X_train <- as.matrix(data_train)
  Y_train <- y_train
  X_test  <- as.matrix(data_test)
  Y_test  <- y_test
  
  cv.fit <- cv.glmnet(X_train,Y_train,alpha=1,family='binomial', type.measure = "class")
  fit<-glmnet(X_train,Y_train,alpha=1,family='binomial')
  
  Coefficients <- coef(fit, s = cv.fit$lambda.min)
  Active.Index <- which(Coefficients != 0)
  Active.Weighted <- Coefficients[Active.Index]
  Active.Feature <-row.names(Coefficients)[Active.Index]
  index_lasso <- Active.Index[-1]-1
  return(index_lasso)
}

roc_metrics_calc <- function(score){
  vName <- colnames(score)
  roc_sum <- data.frame(matrix(nrow = 0, ncol = 5))
  for (i in 2:ncol(score)) {
    ROC <- roc(score[,1], as.numeric(score[, i]))
    auc95 <- paste0(round(ci(ROC)[2],3), 
                    ' (', round(ci(ROC)[1],3), 
                    '-', round(ci(ROC)[3],3), ')')
    uniroc<-data.frame('model' = vName[i],
                       'AUC(95%CI)' = auc95)
    youden <- coords(ROC, "best", 
                     ret=c("accuracy", "sensitivity", "specificity"))
    if (nrow(youden) > 1){
      index <- which.max(youden[, 2])
      youden <- youden[index, ]
    }
    
    uniroc <- cbind(uniroc, round(youden, 3))
    roc_sum <- rbind(roc_sum, uniroc)
  }
  return(roc_sum)
}
#########################################################################
#########################################################################
#########################################################################
data <- as.data.frame(fread("data.csv"))
data <- data[, -1]
Index_Clinical <- c(3:9)
clName<-colnames(data[,Index_Clinical])
vName<-colnames(data[,-(1:9)])
data$Label <- factor(data$Label)
data$SEX <- factor(data$SEX)
#########################################################################
#########################################################################
#########################################################################
p = 0.7
N1 <- round(length(which(data$Label==0))*p)
N2 <- round(length(which(data$Label==1))*p) 
labels <- c("R1", "R2","R3", "R4","R5", "R0")
core <- makeCluster(4)
score_train <- data.frame(matrix(nrow = 188, ncol = 0))
score_test  <- data.frame(matrix(nrow = 81, ncol = 0))
#########################################################################
#########################################################################
#########################################################################
  g<-49748
  set.seed(g)
  sub_str<-strata(data, stratanames=("Label"),size=c(N1,N2),method="srswor")
  data_train<-data[sub_str$ID_unit, ]
  data_test<-data[-sub_str$ID_unit, ]
  score_train$Label <- data_train$Label
  score_test$Label  <- data_test$Label
  ##########################step.1 Variance###############################
  auc  <- data.frame(matrix(nrow = 1, ncol = 0))
  auc["g"] <- g
  cFeatures <- c()
  ##########################step.2 LR#####################################
  print(paste0(g, ": LR==Started==============="))
  pValue <- parSapply(core, vName, logit_sig_p, label="Label", data=data_train)
  Index_logit<-which(pValue<0.05)
  #############################################################################
  for (i in 1:length(labels)) {
      # i <- 5
      keyw <- labels[i]
      cName <- paste0("score_", keyw)
      cNameTrain <- paste0(keyw, "_train")
      cNameTest <- paste0(keyw, "_test")
      #############################################################################
      Index_keyw <- grep(keyw, vName[Index_logit])
      Index_Rad <- Index_logit[Index_keyw]
      #######################step.3 MRMR#####################################
      if (length(Index_Rad) > 25) {
        Index_mrmr <- mrmr_calc(data_train[, Index_Rad+9], data_train$Label, 20)
        if(max(Index_mrmr) <= length(Index_keyw))
          Index_Rad <- Index_Rad[Index_mrmr]
      }
      #######################step.4 LASSO####################################
      X_train <- as.matrix(data_train[, Index_Rad + 9])
      Y_train <- data_train$Label
      X_test  <- as.matrix(data_test[, Index_Rad + 9])
      Y_test  <- data_test$Label
      ######################################################################
      set.seed(g)
      cv.fit <- cv.glmnet(X_train, Y_train, alpha=1, family='binomial', type.measure = "class")
      # plot(cv.fit)
      # abline(v=log(c(cv.fit$lambda.min, cv.fit$lambda.1se)),
      #        col =c("red","black"),
      #        lty=c(2,2))
      ####################################################################
      fit<-glmnet(X_train, Y_train,alpha=1, family='binomial')
      # plot(fit, xvar = "lambda", label = TRUE)
      # abline(v=log(c(cv.fit$lambda.min, cv.fit$lambda.1se)),
      #        col =c("red","black"),
      #        lty =c(2,2))
      ####################################################################
      Coefficients <- coef(fit, s = cv.fit$lambda.min)
      Active.Index <- which(Coefficients != 0)
      Active.Weighted <- Coefficients[Active.Index]
      Active.Feature<-row.names(Coefficients)[Active.Index]
      if(i<6){
        cFeatures <- c(cFeatures, Active.Feature[-1])
      }
      # output<-data.frame(Active.Index,Active.Weighted,Active.Feature)
      # write.csv(output, file = "LASSO_output.csv")
      
      ##################################################################################
      #########################output#####################################
      score_train[cName] <-predict(fit,type="response", newx=X_train,
                                   s=cv.fit$lambda.min)
      score_test[cName]  <-predict(fit,type="response", newx=X_test,
                                   s=cv.fit$lambda.min)
      auc[cNameTrain]  <- auc(roc(score_train$Label, as.numeric(score_train[[cName]])))
      auc[cNameTest]   <- auc(roc(score_test$Label, as.numeric(score_test[[cName]])))
      ####################################################################
      # print(paste0(g, ": ", keyw, ": Finished=============================================="))
    }#end for label(i)
    # print(Flag)
  
  ##################################################################################
  #Comb_selected
  cName <- "Comb_selected"
  cNameTrain <- "selected_train"
  cNameTest <- "selected_test"
  Index_Rad <- grep(paste0(cFeatures, collapse = '|'), vName)
  #######################step.3 MRMR#####################################
  if (length(Index_Rad) > 25) {
    Index_mrmr <- mrmr_calc(data_train[, Index_Rad+9], data_train$Label, 20)
    if(max(Index_mrmr) <= length(Index_Rad))
      Index_Rad <- Index_Rad[Index_mrmr]
  }
  #######################step.4 LASSO####################################
  X_train <- as.matrix(data_train[, Index_Rad + 9])
  Y_train <- data_train$Label
  X_test  <- as.matrix(data_test[, Index_Rad + 9])
  Y_test  <- data_test$Label
  ######################################################################
  set.seed(g)
  cv.fit <- cv.glmnet(X_train, Y_train, alpha=1, family='binomial', type.measure = "class")
  plot(cv.fit)
  abline(v=log(c(cv.fit$lambda.min, cv.fit$lambda.1se)),
         col =c("red","black"),
         lty=c(2,2))
  ####################################################################
  fit<-glmnet(X_train, Y_train,alpha=1, family='binomial')
  plot(fit, xvar = "lambda", label = TRUE)
  abline(v=log(c(cv.fit$lambda.min, cv.fit$lambda.1se)),
         col =c("red","black"),
         lty =c(2,2))
  ####################################################################
  Coefficients <- coef(fit, s = cv.fit$lambda.min)
  Active.Index <- which(Coefficients != 0)
  Active.Weighted <- Coefficients[Active.Index]
  Active.Feature<-row.names(Coefficients)[Active.Index]
  output<-data.frame(Active.Index,Active.Weighted,Active.Feature)
  write.csv(output, file = "LASSO_output.csv")
  #########################output#####################################
  score_train[cName] <-predict(fit,type="response", newx=X_train,
                               s=cv.fit$lambda.min)
  score_test[cName]  <-predict(fit,type="response", newx=X_test,
                               s=cv.fit$lambda.min)
  auc[cNameTrain]  <- auc(roc(score_train$Label, as.numeric(score_train[[cName]])))
  auc[cNameTest]   <- auc(roc(score_test$Label, as.numeric(score_test[[cName]])))
  ####################################################################
  model_All <- glm("Label ~ score_R1 + score_R2 + score_R3 +score_R4 + score_R5",
                   family=binomial(link=logit), data=score_train)
  score_train["score_all_LR"] <-predict(model_All, type="response", newdata = score_train)
  score_test["score_all_LR"]  <-predict(model_All, type="response", newdata = score_test)
  auc["score_all_LR_train"]  <- auc(roc(score_train$Label, as.numeric(score_train$score_all_LR)))
  auc["score_all_LR_test"]   <- auc(roc(score_test$Label, as.numeric(score_test$score_all_LR)))
  # print(paste0(g, ": All_LR Finished=============================================="))
  ##################################################################################
  model_All.step <- step(model_All, direction="both")
  score_train["score_all_step"] <-predict(model_All.step, type="response", 
                                          newdata = score_train)
  score_test["score_all_step"]  <-predict(model_All.step, type="response", 
                                          newdata = score_test)
  auc["score_all_step_train"]  <- auc(roc(score_train$Label, 
                                          as.numeric(score_train$score_all_step)))
  auc["score_all_step_test"]   <- auc(roc(score_test$Label, 
                                          as.numeric(score_test$score_all_step)))
  # print(paste0(g, ": All.step Finished=============================================="))
  ##################################################################################
  # print(paste0(g, ": LR_clinical==Started==============="))
  pValue <- parSapply(core, clName, logit_sig_p, label="Label", data=data_train)
  index_cls<-which(pValue<0.05)
  ##################################################################################
  clsName<-clName[index_cls]
  all_train <- cbind(data_train, score_train[,-1])
  all_test <- cbind(data_test, score_test[,-1])
  
  model_clinical <- glm(paste0("Label~", paste0(clsName, collapse = '+')),
                   family=binomial(link=logit),data=data_train)
  score_train["Clinical"] <-predict(model_clinical, type="response", newdata = data_train)
  score_test["Clinical"]  <-predict(model_clinical, type="response", newdata = data_test)
  auc["Clinical_train"]  <- auc(roc(score_train$Label, 
                                          as.numeric(score_train$Clinical)))
  auc["Clinical_test"]   <- auc(roc(score_test$Label, 
                                          as.numeric(score_test$Clinical)))
  
  model_comb_all <- glm(paste0("Label ~ score_all_step+", paste0(clsName, collapse = '+')),
                        family=binomial(link=logit),data=all_train)
  score_train["Combined"] <-predict(model_comb_all, type="response", newdata = all_train)
  score_test["Combined"]  <-predict(model_comb_all, type="response", newdata = all_test)
  auc["Combined_train"]  <- auc(roc(score_train$Label, 
                                    as.numeric(score_train$Combined)))
  auc["Combined_test"]   <- auc(roc(score_test$Label, 
                                    as.numeric(score_test$Combined)))
##################################################################################
##################################################################################
write.csv(score_train, file = "score_train.csv")
write.csv(score_test, file = "score_test.csv")
roc_metrics_train <- roc_metrics_calc(score_train)
roc_metrics_test <- roc_metrics_calc(score_test)
roc_metrics <- data.frame(matrix(nrow = 0, ncol = 5))
nRoc <- nrow(roc_metrics_train)
for (i in 1:nRoc) {
  roc_metrics <- rbind(roc_metrics, roc_metrics_train[i,])
  roc_metrics <- rbind(roc_metrics, roc_metrics_test[i,])
}
write.csv(roc_metrics_train, file = "roc_metrics_train.csv")
write.csv(roc_metrics_test, file = "roc_metrics_test.csv")
write.csv(roc_metrics, file = "roc_metrics.csv")

model_test <- glm(Label ~ score_R1+score_R2+score_R3+score_R4+score_R5+WOMSTF+WOMADL,
                  family=binomial(link=logit),data=all_train)
model_test.step <- step(model_test, direction="both")
s_train <-predict(model_test, type="response", newdata = all_train)
s_test  <-predict(model_test, type="response", newdata = all_test)
s_train <-predict(model_test.step, type="response", newdata = all_train)
s_test  <-predict(model_test.step, type="response", newdata = all_test)
auc_Train  <- auc(roc(score_train$Label, as.numeric(s_train)))
auc_Test  <- auc(roc(score_test$Label, as.numeric(s_test)))
