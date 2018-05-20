library(tidyverse)
library(data.table)
library(dplyr)
library(date)
library(chron)
library(labeling)
library(jsonlite)
library(ggplot2)
library(randomForest)
library(stringr)
library(rAverage)
library(lubridate)
library(ggplot2)
library(pROC)
library(rpart)
library(rpart.plot)
library(stats)
library(ROCR)
library(gridExtra)
library(caret)
library(fpc)
library(factoextra)
library(mclust)
library(NbClust)
library(corrplot)
library(GGally)
library(sna)
library(RColorBrewer)

users <- fread("/users.csv")
head(users)

#tunnuse lisamine vastavalt kasuliku kasutaja definitsioonile
users <- users %>%
  mutate(hasReferred1 = ifelse(Referrals>=1,1,0)) %>%
  mutate(hasReferred2 = ifelse(Referrals>=2,1,0)) %>%
  mutate(hasReferred5 = ifelse(Referrals>=5,1,0)) %>%
  mutate(hasReferred10 = ifelse(Referrals>=6,1,0))
  
#engagementi numbriks tegemine
users_modified <- users %>%
  mutate(engagements=str_sub(Engagement, 1, str_length(Engagement)-1)) %>%
  mutate(engagements = sapply(engagements, as.numeric, simplify = T)) %>%
  mutate(engagements = unname(engagements))
users_modified$Engagement <- NULL

#kuupäevade teisendamine kuupäeva formaati ja seejärel sekunditeks
lct <- Sys.getlocale("LC_TIME"); Sys.setlocale("LC_TIME", "C")
users_modified <- users_modified %>%
  mutate(Registration_date = as.Date(Registration_date, "%B %d %Y"))
users_modified <- users_modified[order(users_modified$Registration_date),] 
users_modified <- users_modified %>% 
  mutate(Registration_date = as.numeric(Registration_date))

#faktoriaalsete tunnuste faktoriks teisendamine
cols_to_factor <- c("Gender", "City", "Registration_time", "isReferred", "hasPosted", "hasReferred1", "hasReferred2", "hasReferred5", "hasReferred10")
users_features_full <- users_modified %>%
  mutate_at(cols_to_factor, funs(factor(.)))

#andmestiku puhastamine ebavajaliikest tunnustest
users_features_full <- users_features_full[!duplicated(users_features_full),]
users_features <- users_features_full[,c(1:6,9,12:16)]

#treening- ja testandmestike eraldamine
set.seed(4698)
str(users_features)
train_idx <- sample(nrow(users_features), 4196, replace = F) # 8392 rida on kokku, seega 4196 train ja 4117 testiks
train <- users_features[train_idx,]
test <- users_features[-train_idx,]

#andmete balansseerimine (kasulik kasutaja on defineeritud kui kasutaja, kes on kutsunud liituma vähemalt ühe uue kasutaja)

set.seed(258)
hasReferred <- filter(users_features, hasReferred1==1) # kasutajad, kelle soovituslingi läbi on liitunud vähemalt üks uus kasutaja
hasnotReferred <- filter(users_features, hasReferred1==0) # kasutajad, kelle soovituslingi läbi pole liitunud ühtegi uut kasutajat
train_idx_hasReferred <- sample(nrow(hasReferred), nrow(hasReferred)/2, replace = F)
train_hasReferred <- hasReferred[train_idx_hasReferred,] 
test_hasReferred <- hasReferred[-train_idx_hasReferred,]
train_idx_hasnotReferred <- sample(nrow(hasnotReferred), nrow(train_hasReferred), replace = F)  
train_hasnotReferred <- hasnotReferred[train_idx_hasnotReferred,] 
test_hasnotReferred_leftovers <- hasnotReferred[-train_idx_hasnotReferred,]
test_idx_hasnotReferred <- sample(nrow(test_hasnotReferred_leftovers), nrow(test_hasReferred)/0.14, replace=F)
test_hasnotReferred <- hasnotReferred[test_idx_hasnotReferred,]
train_balanced <- rbind.data.frame(train_hasReferred,train_hasnotReferred)
test_original_ratio <- rbind.data.frame(test_hasReferred,test_hasnotReferred)

#uurin andmeid
table(users_features$hasReferred1) # kui paljud kasutajad on uusi kasutajaid kutsunud 
table(train_balanced$hasReferred1) # kas tasakaalustatud treeningandmestik on tasakaalus
table(test_original_ratio$hasReferred1) # kui palju on uues testandmestikus kasulikke kasutajaid
table(users_features_full$isReferred) # milline on soovitusmootori kaudu liitunud kasutajate arv

#esitustäpsuse ja saagise arvutamise funktsioon
precision_recall = function(real, predicted){
  tbl = as.data.frame(table(real=real, predicted=predicted))
  tp = filter(tbl, real==1 & predicted==1)$Freq
  fp <- filter(tbl, real==0 & predicted==1)$Freq
  fn <- filter(tbl, real==1 & predicted==0)$Freq
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  return(c(precision, recall))
}

#PEATÜKK 3.2 Uute kasutajate kasulikkuse ennustamine # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#kasulik kasutaja on defineeritud kui kasutaja, kes on kutsunud liituma vähemalt ühe uue kasutaja)

#JUHUMETSA ALGORITM
model_3 <- randomForest(data=train[,-1], hasReferred1~. - Referrals - hasReferred2 - hasReferred5 - hasReferred10, na.action = na.omit, do.trace=FALSE)
varImpPlot(model_3)
plot(model_3)
summary(model_3)

#ennustan testandmete kasutajate kasulikkust
rf_pred_3 <- predict(model_3, newdata=test[,-1])
table(rf_pred_3, test$hasReferred1)
test$rf_pred_3 <- rf_pred_3
precision_recall(test$hasReferred1, rf_pred_3)

rf_pred_3_p <- predict(model_3, newdata=test[,-1], type = 'prob')[,1]
test$rf_pred_3_p <- rf_pred_3_p
roc_curve <- roc(test$hasReferred1, as.numeric(rf_pred_3_p), plot = T) #ROC kõver
roc_rf <- roc(test$hasReferred1, as.numeric(rf_pred_3_p), plot = T) #uue muutuja loomine, et seda hiljem mudelite võrdlemisel kasutada
roc_curve$auc #ROC kõvera alune ala

#maksimaalse saagise leidmine, mille korral esitustäpsus on üle 60%
pred <- prediction(test$rf_pred_3_p, test$hasReferred1)
perf <- performance(pred,"prec","rec") #arvutab pika listi, kus on esitustäpsus ja saagis
precision_vector = perf@x.values[[1]] #võtan need vektorid sealt listist
recall_vector = perf@y.values[[1]]
#tahame olla 60% kindlad et me valime õigeid inimesi
max(recall_vector[precision_vector>0.6]) 

#JUHUMETSA ALGORITM TASAKAALUSTATUD ANDMETEGA
model_4 <- randomForest(data=train_balanced[,-1], hasReferred1~. - Referrals - hasReferred2 - hasReferred5 - hasReferred10, na.action = na.omit, do.trace=FALSE)
varImpPlot(model_4)
plot(model_4)

rf_pred_4 <- predict(model_4, newdata=test_original_ratio[,-1])
table(rf_pred_4, test_original_ratio$hasReferred1)
test_original_ratio$rf_pred_4 <- rf_pred_4
precision_recall(test_original_ratio$hasReferred1, rf_pred_4)

rf_pred_4_p <- predict(model_4, newdata=test_original_ratio[,-1], type = 'prob')[,1]
test_original_ratio$rf_pred_4_p <- rf_pred_4_p
roc_curve <- roc(test_original_ratio$hasReferred1, as.numeric(rf_pred_4_p), plot = T)
roc_curve$auc

pred <- prediction(test_original_ratio$rf_pred_4_p, test_original_ratio$hasReferred1)
perf <- performance(pred,"prec","rec")
precision_vector = perf@x.values[[1]]
recall_vector = perf@y.values[[1]]
max(recall_vector[precision_vector>0.6])  #leian maksimaalse saagise 60% esitustäpsuse juures

#LOGISTILISE REGRESSIOONI ABIL
logistic <- glm(data = train[,-1], formula = as.factor(hasReferred1)~ Age + Gender + City + Followers + AVG_Likes + engagements, 
                family = 'binomial')
summary(logistic)
exp(logistic$coefficients)

#ennustan testandmete kasutajate kasulikkust
test$logistic_predictions <- predict(newdata=test[,-1], logistic, type='response')
ggplot(data = test, aes(x=logistic_predictions, fill=hasReferred1)) + geom_density(alpha=0.3) + theme_bw()
head(test)

roc_curve <- roc(test$hasReferred1, test$logistic_predictions, plot = T, auc=T) #ROC kõver
roc_lr <- roc(test$hasReferred1, test$logistic_predictions, plot = T, auc=T) #uue muutuja loomine, et seda hiljem mudelite võrdlemisel kasutada
roc_curve$auc #ROC kõvera alune ala

pred <- prediction(test$logistic_predictions, test$hasReferred1)
perf <- performance(pred,"prec","rec")
precision_vector = perf@x.values[[1]]
recall_vector = perf@y.values[[1]]
max(recall_vector[precision_vector>0.6]) #leian maksimaalse saagise 60% esitustäpsuse juures

bestthr <- coords(roc_curve, "best", ret = "threshold")
test <- mutate(test, logistic_predictions_binary = ifelse(logistic_predictions >= bestthr, 1, 0))
tab <- table(real = test$hasReferred1, predicted = test$logistic_predictions_binary) #eksimismaariks
precision_recall(test$hasReferred1, test$logistic_predictions_binary) #esitustäpsus ja saagis

# LOGISTILINE REGRESSIOON tasakaalustatud treeningadmetega

logistic_balanced <- glm(data = train_balanced[,-1], formula = as.factor(hasReferred1)~ Age + Gender + City + Followers + AVG_Likes + engagements, family = 'binomial')

summary(logistic_balanced)
exp(logistic_balanced$coefficients)

test_original_ratio$logistic_predictions_balanced <- predict(newdata=test_original_ratio[,-1], logistic_balanced, type='response')
ggplot(data = test_original_ratio, aes(x=logistic_predictions_balanced, fill=hasReferred1)) + geom_density(alpha=0.3) + theme_bw() + theme(legend.position="top")

roc_curve_balanced <- roc(test_original_ratio$hasReferred1, test_original_ratio$logistic_predictions_balanced, plot = T)
roc_curve_balanced$auc

pred <- prediction(test_original_ratio$logistic_predictions_balanced, test_original_ratio$hasReferred1)
perf <- performance(pred,"prec","rec")
precision_vector = perf@x.values[[1]]
recall_vector = perf@y.values[[1]]
max(recall_vector[precision_vector>0.6]) 

bestthr_balanced <- coords(roc_curve_balanced, "best", ret = "threshold")
test_balanced <- mutate(test_original_ratio, logistic_predictions_balanced_binary = ifelse(logistic_predictions_balanced >= bestthr_balanced, 1, 0))
tab <- table(real = test_balanced$hasReferred1, predicted = test_balanced$logistic_predictions_balanced_binary)
tab
precision_recall(test_balanced$hasReferred1, test_balanced$logistic_predictions_balanced_binary)

#ROC kõverate võrdlus (mõlema meetodi tasakaalustamata treeningandmete põhjal loodud mudelid)
roc_vordlus <- ggroc(list("Juhumetsa abil loodud ROC kõver"=roc_rf, "Logistilise regressiooni abil loodud ROC kõver"=roc_lr), size = 1.5) + theme(legend.position="top") + theme_bw()
roc_vordlus

# Kasuliku kasutaja definitsiooni muutmise mõju mudelile

# Eesmärk on logistilise regressiooni abil aru saada, kas kasuliku kasutaja definitsiooni muutmine muudab mudelite ennustusvõimet paremaks või halvemaks
# Kasulik kasutaja on defineeritud kui kasutaja, kes on kutsunud liituma vähemalt kaks uut kasutajat)

logistic_2 <- glm(data = train[,-1], formula = as.factor(hasReferred2) ~ Age + Gender + City + Followers + AVG_Likes + engagements, 
                       family = 'binomial')
summary(logistic_2)
exp(logistic_2$coefficients)

test$hasReferred2_logistic_predictions <- predict(newdata=test[,-1], logistic_2, type='response')
roc_curve2 <- roc(test$hasReferred2, test$hasReferred2_logistic_predictions, plot = T, auc = T)
roc_curve2$auc

pred2 <- prediction(test$hasReferred2_logistic_predictions, test$hasReferred2)
perf2 <- performance(pred2,"prec","rec")
precision_vector2 = perf2@x.values[[1]]
recall_vector2 = perf2@y.values[[1]]
max(recall_vector2[precision_vector2>0.6])

# Kasulik kasutaja on defineeritud kui kasutaja, kes on kutsunud liituma vähemalt viis uut kasutajat)

logistic_5 <- glm(data = train[,-1], formula = as.factor(hasReferred5) ~ Age + Gender + City + Followers + AVG_Likes + engagements, 
                  family = 'binomial')
summary(logistic_5)
exp(logistic_5$coefficients)

test$hasReferred5_logistic_predictions <- predict(newdata=test[,-1], logistic_5, type='response')
roc_curve5 <- roc(test$hasReferred5, test$hasReferred5_logistic_predictions, plot = T, auc = T)
roc_curve5$auc

pred5 <- prediction(test$hasReferred5_logistic_predictions, test$hasReferred5)
perf5 <- performance(pred5,"prec","rec")
precision_vector5 = perf5@x.values[[1]]
recall_vector5 = perf5@y.values[[1]]
max(recall_vector5[precision_vector5>0.6])

# Kasulik kasutaja on defineeritud kui kasutaja, kes on kutsunud liituma vähemalt kümme  uut kasutajat)

logistic_10 <- glm(data = train[,-1], formula = as.factor(hasReferred10) ~ Age + Gender + City + Followers + AVG_Likes + engagements, 
                  family = 'binomial')
summary(logistic_10)
exp(logistic_10$coefficients)

test$hasReferred10_logistic_predictions <- predict(newdata=test[,-1], logistic_10, type='response')
roc_curve10 <- roc(test$hasReferred10, test$hasReferred10_logistic_predictions, plot = T, auc = T)
roc_curve10$auc

pred10 <- prediction(test$hasReferred10_logistic_predictions, test$hasReferred10)
perf10 <- performance(pred10,"prec","rec")
precision_vector10 = perf10@x.values[[1]]
recall_vector10 = perf10@y.values[[1]]
max(recall_vector10[precision_vector10>0.6])

# ROC kõverate võrdlus

roc_erinevad <- ggroc(list("Referrals >= 1"=roc_curve,
                     "Referrals >= 2"=roc_curve2, 
                     "Referrals >= 5"=roc_curve5, 
                     "Referrals >= 10"=roc_curve10), 
                     size = 1.5) + theme_bw()
roc_erinevad


#PEATÜKK 3.1 Postituste avaldamine kui kasvumootor # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#impordin postituste ja külastuste andmestikud
posts <- fread("/promotyposts.csv")
visitors <- fread("/Visitors.csv")

#muudan faktoriaalsed tunnused faktoriteks
cols_to_factor <- c("Campaign")
posts_features <- posts_modified %>%
  mutate_at(cols_to_factor, funs(factor(.)))
posts_features <- posts_features[!duplicated(posts_features),]

#grupeerin postituste statistika kuupäevade kaupa
#detach(package:plyr)
Posts_by_dates <- data.frame(posts_features %>%
                               group_by(Date) %>%
                               summarise(Total_posts = n(),
                                         Total_likes = sum(Likes),
                                         Total_comments = sum(Comments),
                                         Total_reach = sum(Reach)))

# grupeerin liitunud kasutajate info kuupäevade kaupa
users_features_full$Registration_date <- users$Registration_date
users_features_full$isReferred <- users$isReferred
Users_by_dates <- data.frame(users_features_full %>%
                               group_by(Date = as.Date(Registration_date, "%B %d %Y")) %>%
                               summarize(Joined_users = n(),
                                         Referred_users = sum(isReferred),
                                         Not_referred_users = (Joined_users - Referred_users)))

# ühendan postituste, külastuste ja liitumiste andmed üheks data frame'iks
Growth_by_dates_full <- merge(Posts_by_dates, visitors_modified, by="Date", all=TRUE)
Growth_by_dates_full <- merge(Growth_by_dates_full, Users_by_dates, by="Date", all=TRUE)
Growth_by_dates_full[is.na(Growth_by_dates_full)] <- 0

# eemaldan esimeste kuude liitumiste andmed, kui postitusi veel polnud
Growth_by_dates <- subset(Growth_by_dates_full, Date > "2017-12-08")

#leian korrelatsioonitabeli ja visualiseerin selle
GBDcor <- cor(Growth_by_dates[,-1], method = c("spearman"))
corrplot(as.matrix(GBDcor), addrect = 2, col = cm.colors(100))

# visualiseerin sooituslingi kaudu ja mitte soovituslingi kaudu liitunud kasutajate arvu
ggplot(data=Growth_by_dates_full, aes(Date)) +
  geom_line(aes(y = Referred_users, colour = "Soovituslingi kaudu liitunud kasutajad"), size=1.2) + 
  geom_line(aes(y = Not_referred_users, colour = "Mitte soovituslingi kaudu liitunud kasutajad"), size=1.2) + theme_bw() + theme(legend.position="top")
res <- wilcox.test(Growth_by_dates_full$Referred_users,Growth_by_dates_full$Not_referred_users, paired=TRUE)

# Visualiseerin viiruslikkuse muutumise ajas
Growth_with_coefficients <- Growth_by_dates_full[c(1,7,8,9)]
Growth_with_coefficients$Total_users <- cumsum(Growth_with_coefficients$Joined_users)
Growth_with_coefficients$Total_users_minus_referred <- (Growth_with_coefficients$Total_users -Growth_with_coefficients$Referred_users)
Growth_with_coefficients$Viral_coefficient <- Growth_with_coefficients$Referred_users/Growth_with_coefficients$Total_users

ggplot(data=Growth_with_coefficients, aes(Date)) + 
  geom_line(aes(y = Viral_coefficient), size=1.2) + theme_bw()

# visualiseerin kasutajate koguarvu muutumise
ggplot(data=Growth_with_coefficients, aes(Date)) + 
  geom_line(aes(y = Total_users), size=1.2) + theme_bw()


#PEATÜKK 3.4 Klasterdamine # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


#viin kõik tunnused ühtsele skaalale
users_bins <- mutate(users_modified, gender_bins = cut(Gender, breaks=c(-1,0,1)))
table(users_bins$gender_bins) #kategoriseerin soo jĆ¤rgi (1-male, 0-female)

users_bins <- mutate(users_modified, city_bins = cut(City, breaks=c(0,1,2,3)))
table(users_bins$city_bins) #kategoriseerin linna  jĆ¤rgi (1 - suurlinn, 2 - keskmine, 3-väikelinn)

users_bins <- mutate(users_modified, followers_bins = cut(Followers, breaks=c(0,288,502,835,194829)))
table(users_bins$followers_bins)

users_bins <- mutate(users_modified, age_bins = cut(Age, breaks=c(9,15,18,21,52)))
table(users_bins$age_bins)

uusers_bins <- mutate(users_modified, likes_bins = cut(AVG_Likes, breaks=c(0,50,98,175,6465)))
table(users_bins$likes_bins)

users_bins <- mutate(users_modified, eng_bins = cut(engagements, breaks=c(0,14,21,29,100)))
table(users_bins$eng_bins)

users_bins <- mutate(users_modified, referral_bins = cut(Referrals, breaks=c(-1,0,1,2,10,279)))
table(users_bins$referral_bins)

#skaleerin vaadeldavad nĆ¤itajad
users_clustering <- na.omit(users_modified) %>%
  mutate(Gender=scale(Gender),
         City=scale(City),
         Followers=scale(Followers),
         Age=scale(Age), 
         AVG_Likes=scale(AVG_Likes),
         engagements=scale(engagements),
         Referrals=scale(Referrals))
head(users_clustering)
users_clustering <- users_clustering[c(1:6,9,16)] #jätame alles vaid klasterdamise jaoks olulised tunnused

#leian sobivaima klastrite arvu Elbow meetodil
elbow_method <- function(data, max_k=50){
  require(ggplot2)
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 1:max_k){
    wss[i] <- sum(kmeans(users_clustering[,-1], centers=i)$withinss)
  }
  p <- data.frame(number_of_clusters=c(1:max_k), wss=wss) %>%
    ggplot(aes(x=number_of_clusters, y=wss, group=1)) + geom_point() + 
    geom_line() + theme_bw() + ylab("Within groups sum of squares")
  return(print(p))
}
elbow_method(users_clustering, max_k=15)

#kui Elbow meetod ei anna selget vastust, leian sobivaima klastrite arvu NbClust meetodiga
nb <- NbClust(users_clustering[-1], distance = "euclidean", min.nc = 2,
max.nc = 10, method = "complete", index ="all") # võtab väga kaua aega, väga

clusters <- kmeans(users_clustering[,-1], 3, nstart = 20)
users_clustering$cluster <- as.numeric(clusters$cluster)
p1 <- fviz_cluster(clusters, geom = "point", data = users_clustering[,-1]) + ggtitle("k = 3") + theme_bw()

#vaatlen kõiki klastreid eraldi
users_with_clusters_draft <- merge(users_features, users_clustering, by="ID") #loon uue tabeli, kuhu lisan klastrid
users_with_clusters <- users_with_clusters_draft[c(1:9,12,20)]
head(users_with_clusters)

summary(subset(users_with_clusters, cluster==1))
summary(subset(users_with_clusters, cluster==2))
summary(subset(users_with_clusters, cluster==3))

#võtan mudelisse ainult Instagramiga seotud näitajad, vanuse ja lisaks Promoty referrals

ig_users_clustering <- na.omit(users_modified) %>%
  mutate(Gender=scale(Gender),
         Followers=scale(Followers),
         AVG_Likes=scale(AVG_Likes),
         engagements=scale(engagements),
         Referrals=scale(Referrals))
ig_users_clustering <- ig_users_clustering[c(1:3,6,9,16)] #jätame alles vaid klasterdamise jaoks olulised tunnused
elbow_method(ig_users_clustering, max_k=15) # Elbow meetod näitab, et sobivaim klastrite arv on 5

ig_clusters <- kmeans(ig_users_clustering[,-1], 5, nstart = 20)
ig_users_clustering$cluster <- as.numeric(ig_clusters$cluster)
p2 <- fviz_cluster(ig_clusters, geom = "point", data = ig_users_clustering[,-1]) + theme_bw()

#vaatlen kõiki klastreid eraldi
ig_users_with_clusters_draft <- merge(users_features, ig_users_clustering, by="ID") #loon uue tabeli, kuhu lisan klastrid
ig_users_with_clusters <- ig_users_with_clusters_draft[c(1:9,12,17)]
head(ig_users_with_clusters)

summary(subset(ig_users_with_clusters, cluster==1))
summary(subset(ig_users_with_clusters, cluster==2))
summary(subset(ig_users_with_clusters, cluster==3))
summary(subset(ig_users_with_clusters, cluster==4))
summary(subset(ig_users_with_clusters, cluster==5))

# kasutajate referral käitumise analüüs

# andmestikud, mis näitavad, kes kutsus keda
network <- fread("/Users/Marelle Ellen/Documents/04 magistritöö/whoreferredwho.csv")

network <- network %>%
  mutate(Registration_date = as.Date(Registration_date, "%B %d %Y"))
network <- network[order(network$Registration_date),] 
head(network)
str(network)

network_by_referrers <- data.frame(network %>%
                                     group_by(Who_referred, Registration_date) %>%
                                     summarise(Referrals = n()))
# network_by_referrers$Who_referred <- as.numeric(network_by_referrers$Who_referred)
head(network_by_referrers) #4381 andmereast sai 2260 andmerida

#valin igast klastrist käsitsi kaks kasutajat, kes vastavad klastri keskmisele
#ja visualiseerin iga valitud kasutaja referralid

referrals20 <- network_by_referrers[network_by_referrers$Who_referred == "20",]
users_features_full[users_features_full$ID == 20,]$Registration_date # kontrollin antud kasutaja registreerumise aega
referrals20row1 <- c("20", "2017-10-29", "0") #lisan esimese rea, mille referralite arv on antud juhul 0
referrals20 <- rbind(referrals20, referrals20row)
referrals20$Referrals <- as.numeric(referrals20$Referrals)
referrals20plot <- ggplot(data=referrals20, aes(Registration_date)) +
  geom_line(aes(y = Referrals), colour="#f3756d", size=1.2) + theme_bw()

referrals1441 <- network_by_referrers[network_by_referrers$Who_referred == "1 441",]
referrals1441plot <- ggplot(data=referrals1441, aes(Registration_date)) +
  geom_line(aes(y = Referrals), colour="#f3756d", size=1.2) +theme_bw()

referrals19 <- network_by_referrers[network_by_referrers$Who_referred == "19",]
referrals19plot <- ggplot(data=referrals19, aes(Registration_date)) +
  geom_line(aes(y = Referrals), colour="#19bdc2", size=1.2) + theme_bw()

referrals2966 <- network_by_referrers[network_by_referrers$Who_referred == "2 966",]
(users_features_full[users_features_full$ID == 2966,])$Registration_date
referrals2966row1 <- c("2966", "2017-10-27", "0") #lisan esimese rea, mille referralite arv on antud juhul 0
referrals2966 <- rbind(referrals2966, referrals2966row1)
referrals2966$Referrals <- as.numeric(referrals2966$Referrals)
referrals2966plot <- ggplot(data=referrals2966, aes(Registration_date)) +
  geom_line(aes(y = Referrals), colour="#19bdc2", size=1.2) + theme_bw()
referrals2966plot

grid.arrange(referrals20plot, referrals1441plot, referrals19plot, referrals2966plot, nrow=2)
