#install.packages('BGLR')
library(BGLR)
Args <- commandArgs(trailingOnly=TRUE)
phefile=Args[1]
genofile=Args[2]
result=Args[3]

phe<-read.table(phefile)
y<-phe$V2

geno<-read.table(genofile,header = F)
geno<-as.matrix(geno[,2:ncol(geno)])

nIter=20000;
burnIn=3000;
thin=10;
saveAt='';
S0=NULL;
weights=NULL;

R2=0.5;
ETA<-list(list(X=geno,model='BayesA'))

fit=BGLR(y=y,ETA=ETA,nIter=nIter,burnIn=burnIn,thin=thin,saveAt=saveAt,df0=5,S0=S0,weights=weights,R2=R2)
write.table(fit$yHat,result,quote = F,row.names =F,col.names =F)

SNPeffect<-fit$ETA[[1]]$b
write.table(SNPeffect,'SNPeffect.txt',quote = F,row.names =F,col.names =F)

pi<-fit$ETA[[1]]$probIn
write.table(pi,'pi.txt',quote = F,row.names =F,col.names =F)