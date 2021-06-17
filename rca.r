library(kernlab)

geigen <- function (A,B,C,top) {
  p       <- nrow(B) 
  q       <- nrow(C) 
  s       <- min(c(p,q))
  B       <- (B+t(B))/2
  C       <- (C+t(C))/2
  Bfac    <- chol(B)
  Cfac    <- chol(C)
  Bfacinv <- solve(Bfac)
  Cfacinv <- solve(Cfac)
  D       <- t(Bfacinv)%*%A%*%Cfacinv
  if (p >= q) {
    result <- svd(D,nu=top,nv=top)
    values <- result$d
    L      <- Bfacinv %*% result$u
    M      <- Cfacinv %*% result$v
  } else {
    result <- svd(t(D),nu=top,nv=top)
    values <- result$d
    L      <- Bfacinv %*% result$v
    M      <- Cfacinv %*% result$u
  }
  list(cor=values, xcoef=L, ycoef=M)
}

rcc <- function (X,Y,l1,l2,top) {
  geigen(cov(X,Y),var(X)+diag(l1,ncol(X)),
         var(Y)+diag(l2,ncol(Y)),top)
}

aug <- function(x,k,act,type="fourier") {
  s <- sigest(x,scaled=NULL)[2]
  
  if(type == "linear") {
    return(function(x0) x0)
  }
  
  if(type == "nystrom") {
    w <- x[sample(1:nrow(x),k),]
    return(function(x0) kernelMatrix(rbfdot(s),x0,w))
  }
  
  if(type == "fourier") {
    w <- matrix(sign(rnorm(ncol(x)*k,sd=sqrt(2*s)),ncol(x)))
    #b <- runif(k,0,2*pi)
    #f <- function(x0) x0%*%w+t(matrix(b,k,nrow(x0)))
    f <- function(x0) x0%*%w
    if(act=="cos"){
      return(function(x0) cos(f(x0)))
    }
    if(act=="sign"){
      return(function(x0) sign(f(x0)))
    }
    if(act=="relu"){
      return(function(x0) max(f(x0),0))
    }
    if(act=="ternary"){
      s_minus = 0.0252 ## 0.0090
      s_plus = 0.7617 ##0.8402
      r = 0.2736
      t = 1.4253
      return(function(x0) ((f(x0)>(sqrt(2)*s_minus)) - (r*t*(f(x0)<(sqrt(2)*s_minus)))))
    }
    
  }
}

rcca_fit <- function(x,y,kx,ky,type,top,act) {
  augx <- aug(x,kx,act,type)
  augy <- aug(y,ky,act,type)
  C    <- rcc(augx(x),augy(y),1e-10,1e-10,top)
  list(cor=sum(abs(C$cor[1:top])),a=C$xcoef,b=C$ycoef,augx=augx,augy=augy)
}

rcca_eval <- function(rcca,x,y){
  list(x=rcca$augx(x)%*%rcca$a,y=rcca$augy(y)%*%rcca$b)
}

rpca_fit <- function(x,k,type) {
  augx <- aug(x,k,type)
  list(augx=augx,pca=prcomp(augx(x)))
}

rpca_eval <- function(rpca,x) {
  predict(rpca$pca,rpca$augx(x))
}

dataset   <- "xrmb"
algorithm <- "nystrom"
act <- "cos"

if(dataset == "xrmb") {
  system('cat xrmb* > xrmb.data')
  load('xrmb.data')
  x_tr <- x_tr[1:30000,]
  y_tr <- y_tr[1:30000,]
  top    <- 112
} else {
  load('mnist.data')
  top  <- 50
}

#for(k in c(1000,2000,3000,4000,5000,6000)){
for(k in c(3000)){
  t1     <- Sys.time()
  print(sum(diag(t(x_tr)%*%x_tr))/nrow(x_tr)/ncol(x_tr))
  print(sum(diag(t(y_tr)%*%y_tr))/nrow(y_tr)/ncol(y_tr))
  cca    <- rcca_fit(x_tr,y_tr,k,k,algorithm,top,act)
  t2     <- Sys.time()
  print(t2-t1)
  cca_te <- rcca_eval(cca,x_te,y_te)
  r_te   <- sum(sapply(1:top,function(i) abs(cor(cca_te$x[,i],cca_te$y[,i]))))
  print(sprintf("%s-%i on %s with %s activation has %.2f/%.2f in %.2f",algorithm,k,dataset, act, cca$cor,r_te,t2-t1))
}