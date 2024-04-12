#' Calculate lambda sequence for L1 logistic regression
#'
#' This function calculate the smallest lamda that shrink all
#' coefficients but intercept to zero and return a sequence of lambda
#' used for glmlasso
#' @param X design matrix with first column being 1
#' @param y response vector
#' @param nlam length of the lambda sequence
#' @param minlam smallest lambda for the lambda sequence
#' @param dweigtht vector of length ncol(X) with penalty factors for
#' each coefficient
#' @return a list containing the largest lambda, the lambda sequence
#' and the estimated intercept at the largest lambda
#' @export
#' @examples
#' n <- 1000
#' p <- 10
#' Z <- matrix(rnorm(n*(p-2)),nrow=n,ncol=(p-2))
#' v2 <- c(rep(1,n/2),rep(0,n/2))
#' X <- cbind(1,v2,Z)
#' beta.star <- c(p^(-1/2)*rnorm(p-1),2)
#' y <- X%*%beta.star+ rnorm(n)
#' logLamRange(X,y,100,0.0001,c(0,rep(1,p-1)))


logLamRange <- function(X,y,nlam,minlam,dweight){
    n <- length(y)
    n1 <- sum(y)
    betamax1 <- log(n1)-log(n-n1)
    px <- exp(betamax1)/(1+exp(betamax1))
    XX <- (y-px)*X
    grad <- colSums(XX)
    lammax <- max(abs(grad[-1]/dweight[-1]))
    if(is.null(minlam)){
        re <-
            list(lammax=lammax,lambda=seq(from=lammax/n,to=lammax/1000/n,length.out=nlam),betamax1=betamax1)
    }else{
        re <-
            list(lammax=lammax,lambda=seq(from=lammax/n,to=minlam/n,length.out=nlam),betamax1=betamax1)
        }                       
    return(re)
}


#' Calculate lambda sequence for L1 linear regression
#'
#' This function calculate the smallest lamda that shrink all
#' coefficients but intercept to zero and return a sequence of lambda
#' used for glmlasso
#' @param X design matrix with first column being 1
#' @param y response vector
#' @param nlam length of the lambda sequence
#' @param minlam smallest lambda for the lambda sequence
#' @param dweigtht vector of length ncol(X) with penalty factors for
#' each coefficient
#' @return a list containing the largest lambda, the lambda sequence
#' and the estimated intercept at the largest lambda
#' @export
#' @examples
#' ilogit <- function(x) exp(x)/(1+exp(x))
#' n <- 1000
#' p <- 10
#' Z <- matrix(rnorm(n*(p-2)),nrow=n,ncol=(p-2))
#' v2 <- c(rep(1,n/2),rep(0,n/2))
#' X <- cbind(1,v2,Z)
#' beta.star <- c(p^(-1/2)*rnorm(p-1),2)
#' y <- rbinom(n=n,size=1,prob=ilogit(X%*%beta.star))
#' lmLamRange(X,y,100,0.0001,c(0,rep(1,p-1)))


lmLamRange <- function(X,y,nlam,minlam,dweight){
    betamax1 <- mean(y)
    grad <- as.vector(crossprod(X,betamax1*X[,1])-crossprod(X,y))
    lammax <- max(abs(grad[-1]/dweight[-1]))
    if(is.null(minlam)){
        re <-
            list(lammax=lammax,lambda=seq(from=lammax/n,to=lammax/1000/n,length.out=nlam),betamax1=betamax1)
    }else{
        re <-
            list(lammax=lammax,lambda=seq(from=lammax/n,to=minlam/n,length.out=nlam),betamax1=betamax1)
    }              
    return(re)
}


#'  L1 penalized logistic regression and linear regression
#'
#' Use coordinate-desecent algorithm to provide the solution path for
#' L1 penalized logistic regression and linear regression
#' 
#'@param X design matrix with first column being 1
#'@param y the response vector
#'@param family the distribution of the response. Currently only
#'     support two type with "gaussian" doing the L1 linear regression
#'     and "binomial" doing the L1 logistic regression
#'@param nlambda length of the lamda sequence with default value 100
#'@param minlam the minimun of the lambda sequence. By defalt, it is
#'     lambdaMax/1000
#'@param lambda user specified lambda sequence. By default it is null
#'     and the function will call logLamRange() or lmLamRange() to
#'     calculate the lambda sequence
#'@param penalty.factor the vector containing the individual penalty
#'     factor for each coefficient. By default, it is
#'     c(0,rep(1,ncol(X)-1))
#' @param tol the precision of the estimated coefficients.
#'@return an object of S3 class "glmlasso" containing the lambda
#'     sequence and an matrix for the solution path of coefficients
#' @export
#' @examples
#' ilogit <- function(x) exp(x)/(1+exp(x))
#' n <- 1000
#' p <- 10
#' Z <- matrix(rnorm(n*(p-2)),nrow=n,ncol=(p-2))
#' v2 <- c(rep(1,n/2),rep(0,n/2))
#' X <- cbind(1,v2,Z)
#' beta.star <- c(p^(-1/2)*rnorm(p-1),2)
#' y <- rbinom(n=n,size=1,prob=ilogit(X%*%beta.star))
#' glmlasso(X,y,family="binomial")

glmlasso <-
    function(X,y,family=c("gaussian","binomial"),nlambda=100,minlam=NULL,
             lambda=NULL,
             penalty.factor=c(0,rep(1,ncol(X)-1)),tol=1e-6){

        p <- ncol(X)
        n <- nrow(X)
        
        if( family=="gaussian"){
            if(is.null(lambda)){
                lambda <-lmLamRange(X,y,nlambda,minlam,penalty.factor)$lambda
            }

            cout <-
                .C("lslasso",x=as.double(X),y=as.double(y),ix=as.integer(n),
                   jx=as.integer(p),active=as.integer(rep(0,p)),df=as.integer(1),
                   lambda=as.double(lambda),nlam=as.integer(length(lambda)),
                   weight=as.double(rep(1,n)),dweight=as.double(penalty.factor),
                   start=as.double(c(mean(y),rep(0,p-1))),
                   beta=as.double(rep(0,p*length(lambda))),tol=as.double(tol),
                   PACKAGE = "glmlasso")
            re <-
                list(lambda=lambda,beta.matrix=matrix(cout$beta,nrow=p))
            class(re) <- "glmlasso"
            return(re)
            
        }else if(family=="binomial"){
            if(is.null(lambda)){
                lambda <-logLamRange(X,y,nlambda,minlam,penalty.factor)$lambda
            }

            cout <-
               .C("loglasso",x=as.double(X),y=as.double(y),ix=as.integer(n),
                  jx=as.integer(p),active=as.integer(rep(0,p)),df=as.integer(1),
                  lambda=as.double(lambda),nlam=as.integer(length(lambda)),
                  dweight=as.double(penalty.factor),
                  start=as.double(c(log(sum(y))-log(n-sum(y)),rep(0,p-1))),
                  beta=as.double(rep(0,p*length(lambda))),tol=as.double(tol),
                  PACKAGE="glmlasso")
            re <-
                list(lambda=lambda,beta.matrix=matrix(cout$beta,nrow=p))
            class(re) <- "glmlasso"
            return(re)
        }else{
            stop("family not matched")
        }
    }


#' Plot the solution path glmlasso
#'
#' @param x the S3 class "glmlasso" object given function glmlasso
#' @return a plot with the Lasso solution path
#' @export
#' @examples
#' ilogit <- function(x) exp(x)/(1+exp(x))
#' n <- 1000
#' p <- 10
#' Z <- matrix(rnorm(n*(p-2)),nrow=n,ncol=(p-2))
#' v2 <- c(rep(1,n/2),rep(0,n/2))
#' X <- cbind(1,v2,Z)
#' beta.star <- c(p^(-1/2)*rnorm(p-1),2)
#' y <- rbinom(n=n,size=1,prob=ilogit(X%*%beta.star))
#' out <- glmlasso(X,y,family="binomial")
#' plot(out)
                                    
            
                
plot.glmlasso <- function(x){
    maxrange <- max(x$beta.matrix[-1,])
    minrange <- min(x$beta.matrix[-1,])
    l1norm <- colSums(x$beta.matrix[-1,])
    plot(l1norm,x$beta.matrix[2,],ylim=c(minrange,maxrange),xlab="L1 Norm",type="l",
         ylab="coefficients",col=1)
    for(i in 3:nrow(x$beta.matrix))
        lines(l1norm,x$beta.matrix[i,],type="l",col=i-1)
    abline(h=0)
}

    


        
