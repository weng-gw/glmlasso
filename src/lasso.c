/* lasso.c --- 
 * Time-stamp: <2017-04-27 13:46:32 wgw>
 * Author: wgw
 * Version: $Id: lasso.c,v 0.0 2017/04/14 03:24:33 wgw Exp$
 *\revision$Header: /home/wgw/Documents/StatMs/stat8054/project/lasso.c,v 0.0 2017/04/14 03:24:33 wgw Exp$
 */
#include <stdio.h>
#include <math.h>
#include <R_ext/Print.h>


/* Soft Threshold Function*/
double soft(double x, double thresh){
  if(x > thresh)
    return x-thresh;
  else if(x<-thresh)
    return x+thresh;
  else
    return 0;
}

/* Matrix-id-to-Vector-id Function*/
int vec_id(int i, int j, int row){
  return j*row+i;
}

/* Inner Product Fucntion*/
double inner_prod(double *x, double *y, int n){
  double iprod = 0.0;
  int i;
  for(i=0;i<n;i++)
    iprod += x[i]*y[i];

  return iprod;
}

/* Maximum absolute difference of two Vector over active set*/
double vec_maxdiff(double *x, double *y,int *active, int df){
  double max=fabs(x[active[0]]-y[active[0]]),diff;
  int i;
  for(i=1;i<df;i++){
    diff = fabs(x[active[i]]-y[active[i]]);
    if(diff>max) max = diff;
  }
  return max;
}

/* Calculate Residual of Y=X*beta over active set */
void residual(double *x, double *y, double *beta,int row,int
	      *active,int df, double *residual){
  int i,j;
  for(i=0;i<row;i++) residual[i]=y[i];

  for(j=0;j<df;j++){
    for(i=0;i<row;i++)
      residual[i]-=beta[active[j]]*x[vec_id(i,active[j],row)];
  }
}

/* Residual function for only one predictor */
void residualone(double *x, double *y, double *beta,int row, double *residual){
  int i;
    for(i=0;i<row;i++) residual[i]=y[i]-*beta*x[i];
}

void comp_wx(double *x, double *weight, int row, int *active,int df, double
*wx){
  int i,j;
  for(j=0;j<df;j++){
    for(i=0;i<row;i++)
      wx[vec_id(i,active[j],row)]= x[vec_id(i,active[j],row)]*weight[i];
  }
}

void comp_wx2(double *x, double *weight,int row, int *active, int df, double
*wx2){
  int i, j;
  double wx2j;
  for(j=0;j<df;j++){
    wx2j = 0;
    for(i=0;i<row;i++)
      wx2j+=weight[i]*x[vec_id(i,active[j],row)]*x[vec_id(i,active[j],row)];
    wx2[active[j]]=wx2j;
  }
}


/* Calculate Threshold in for lm-lasso */
void threshold(int row, int *active, int df, double lambda, double *dweight, double
		*weightx2, double *th){
  int j;
  for(j=0;j<df;j++)
    th[active[j]] = row*lambda*dweight[active[j]]/weightx2[active[j]];
}


/* Single lambda lslasso with given starting value and active set*/
void singlelslasso(double *x, double *y, int *ix, int *jx, int *active, int
	     *df, double *weight, double *dweight, double *lambda,
	     double *start,double *tol, double *beta){
  int n=ix[0], p=jx[0];
  int j,i;
  double oldbeta[p],ytilde[n],weightx[n*p],weightx2[p],thresh[p];
  double betajdiff;
  double iprod;
  for(j=0;j<p;j++){
    oldbeta[j]=beta[j]=start[j];
    
  }
  for(j=0;j<*df;j++) oldbeta[active[j]]=beta[active[j]]+1;

  residual(x,y,beta,n,active,*df,ytilde);
  comp_wx(x,weight,n,active,*df,weightx);
  comp_wx2(x,weight,n,active,*df,weightx2);
  threshold(n,active,*df, *lambda,dweight,weightx2,thresh);
  
  while(vec_maxdiff(beta,oldbeta,active,*df)>(*tol)){
    for(j=0;j<*df;j++) oldbeta[active[j]]=beta[active[j]];
    for(j=0;j<*df;j++){
      beta[active[j]] =
	soft(oldbeta[active[j]]+inner_prod(weightx+active[j]*n,ytilde,n)/weightx2[active[j]],thresh[active[j]]);
      betajdiff = beta[active[j]]-oldbeta[active[j]];
      residualone(x+active[j]*n,ytilde,&betajdiff,n,ytilde);
    }
  }
}


/* inverse logit function*/
void ilogit(double *x, int nx, double *fvalue){
  int i;
  for(i=0;i<nx;i++) fvalue[i]=exp(x[i])/(1+exp(x[i]));
}

/* Calculate linear fit */
void yhat(double *x,double *beta, int row, int *active, int df, double *fitted){
  int i,j;
  for(i=0;i<row;i++) fitted[i]=0;
  for(j=0;j<df;j++){
    for(i=0;i<row;i++){
      fitted[i]+=beta[active[j]]*x[vec_id(i,active[j],row)];
    }
  }
}


/* loss function for loglasso */
void logloss(double *x, double *y, int *ix, double *beta,int *active, int *df,
	     double *dweight, double *lambda, double *fvalue){
  int i, j;
  double linear;

  (*fvalue)=0;
  for(i=0;i<(*ix);i++){
    linear = 0;
    for(j=0;j<(*df);j++)
      linear+=x[vec_id(i,active[j],*ix)]*beta[active[j]];
    (*fvalue)+=log(1+exp(linear))-y[i]*linear;
  }
  for(j=0;j<(*df);j++) (*fvalue)+=(*ix)*dweight[active[j]]*(*lambda)*fabs(beta[active[j]]);
}


/* Single lambda log lasso*/
void singleloglasso(double *x, double *y, int *ix, int *jx, int *active,
		  int *df,double *dweight,
	    double *lambda, double *tol, double *beta){
  int n = *ix, p=*jx;
  double betaold[p],pk[n],wk[n],zk[n],fittedy[n];
  int i,j;
  double loss,oldloss,delta;
  

  for(j=0;j<p;j++) betaold[j]=beta[j];
  for(j=0;j<*df;j++) betaold[active[j]]+=1;

  //  logloss(x,y,ix,beta,active, df, dweight, lambda, &oldloss);
  
  while(vec_maxdiff(beta,betaold,active, *df)>*tol){
    for(j=0;j<*df;j++) betaold[active[j]]=beta[active[j]];
    yhat(x,beta,n,active,*df,fittedy);
    ilogit(fittedy,n,pk);
    for(i=0;i<n;i++){
      wk[i]=pk[i]*(1-pk[i]);
      zk[i]=(y[i]-pk[i])/wk[i]+fittedy[i];
    }

    singlelslasso(x,zk,ix,jx,active,df,wk,dweight,lambda,betaold,tol,beta);
    //  delta=0.8;
    /*    logloss(x,y,ix,beta,active, df, dweight, lambda,&loss);
    while(loss>oldloss){
      for(i=0;i<n;i++){
	zk[i]=(y[i]-pk[i])/wk[i]*delta+fittedy[i];
      }
	singlelslasso(x,zk,ix,jx,active,df,wk,dweight,lambda,betaold,tol,beta);
	logloss(x,y,ix,beta,active, df, dweight, lambda,&loss);
	delta*=0.5;
	}
	oldloss = loss;*/
}
}

/* gradient check for loglasso */
void logcheckgrad(double *x, double *y, int *ix, int *jx, double *beta,
	       double *lam, double *dweight,int *active, int *df, int
	       *check, int *flag,double *tol){
  double yfit[*ix],px[*ix],gradl[*jx];
  int i,j;
  flag[0]=0;
  yhat(x,beta,*ix,active,*df,yfit);
  ilogit(yfit,*ix,px);
  for(j=0;j<*jx;j++){
    gradl[j]=0;
    for(i=0;i<*ix;i++)
      gradl[j]+=(px[i]-y[i])*x[vec_id(i,j,*ix)];
    if(beta[j]>*tol){
      if(fabs(gradl[j]+(*lam)*dweight[j]*(*ix))<*tol){
	check[j]=0;
      }else{
	check[j]=1;
	flag[0]=1;
      }
    }else if(beta[j]< -(*tol)){
      if(fabs(gradl[j]-(*lam)*dweight[j]*(*ix))<*tol){
	check[j]=0;
      }else{
	check[j]=1;
	flag[0]=1;
      }
    }else{
      if(fabs(gradl[j])<(*lam)*dweight[j]*(*ix)){
	check[j]=0;
      }else{
	check[j]=1;
	flag[0]=1;
      }
    }
  }
}

/* gradient check for lmlasso */
void lscheckgrad(double *x, double *y, int *ix, int *jx, double *beta,
		 double *lam,double *dweight, int *active, int *df,
		 int *check, int *flag, double *tol){
  double res[*ix],gradl;
  int i, j;
  flag[0]=0;
  residual(x,y,beta,*ix,active,*df,res);
  for(j=0;j<(*jx);j++){
    gradl=0;
    for(i=0;i<(*ix);i++)
      gradl+=-res[i]*x[vec_id(i,j,*ix)];
    if(beta[j]>*tol){
      if(fabs(gradl+(*lam)*dweight[j]*(*ix))<*tol){
	check[j]=0;
      }else{
	check[j]=1;
	flag[0]=1;
      }
    }else if(beta[j]< -(*tol)){
      if(fabs(gradl-(*lam)*dweight[j]*(*ix))<*tol){
	check[j]=0;
      }else{
	check[j]=1;
	flag[0]=1;
      }
    }else{
      if(fabs(gradl)<(*lam)*dweight[j]*(*ix)){
	check[j]=0;
      }else{
	check[j]=1;
	flag[0]=1;
      }
    }
  }
}
    


/* update active set*/
void updateactive(int *check,int *jx, int *active, int *df){
  int j,i, exist;
  for(j=0;j<(*jx);j++){
    if(check[j]==0) continue;
    exist=0;
    for(i=0;i<(*df);i++){
      if(j==active[i]){
	exist=1;
	break;
      }
    }
    if(exist==0){
      active[*df]=j;
      (*df)+=1;
    }
  }
}



/* single lambda loglasso using active set update */
void logactive(double *x, double *y, int *ix, int *jx, int *active,
		  int *df,int *check,int *flag, double *lambda, double
		  *dweight, double *start, double *beta,double *tol){
  int i,j;
  for(j=0;j<(*jx);j++) beta[j]=start[j];
  logcheckgrad(x,y, ix,jx, beta,lambda,dweight, active, df,
	       check,flag, tol);
  while(*flag){
    updateactive(check,jx,active,df);
    singleloglasso(x, y, ix, jx, active,df,dweight,lambda, tol,beta);
    logcheckgrad(x,y, ix,jx, beta,lambda,dweight, active, df,
	       check,flag, tol);
  }
}

/* multiple lambda loglasso */
void loglasso(double *x, double *y, int *ix, int *jx, int *active, int
	    *df, double *lambda, int *nlam, double *dweight, double
	    *start, double *beta, double *tol){
  int i,flag;
  int check[*jx];

  logactive(x,
  y,ix,jx,active,df,check,&flag,lambda,dweight,start,beta,tol);

  for(i=1;i<(*nlam);i++)
      logactive(x,
		   y,ix,jx,active,df,check,&flag,lambda+i,dweight,beta+(i-1)*(*jx),beta+i*(*jx),tol);
}
      

/* single lambda lmlasso using active set update */
void lsactive(double *x, double *y, int *ix, int *jx, int *active,
	      int *df, int *check, int *flag, double *lambda, double
	      *weight, double *dweight, double *start, double *beta,
	      double *tol){
  int i,j;
  for(j=0;j<(*jx);j++) beta[j]=start[j];
  lscheckgrad(x,y,ix,jx,beta,lambda,dweight,active,df,check,flag,tol);

  while(*flag){
    updateactive(check,jx,active,df);
    singlelslasso(x,y,ix,jx,active,df ,weight,dweight,lambda,beta,tol,beta);
    lscheckgrad(x,y,ix,jx,beta,lambda,dweight,active,df,check,flag,tol);
  }
}


/* multiple lambda loglasso */
void lslasso(double *x, double *y, int *ix, int *jx, int *active, int
	     *df, double *lambda, int *nlam,double *weight, double
	     *dweight, double *start, double *beta, double *tol){
  int i, flag;
  int check[*jx];

  lsactive(x,y,ix,jx,active,df,check,&flag,lambda,weight,dweight,start,beta,tol);

  for(i=1;i<(*nlam);i++)
    lsactive(x,y,ix,jx,active,df,check,&flag,lambda+i,weight,dweight,beta+(i-1)*(*jx),beta+i*(*jx),tol);
}

