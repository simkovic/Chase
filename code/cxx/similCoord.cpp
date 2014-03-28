#include <sstream>

#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cmath>
#include <string.h>

#include "cnpy.h"
#include <cstdlib>
#include <iostream>
#include <map>

double norm(double a, double b){
	return sqrt(pow(a,2)+pow(b,2));
};

double min(double a, double b){
	if (a>b) return b;
	else return a;
}
// TODO handle nans in E
int main(int argc, char** argv){
	cnpy::NpyArray arr = cnpy::npy_load("E.npy");
	double* E = reinterpret_cast<double*>(arr.data);
	//cnpy::NpyArray arr2 = cnpy::npy_load("T.npy");
	//double* T = reinterpret_cast<double*>(arr2.data);
	cnpy::NpyArray arr3 = cnpy::npy_load("Q.npy");
	double* Q = reinterpret_cast<double*>(arr3.data);
	//for (int a=0;a<48;a++){printf("T[%d]=%.03f\n",a,T[a]);}
	const unsigned int N= 50;//arr.shape[0];
	const unsigned int F= arr.shape[1]/2;
	const unsigned int R= 24;
	const unsigned int A= arr.shape[2];
	const unsigned int Qmax= arr3.shape[0]-1;
	const unsigned int M= 2;
	const unsigned int DIST= 5;// [deg]
	double* T= new double[R*2];
	for (int r=0;r<R;r++){
		T[r*2]=cos(r/double(R)*2*M_PI);
		T[r*2+1]=sin(r/double(R)*2*M_PI);}
	//int* INDEX=new int[A*2];
	double* avgDist=new double[A*2];
	double* TMP=new double[A*A];
	double* MINs=new double[A*2];
	int Edim[]={F*2*A*2,A*2,2};
	double* S = new double[N*N*F*F*R*M];
	const unsigned int shp[]={N,N,F,F,R,M};
	for (int i=0;i<N*N*F*F*R*M;i++) S[i]=NAN;
	//printf("E[%d,%d,%d,1]=%f\n",2,0,2,E[2*2*F*A*2+0*A*2+2*2]);
	//for (int i=0;i<argc;i++) printf("argv[%d]=%s\n",i,argv[i]);	
	//int n1=0; if (argc>1) n1=atoi(argv[1]); 
	
	double x,y,temp5;
	int temp1,temp2,index,temp3,temp4;
	int denom=1;
	// check whether within window
	bool* VALID=new bool[N*A];
	for (int n=0;n<N;n++){
	for (int a=0;a<A;a++){
		temp5=0;
		for (int f=0;f<F*2;f++){
			temp1=n*Edim[0]+f*Edim[1]+a*Edim[2];
			temp5+= norm(E[temp1],E[temp1+1]);} 
		VALID[n*A+a]= temp5/(F*2)<DIST;
	}}
	const unsigned int bla[]={N,A};
	cnpy::npy_save("V.npy",VALID,bla,2,"w");
	// start computation
	for (int n1=0;n1<N;n1++){
		printf("n1=%d\n",n1);
		temp1=0; for (int a=0;a<A;a++) temp1+= int(VALID[n1*A+a]);
	for (int n2=0;n2<N;n2++){
		temp2=0; for (int a=0;a<A;a++) temp2+= int(VALID[n2*A+a]);
		//printf("n2=%d\n",n2);
		if (n1==n2) continue;
		for (int f1=0;f1<F;f1++){
			if (isnan(E[n1*F*2*A*2+(f1)*A*2]) || 
				isnan(E[n1*F*2*A*2+(f1+F-1)*A*2])) 
				continue;
		for (int f2=0;f2<F;f2++){
			if (isnan(E[n2*F*2*A*2+(f2)*A*2]) || 
				isnan(E[n2*F*2*A*2+(f2+F-1)*A*2])) 
				continue;
		for (int r1=0;r1<R;r1++){
		for (int m1=0;m1<M;m1++){
			index=((((n1*N+n2)*F+f1)*F +f2)*R+r1)*M+m1;
			if (temp1==0 || temp2==0)  continue;
			else S[index]=0;
			// compute similarity
			for (int f=0;f<F;f++){
			for (int a1=0;a1<A;a1++){
				if (!VALID[n1*A+a1]) continue;
				// counter clock-wise rotation and mirroring
				temp4=n1*Edim[0]+(f1+f)*Edim[1]+a1*Edim[2];
				x= (T[2*r1+0]*E[temp4]-T[2*r1+1]*E[temp4+1])*pow(-1,m1); 
				y= T[2*r1+1]*E[temp4]+T[2*r1+0]*E[temp4+1];
			for (int a2=0;a2<A;a2++){
				if (!VALID[n2*A+a2]) continue;
				temp4=n2*Edim[0]+(f2+f)*Edim[1]+a2*Edim[2];
				temp3=int(round(norm(x-E[temp4],y-E[temp4+1])*100));
				if (temp3> Qmax) TMP[a1*A+a2]=Q[Qmax];
				else TMP[a1*A+a2]=Q[temp3];
			}}
			//if (r1==0 && f1==0 && m1==0) printf("shape=(%d,%d)\n",temp1,temp2);
			//min step over the larger dimension
			for (int a=0;a<2*A;a++) MINs[a]=1;
			for (int a1=0;a1<A;a1++){ 
			if (!VALID[n1*A+a1]) continue; 
			for (int a2=0;a2<A;a2++){
				if (!VALID[n2*A+a2]) continue;
				if (MINs[a1]>TMP[a1*A+a2]) MINs[a1]=TMP[a1*A+a2];
				if (MINs[a2+A]>TMP[a1*A+a2]) MINs[a2+A]=TMP[a1*A+a2];
			}}
			// mean step over the smaller dimension
			x=0; y=0;
			for (int a=0;a<A;a++){ 
				if (temp1<=temp2 && VALID[n1*A+a]) x+= MINs[a];
				if (temp1>=temp2 && VALID[n2*A+a]) y+= MINs[a+A];}
			if (temp1<temp2) S[index]+= x;// /temp1;
			else if (temp1>temp2) S[index]+= y;// /temp2;
			else S[index]+= min(x,y);// /temp1;

		};
		S[index]= S[index]/F;
		}}}}
	}}
	cnpy::npy_save("S.npy",S,shp,6,"w");
}
