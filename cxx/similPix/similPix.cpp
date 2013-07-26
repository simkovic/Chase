//      main.cxx
//      
//      Copyright 2012 matus <matus@matus-ThinkPad-T400>
//      
//      This program is free software; you can redistribute it and/or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation; either version 2 of the License, or
//      (at your option) any later version.
//      
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//      GNU General Public License for more details.
//      
//      You should have received a copy of the GNU General Public License
//      along with this program; if not, write to the Free Software
//      Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//      MA 02110-1301, USA.
//      
//      
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
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>



int main(int argc, char** argv){
	//for (int i=0;i<argc;i++){printf("ta %s\n",argv[i]);}
	char buffer [50];
	
	sprintf(buffer,"PF%d.npy",atoi(argv[1]));
	cnpy::NpyArray arr = cnpy::npy_load(buffer);
	//printf("dim = %d %u\n",arr1.shape.size(),arr1.word_size);
	//for (int a=0;a<arr.shape.size();a++){printf("pfsize, %u\n",arr.shape[a]);}
	const unsigned int N1= arr.shape[0];
	unsigned char* D1 = reinterpret_cast<unsigned char*>(arr.data);
	
	sprintf(buffer,"PF%s.npy",argv[2]);
	cnpy::NpyArray arr2 = cnpy::npy_load(buffer);
	const unsigned int N2= arr.shape[0];
	unsigned char* D2 = reinterpret_cast<unsigned char*>(arr.data);
	
	//double res1,res2,temp;
	int i=-1; int j=-1;
	const unsigned int P=arr.shape[1];
	const unsigned int R=arr.shape[3];
	const unsigned int F=arr.shape[4];
	const unsigned int Fws=F/2; // F window size
	
	int Ddim[]={P*P*R*F,P*R*F,R*F,F};
	double* S = new double[N1*N2*R*(F-Fws)*(F-Fws)*2];
	const unsigned int shape[] = {N1,N2,R*4,F-Fws,F-Fws,2};
	
	int Sdim[]={N2*4*R*F*(F-Fws)*2,4*R*F*(F-Fws)*2,pow(F-Fws,2)*2,(F-Fws)*2,2};
	
	//printf("D1[0]= %u", D1[0]);
	//printf("D1[1]= %u", D1[234*m1+12*m2+12*m3+0*m4+45]);
	
	
	double res1,res2,temp;
	double test=0;
	
	int index1,ym;
	int r2=0; int f2=F/2;
	for (int n1=0;n1<1;n1++){
	printf("n1=%d\n",n1);
	//if (n1 % 100==0 && n1>0) { sprintf(buffer,"S%02d.npy",n1/100);
	//	cnpy::npy_save(buffer,S,shape,5,"w");}
	for (int n2=0;n2<10;n2++){
		printf("n2=%d\n",n2);
	for (int r1=0;r1<R;r1++){
	for (int f1=0;f1<F-Fws;f1++){
	for (int m1=0;m1<2;m1++){
	for (int ori=0;ori<4;ori++){
	//for (int f2=0;f2<F-Fws;f2++){
		//res1=0;res2=0;
		index1= n1*Sdim[0]+n2*Sdim[1]+(ori*5+r1)*Sdim[2]+f1*Sdim[3]+f2*Sdim[4]+m1;
		S[index1]=0;
	for (int x=0;x<P;x++){
	for (int y=0;y<P;y++){
	for (int f=0;f<Fws;f++){
		// index rotation
		if (m1==1) ym=P-y-1; else ym=y;
		if (ori==0) {i=x;j=ym;}
		if (ori==1){i=ym;j=P-x-1;}
		if (ori==2){i=P-x-1;j=P-ym-1;}
		if (ori==3){i=P-ym-1;j=x;}
		temp= D1[n1*Ddim[0]+i*Ddim[1]+j*Ddim[2]+r1*Ddim[3]+f+f1]
			-D2[n2*Ddim[0]+x*Ddim[1]+y*Ddim[2]+r2*Ddim[3]+f+f2];
		//res1=res1+temp*temp;
		S[index1]=S[index1]+temp*temp;
	}}}
	S[index1]=sqrt(S[index1]);
	//if (a==0 && b==3) {printf("%d, %f, %f\n",r,res1,res2);}
	//res1=sqrt(res1);
	//res2=sqrt(res2);
	//find minimum orientation
	//if (res1<res2) {
	//	if (res1<S[a*N+b]) {S[a*N+b]=res1;S[a+b*N]=r;}
	//} else {
	//	if (res2<S[a*N+b]) {S[a*N+b]=res2;S[a+b*N]=r+4;}
	//}
	//printf("res1= %f\n", sqrt(res1));
	}}}}}}//}
	//printf("S[0]= %f\n", S[0*N*N+0*N+3]);
	sprintf(buffer,"S%03dx%03d.npy",atoi(argv[1]),atoi(argv[2]));
    cnpy::npy_save(buffer,S,shape,6,"w");
	//printf("finished\n");
}	
