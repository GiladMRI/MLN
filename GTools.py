import tensorflow as tf
import pdb
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy

def getHome():
	# return '/home/deni/'
	# return '/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/'
	return '/media/a/H2/home/a/'

def getDatasetsBase():
	# return '/home/deni/'
	return '/media/a/H1/TFDatasets/'

""" Example use of TF_TSNUFFT:
B0Data=scipy.io.loadmat('/media/a/H1/MoreDataForTFNUFT.mat')
Sens=B0Data['Sens']
TSBF=B0Data['TSBF']
TSC=B0Data['TSC']

NUFTData=scipy.io.loadmat('/media/a/DATA/180628_AK/meas_MID244_gBP_VD11_U19_G35S155_4min_FID22439/TrajForNUFT.mat')
Kd=NUFTData['Kd']
P=NUFTData['P']
SN=NUFTData['SN']
Trajm2=NUFTData['Trajm2']

SmpI=scipy.io.loadmat('/media/a/H1/SmpI.mat')
SmpI=SmpI['SmpI']

nTraj=Trajm2.shape[1]
nCh=Sens.shape[2]
nTSC=TSC.shape[2]

SNc,paddings,sp_R,sp_I,TSBFX=GT.TF_TSNUFFT_Prepare(SN,Sens,TSC,TSBF,Kd,P)
Out=GT.TF_TSNUFFT_Run(SmpI,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX)

SOut={}
SOut['Out']=Out
scipy.io.savemat('/media/a/H1/TFTSNUFTOut.mat',SOut)
"""
def TF_TSNUFFT_Run(InImage,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX):
	# SNx=tf.reshape(SNx,[SNx.shape[0],SNx.shape[1],1])
	InImage=tf.reshape(InImage,[InImage.shape[0],InImage.shape[1],1])
	Step1=tf.multiply(InImage,SNc)
	Padded=tf.pad(Step1, paddings, "CONSTANT")
	Step2=tf.transpose(tf.fft(tf.transpose(tf.fft(tf.transpose(Padded,perm=[2,0,1])),perm=[0,2,1])),perm=[1,2,0])
	# Step2=tf.fft(tf.transpose(tf.fft(Padded),perm=[1,0]))
	Col=tf.reshape(Step2,[-1,nTSC*nCh])
	ColR=tf.real(Col)
	ColI=tf.imag(Col)
	RR=tf.sparse_tensor_dense_matmul(sp_R,ColR)
	RI=tf.sparse_tensor_dense_matmul(sp_R,ColI)
	IR=tf.sparse_tensor_dense_matmul(sp_I,ColR)
	II=tf.sparse_tensor_dense_matmul(sp_I,ColI)
	R=RR-II
	I=RI+IR
	C=tf.complex(R,I)

	# pdb.set_trace()

	# CX=np.reshape(C,(nTraj,nTSC,nCh))
	CX=tf.reshape(C,[nTraj,nTSC,nCh])

	WithTSB=CX*TSBFX

	WithTSBR=tf.reduce_sum(WithTSB,axis=1)
	return WithTSBR

def TF_TSNUFFT_Prepare(SN,Sens,TSC,TSBF,Kd,P):
	nTraj=TSBF.shape[1]
	nTSC=TSC.shape[2]
	InputIShape=Sens.shape[0:2]
	nCh=Sens.shape[2]

	TSCX=np.reshape(TSC,np.concatenate((TSC.shape,[1]),axis=0))
	SensP=np.transpose(np.reshape(Sens,np.concatenate((Sens.shape,[1]),axis=0)),(0,1,3,2))
	SensWithTSC=SensP*TSCX
	SensWithTSCX=np.reshape(SensWithTSC,(InputIShape[0],InputIShape[1],nCh*nTSC))
	SNX=np.reshape(SN,np.concatenate((SN.shape,[1]),axis=0))

	SensWithTSCXWithSN=SensWithTSCX*SNX

	# SNc=tf.constant(tf.cast(SensWithTSCXWithSN,tf.complex64))
	SNc=tf.constant(np.complex64(SensWithTSCXWithSN))

	TSBFX=np.transpose(np.reshape(TSBF,(nTSC,1,nTraj)),axes=(2,0,1))
	TSBFX=tf.constant(np.complex64(TSBFX))

	ToPad=[Kd[0,0]-InputIShape[0],Kd[0,1]-InputIShape[1]]

	paddings = tf.constant([[0, ToPad[0]], [0, ToPad[1]],[0,0]])
	# paddings = tf.constant([[0, 68], [0, 60]])
	
	Idx=scipy.sparse.find(P)
	I2=np.vstack([Idx[0],Idx[1]]).T

	I2=tf.constant(np.int64(I2))

	ValR=tf.constant(np.float32(np.real(Idx[2])))
	ValI=tf.constant(np.float32(np.imag(Idx[2])))
	

	sp_R = tf.SparseTensor(I2, ValR, [P.shape[0],P.shape[1]])
	sp_I = tf.SparseTensor(I2, ValI, [P.shape[0],P.shape[1]])	

	# sp_R = tf.SparseTensor(I2, tf.cast(np.real(Idx[2]),tf.float32), [P.shape[0],P.shape[1]])
	# sp_I = tf.SparseTensor(I2, tf.cast(np.imag(Idx[2]),tf.float32), [P.shape[0],P.shape[1]])

	return SNc,paddings,sp_R,sp_I,TSBFX


def TF_NUFT(A,SN,Kd,P):
	# A is data, e.g. of size H,W,nMaps
	# SN should be from Fessler, .* Channel maps; so finally H,W,nMaps
	# Kd is the final size for the overFT, e.g. H*2,W*2
	# P is a sparse matrix of nTraj x H*W ; <101x16320 sparse matrix of type '<class 'numpy.complex128'>'	with 2525 stored elements in Compressed Sparse Column format>

	# MData=scipy.io.loadmat('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/gUM/ForTFNUFT.mat')
	# A=MData['A']
	# SN=MData['SN']
	# Kd=MData['Kd']
	# P=MData['P']

	# NUbyFS3=MData['NUbyFS3'].T

	ToPad=[Kd[0,0]-A.shape[0],Kd[0,1]-A.shape[1]]
	
	paddings = tf.constant([[0, ToPad[0]], [0, ToPad[1]],[0,0]])
	# paddings = tf.constant([[0, 68], [0, 60]])
	nMaps=2 # A.shape[1]

	Idx=scipy.sparse.find(P)
	I2=np.vstack([Idx[0],Idx[1]]).T

	sp_R = tf.SparseTensor(I2, tf.cast(np.real(Idx[2]),tf.float32), [101,16320])
	sp_I = tf.SparseTensor(I2, tf.cast(np.imag(Idx[2]),tf.float32), [101,16320])

	SNx=tf.constant(tf.cast(SN,tf.complex64))
	Ax=tf.constant(tf.cast(A,tf.complex64))

	SNx=tf.reshape(SNx,[SNx.shape[0],SNx.shape[1],1])
	Step1=tf.multiply(Ax,SNx)
	Padded=tf.pad(Step1, paddings, "CONSTANT")
	Step2=tf.transpose(tf.fft(tf.transpose(tf.fft(tf.transpose(Padded,perm=[2,0,1])),perm=[0,2,1])),perm=[1,2,0])
	# Step2=tf.fft(tf.transpose(tf.fft(Padded),perm=[1,0]))
	Col=tf.reshape(Step2,[-1,nMaps])
	ColR=tf.real(Col)
	ColI=tf.imag(Col)
	RR=tf.sparse_tensor_dense_matmul(sp_R,ColR)
	RI=tf.sparse_tensor_dense_matmul(sp_R,ColI)
	IR=tf.sparse_tensor_dense_matmul(sp_I,ColR)
	II=tf.sparse_tensor_dense_matmul(sp_I,ColI)
	R=RR-II
	I=RI+IR
	C=tf.complex(R,I)

	return C

def GenerateNeighborsMap(Traj,kMax,osN,ncc,nChToUseInNN,nNeighbors):
	# kMax=np.ceil(np.amax(np.abs(CurBartTraj)))
	# osfForNbrhd=1.3;
	# osN=(np.ceil(kMax*osfForNbrhd)*2+1).astype(int)

	# nChToUseInNN=8
	# ncc=8
	nTrajAct=Traj.shape[1]
	
	# nNeighbors=12
	NMap=np.zeros([osN,osN,nNeighbors],dtype='int32')

	C=linspaceWithHalfStep(-kMax,kMax,osN)

	for i in np.arange(0,osN):
	    for j in np.arange(0,osN):
	    	CurLoc=np.vstack([C[i], C[j]])
	    	D=Traj-CurLoc
	    	R=np.linalg.norm(D,ord=2,axis=0)/np.sqrt(2)
	    	Idx=np.argsort(R)
	    	NMap[i,j,:]=Idx[0:nNeighbors]


	a=np.reshape(np.arange(0,nChToUseInNN)*nTrajAct,(1,1,1,nChToUseInNN))
	NMapC=np.reshape(NMap,(NMap.shape[0],NMap.shape[1],NMap.shape[2],1))+a
	NMapC=np.transpose(NMapC,(0,1,2,3))
	NMapCX=np.reshape(NMapC,(osN,osN,nNeighbors*nChToUseInNN))
	NMapCR=np.concatenate((NMapCX,NMapCX+nTrajAct*ncc),axis=2)
	return NMapCR

	# T=scipy.io.loadmat('/media/a/H1/NMapTest.mat')
	# Traj=T['Traj'][0:2,:]
	# NMapRef=T['NMap']-1
	# NMapCRef=T['NMapC']-1
	# NMapCXRef=T['NMapCX']-1
	# NMapCRRef=T['NMapCR']

	# Out=np.amax(np.abs(NMap-NMapRef))
	# OutC=np.amax(np.abs(NMapC-NMapCRef))
	# OutCX=np.amax(np.abs(NMapCX-NMapCXRef))
	# OutCR=np.amax(np.abs(NMapCR-NMapCRRef))

	# [Out, OutC,OutCX,OutCR]
	# Result: [0, 0, 0, 0]


def MoveWithCopiedBackwards(N,L):
    out=tf.concat([tf.range(L,N), tf.range(N-2,N-2-L,-1)],axis=0)
    return out

def MoveWithCopiedForwards(N,L):
    out=tf.concat([tf.range(L,0,-1), tf.range(0,N-L)],axis=0)
    return out

def ExpandWithBackwardsOn2(A,N,K):
	B=A
	for x in range(1, K):
	    CurMove=MoveWithCopiedBackwards(N,x)
	    CurB=tf.gather(A,CurMove,axis=0)
	    B=tf.concat([B, CurB],axis=2)
	return B

def ExpandWithForwardsOn2(A,N,K):
	B=A
	for x in range(1, K):
	    CurMove=MoveWithCopiedForwards(N,x)
	    CurB=tf.gather(A,CurMove,axis=0)
	    B=tf.concat([B, CurB],axis=2)
	return B

def ExpandWithCopiesOn2(A,N,K):
	Back=ExpandWithBackwardsOn2(A,N,K)
	Forward=ExpandWithForwardsOn2(A,N,K)
	B=tf.concat([Back,A,Forward],axis=2)
	return B


def gfft_TFOn3D(x,H,dim=0):
	HalfH=H/2
	Id=np.hstack([np.arange(HalfH,H), np.arange(0,HalfH)])
	Id=Id.astype(int)
	if dim==0 :
		x = tf.transpose(x, perm=[2,1,0])
	if dim==1 :
		x = tf.transpose(x, perm=[0,2,1])
	x = tf.gather(x,Id,axis=2)
	out=tf.fft(x)
	out=tf.divide(out,tf.sqrt(tf.cast(H,tf.complex64)))
	out = tf.gather(out,Id,axis=2)
	if dim==0 :
		out = tf.transpose(out, perm=[2,1, 0])
	if dim==1 :
		out = tf.transpose(out, perm=[0,2,1])
	return out

def gfft_TF(x,H,dim=0):
	HalfH=H/2
	Id=np.hstack([np.arange(HalfH,H), np.arange(0,HalfH)])
	Id=Id.astype(int)
	# IQ2=tf.reshape(IQ,IQ.shape[0:2])
	if dim==1 :
		x = tf.transpose(x, perm=[1, 0])
	x = tf.gather(x,Id,axis=1)
	out=tf.fft(x)
	out=tf.divide(out,tf.sqrt(tf.cast(H,tf.complex64)))
	out = tf.gather(out,Id,axis=1)
	if dim==1 :
		out = tf.transpose(out, perm=[1,0])
	return out

def gfft(x,dim=0):
    out=np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x,axes=dim),axis=dim),axes=dim)
    out=out/np.sqrt(x.shape[dim])
    return out

def gifft(x,dim=0):
    out=np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x,axes=dim),axis=dim),axes=dim)
    out=out*np.sqrt(x.shape[dim])
    return out

def IDFT_matrix(N):
	HalfN=N/2
	Id=np.hstack([np.arange(HalfN,N), np.arange(0,HalfN)])
	i, j = np.meshgrid(Id, Id)
	omega = np.exp(  2 * np.pi * 1J / N )
	W = np.power( omega, i * j ) / np.sqrt(N)
	return W

def linspaceWithHalfStep(Start,End,N):
    HalfStep=(End-Start)/(2*N)
    Out=np.linspace(Start+HalfStep,End-HalfStep,N)
    return Out

def gDFT_matrix(WhichFreqs,N2):
    OneCol=linspaceWithHalfStep(-np.pi,np.pi,N2);
    OneCol=np.reshape(OneCol,[-1, 1])
    WhichFreqs=np.reshape(WhichFreqs,[1,-1])
    Out=np.exp(1J*OneCol*WhichFreqs)
    return Out
	
	# Out=exp(1i*WhichFreqs*OneCol)
	# HalfN=N/2
	# Id=np.hstack([np.arange(HalfN,N), np.arange(0,HalfN)])
	# i, j = np.meshgrid(Id, Id)
	# omega = np.exp( - 2 * np.pi * 1J / N )
	# W = np.power( omega, i * j ) / np.sqrt(N)
	# return W
# function Out=gdftmtx(WhichFreqs,Npe)
# OneCol=linspaceWithHalfStep(-pi,pi,Npe);
# Out=exp(1i*WhichFreqs.'*OneCol);

def DFT_matrix(N):
	HalfN=N/2
	Id=np.hstack([np.arange(HalfN,N), np.arange(0,HalfN)])
	i, j = np.meshgrid(Id, Id)
	omega = np.exp( - 2 * np.pi * 1J / N )
	W = np.power( omega, i * j ) / np.sqrt(N)
	return W

def TFGenerateDCPhase(nx=100,ny=120,LFac=5,QFac=0.1,SFac=2):
	# LFac = 5
	# QFac = 0.1
	# nx, ny = (3, 2)
	Linx = tf.linspace(-np.pi, np.pi, nx)
	Liny = tf.linspace(-np.pi, np.pi, ny)
	X, Y = tf.meshgrid(Linx, Liny,indexing='ij')

	Rnd=tf.random_uniform([11])

	AL=(Rnd[0]-0.5)*LFac*X+(Rnd[1]-0.5)*LFac*Y+(Rnd[2]-0.5)*QFac*( tf.pow(X,2)  )+(Rnd[3]-0.5)*QFac*(  tf.pow(Y,2)  );
	BL=(Rnd[4]-0.5)*LFac*X+(Rnd[5]-0.5)*LFac*Y+(Rnd[6]-0.5)*QFac*( tf.pow(X,2)  )+(Rnd[7]-0.5)*QFac*(  tf.pow(Y,2) );
	PX=(Rnd[8]-0.5)*tf.sin(AL)+(Rnd[9]-0.5)*tf.sin(BL);
	DCPhase=Rnd[10]*2*np.pi-np.pi;
	PX=PX*2*SFac*np.pi+DCPhase;
	Out=tf.exp(tf.complex(PX*0, PX));
	return Out

def TFGenerateRandomSinPhase(nx=100,ny=120,LFac=5,QFac=0.1,SFac=2):
	# LFac = 5
	# QFac = 0.1
	# nx, ny = (3, 2)
	Linx = tf.linspace(-np.pi, np.pi, nx)
	Liny = tf.linspace(-np.pi, np.pi, ny)
	X, Y = tf.meshgrid(Linx, Liny,indexing='ij')

	Rnd=tf.random_uniform([11])

	AL=(Rnd[0]-0.5)*LFac*X+(Rnd[1]-0.5)*LFac*Y+(Rnd[2]-0.5)*QFac*( tf.pow(X,2)  )+(Rnd[3]-0.5)*QFac*(  tf.pow(Y,2)  );
	BL=(Rnd[4]-0.5)*LFac*X+(Rnd[5]-0.5)*LFac*Y+(Rnd[6]-0.5)*QFac*( tf.pow(X,2)  )+(Rnd[7]-0.5)*QFac*(  tf.pow(Y,2) );
	PX=(Rnd[8]-0.5)*tf.sin(AL)+(Rnd[9]-0.5)*tf.sin(BL);
	DCPhase=Rnd[10]*2*np.pi-np.pi;
	PX=PX*2*SFac*np.pi+DCPhase;
	Out=tf.exp(tf.complex(PX*0, PX));
	return Out

def GenerateRandomSinPhase(nx=100,ny=120,LFac=5,QFac=0.1,SFac=2):
	# LFac = 5
	# QFac = 0.1
	# nx, ny = (3, 2)
	Linx = np.linspace(-np.pi, np.pi, nx)
	Liny = np.linspace(-np.pi, np.pi, ny)
	X, Y = np.meshgrid(Linx, Liny)

	Rnd=np.random.rand(11)
	AL=(Rnd[0]-0.5)*LFac*X+(Rnd[1]-0.5)*LFac*Y+(Rnd[2]-0.5)*QFac*( np.power(X,2)  )+(Rnd[3]-0.5)*QFac*(  np.power(Y,2)  );
	BL=(Rnd[4]-0.5)*LFac*X+(Rnd[5]-0.5)*LFac*Y+(Rnd[6]-0.5)*QFac*( np.power(X,2) )+(Rnd[7]-0.5)*QFac*( np.power(Y,2) );
	PX=(Rnd[8]-0.5)*np.sin(AL)+(Rnd[9]-0.5)*np.sin(BL);
	DCPhase=Rnd[10]*2*np.pi-np.pi;
	PX=PX*2*SFac*np.pi+DCPhase;
	Out=np.exp(1j*PX);
	return Out

def GShow(A):
	ax = plt.subplot(111)
	im = ax.imshow(A)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	plt.show()
	ax.set_title('Title')

def GShowC(C):
	ax = plt.subplot(121)
	im = ax.imshow(np.abs(C))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax.set_title('Title')

	ax2 = plt.subplot(122)
	im = ax2.imshow(np.angle(C))
	divider = make_axes_locatable(ax2)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax2.set_title('Title2')

	plt.show()
	
def GShowC4(C):
	if C.ndim==3:
		if C.shape[2]==1:
			C=np.reshape(C,C.shape[0:2])
	ax = plt.subplot(221)
	im = ax.imshow(np.abs(C))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax.set_title('abs')

	ax2 = plt.subplot(222)
	im = ax2.imshow(np.angle(C))
	divider = make_axes_locatable(ax2)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax2.set_title('angle')

	ax = plt.subplot(223)
	im = ax.imshow(np.real(C))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax.set_title('real')

	ax2 = plt.subplot(224)
	im = ax2.imshow(np.imag(C))
	divider = make_axes_locatable(ax2)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax2.set_title('imag')

	plt.show()


print('hello')
Q=GenerateRandomSinPhase()

# plt.matshow(samplemat((15, 15)))

# plt.show()

"""
function Out=GenerateRandomSinPhase(N,LFac,QFac)
if(numel(N)==1)
    N=[N N];
end
if(nargin<4)
    nP=2;
end
if(nargin<3)
    QFac=1;
end
if(nargin<2)
    LFac=5;
end
Linx=linspace(-pi,pi,N(1));
Liny=linspace(-pi,pi,N(2));

[X,Y]=ndgrid(Linx,Liny);

AL=(rand-0.5)*LFac*X+(rand-0.5)*LFac*Y+(rand-0.5)*QFac*(X.^2)+(rand-0.5)*QFac*(Y.^2);
BL=(rand-0.5)*LFac*X+(rand-0.5)*LFac*Y+(rand-0.5)*QFac*(X.^2)+(rand-0.5)*QFac*(Y.^2);
PX=(rand-0.5)*sin(AL)+(rand-0.5)*sin(BL);
DCPhase=rand*2*pi-pi;
PX=PX*pi+DCPhase;
Out=exp(1i*PX);
"""
