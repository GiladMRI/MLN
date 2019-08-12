import tensorflow as tf
import pdb
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy
import myParams


def getHome():
	# return '/home/deni/'
	# return '/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/'
# 	return '/media/a/H2/home/a/'
    return '/opt/data/'

def getDatasetsBase():
	# return '/home/deni/'
	return '/media/a/H1/TFDatasets/'

def getParam_tmpF(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            try:
                return np.array(list(map(int, s.split(','))))
            except ValueError:
                try:
                    return np.array(list(map(float, s.split(','))))
                except ValueError:
                    return s
            
def readParamsTxt(ParamFN):
    ParamsD = {}
    with open(ParamFN) as f:
        for line in f:
            if len(line)<3:
                continue
            # print(line)
            #print(line.replace("\n",""))
            (key,val,X)=(line+' a').split(maxsplit=2)
#             (key, val) = line.split()
            valx=getParam_tmpF(val)
            ParamsD[key] = valx
            myParams.myDict[key]=ParamsD[key]
#             print(key + " : " + str(val) + " " + type(valx).__name__)

def getparam(S):
    try:
        return myParams.myDict[S]
    except ValueError:
        print('Couldnt find parameter: ' + S)
        return 0
    
def setparam(S,V):
    myParams.myDict[S]=V
    return
    
def ConcatCOnDim(X,dim):
#     return tf.cast(tf.concat([tf.real(X),tf.imag(X)],axis=dim),tf.float32)
    return tf.concat([tf.real(X),tf.imag(X)],axis=dim)

def ConcatRIOn0(X): return tf.concat([tf.real(X),tf.imag(X)],axis=0)
def ConcatRIOn1(X): return tf.concat([tf.real(X),tf.imag(X)],axis=1)
def ConcatRIOn2(X): return tf.concat([tf.real(X),tf.imag(X)],axis=2)
def ConcatRIOn3(X): return tf.concat([tf.real(X),tf.imag(X)],axis=3)
def ConcatRIOn4(X): return tf.concat([tf.real(X),tf.imag(X)],axis=4)
def ConcatRIOn5(X): return tf.concat([tf.real(X),tf.imag(X)],axis=5)
def ConcatRIOn6(X): return tf.concat([tf.real(X),tf.imag(X)],axis=6)
def ConcatRIOn7(X): return tf.concat([tf.real(X),tf.imag(X)],axis=7)

def ConcatCOnDimWithStack(X,dim):
#     return tf.cast(tf.concat([tf.stack([tf.real(X)],axis=dim),tf.stack([tf.imag(X)],axis=dim)],axis=dim),tf.float32)
    return tf.concat([tf.stack([tf.real(X)],axis=dim),tf.stack([tf.imag(X)],axis=dim)],axis=dim)

def NP_ConcatCOnDim(X,dim):
    return np.float32(np.concatenate((np.real(X),np.imag(X)),axis=dim))

def NP_ConcatRIOn0(X): return NP_ConcatCOnDim(X,0)
def NP_ConcatRIOn1(X): return NP_ConcatCOnDim(X,1)
def NP_ConcatRIOn2(X): return NP_ConcatCOnDim(X,2)
def NP_ConcatRIOn3(X): return NP_ConcatCOnDim(X,3)
def NP_ConcatRIOn4(X): return NP_ConcatCOnDim(X,4)
def NP_ConcatRIOn5(X): return NP_ConcatCOnDim(X,5)
def NP_ConcatRIOn6(X): return NP_ConcatCOnDim(X,6)

def NP_fft2d_on6d(X): return np.transpose(np.fft.fft2(np.transpose(X,(2,3,4,5,0,1))),(4,5,0,1,2,3))
def NP_ifft2d_on6d(X): return np.transpose(np.fft.ifft2(np.transpose(X,(2,3,4,5,0,1))),(4,5,0,1,2,3))

# def RItoCon4(X):
#     return tf.squeeze(tf.complex(tf.slice(X,[0,0,0,0],[-1,-1,-1,1]),tf.slice(X,[0,0,0,1],[-1,-1,-1,1])))

# def RItoCon4(X):
#     return tf.squeeze(tf.complex(tf.slice(X,[0,0,0,0],[batch_size,H,W,1]),tf.slice(X,[0,0,0,1],[batch_size,H,W,1])))

def NP_addDim(X): return np.stack([X],axis=-1)
def TF_addDim(X): return tf.stack([X],axis=-1)

def TF_2d_to_3d(X): return tf.stack([X],axis=2)
def TF_3d_to_4d(X): return tf.stack([X],axis=3)
def TF_4d_to_5d(X): return tf.stack([X],axis=4)
def TF_5d_to_6d(X): return tf.stack([X],axis=5)

def TF_2d_to_4d(X): return TF_3d_to_4d(TF_2d_to_3d(X))
def TF_2d_to_5d(X): return TF_4d_to_5d(TF_3d_to_4d(TF_2d_to_3d(X)))

def TF_3d_to_5d(X): return TF_4d_to_5d(TF_3d_to_4d(X))

def TF_fft2d_on5d(X): return tf.transpose(tf.fft2d(tf.transpose(X,[2,3,4,0,1])),[3,4,0,1,2])
def TF_ifft2d_on5d(X): return tf.transpose(tf.ifft2d(tf.transpose(X,[2,3,4,0,1])),[3,4,0,1,2])
def TF_fft2d_on6d(X): return tf.transpose(tf.fft2d(tf.transpose(X,[2,3,4,5,0,1])),[4,5,0,1,2,3])
def TF_ifft2d_on6d(X): return tf.transpose(tf.ifft2d(tf.transpose(X,[2,3,4,5,0,1])),[4,5,0,1,2,3])
def TF_fft2d_on7d(X): return tf.transpose(tf.fft2d(tf.transpose(X,[2,3,4,5,6,0,1])),[5,6,0,1,2,3,4])
def TF_ifft2d_on7d(X): return tf.transpose(tf.ifft2d(tf.transpose(X,[2,3,4,5,6,0,1])),[5,6,0,1,2,3,4])

def TF_fft2d_onNd(X,N): return tf.transpose(tf.fft2d(tf.transpose(X,np.concatenate((np.arange(2,N),[0,1]),axis=0))),np.concatenate(([N-2,N-1],np.arange(0,N-2)),axis=0))
def TF_ifft2d_onNd(X,N): return tf.transpose(tf.ifft2d(tf.transpose(X,np.concatenate((np.arange(2,N),[0,1]),axis=0))),np.concatenate(([N-2,N-1],np.arange(0,N-2)),axis=0))

def TF_fft2d_on3d(X): return tf.transpose(tf.fft2d(tf.transpose(X,[2,0,1])),[1,2,0])
def TF_ifft2d_on3d(X): return tf.transpose(tf.ifft2d(tf.transpose(X,[2,0,1])),[1,2,0])

def tfrm(X): return tf.reduce_mean(tf.abs(X))

def rms(X): return np.sqrt(np.mean(np.square(np.abs(X))))
def TF_rms(X): return tf.sqrt(tf.reduce_mean(tf.square(tf.abs(X))))

def QuickCompare(Ref,X):
    return [rms(Ref),rms(X),rms(Ref-X),rms(Ref)/rms(Ref-X)]

def toep(X,Kern,H,W):
    return np.fft.ifft2(np.fft.fft2(np.pad(X,((0,H),(0,W)),'constant'),axes=(0,1))*Kern,axes=(0,1))[:H,:W]

def TF_toep(X,Kern,H,W):
    return tf.ifft2d(tf.fft2d(tf.pad(X,((0,H),(0,W)),'constant'))*Kern)[:H,:W]

def cgp(x0, A, b, mit, stol, bbA):
# def [x, k] = cgp(x0, A, C, b, mit, stol, bbA, bbC):
# https://en.wikipedia.org/wiki/Conjugate_gradient_method#Example_code_in_MATLAB_/_GNU_Octave_2
    x = x0;
    ha = 0;
    hp = 0;
    hpp = 0;
    ra = 0;
    rp = 0;
    rpp = 0;
    u = 0;
    k = 0;

    ra = b - bbA(A, x0); # <--- ra = b - A * x0;
    while rms(ra) > stol:
        ha=ra
        k = k + 1;
        if (k == mit):
            print('GCP:MAXIT: mit reached, no conversion.');
            return x,k
        hpp = hp;
        rpp = rp;
        hp = ha;
        rp = ra;
        t = np.sum(np.conj(rp)*hp)
        if k == 1:
            u = hp;
        else:
            u = hp + (t / np.sum(np.conj(rpp)*hpp)) * u;
        Au = bbA(A, u) # <--- Au = A * u;
        Fac=np.sum(np.conj(u)*Au)
        a = t / Fac
        x = x + a * u;
        ra = rp - a * Au;
    return x,k

def TF_cgp(x0, A, b, mit, stol, bbA):
    x = x0;
    ha = 0;
    hp = 0;
    hpp = 0;
    ra = 0;
    rp = 0;
    rpp = 0;
    u = 0;
    k = 0;

    ra = b - bbA(A, x0); # <--- ra = b - A * x0;
    while TF_rms(ra) > stol:
        ha=ra
        k = k + 1;
        if (k == mit):
            print('GCP:MAXIT: mit reached, no conversion.');
            return x,k
        hpp = hp;
        rpp = rp;
        hp = ha;
        rp = ra;
        t = tf.reduce_sum(tf.conj(rp)*hp)
        if k == 1:
            u = hp;
        else:
            u = hp + (t / tf.reduce_sum(tf.conj(rpp)*hpp)) * u;
        Au = bbA(A, u) # <--- Au = A * u;
        Fac=tf.reduce_sum(tf.conj(u)*Au)
        a = t / Fac
        x = x + a * u;
        ra = rp - a * Au;
    return x,k

def NP_NUFFT_forw(X,SN,P,H,W):
    return P*np.reshape(np.fft.fft2(np.pad(X*SN,((0,H),(0,W)),'constant')),-1)
    
# def back(X,SN,P,H,W):
#     return np.fft.ifft2(np.reshape(np.conj(P.T)*X,((H*2,W*2))),axes=(0,1))[:H,:W]*np.conj(SN)
def NP_NUFFT_back(X,SN,P,H,W):
    return (np.fft.ifft2(np.reshape(np.conj(np.transpose(P))*X,(H*2,W*2)))[:H,:W])*np.conj(SN)

def NP_NUFFT_forwWback(X,Wx,SN,P,H,W):
    return NP_NUFFT_back(NP_NUFFT_forw(X,SN,P,H,W)*Wx,SN,P,H,W)

def NP_NUFFTHNUFFT_WithW(I,SN,P,CurW,H,W):
    Step1=I*SN
    Pad=np.pad(Step1,((0,H),(0,W)),'constant')
    F=np.fft.fft2(Pad)
    Col=np.reshape(F,(-1))
    Sig=P*Col

    Sig=Sig*CurW
#     Out=back(Sig,SN,P,H,W)
    Step1=np.conj(np.transpose(P))*Sig
    Step1=np.reshape(Step1,(H*2,W*2))
    F=np.fft.ifft2(Step1)
    Cropped=F[:H,:W]
    Out=Cropped*np.conj(SN)
    return Out

def NUFFT_to_ToepKern(Wx,SN,P,H,W):
    # NUFFT to ToepKern
    v11=np.zeros((H,W),np.complex128)
    v12=np.zeros((H,W),np.complex128)
    v21=np.zeros((H,W),np.complex128)
    v22=np.zeros((H,W),np.complex128)
    v11[0,0]=1
    v12[0,-1]=1
    v21[-1,0]=1
    v22[-1,-1]=1

    block11=NP_NUFFTHNUFFT_WithW(v11,SN,P,Wx,H,W)
    block12=NP_NUFFTHNUFFT_WithW(v12,SN,P,Wx,H,W)
    block21=NP_NUFFTHNUFFT_WithW(v21,SN,P,Wx,H,W)
    block22=NP_NUFFTHNUFFT_WithW(v22,SN,P,Wx,H,W)

    Big=np.zeros((H*2,W*2),np.complex128)
    Big[:H,:W]=block22;
    Big[H-1:-1,W-1:-1]=block11;
    Big[:H,W-1:-1]=block21;
    Big[H-1:-1,:W]=block12;
    Bigc=np.roll(Big,(-H+1,-W+1),(0,1))
    TKern=np.fft.fft2(Bigc)
    return TKern
    # QuickCompare(TKern,TKern1)

def _glorot_initializer_g(units, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
        stddev  = np.sqrt(stddev_factor / np.sqrt(np.prod(units)))
        return tf.truncated_normal(units,mean=0.0, stddev=stddev)
    
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
# def TS_NUFFT_OPHOP(InImage,TSCSens,H,W,batch_size,paddingsY,nTSC,nCh,fftkernc5D):
#     InImage=tf.stack([tf.stack([InImage],axis=3)],axis=4)
#     InImage=tf.transpose(InImage,[1,2,3,4,0])
#     Step1=tf.multiply(InImage,TSCSens)
#     Padded=tf.pad(Step1, paddingsY, "CONSTANT")
#     Step2=tf.transpose(tf.fft2d(tf.transpose(Padded,perm=[2,3,4,0,1])),[3,4,0,1,2])
#     Step2=tf.multiply(Step2,fftkernc5D)
#     Step2=tf.transpose(tf.ifft2d(tf.transpose(Step2,perm=[2,3,4,0,1])),[3,4,0,1,2])
#     Cropped=tf.slice(Step2,[0,0,0,0,0],[H,W,nTSC,nCh,batch_size])
#     Step3=tf.multiply(Cropped,tf.conj(TSCSens))
#     Step3=tf.reduce_sum(Step3,axis=[2,3])
#     Step3=tf.transpose(Step3,[2,0,1])
#     return Step3

def blocksToFftkern(block1,block2):
    (N1,N2)=block1.shape
    z1 = np.zeros((N1,1))
    z2 = np.zeros((N1-1,1))
    Row1=np.concatenate((block1,z1,np.conj(np.flip(np.concatenate((block1[0:1,1:],block2[1:,1:]),axis=0),axis=1))  ),axis=1)
    Row2=np.concatenate((np.flip(block2[1:,:],axis=0),z2,np.flip(np.flip(np.conj(block1[1:,1:]),axis=0),axis=1)),axis=1)
    tmp1a=np.concatenate((Row1,np.zeros((1,N2*2)),Row2),axis=0)
    tmp2a=np.conj(np.flip(np.flip(np.roll(np.roll(tmp1a,-1,axis=0),-1,axis=1),axis=0),axis=1))
    kern=(tmp1a+tmp2a)/2
    fftkerna=np.fft.fft2(kern)
    fftkerna=np.real(fftkerna)
    return fftkerna

def GetTSCoeffsByLinear(N,L):
    M=np.zeros((N,L))
    Ttimes=np.linspace(0,1,L);
    xnew = np.linspace(0, 1, N)

    for i in range(0,L):
#         print(i)
        tmp=np.zeros((L))
        tmp[i]=1
        f=scipy.interpolate.interp1d(Ttimes,tmp)
        M[:,i]=f(xnew)
    return M

def NP_Cartesian_OPHOP_ITS_MB(InImage,Sens6,Msk):
    # InImage is batch_size,H,W,nTSC,MB
    # Sens6 is H,W,/nTSC/,nCh,MB,batch_size
    InImage=NP_addDim(InImage)
    InImage=np.transpose(InImage,(1,2,3,5,4,0)) # H,W,nTSC,/nCh/,MB,batch_size
    Step1=InImage*Sens6 # H,W,nTSC,nCh,MB,batch_size
    F=NP_fft2d_on6d(Step1)
    MF=F*Msk
    IMF=NP_ifft2d_on6d(MF)
    SIMF=IMF*np.conj(Sens6)
    Step2=np.sum(SIMF,axis=3) # H,W,nTSC,MB,batch_size
    Step3=np.transpose(Step2,(4,0,1,2,3)) # batch_size,H,W,nTSC,MB
    return Step3 # batch_size,H,W,nTSC,MB

def Cartesian_OPHOP_ITS_MB(InImage,Sens6,Msk):
    # InImage is batch_size,H,W,nTSC,MB
    # Sens6 is H,W,/nTSC/,nCh,MB,batch_size
    InImage=TF_addDim(InImage)
    InImage=tf.transpose(InImage,[1,2,3,5,4,0]) # H,W,nTSC,/nCh/,MB,batch_size
    Step1=InImage*Sens6 # H,W,nTSC,nCh,MB,batch_size
    F=TF_fft2d_on6d(Step1)
    MF=F*Msk
    IMF=TF_ifft2d_on6d(MF)
    SIMF=IMF*tf.conj(Sens6)
    Step2=tf.reduce_sum(SIMF,axis=[3]) # H,W,nTSC,MB,batch_size
    Step3=tf.transpose(Step2,[4,0,1,2,3]) # batch_size,H,W,nTSC,MB
    return Step3 # batch_size,H,W,nTSC,MB

def TS_NUFFT_OPHOP_ITS_MB(InImage,Sens6,H,W,batch_size,paddingsYMB,nTSC,nCh,fftkernc7):
    # InImage is batch_size,H,W,nTSC,MB
    # Sens6 is H,W,/nTSC/,nCh,MB,batch_size
    # fftkernc7 is # H*2,W*2,nTSC,/nCh/,MB,/batch_size/,MBaux
    InImage=TF_addDim(InImage) # batch_size,H,W,nTSC,MB,/nCh/
    InImage=tf.transpose(InImage,[1,2,3,5,4,0]) # H,W,nTSC,/nCh/,MB,batch_size
    Step1=InImage*Sens6 # H,W,nTSC,nCh,MB,batch_size
    Padded=tf.pad(Step1, paddingsYMB, "CONSTANT") # H*2,W*2,nTSC,nCh,MB,batch_size
    Step2=TF_fft2d_on6d(Padded) # H*2,W*2,nTSC,nCh,MB,batch_size
    Step2=TF_addDim(Step2) # H*2,W*2,nTSC,nCh,MB,batch_size,/MBaux/
    Step2=Step2*fftkernc7 # H*2,W*2,nTSC,nCh,MB,batch_size,MBaux
    Step2=TF_ifft2d_on7d(Step2) # H*2,W*2,nTSC,nCh,MB,batch_size,MBaux
#     Cropped=tf.slice(Step2,[0,0,0,0,0],[H,W,-1,-1,-1])
    Cropped=Step2[:H,:W,:,:,:,:,:] # H,W,nTSC,nCh,MB,batch_size,MBaux
    Step3a=Cropped*tf.conj(TF_addDim(Sens6))
    Step3=tf.reduce_sum(Step3a,axis=[3,4]) # H,W,nTSC,batch_size,MBaux
    Step3=tf.transpose(Step3,[3,0,1,2,4]) # batch_size,H,W,nTSC,MB?aux?
    return Step3 # batch_size,H,W,nTSC,MB?aux?

def TS_NUFFT_OPHOP_ITS(InImage,Sens5,H,W,batch_size,paddingsY,nTSC,nCh,fftkernc5):
    # InImage is batch_size,H,W,nTSC
    # Sens5 is H,W,1,nCh,batch_size
    # fftkernc5D is H*2,W*2,nTSC,1,1
    InImage=TF_addDim(InImage) # batch_size,H,W,nTSC,1
    InImage=tf.transpose(InImage,[1,2,3,4,0]) # H,W,nTSC,1,batch_size
    Step1=InImage*Sens5 # H,W,nTSC,nCh,batch_size
    Padded=tf.pad(Step1, paddingsY, "CONSTANT") # H*2,W*2,nTSC,nCh,batch_size
    Step2=TF_fft2d_on5d(Padded)
    # Step2=tf.transpose(Step2,[1,0,2,3,4])
    Step2=Step2*fftkernc5
    # Step2=tf.transpose(Step2,[1,0,2,3,4])
    Step2=TF_ifft2d_on5d(Step2)
    Cropped=tf.slice(Step2,[0,0,0,0,0],[H,W,-1,-1,-1])
    Step3a=Cropped*tf.conj(Sens5)
    Step3=tf.reduce_sum(Step3a,axis=[3]) # H,W,nTSC,batch_size
    Step3=tf.transpose(Step3,[3,0,1,2]) # batch_size,H,W,nTSC
    return Step3 # batch_size,H,W,nTSC

def TS_NUFFT_OPHOP(InImage,TSCSens,H,W,batch_size,paddingsY,nTSC,nCh,fftkernc5D,SumOver=True):
    InImage=TF_3d_to_5d(InImage)
    InImage=tf.transpose(InImage,[1,2,3,4,0])
    Step1=tf.multiply(InImage,TSCSens)
    Padded=tf.pad(Step1, paddingsY, "CONSTANT")
    Step2=TF_fft2d_on5d(Padded)
    # Step2=tf.transpose(Step2,[1,0,2,3,4])
    Step2=tf.multiply(Step2,fftkernc5D)
    # Step2=tf.transpose(Step2,[1,0,2,3,4])
    Step2=TF_ifft2d_on5d(Step2)
    Cropped=tf.slice(Step2,[0,0,0,0,0],[H,W,nTSC,nCh,batch_size])
    Step3a=tf.multiply(Cropped,tf.conj(TSCSens))
    if SumOver:
        Step3=tf.reduce_sum(Step3a,axis=[2,3])
        Step3=tf.transpose(Step3,[2,0,1])
        return Step3
    else:
        return Step3a

def TS_NUFFT_OP(InImage,TSCSens,SNc,H,W,batch_size,paddingsX,nTraj,nTSC,nCh,sp_C,TSBFXc):
    InImage=tf.stack([tf.stack([InImage],axis=3)],axis=4)
    InImage=tf.transpose(InImage,[1,2,3,4,0])
    Step1=tf.multiply(InImage,SNc)
    Step1=tf.multiply(Step1,TSCSens)
    Step1=tf.reshape(Step1,[H,W,nTSC*nCh*batch_size])
    Padded=tf.pad(Step1, paddingsX, "CONSTANT")
    Step2a=TF_fft2d_on3d(Padded)
    Step2=tf.transpose(Step2a,[1,0,2])
    Col=tf.reshape(Step2,[-1,nTSC*nCh*batch_size])
    C=tf.sparse_tensor_dense_matmul(sp_C,Col)
    CX=tf.reshape(C,[nTraj,nTSC,nCh,batch_size])
    WithTSB=CX*TSBFXc
    WithTSBR=tf.reduce_sum(WithTSB,axis=1)
    Sig=tf.transpose(WithTSBR,[2,0,1])
    return Sig

def TS_NUFFT_OP_H(Sig,TSCSens,SNc,H,W,batch_size,paddingsX,nTraj,nTSC,nCh,sp_C,TSBFXc,SumOver=True):
    SigP=tf.transpose(tf.stack([Sig],axis=3),[1,3,2,0])
    SWithTSB=tf.multiply(tf.conj(TSBFXc),SigP)
    SWithTSB=tf.reshape(SWithTSB,[nTraj,nTSC*nCh*batch_size])
    C=tf.conj(tf.sparse_tensor_dense_matmul(sp_C,tf.conj(SWithTSB),adjoint_a=True))
#     C=tf.sparse_tensor_dense_matmul(sp_C,SWithTSB,adjoint_a=True)
    PaddedH=tf.reshape(C,[H*2,W*2,nTSC*nCh*batch_size])
    PaddedH=tf.transpose(PaddedH,[1,0,2])
    Step2=TF_ifft2d_on3d(PaddedH)*H*W*2*2
    Cropped=tf.slice(Step2,[0,0,0],[H,W,nTSC*nCh*batch_size])
    Cropped=tf.reshape(Cropped,[H,W,nTSC,nCh,batch_size])
    Step1=tf.multiply(Cropped,tf.conj(TSCSens))
    Step1=tf.multiply(Step1,tf.conj(SNc))
    if SumOver:
        yNew=tf.reduce_sum(Step1,axis=[2,3])
        yNew=tf.transpose(yNew,[2,0,1])
        return yNew
    else:
        return Step1

# def TS_NUFFT_OP_H(Sig,TSCSens,SNc,H,W,batch_size,paddingsX,nTraj,nTSC,nCh,sp_C,TSBFXc):
#     SigP=tf.transpose(tf.stack([Sig],axis=3),[1,3,2,0])
#     SWithTSB=tf.multiply(tf.conj(TSBFXc),SigP)
#     SWithTSB=tf.reshape(SWithTSB,[nTraj,nTSC*nCh*batch_size])
#     C=tf.conj(tf.sparse_tensor_dense_matmul(sp_C,tf.conj(SWithTSB),adjoint_a=True))
# #     C=tf.sparse_tensor_dense_matmul(sp_C,SWithTSB,adjoint_a=True)
#     PaddedH=tf.reshape(C,[H*2,W*2,nTSC*nCh*batch_size])
#     Step2=tf.transpose(tf.ifft(tf.transpose(tf.ifft(tf.transpose(PaddedH,perm=[2,0,1])),perm=[0,2,1])),perm=[1,2,0])*np.sqrt(2*2*H*W)
#     Cropped=tf.slice(Step2,[0,0,0],[H,W,nTSC*nCh*batch_size])
#     Cropped=tf.reshape(Cropped,[H,W,nTSC,nCh,batch_size])
#     Step1=tf.multiply(Cropped,tf.conj(TSCSens))
#     Step1=tf.multiply(Step1,tf.conj(SNc))
#     yNew=tf.reduce_sum(Step1,axis=[2,3])
#     yNew=tf.transpose(yNew,[2,0,1])
#     return yNew
    
# def TS_NUFFT_OP(InImage,TSCSens,SNc,H,W,batch_size,paddingsX,nTraj,nTSC,nCh,sp_C,TSBFXc):
#     InImage=tf.stack([tf.stack([InImage],axis=3)],axis=4)
#     InImage=tf.transpose(InImage,[1,2,3,4,0])
#     Step1=tf.multiply(InImage,SNc)
#     Step1=tf.multiply(Step1,TSCSens)
#     Step1=tf.reshape(Step1,[H,W,nTSC*nCh*batch_size])
#     Padded=tf.pad(Step1, paddingsX, "CONSTANT")
#     Step2=tf.transpose(tf.fft(tf.transpose(tf.fft(tf.transpose(Padded,perm=[2,0,1])),perm=[0,2,1])),perm=[1,2,0])/np.sqrt(2*2*H*W)
#     Col=tf.reshape(Step2,[-1,nTSC*nCh*batch_size])
#     C=tf.sparse_tensor_dense_matmul(sp_C,Col)
#     CX=tf.reshape(C,[nTraj,nTSC,nCh,batch_size])
#     WithTSB=CX*TSBFXc
#     WithTSBR=tf.reduce_sum(WithTSB,axis=1)
#     Sig=tf.transpose(WithTSBR,[2,0,1])
#     return Sig

def TF_TSNUFFT_Run_TSCin(InImage,TSCin,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX):
    # SNx=tf.reshape(SNx,[SNx.shape[0],SNx.shape[1],1])
    InImage=InImage*TSCin
    # InImage=tf.reshape(InImage,[InImage.shape[0],InImage.shape[1],1])
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

def TF_TSNUFFT_Run3(H,W,InImage,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX):
    # SNx=tf.reshape(SNx,[SNx.shape[0],SNx.shape[1],1])
    # InImage=tf.reshape(InImage,[InImage.shape[0],InImage.shape[1],1])
    Step1=tf.multiply(InImage,SNc)
    Step1=tf.reshape(Step1,[H,W,nCh*nTSC])
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

def TF_TSNUFFT_Prepare3(SN,Sens,TSBF,Kd,P):
    nTraj=TSBF.shape[1]
    nTSC=TSBF.shape[0]
    InputIShape=Sens.shape[0:2]
    nCh=Sens.shape[2]

    # TSCX=np.reshape(TSC,np.concatenate((TSC.shape,[1]),axis=0))
    SensP=np.transpose(np.reshape(Sens,np.concatenate((Sens.shape,[1]),axis=0)),(0,1,3,2))
    # SensWithTSC=SensP*TSCX
    # SensWithTSCX=np.reshape(SensWithTSC,(InputIShape[0],InputIShape[1],nCh*nTSC))
    # SNX=np.reshape(SN,np.concatenate((SN.shape,[1]),axis=0))
    SNX=NP_addDim(NP_addDim(SN))

    SensWithSN=SensP*SNX

    # SensWithTSCXWithSN=SensWithTSCX*SNX

    # SNc=tf.constant(tf.cast(SensWithTSCXWithSN,tf.complex64))
    # SNc=tf.constant(np.complex64(SensWithTSCXWithSN))
    SNc=tf.constant(np.complex64(SensWithSN))

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

def TF_TSNUFFT_Prepare2(SN,Sens,TSC,TSBF,Kd,P):
    nTraj=TSBF.shape[1]
    nTSC=TSBF.shape[0]
    InputIShape=Sens.shape[0:2]
    nCh=Sens.shape[2]
    
    # TSCX=np.reshape(TSC,np.concatenate((TSC.shape,[1]),axis=0))
    TSCX=tf.stack([TSC],axis=3)
    SensP=np.transpose(np.reshape(Sens,np.concatenate((Sens.shape,[1]),axis=0)),(0,1,3,2))
    SensPT=tf.constant(np.complex64(SensP))
    SensWithTSC=tf.multiply(SensPT,TSCX)
    SensWithTSCX=tf.reshape(SensWithTSC,[SN.shape[0],SN.shape[1],-1])
    # SensWithTSCX=np.reshape(SensWithTSC,(InputIShape[0],InputIShape[1],nCh*nTSC))
    SNX=np.reshape(SN,np.concatenate((SN.shape,[1]),axis=0))
    SNXT=tf.constant(np.complex64(SNX))
    
    SensWithTSCXWithSN=SensWithTSCX*SNXT
    
    #print('SensPT')
    #print(SensPT.shape)
    #print('TSCX')
    #print(TSCX.shape)
    #print('SensWithTSC')
    #print(SensWithTSC.shape)
    #print('SensWithTSCXWithSN')
    #print(SensWithTSCXWithSN.shape)
    
        
        
    # SNc=tf.constant(tf.cast(SensWithTSCXWithSN,tf.complex64))
    # SNc=tf.constant(np.complex64(SensWithTSCXWithSN))
    # SNc=tf.constant(SensWithTSCXWithSN)
    SNc=SensWithTSCXWithSN
    
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
    ValC=tf.constant(np.complex64(Idx[2]))
    
    sp_R = tf.SparseTensor(I2, ValR, [P.shape[0],P.shape[1]])
    sp_I = tf.SparseTensor(I2, ValI, [P.shape[0],P.shape[1]])
    sp_C = tf.SparseTensor(I2, ValC, [P.shape[0],P.shape[1]])
    
    # sp_R = tf.SparseTensor(I2, tf.cast(np.real(Idx[2]),tf.float32), [P.shape[0],P.shape[1]])
    # sp_I = tf.SparseTensor(I2, tf.cast(np.imag(Idx[2]),tf.float32), [P.shape[0],P.shape[1]])
    
    return SNc,paddings,sp_R,sp_I,TSBFX,sp_C

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

def GenerateNeighborsMapBaseExt(Traj,kMax,osN,nNeighbors):
    NMap=np.zeros([osN,osN,nNeighbors],dtype='int32')
    DMap=np.zeros([osN,osN,nNeighbors,2],dtype='float32')

    C=np.arange(-63,65)
    for i in np.arange(0,osN):
        for j in np.arange(0,osN):
            CurLoc=np.vstack([C[i], C[j]])
            D=Traj-CurLoc
            D1=np.squeeze(D[0,:])
            D2=np.squeeze(D[1,:])
            R=np.linalg.norm(D,ord=2,axis=0)/np.sqrt(2)
            Idx=np.argsort(R)
            Idxs=Idx[0:nNeighbors]
            NMap[i,j,:]=Idxs
            DMap[i,j,:,0]=D1[Idxs]
            DMap[i,j,:,1]=D2[Idxs]
            
    return NMap, DMap

def GenerateNeighborsMapBase(Traj,kMax,osN,nNeighbors):
    nTrajAct=Traj.shape[1]
    NMap=np.zeros([osN,osN,nNeighbors],dtype='int32')
#     C=linspaceWithHalfStep(-kMax,kMax,osN)
    C=np.arange(-63,65)
    for i in np.arange(0,osN):
        for j in np.arange(0,osN):
            CurLoc=np.vstack([C[i], C[j]])
            D=Traj-CurLoc
            R=np.linalg.norm(D,ord=2,axis=0)/np.sqrt(2)
            Idx=np.argsort(R)
            NMap[i,j,:]=Idx[0:nNeighbors]
    
    return NMap

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

def GenerateNeighborsMapC(Traj,kMax,osN,ncc,nChToUseInNN,nNeighbors):
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
	# NMapCR=np.concatenate((NMapCX,NMapCX+nTrajAct*ncc),axis=2)
	return NMapCX

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

def gifft_TFOn3D(x,H,dim=0):
	HalfH=H/2
	Id=np.hstack([np.arange(HalfH,H), np.arange(0,HalfH)])
	Id=Id.astype(int)
	if dim==0 :
		x = tf.transpose(x, perm=[2,1,0])
	if dim==1 :
		x = tf.transpose(x, perm=[0,2,1])
	x = tf.gather(x,Id,axis=2)
	out=tf.ifft(x)
	out=tf.multiply(out,tf.sqrt(tf.cast(H,tf.complex64)))
	out = tf.gather(out,Id,axis=2)
	if dim==0 :
		out = tf.transpose(out, perm=[2,1, 0])
	if dim==1 :
		out = tf.transpose(out, perm=[0,2,1])
	return out

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

def TFRandSumGeneralizedGaussians(Sz=[128,128],Mx=200,N=5):
    RangeL=np.divide(1,[50, 8])
    PRange=[0.4, 2]

    DRangeL=RangeL[1]-RangeL[0];
    DPRange=PRange[1]-PRange[0];

    Z=tf.zeros(Sz)

    pi = tf.constant(np.pi)

    for _ in range(N):
        Rnd=tf.random_uniform([15])
        MxHz=(Rnd[0]*2-1)*Mx
        Prnd=Rnd[1]*DPRange+PRange[0];
        CenterP=tf.reshape(Rnd[2:4]-0.5,[1,1,2,1])
        phi=Rnd[4]*2*pi;
        Lambdas=Rnd[5:7]*DRangeL+RangeL[0];

        Linx = tf.linspace(-0.5, 0.5, Sz[0])
        Liny = tf.linspace(-0.5, 0.5, Sz[1])
        X, Y = tf.meshgrid(Linx, Liny,indexing='ij')

        M=tf.stack([X,Y],axis=2)
        M=tf.stack([M],axis=3)
        MC=M-tf.tile(CenterP,[Sz[0],Sz[1],1,1])
        Lambda=tf.tile(tf.reshape(tf.divide(1,Lambdas),[1,1,2]),[Sz[0],Sz[1], 1]);
        R=tf.stack([tf.stack([tf.cos(phi),-tf.sin(phi)],axis=0),tf.stack([tf.sin(phi),tf.cos(phi)],axis=0)],axis=1)
        RM=tf.tile(tf.reshape(R,[1,1,2,2]),[Sz[0],Sz[1], 1, 1]);
        MCR=tf.squeeze(tf.reduce_sum(tf.multiply(tf.tile(MC,[1,1, 1, 2]),RM),axis=2))
        MCRN=tf.multiply(MCR,Lambda)
        SMCRR=tf.reduce_mean(tf.square(MCRN),axis=2)
        PEMCRR=tf.exp(-0.5*(tf.pow(SMCRR,Prnd)))
        NPEMCRR=PEMCRR*MxHz/tf.reduce_max(PEMCRR)
        Z=Z+NPEMCRR
    
    return Z

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

def TFMontage(X):
    QQ=tf.shape(X)
    N=np.int32(QQ[2])
    fig = plt.figure(figsize = (20,2))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, N),  # creates 2x2 grid of axes
                     axes_pad=0,  # pad between axes in inch.
                     )

    for i in range(N):
        CurI=tf.abs(tf.squeeze(tf.slice(X,[0,0,i],[-1,-1,1])))
    #     grid[i].imshow(np.random.random((10, 10)))  # The AxesGrid object work as a list of axes.
        grid[i].imshow(CurI,cmap='gray')  # The AxesGrid object work as a list of axes.
        grid[i].axis('off')
        grid[i].set_xticks([])
        grid[i].set_yticks([])

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
