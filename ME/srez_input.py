import sys

import tensorflow as tf
import pdb
import numpy as np

import myParams
import GTools as GT

import scipy.io
import h5py

import time

FLAGS = tf.app.flags.FLAGS

def setup_inputs(sess, filenames, image_size=None, capacity_factor=3, TestStuff=False):

    batch_size=myParams.myDict['batch_size']

    channelsIn=myParams.myDict['channelsIn']
    channelsOut=myParams.myDict['channelsOut']
#     DataH=myParams.myDict['DataH']
#     DataW=myParams.myDict['DataW']
    LabelsH=myParams.myDict['LabelsH']
    LabelsW=myParams.myDict['LabelsW']

    H=LabelsH
    W=LabelsW
    
    if myParams.myDict['InputMode'] == 'RegridTry3FME':

        BaseTSDataP=myParams.myDict['BaseTSDataP']
        BaseNUFTDataP=myParams.myDict['BaseNUFTDataP']

        B0Data=scipy.io.loadmat(BaseTSDataP + 'B0TS.mat')
        TSBF=B0Data['TSBF']
        print(TSBF.shape)
        
        RandomB0=myParams.myDict['RandomB0']>0
        
        nTSC=TSBF.shape[0]
        # if RandomB0:
        #     print('Random B0,TSC!!')
        #     TimePoints=tf.constant(np.float32(np.reshape(np.linspace(0.0,12.5,nTSC)*2*np.pi/1000.0,[1,1,-1])))

        #     Z=GT.TFRandSumGeneralizedGaussians([128,128],200,5)
        #     Z3=tf.stack([Z],axis=2)
        #     Rads=tf.multiply(TimePoints,Z3)
        #     TSC=tf.exp(1j*tf.cast(Rads,tf.complex64))
        # else:
        #     # TSC=B0Data['TSC']

        B0_Hz=-B0Data['B0_Hz']
        B0_Hz=tf.constant(GT.NP_addDim(np.float32(B0_Hz)))

        MB=GT.getparam('MB')

        # WhichEchosToRec=GT.getparam('WhichEchosToRec')
        # nEchos=WhichEchosToRec.shape[0]

        SensCC=scipy.io.loadmat(BaseTSDataP + 'SensCC1.mat')
        Sens=SensCC['SensCC']
        SensMsk=SensCC['SensMsk']

        SensMsk=np.reshape(SensMsk,(SensMsk.shape[0],SensMsk.shape[1],1))

        SensMskMB=tf.constant(np.complex64(SensMsk))

        TFMsk = tf.constant(np.complex64(SensMsk))
        
        print('loading images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        # DatasetMatFN=myParams.myDict['DatasetMatFN']
        # f = h5py.File(DatasetMatFN, 'r')
        
        # # nToLoad=10000
        nToLoad=myParams.myDict['nToLoad']
        # LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
        # if LoadAndRunOnData:
        #     nToLoad=3

        # I=f['HCPData'][1:nToLoad]
        # print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        GREBaseP='/opt/data/'
        SFN=GREBaseP+'All_Orientation-0x.mat'
        f = h5py.File(SFN, 'r')
        I=f['CurSetAll'][0:nToLoad]
        print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))

        B0Factor=GT.getparam('B0Factor')
        T2SFactor=GT.getparam('T2SFactor')

        TimePoints_ms=GT.getparam('TimePoints_ms')
        
        for b in range(0, MB):
            TFI = tf.constant(np.int16(I))
            Idx=tf.random_uniform([1],minval=0,maxval=I.shape[0],dtype=tf.int32)

            Data4=tf.squeeze(tf.slice(I,[Idx[0],0,0,0],[1,-1,-1,-1]),axis=0)
            Data4 = tf.image.random_flip_left_right(Data4)
            Data4 = tf.image.random_flip_up_down(Data4)

            u1=tf.random_uniform([1])
            Data4=tf.cond(u1[0]<0.5, lambda: tf.identity(Data4), lambda: tf.image.rot90(Data4))

            Data4 = tf.random_crop(Data4, [H, W, 4])

            def TFexpix(X): return tf.exp(tf.complex(tf.zeros_like(X),X))

            M=tf.slice(Data4,[0,0,0],[-1,-1,1])
            Ph=tf.slice(Data4,[0,0,1],[-1,-1,1])
            feature=tf.cast(M,tf.complex64)*TFexpix(Ph)

            feature=feature*SensMskMB[:,:,b:b+1]

            T2S_ms=tf.slice(Data4,[0,0,2],[-1,-1,1])
            T2S_ms = tf.where( T2S_ms<1.5, 10000 * tf.ones_like( T2S_ms ), T2S_ms )

            # B0_Hz=tf.slice(Data4,[0,0,3],[-1,-1,1])

            T2S_ms = tf.where( tf.is_nan(T2S_ms), 10000 * tf.ones_like( T2S_ms ), T2S_ms )
            # B0_Hz = tf.where( tf.is_nan(B0_Hz), tf.zeros_like( B0_Hz ), B0_Hz )


            # B0_Hz=B0_Hz*B0Factor
            T2S_ms=T2S_ms*T2SFactor

            urand_ms=tf.random_uniform([1])*12
            urand_sec=(tf.random_uniform([1])*2-1)*3/1000

            feature=feature*tf.cast(tf.exp(-urand_ms/T2S_ms),tf.complex64)
            feature=feature*TFexpix(2*np.pi*B0_Hz*urand_sec)

            mx=tf.reduce_max(M)
            mx=tf.maximum(mx,1)
            mx=tf.cast(mx,tf.complex64)

            feature=feature/mx

            # CurIWithPhase=feature

            LFac=myParams.myDict['RandomPhaseLinearFac']
            QFac=myParams.myDict['RandomPhaseQuadraticFac']
            SFac=myParams.myDict['RandomPhaseScaleFac']

            Q=GT.TFGenerateRandomSinPhase(H, W,LFac,QFac,SFac) # (nx=100,ny=120,LFac=5,QFac=0.1,SFac=2):
            
            CurIWithPhase=feature*tf.reshape(Q,[H,W,1])

            TSCM=tf.exp(-TimePoints_ms/T2S_ms)
            TSCP=tf.exp(1j*2*np.pi*tf.cast(B0_Hz*TimePoints_ms/1000,tf.complex64))
            TSC=tf.cast(TSCM,tf.complex64)*TSCP
            
            ITSbase=CurIWithPhase*TSC # ITSbase is H,W,nTSC

            # TSC=GT.TF_addDim(TSC)

        TimePointsForRec_ms=GT.getparam('TimePointsForRec_ms')
        nEchos=TimePointsForRec_ms.shape[0]

        TSCMForRec=tf.exp(-TimePointsForRec_ms/T2S_ms)
        TSCPForREc=tf.exp(1j*2*np.pi*tf.cast(B0_Hz*TimePointsForRec_ms/1000,tf.complex64))
        TSCForRec=tf.cast(TSCMForRec,tf.complex64)*TSCPForREc
            
            # ITSbase=CurIWithPhase*TSC # ITSbase is H,W,nTSC

        # TSCForRec=tf.constant(np.complex64(TSC[:,:,WhichEchosToRec]))
        # TSCForRec=tf.constant(np.complex64(TSC[:,:,WhichEchosToRec]))
        # TSC=tf.constant(np.complex64(TSC))

        H=LabelsH
        W=LabelsW
        
        # # TFI = tf.constant(np.int16(I))
        # # Idx=tf.random_uniform([1],minval=0,maxval=I.shape[0],dtype=tf.int32)
        # # feature=tf.slice(TFI,[Idx[0],0,0],[1,-1,-1])

        # # feature=tf.transpose(feature, perm=[1,2,0])

        # # feature = tf.image.random_flip_left_right(feature)
        # # feature = tf.image.random_flip_up_down(feature)
        # # u1=tf.random_uniform([1])
        # # feature=tf.cond(u1[0]<0.5, lambda: tf.identity(feature), lambda: tf.image.rot90(feature))
        
        # # feature = tf.random_crop(feature, [H, W, 1])

        # # feature = tf.cast(feature, tf.int32)

        # mx=tf.reduce_max(feature)
        # mx=tf.maximum(mx,1)

        # feature = tf.cast(feature/mx, tf.complex64)

        # feature=tf.multiply(feature,TFMsk)

        # LFac=myParams.myDict['RandomPhaseLinearFac']
        # QFac=myParams.myDict['RandomPhaseQuadraticFac']
        # SFac=myParams.myDict['RandomPhaseScaleFac']

        # Q=GT.TFGenerateRandomSinPhase(H, W,LFac,QFac,SFac) # (nx=100,ny=120,LFac=5,QFac=0.1,SFac=2):
        
        # CurIWithPhase=feature*tf.reshape(Q,[H,W,1])
        
        # SeveralEchos=CurIWithPhase*TSC[:,:,WhichEchosToRec]
        SeveralEchos=CurIWithPhase*TSCForRec
        SeveralEchos=tf.transpose(SeveralEchos,[0,2,1]) # H, nTSC, W
        SeveralEchos=tf.reshape(SeveralEchos,[H,W*nEchos*MB,1])
        label=GT.ConcatRIOn2(SeveralEchos)
        # ITS_RI=GT.ConcatRIOn2(ITS)

        # ITS=tf.transpose(ITSbaseMB,[0,3,2,1]) # H, nTSC, W
        # ITS=tf.reshape(ITS,[H,W*nTSC*MB,1])
        # ITS_RI=GT.ConcatRIOn2(ITS)
        # TSC

        # label=GT.ConcatRIOn2(CurIWithPhase)

        # label=tf.concat([tf.real(CurIWithPhase),tf.imag(CurIWithPhase)],axis=2)

        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Kd=NUFTData['Kd']
        P=NUFTData['P']
        SN=NUFTData['SN']
        Trajm2=NUFTData['Trajm2']

        nTraj=Trajm2.shape[1]
        nCh=Sens.shape[2]

        # if RandomB0:
        #     SNc,paddings,sp_R,sp_I,TSBFX=GT.TF_TSNUFFT_Prepare2(SN,Sens,TSC,TSBF,Kd,P)
        # else:
        SNc,paddings,sp_R,sp_I,TSBFX=GT.TF_TSNUFFT_Prepare3(SN,Sens,TSBF,Kd,P)

        print('SNc')
        print(SNc.shape)
        print('ITSbase')
        print(ITSbase.shape)
        
        def ConcatCI(X): return tf.concat([tf.real(X),tf.imag(X)],axis=0)
        
        # feature=ConcatCI(tf.reshape(tf.transpose(GT.TF_TSNUFFT_Run(CurIWithPhase,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX), perm=[1,0]),[nTraj*nCh,1,1]))
        # feature=ConcatCI(tf.reshape(tf.transpose(GT.TF_TSNUFFT_Run_TSCin(CurIWithPhase,TSCin,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX), perm=[1,0]),[nTraj*nCh,1,1]))
        # SNc1=tf.constant(np.complex64(GT.NP_addDim(SN)))
        feature=ConcatCI(tf.reshape(tf.transpose(GT.TF_TSNUFFT_Run3(H,W,GT.TF_addDim(ITSbase),SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX), perm=[1,0]),[nTraj*nCh,1,1]))
        
        # TSCX=tf.stack([TSC],axis=3)
        # SensP=np.transpose(np.reshape(Sens,np.concatenate((Sens.shape,[1]),axis=0)),(0,1,3,2))
        # SensPT=tf.constant(np.complex64(SensP))
        # TSCc=tf.constant(np.complex64(TSC))
        # TSC1D=ConcatCI(tf.reshape(TSCc,[-1,1,1]))
        # Sensc=tf.constant(np.complex64(Sens))
        # Sens1D=ConcatCI(tf.reshape(Sensc,[-1,1,1]))
#             print('TSC')
#             print(TSC.shape) # 12
#             print('Sens')
#             print(Sens.shape) # 13
        
        # feature=tf.concat([feature,TSC1D,Sens1D],axis=0)
        
        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == 'RegridTry3FME1':

        BaseTSDataP=myParams.myDict['BaseTSDataP']
        BaseNUFTDataP=myParams.myDict['BaseNUFTDataP']

        B0Data=scipy.io.loadmat(BaseTSDataP + 'B0TS.mat')
        TSBF=B0Data['TSBF']
        print(TSBF.shape)
        
        RandomB0=myParams.myDict['RandomB0']>0
        
        nTSC=TSBF.shape[0]
        if RandomB0:
            print('Random B0,TSC!!')
            TimePoints=tf.constant(np.float32(np.reshape(np.linspace(0.0,12.5,nTSC)*2*np.pi/1000.0,[1,1,-1])))

            Z=GT.TFRandSumGeneralizedGaussians([128,128],200,5)
            Z3=tf.stack([Z],axis=2)
            Rads=tf.multiply(TimePoints,Z3)
            TSC=tf.exp(1j*tf.cast(Rads,tf.complex64))
        else:
            TSC=B0Data['TSC']

        MB=GT.getparam('MB')

        WhichEchosToRec=GT.getparam('WhichEchosToRec')
        nEchos=WhichEchosToRec.shape[0]
        

        # GREBaseP='/opt/data/'
        # SFN=GREBaseP+'All_Orientation-0x.mat'
        # f = h5py.File(SFN, 'r')
        # I=f['CurSetAll'][0:nToLoad]
        # print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))

        # SendTSCest=GT.getparam('SendTSCest')>0
        # HamPow=GT.getparam('HamPow')

        # B0Factor=GT.getparam('B0Factor')
        # T2SFactor=GT.getparam('T2SFactor')

        # for b in range(0, MB):
        #     TFI = tf.constant(np.int16(I))
        #     Idx=tf.random_uniform([1],minval=0,maxval=I.shape[0],dtype=tf.int32)

        #     Data4=tf.squeeze(tf.slice(I,[Idx[0],0,0,0],[1,-1,-1,-1]),axis=0)
        #     Data4 = tf.image.random_flip_left_right(Data4)
        #     Data4 = tf.image.random_flip_up_down(Data4)

        #     u1=tf.random_uniform([1])
        #     Data4=tf.cond(u1[0]<0.5, lambda: tf.identity(Data4), lambda: tf.image.rot90(Data4))

        #     Data4 = tf.random_crop(Data4, [H, W, 4])

        #     def TFexpix(X): return tf.exp(tf.complex(tf.zeros_like(X),X))

        #     M=tf.slice(Data4,[0,0,0],[-1,-1,1])
        #     Ph=tf.slice(Data4,[0,0,1],[-1,-1,1])
        #     feature=tf.cast(M,tf.complex64)*TFexpix(Ph)

        #     feature=feature*SensMskMB[:,:,b:b+1]

        #     T2S_ms=tf.slice(Data4,[0,0,2],[-1,-1,1])
        #     T2S_ms = tf.where( T2S_ms<1.5, 10000 * tf.ones_like( T2S_ms ), T2S_ms )

        #     B0_Hz=tf.slice(Data4,[0,0,3],[-1,-1,1])

        #     T2S_ms = tf.where( tf.is_nan(T2S_ms), 10000 * tf.ones_like( T2S_ms ), T2S_ms )
        #     B0_Hz = tf.where( tf.is_nan(B0_Hz), tf.zeros_like( B0_Hz ), B0_Hz )


        #     B0_Hz=B0_Hz*B0Factor
        #     T2S_ms=T2S_ms*T2SFactor

        #     urand_ms=tf.random_uniform([1])*12
        #     urand_sec=(tf.random_uniform([1])*2-1)*3/1000

        #     feature=feature*tf.cast(tf.exp(-urand_ms/T2S_ms),tf.complex64)
        #     feature=feature*TFexpix(2*np.pi*B0_Hz*urand_sec)

        #     mx=tf.reduce_max(M)
        #     mx=tf.maximum(mx,1)
        #     mx=tf.cast(mx,tf.complex64)

        #     feature=feature/mx

        #     CurIWithPhase=feature

        #     TSCM=tf.exp(-TimePoints_ms/T2S_ms)
        #     TSCP=tf.exp(1j*2*np.pi*tf.cast(B0_Hz*TimePoints_ms/1000,tf.complex64))
        #     TSC=tf.cast(TSCM,tf.complex64)*TSCP
            
        #     ITSbase=CurIWithPhase*TSC # ITSbase is H,W,nTSC

        #     TSC=GT.TF_addDim(TSC)

        TSCForRec=tf.constant(np.complex64(TSC[:,:,WhichEchosToRec]))
        # TSC=tf.constant(np.complex64(TSC))

        SensCC=scipy.io.loadmat(BaseTSDataP + 'SensCC1.mat')
        Sens=SensCC['SensCC']
        SensMsk=SensCC['SensMsk']

        SensMsk=np.reshape(SensMsk,(SensMsk.shape[0],SensMsk.shape[1],1))

        TFMsk = tf.constant(np.complex64(SensMsk))
        
        print('loading images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        DatasetMatFN=myParams.myDict['DatasetMatFN']
        f = h5py.File(DatasetMatFN, 'r')
        
        # nToLoad=10000
        nToLoad=myParams.myDict['nToLoad']
        LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
        if LoadAndRunOnData:
            nToLoad=3

        I=f['HCPData'][1:nToLoad]
        print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        H=LabelsH
        W=LabelsW
        
        TFI = tf.constant(np.int16(I))
        Idx=tf.random_uniform([1],minval=0,maxval=I.shape[0],dtype=tf.int32)
        feature=tf.slice(TFI,[Idx[0],0,0],[1,-1,-1])

        feature=tf.transpose(feature, perm=[1,2,0])

        feature = tf.image.random_flip_left_right(feature)
        feature = tf.image.random_flip_up_down(feature)
        u1=tf.random_uniform([1])
        feature=tf.cond(u1[0]<0.5, lambda: tf.identity(feature), lambda: tf.image.rot90(feature))
        
        feature = tf.random_crop(feature, [H, W, 1])

        feature = tf.cast(feature, tf.int32)

        mx=tf.reduce_max(feature)
        mx=tf.maximum(mx,1)

        feature = tf.cast(feature/mx, tf.complex64)

        feature=tf.multiply(feature,TFMsk)

        LFac=myParams.myDict['RandomPhaseLinearFac']
        QFac=myParams.myDict['RandomPhaseQuadraticFac']
        SFac=myParams.myDict['RandomPhaseScaleFac']

        Q=GT.TFGenerateRandomSinPhase(H, W,LFac,QFac,SFac) # (nx=100,ny=120,LFac=5,QFac=0.1,SFac=2):
        
        CurIWithPhase=feature*tf.reshape(Q,[H,W,1])
        
        # SeveralEchos=CurIWithPhase*TSC[:,:,WhichEchosToRec]
        SeveralEchos=CurIWithPhase*TSCForRec
        SeveralEchos=tf.transpose(SeveralEchos,[0,2,1]) # H, nTSC, W
        SeveralEchos=tf.reshape(SeveralEchos,[H,W*nEchos*MB,1])
        label=GT.ConcatRIOn2(SeveralEchos)
        # ITS_RI=GT.ConcatRIOn2(ITS)

        # ITS=tf.transpose(ITSbaseMB,[0,3,2,1]) # H, nTSC, W
        # ITS=tf.reshape(ITS,[H,W*nTSC*MB,1])
        # ITS_RI=GT.ConcatRIOn2(ITS)
        # TSC

        # label=GT.ConcatRIOn2(CurIWithPhase)

        # label=tf.concat([tf.real(CurIWithPhase),tf.imag(CurIWithPhase)],axis=2)

        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Kd=NUFTData['Kd']
        P=NUFTData['P']
        SN=NUFTData['SN']
        Trajm2=NUFTData['Trajm2']

        nTraj=Trajm2.shape[1]
        nCh=Sens.shape[2]

        if RandomB0:
            SNc,paddings,sp_R,sp_I,TSBFX=GT.TF_TSNUFFT_Prepare2(SN,Sens,TSC,TSBF,Kd,P)
        else:
            SNc,paddings,sp_R,sp_I,TSBFX=GT.TF_TSNUFFT_Prepare(SN,Sens,TSC,TSBF,Kd,P)

        print('SNc')
        print(SNc.shape)
        
        def ConcatCI(X): return tf.concat([tf.real(X),tf.imag(X)],axis=0)
        
        feature=ConcatCI(tf.reshape(tf.transpose(GT.TF_TSNUFFT_Run(CurIWithPhase,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX), perm=[1,0]),[nTraj*nCh,1,1]))
        # feature=ConcatCI(tf.reshape(tf.transpose(GT.TF_TSNUFFT_Run_TSCin(CurIWithPhase,TSCin,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX), perm=[1,0]),[nTraj*nCh,1,1]))
        
        # TSCX=tf.stack([TSC],axis=3)
        # SensP=np.transpose(np.reshape(Sens,np.concatenate((Sens.shape,[1]),axis=0)),(0,1,3,2))
        # SensPT=tf.constant(np.complex64(SensP))
        # TSCc=tf.constant(np.complex64(TSC))
        # TSC1D=ConcatCI(tf.reshape(TSCc,[-1,1,1]))
        # Sensc=tf.constant(np.complex64(Sens))
        # Sens1D=ConcatCI(tf.reshape(Sensc,[-1,1,1]))
#             print('TSC')
#             print(TSC.shape) # 12
#             print('Sens')
#             print(Sens.shape) # 13
        
        # feature=tf.concat([feature,TSC1D,Sens1D],axis=0)
        
        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == 'Cart3SB':
#         nTSC=GT.getparam('nTimeSegments')
        MB=GT.getparam('MB')
#         MB=1

#         AcqTime_ms=20
#         TimePoints_ms=np.linspace(0.0,AcqTime_ms,nTSC)
        TimePoints_ms=GT.getparam('TimePoints_ms')
        nTSC=TimePoints_ms.shape[0]
        GT.setparam('nTimeSegments',nTSC)
#         GT.setparam('TimePoints_ms',TimePoints_ms)
            
        nCh=GT.getparam('nccToUse')
#         SnsFN='/media/a/H2/CCSensMaps.mat'
        SnsFN='/opt/data/CCSensMaps.mat'
        fS = h5py.File(SnsFN, 'r')
        SensMaps=fS['SensCC']
        SensMaps=SensMaps['real']+1j*SensMaps['imag']
        SensMapsSz=SensMaps.shape
        print('Loaded SensMaps, Shape %d %d %d %d' % (SensMapsSz[0],SensMapsSz[1],SensMapsSz[2],SensMapsSz[3]))
        SensMaps=SensMaps[:,:,:,:nCh]
        SensMaps=tf.constant(SensMaps)

        NumSensMapsInFile=SensMaps.shape[0]
        IdxS=tf.random_uniform([1],minval=0,maxval=NumSensMapsInFile,dtype=tf.int32)
        for b in range(0, MB):
            if b==1:
                IdxB2=tf.random_uniform([1],minval=12,maxval=19,dtype=tf.int32)
                IdxS=IdxS+IdxB2[0]
                IdxS=tf.cond(IdxS[0]>=NumSensMapsInFile, lambda: IdxS-NumSensMapsInFile, lambda: IdxS)
            
            Sens=tf.squeeze(tf.slice(SensMaps,[IdxS[0],0,0,0],[1,-1,-1,-1]),axis=0)

            Sens=tf.random_crop(Sens,[H,W,nCh])

            Sens = tf.image.random_flip_left_right(Sens)
            Sens = tf.image.random_flip_up_down(Sens)
            uS=tf.random_uniform([1])
            Sens=tf.cond(uS[0]<0.5, lambda: tf.identity(Sens), lambda: tf.image.rot90(Sens))
            SensMsk=tf.cast(GT.TF_addDim(tf.reduce_sum(tf.abs(Sens),axis=2)>0),tf.complex64)
            Sens=GT.TF_addDim(Sens)
            if b==0:
                SensMB=Sens
                SensMskMB=SensMsk
            else:
                SensMB=tf.concat([SensMB,Sens],axis=3) #     SensMB H W nCh MB
                SensMskMB=tf.concat([SensMskMB,SensMsk],axis=2) #     SensMskMB H W MB
        
        nToLoad=myParams.myDict['nToLoad']
        LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
        if LoadAndRunOnData:
            nToLoad=3

        print('loading images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
#         GREBaseP='/media/a/DATA/GREforBen/'
        GREBaseP='/opt/data/'
        SFN=GREBaseP+'All_Orientation-0x.mat'
        f = h5py.File(SFN, 'r')
        I=f['CurSetAll'][0:nToLoad]
        print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))

        SendTSCest=GT.getparam('SendTSCest')>0
        HamPow=GT.getparam('HamPow')

        B0Factor=GT.getparam('B0Factor')
        T2SFactor=GT.getparam('T2SFactor')

        for b in range(0, MB):
            TFI = tf.constant(np.int16(I))
            Idx=tf.random_uniform([1],minval=0,maxval=I.shape[0],dtype=tf.int32)

            Data4=tf.squeeze(tf.slice(I,[Idx[0],0,0,0],[1,-1,-1,-1]),axis=0)
            Data4 = tf.image.random_flip_left_right(Data4)
            Data4 = tf.image.random_flip_up_down(Data4)

            u1=tf.random_uniform([1])
            Data4=tf.cond(u1[0]<0.5, lambda: tf.identity(Data4), lambda: tf.image.rot90(Data4))

            Data4 = tf.random_crop(Data4, [H, W, 4])

            def TFexpix(X): return tf.exp(tf.complex(tf.zeros_like(X),X))

            M=tf.slice(Data4,[0,0,0],[-1,-1,1])
            Ph=tf.slice(Data4,[0,0,1],[-1,-1,1])
            feature=tf.cast(M,tf.complex64)*TFexpix(Ph)

            feature=feature*SensMskMB[:,:,b:b+1]

            T2S_ms=tf.slice(Data4,[0,0,2],[-1,-1,1])
            T2S_ms = tf.where( T2S_ms<1.5, 10000 * tf.ones_like( T2S_ms ), T2S_ms )

            B0_Hz=tf.slice(Data4,[0,0,3],[-1,-1,1])

            T2S_ms = tf.where( tf.is_nan(T2S_ms), 10000 * tf.ones_like( T2S_ms ), T2S_ms )
            B0_Hz = tf.where( tf.is_nan(B0_Hz), tf.zeros_like( B0_Hz ), B0_Hz )


            B0_Hz=B0_Hz*B0Factor
            T2S_ms=T2S_ms*T2SFactor

            # if SendTSCest:
            # HamPowA=10
            # HamA=np.roll(np.hamming(H),np.int32(H/2))
            # HamA=np.power(HamA,HamPowA)
            # HamXA=np.reshape(HamA,(1,H,1))
            # HamYA=np.reshape(HamA,(1,1,W))

            # B0_Hz_Smoothed=tf.transpose(tf.cast(B0_Hz,tf.complex64),[2,0,1])
            # B0_Hz_Smoothed=tf.fft2d(B0_Hz_Smoothed)
            # B0_Hz_Smoothed=B0_Hz_Smoothed*HamXA
            # B0_Hz_Smoothed=B0_Hz_Smoothed*HamYA
            # B0_Hz_Smoothed=tf.ifft2d(B0_Hz_Smoothed)
            # B0_Hz_Smoothed=tf.transpose(B0_Hz_Smoothed,[1,2,0])
            # B0_Hz_Smoothed=tf.real(B0_Hz_Smoothed)

            # B0_Hz=B0_Hz-B0_Hz_Smoothed
            if SendTSCest:
                # HamPowA=10
                HamPowA=HamPow
                HamA=np.roll(np.hamming(H),np.int32(H/2))
                HamA=np.power(HamA,HamPowA)
                HamXA=np.reshape(HamA,(1,H,1))
                HamYA=np.reshape(HamA,(1,1,W))

                B0_Hz_Smoothed=tf.transpose(tf.cast(B0_Hz,tf.complex64),[2,0,1])
                B0_Hz_Smoothed=tf.fft2d(B0_Hz_Smoothed)
                B0_Hz_Smoothed=B0_Hz_Smoothed*HamXA
                B0_Hz_Smoothed=B0_Hz_Smoothed*HamYA
                B0_Hz_Smoothed=tf.ifft2d(B0_Hz_Smoothed)
                B0_Hz_Smoothed=tf.transpose(B0_Hz_Smoothed,[1,2,0])
                B0_Hz_Smoothed=tf.real(B0_Hz_Smoothed)
                
                TSCest=tf.exp(1j*2*np.pi*tf.cast(B0_Hz_Smoothed*TimePoints_ms/1000,tf.complex64))
                TSCest=GT.TF_addDim(TSCest)


            urand_ms=tf.random_uniform([1])*12
            urand_sec=(tf.random_uniform([1])*2-1)*3/1000

            feature=feature*tf.cast(tf.exp(-urand_ms/T2S_ms),tf.complex64)
            feature=feature*TFexpix(2*np.pi*B0_Hz*urand_sec)

            mx=tf.reduce_max(M)
            mx=tf.maximum(mx,1)
            mx=tf.cast(mx,tf.complex64)

            feature=feature/mx

            CurIWithPhase=feature

            TSCM=tf.exp(-TimePoints_ms/T2S_ms)
            TSCP=tf.exp(1j*2*np.pi*tf.cast(B0_Hz*TimePoints_ms/1000,tf.complex64))
            TSC=tf.cast(TSCM,tf.complex64)*TSCP
            
            ITSbase=CurIWithPhase*TSC # ITSbase is H,W,nTSC

            TSC=GT.TF_addDim(TSC)
            ITSbase=GT.TF_addDim(ITSbase)
            if b==0:
                CurIWithPhaseMB=CurIWithPhase
                TSCMB=TSC
                ITSbaseMB=ITSbase                
                if SendTSCest:
                    TSCMBest=TSCest

            else:
                CurIWithPhaseMB=tf.concat([CurIWithPhaseMB,CurIWithPhase],axis=2) #     CurIWithPhaseMB H W MB
                TSCMB=tf.concat([TSCMB,TSC],axis=3) #     TSCMB H W nTSC MB
                ITSbaseMB=tf.concat([ITSbaseMB,ITSbase],axis=3) #     ITSbaseMB H W nTSC MB
                if SendTSCest:
                    TSCMBest=tf.stack([TSCMBest,TSCest],axis=3)
        print('ok 2')
        ITS_P=tf.transpose(GT.TF_addDim(ITSbaseMB),[4,0,1,2,3]) # /batch_size/,H,W,nTSC,MB


#         Msk3=np.zeros((H,W,1,1,1,1))
#         Msk3[::3,:,:,:,:,:]=1
        Msk3=np.zeros((H,W,nTSC,1,1,1))
        # Msk3[3::6,:,::2,:,:,:]=1
        # Msk3[::6,:,1::2,:,:,:]=1
        # Msk3[::4,:,0,:,:,:]=1
        # Msk3[2::4,:,1,:,:,:]=1
        # Msk3[1::4,:,2,:,:,:]=1
        # Msk3[3::4,:,3,:,:,:]=1

        PEShifts=GT.getparam('PEShifts')
        PEJump=GT.getparam('PEJump')
        print('Using PEShifts')
        for i in range(nTSC):
            Msk3[PEShifts[i]::PEJump,:,i,:,:,:]=1

        Msk3=tf.constant(np.complex64(Msk3))
        
        GT.setparam('CartMask',Msk3)

        Sens6=SensMB[:,:,tf.newaxis,:,:,tf.newaxis] # H,W,/nTS/,nCh,MB,/batch_size/

        AHA_ITS=GT.Cartesian_OPHOP_ITS_MB(ITS_P,Sens6,Msk3)
        # new simpler approach
        if SendTSCest:
            # for sending
            TSCMBest_P=tf.transpose(GT.TF_addDim(TSCMBest),[4,0,1,2,3]) # /batch_size/,H,W,nTSC,MB
            AHA_ITS=AHA_ITS*tf.conj(TSCMBest_P)
            # for label
            # ITSbaseMB=ITSbaseMB/TSCMBest
            ITSbaseMB=ITSbaseMB*tf.conj(TSCMBest)
        
        ITS=tf.transpose(ITSbaseMB,[0,3,2,1]) # H, nTSC, W
        ITS=tf.reshape(ITS,[H,W*nTSC*MB,1])
        ITS_RI=GT.ConcatRIOn2(ITS)

        label=ITS_RI
        
        Iters=GT.getparam('Iterations')
        nIter=Iters.shape[0]
        label=tf.tile(label,[nIter+1,1,1])
        
        #         send sensitivity maps
        Sensc=SensMB
        Sens1D=GT.ConcatRIOn0(tf.reshape(Sensc,[-1,1,1]))
        feature=Sens1D
            
        #         send AHA_ITS
        AHA_ITS_1D=GT.ConcatRIOn0(tf.reshape(AHA_ITS,[-1,1,1]))
        feature=tf.concat([feature,AHA_ITS_1D],axis=0)

        if SendTSCest:
            TSCest1D=GT.ConcatRIOn0(tf.reshape(TSCMBest,[-1,1,1]))
            feature=tf.concat([feature,TSCest1D],axis=0)

        SendWarmStart=GT.getparam('SendWarmStart')>0
        if SendWarmStart:
            # ITSbaseMB H W nTSC MB
            HamW=np.roll(np.hamming(H),np.int32(H/2))
            HamXW=np.reshape(HamW,(1,1,H,1))
            HamYW=np.reshape(HamW,(1,1,1,W))
            HamXYW=HamXW*HamYW

            HamXYW=tf.constant(np.complex64(HamXYW))
            
            CurHamPow=tf.random_uniform([1])
            CurHamPow=CurHamPow[0]*5+2 # between 2 and 7 
            CurHamXYW=tf.pow(HamXYW,tf.cast(CurHamPow,tf.complex64))

            WSmoothed=tf.transpose(ITSbaseMB,[2,3,0,1])
            WSmoothed=tf.fft2d(WSmoothed)
            WSmoothed=WSmoothed*CurHamXYW
            WSmoothed=tf.ifft2d(WSmoothed)
            WSmoothed=tf.transpose(WSmoothed,[2,3,0,1])

            ITS_P=tf.transpose(GT.TF_addDim(WSmoothed),[4,0,1,2,3]) # /batch_size/,H,W,nTSC,MB
            # WarmStart=ITSbaseMB
            WarmStart=ITS_P # /batch_size/,H,W,nTSC,MB
            
            Wfac=tf.random_uniform([1])
            WfacD=0.2
            Wfac=tf.cast(1+(Wfac[0]*2-1)*WfacD,tf.complex64)

            WarmStart=WarmStart*Wfac

            WarmStart1D=GT.ConcatRIOn0(tf.reshape(WarmStart,[-1,1,1]))
            feature=tf.concat([feature,WarmStart1D],axis=0)


        # feature=tf.Print(feature,[],message='feature ' + ' shape: '+str(feature.get_shape())+' '+str(feature.dtype))
        # feature=tf.Print(feature,[],message='feature ' + ' shape: '+str(feature.get_shape())+' '+str(feature.dtype))
        # feature=tf.Print(feature,[],message='feature ' + ' shape: '+str(feature.get_shape())+' '+str(feature.dtype))
        # feature=tf.Print(feature,[],message='feature ' + ' shape: '+str(feature.get_shape())+' '+str(feature.dtype))
        # feature=tf.Print(feature,[],message='feature ' + ' shape: '+str(feature.get_shape())+' '+str(feature.dtype))
        # feature=tf.Print(feature,[],message='feature ' + ' shape: '+str(feature.get_shape())+' '+str(feature.dtype))
        
        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        return features, labels
    
    
    if myParams.myDict['InputMode'] == 'RegridTry3F_B0T2S_ITS_MB':
#         BaseNUFTDataP=GT.getparam('BaseNUFTDataP')
#         NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        
        H=LabelsH
        W=LabelsW
        
        HamPow=GT.getparam('HamPow')
        Ham=np.roll(np.hamming(H),np.int32(H/2))
        Ham=np.power(Ham,HamPow)
        HamX=np.reshape(Ham,(1,H,1))
        HamY=np.reshape(Ham,(1,1,W))

        kMax=GT.getparam('kMax')
        aDataH=GT.getparam('aDataH')
        nNeighbors=GT.getparam('nNeighbors')
        
        NUFTDataFN=GT.getparam('NUFTDataFN')
#         NUFTData=scipy.io.loadmat('/media/a/DATA/TrajForNUFTm_6TI.mat')
#         NUFTData=scipy.io.loadmat('/media/a/H2/home/a/gUM/TrajForNUFT2Echo.mat')
        NUFTData=scipy.io.loadmat(NUFTDataFN)
        Kd=NUFTData['Kd']
        P=NUFTData['P']
        SN=NUFTData['SN']
        Trajm2=NUFTData['Trajm2']
        nTraj=Trajm2.shape[1]
        nTSC=GT.getparam('nTimeSegments')
        
        # MB
        MB=GT.getparam('MB')
        if MB==1:
            cCAIPIVecZ=np.ones((1,nTraj),np.complex128)
        else:
            cCAIPIVecZ=NUFTData['cCAIPIVecZ']

        ToPad=[Kd[0,0]-H,Kd[0,1]-W]
    
        paddings = tf.constant([[0, ToPad[0]], [0, ToPad[1]],[0,0]])
        paddingsX=tf.gather(paddings,[0,1,2],axis=0)
        paddingsYMB=tf.gather(paddings,[0,1,2,2,2,2],axis=0)

        AcqTime_ms=(nTraj-1)*2.5/1000.0
        TimePoints_ms=np.reshape(np.linspace(0.0,AcqTime_ms,nTSC),(1,1,nTSC))
        TSBF=GT.GetTSCoeffsByLinear(nTraj,nTSC)
        TSBF=np.transpose(TSBF,(1,0))
        
        fftkerns=np.zeros((H*2,W*2,nTSC,MB,MB),np.complex64)
        for i in range(nTSC):
            for b in range(0, MB):
                for b2 in range(0, MB):
                    CurW=TSBF[i,:]*np.conj(cCAIPIVecZ[b2,:])*cCAIPIVecZ[b,:]
                    fftkerns[:,:,i,b2,b]=GT.NUFFT_to_ToepKern(CurW,SN,P,H,W)*H*W # H W nTSC MB MBaux 

        fftkernc7=fftkerns[:,:,:,tf.newaxis,:,tf.newaxis,:] # H*2,W*2,nTSC,/nCh/,MB,/batch_size/,MBaux
        GT.setparam('paddingsYMB',paddingsYMB)
        GT.setparam('fftkernc7',fftkernc7)
        GT.setparam('Kd',Kd)
        GT.setparam('nTraj',nTraj)
        GT.setparam('TSBF',TSBF)
        GT.setparam('SN',SN)
        GT.setparam('P',P)
        GT.setparam('cCAIPIVecZ',cCAIPIVecZ)
        GT.setparam('TimePoints_ms',TimePoints_ms)
            
        SendSig=False
        if SendSig:
            NMap=GT.GenerateNeighborsMapBase(Trajm2[0:2,:],kMax,aDataH,nNeighbors)
            NMap=tf.constant(NMap)
            GT.setparam('NMap',NMap)

        nCh=GT.getparam('nccToUse')
#         SnsFN='/media/a/H2/CCSensMaps.mat'
        SnsFN='/opt/data/CCSensMaps.mat'
        fS = h5py.File(SnsFN, 'r')
        SensMaps=fS['SensCC']
        SensMaps=SensMaps['real']+1j*SensMaps['imag']
        SensMaps=SensMaps[:,:,:,:nCh]
        SensMaps=tf.constant(SensMaps)

        NumSensMapsInFile=SensMaps.shape[0]
        IdxS=tf.random_uniform([1],minval=0,maxval=NumSensMapsInFile,dtype=tf.int32)
        for b in range(0, MB):
            if b==1:
                IdxB2=tf.random_uniform([1],minval=12,maxval=19,dtype=tf.int32)
                IdxS=IdxS+IdxB2[0]
                IdxS=tf.cond(IdxS[0]>=NumSensMapsInFile, lambda: IdxS-NumSensMapsInFile, lambda: IdxS)
            
            Sens=tf.squeeze(tf.slice(SensMaps,[IdxS[0],0,0,0],[1,-1,-1,-1]),axis=0)

            Sens=tf.random_crop(Sens,[H,W,nCh])

            Sens = tf.image.random_flip_left_right(Sens)
            Sens = tf.image.random_flip_up_down(Sens)
            uS=tf.random_uniform([1])
            Sens=tf.cond(uS[0]<0.5, lambda: tf.identity(Sens), lambda: tf.image.rot90(Sens))
            SensMsk=tf.cast(GT.TF_addDim(tf.reduce_sum(tf.abs(Sens),axis=2)>0),tf.complex64)
            Sens=GT.TF_addDim(Sens)
            if b==0:
                SensMB=Sens
                SensMskMB=SensMsk
            else:
                SensMB=tf.concat([SensMB,Sens],axis=3) #     SensMB H W nCh MB
                SensMskMB=tf.concat([SensMskMB,SensMsk],axis=2) #     SensMskMB H W MB
        
        nToLoad=myParams.myDict['nToLoad']
        LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
        if LoadAndRunOnData:
            nToLoad=3

        print('loading images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
#         GREBaseP='/media/a/DATA/GREforBen/'
        GREBaseP='/opt/data/'
        SFN=GREBaseP+'All_Orientation-0x.mat'
        f = h5py.File(SFN, 'r')
        I=f['CurSetAll'][0:nToLoad]
        print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))

        H=LabelsH
        W=LabelsW

        SendTSCest=GT.getparam('SendTSCest')>0
        
        for b in range(0, MB):
            TFI = tf.constant(np.int16(I))
            Idx=tf.random_uniform([1],minval=0,maxval=I.shape[0],dtype=tf.int32)

            Data4=tf.squeeze(tf.slice(I,[Idx[0],0,0,0],[1,-1,-1,-1]),axis=0)
            Data4 = tf.image.random_flip_left_right(Data4)
            Data4 = tf.image.random_flip_up_down(Data4)

            u1=tf.random_uniform([1])
            Data4=tf.cond(u1[0]<0.5, lambda: tf.identity(Data4), lambda: tf.image.rot90(Data4))

            Data4 = tf.random_crop(Data4, [H, W, 4])

            def TFexpix(X): return tf.exp(tf.complex(tf.zeros_like(X),X))

            M=tf.slice(Data4,[0,0,0],[-1,-1,1])
            Ph=tf.slice(Data4,[0,0,1],[-1,-1,1])
            feature=tf.cast(M,tf.complex64)*TFexpix(Ph)

            feature=feature*SensMskMB[:,:,b:b+1]

            T2S_ms=tf.slice(Data4,[0,0,2],[-1,-1,1])
            T2S_ms = tf.where( T2S_ms<1.5, 10000 * tf.ones_like( T2S_ms ), T2S_ms )

            B0_Hz=tf.slice(Data4,[0,0,3],[-1,-1,1])

            T2S_ms = tf.where( tf.is_nan(T2S_ms), 10000 * tf.ones_like( T2S_ms ), T2S_ms )
            B0_Hz = tf.where( tf.is_nan(B0_Hz), tf.zeros_like( B0_Hz ), B0_Hz )

            urand_ms=tf.random_uniform([1])*12
            urand_sec=(tf.random_uniform([1])*2-1)*3/1000

            feature=feature*tf.cast(tf.exp(-urand_ms/T2S_ms),tf.complex64)
            feature=feature*TFexpix(2*np.pi*B0_Hz*urand_sec)

            mx=tf.reduce_max(M)
            mx=tf.maximum(mx,1)
            mx=tf.cast(mx,tf.complex64)

            feature=feature/mx

            CurIWithPhase=feature

            TSCM=tf.exp(-TimePoints_ms/T2S_ms)
            TSCP=tf.exp(1j*2*np.pi*tf.cast(B0_Hz*TimePoints_ms/1000,tf.complex64))
            TSC=tf.cast(TSCM,tf.complex64)*TSCP
            
            if SendTSCest:
                HamPowA=10
                HamA=np.roll(np.hamming(H),np.int32(H/2))
                HamA=np.power(HamA,HamPowA)
                HamXA=np.reshape(HamA,(1,H,1))
                HamYA=np.reshape(HamA,(1,1,W))

                B0_Hz_Smoothed=tf.transpose(tf.cast(B0_Hz,tf.complex64),[2,0,1])
                B0_Hz_Smoothed=tf.fft2d(B0_Hz_Smoothed)
                B0_Hz_Smoothed=B0_Hz_Smoothed*HamXA
                B0_Hz_Smoothed=B0_Hz_Smoothed*HamYA
                B0_Hz_Smoothed=tf.ifft2d(B0_Hz_Smoothed)
                B0_Hz_Smoothed=tf.transpose(B0_Hz_Smoothed,[1,2,0])
                B0_Hz_Smoothed=tf.real(B0_Hz_Smoothed)
                
                TSCest=tf.exp(1j*2*np.pi*tf.cast(B0_Hz_Smoothed*TimePoints_ms/1000,tf.complex64))

            ITSbase=CurIWithPhase*TSC # ITSbase is H,W,nTSC

            TSC=GT.TF_addDim(TSC)
            ITSbase=GT.TF_addDim(ITSbase)
            if b==0:
                CurIWithPhaseMB=CurIWithPhase
                TSCMB=TSC
                ITSbaseMB=ITSbase
                if SendTSCest:
                    TSCMBest=TSCest
                
            else:
                CurIWithPhaseMB=tf.concat([CurIWithPhaseMB,CurIWithPhase],axis=2) #     CurIWithPhaseMB H W MB
                TSCMB=tf.concat([TSCMB,TSC],axis=3) #     TSCMB H W nTSC MB
                ITSbaseMB=tf.concat([ITSbaseMB,ITSbase],axis=3) #     ITSbaseMB H W nTSC MB
                if SendTSCest:
                    TSCMBest=tf.stack([TSCMBest,TSCest],axis=3)
        print('ok 2')
        ITS_P=tf.transpose(GT.TF_addDim(ITSbaseMB),[4,0,1,2,3]) # /batch_size/,H,W,nTSC,MB

        TSC6=TSCMB[:,:,:,tf.newaxis,:,tf.newaxis] # H,W,nTS,/nCh/,MB,/batch_size/
        Sens6=SensMB[:,:,tf.newaxis,:,:,tf.newaxis] # H,W,/nTS/,nCh,MB,/batch_size/

        if SendTSCest:
            TSCest6=TSCMBest[:,:,:,tf.newaxis,:,tf.newaxis] # H,W,nTS,/nCh/,MB,/batch_size/
            
        AHA_ITS=GT.TS_NUFFT_OPHOP_ITS_MB(ITS_P,Sens6,H,W,1,paddingsYMB,nTSC,nCh,fftkernc7)

        # AHA_ITS_sum=tf.reduce_sum(tf.squeeze(AHA_ITS,axis=0)*tf.conj(TSCMB),axis=2)

#         GT.TFMontage(CurIWithPhaseMB)
#         GT.TFMontage(AHA_ITS_sum)

        ITSP=tf.transpose(ITSbaseMB,[2,3,0,1])
        ITSPF=tf.fft2d(ITSP)
        ITSPF=ITSPF*HamX
        ITSPF=ITSPF*HamY
        ITSPI=tf.ifft2d(ITSPF)
        ITSb=tf.transpose(ITSPI,[2,3,0,1])
        
#         GT.TFMontage(ITSbaseMB[:,:,:,0])
#         GT.TFMontage(ITSbaseMB[:,:,:,1])
#         GT.TFMontage(ITSb[:,:,:,0])
#         GT.TFMontage(ITSb[:,:,:,1])

        ITS=tf.transpose(ITSbaseMB,[0,3,2,1]) # H, nTSC, W
        ITS=tf.reshape(ITS,[H,W*nTSC*MB,1])
        ITS_RI=GT.ConcatRIOn2(ITS)

        label=ITS_RI
        
        Iters=GT.getparam('Iterations')
        nIter=Iters.shape[0]
        label=tf.tile(label,[nIter+1,1,1])
        
#         GT.TFMontage(ITS)
        # Send sens
#         Sensc=tf.constant(np.complex64(Sens))
        Sensc=SensMB
        Sens1D=GT.ConcatRIOn0(tf.reshape(Sensc,[-1,1,1]))
        feature=Sens1D

        if SendTSCest:
            TSCest1D=GT.ConcatRIOn0(tf.reshape(TSCest6,[-1,1,1]))
            feature=tf.concat([feature,TSCest1D],axis=0)
            
        if SendSig:
            SNc,paddings,sp_R,sp_I,TSBFX,sp_C=GT.TF_TSNUFFT_Prepare2(SN,Sens,TSC,TSBF,Kd,P)

            GT.setparam('sp_C',sp_C)

            Sig=GT.TF_TSNUFFT_Run(CurIWithPhase,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX)
            Sig=tf.reshape(tf.transpose(Sig, perm=[1,0]),[nTraj*nCh,1,1])
            Sig1D=GT.ConcatRIOn0(Sig)
            feature=tf.concat([feature,Sig1D],axis=0)
        else: # Send warm start ITSb
            ITSb1D=GT.ConcatRIOn0(tf.reshape(ITSb,[-1,1,1]))
            feature=tf.concat([feature,ITSb1D],axis=0)
            
        #         send AHA_ITS
        AHA_ITS_1D=GT.ConcatRIOn0(tf.reshape(AHA_ITS,[-1,1,1]))
        feature=tf.concat([feature,AHA_ITS_1D],axis=0)

        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        return features, labels
    
    if myParams.myDict['InputMode'] == 'RegridTry3F_B0T2S_ITS':
#         BaseNUFTDataP=GT.getparam('BaseNUFTDataP')
#         NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        
        H=LabelsH
        W=LabelsW
        
        kMax=myParams.myDict['kMax']
        aDataH=myParams.myDict['aDataH']
        nNeighbors=myParams.myDict['nNeighbors']
        
        NUFTData=scipy.io.loadmat('/media/a/DATA/TrajForNUFTm_6TI.mat')
        Kd=NUFTData['Kd']
        P=NUFTData['P']
        SN=NUFTData['SN']
        Trajm2=NUFTData['Trajm2']
        Pm=NUFTData['Pm']
        nTraj=Trajm2.shape[1]
        nTSC=GT.getparam('nTimeSegments')
        
        ToPad=[Kd[0,0]-H,Kd[0,1]-W]
    
        paddings = tf.constant([[0, ToPad[0]], [0, ToPad[1]],[0,0]])
        paddingsX=tf.gather(paddings,[0,1,2],axis=0)
        paddingsY=tf.gather(paddings,[0,1,2,2,2],axis=0)

        PT=np.conj(np.transpose(P))
        PmT=np.conj(np.transpose(Pm))

        AcqTime_ms=(nTraj-1)*2.5/1000.0
        TimePoints_ms=np.reshape(np.linspace(0.0,AcqTime_ms,nTSC),(1,1,nTSC))
        TSBF=GT.GetTSCoeffsByLinear(nTraj,nTSC)
        TSBF=np.transpose(TSBF,(1,0))
        
        fftkerns=np.zeros((H*2,W*2,nTSC),np.complex64)
        for i in range(nTSC):
            CurW=TSBF[i,:]

            fftkerns[:,:,i]=GT.NUFFT_to_ToepKern(CurW,SN,P,H,W)*H*W

        fftkernc5=GT.TF_addDim(GT.TF_addDim(fftkerns))
        
        SendSig=False
        if SendSig:
            NMap=GT.GenerateNeighborsMapBase(Trajm2[0:2,:],kMax,aDataH,nNeighbors)
            NMap=tf.constant(NMap)
            GT.setparam('NMap',NMap)

        nCh=GT.getparam('nccToUse')
        SnsFN='/media/a/H2/CCSensMaps.mat'
        fS = h5py.File(SnsFN, 'r')
        SensMaps=fS['SensCC']
        SensMaps=SensMaps['real']+1j*SensMaps['imag']
        SensMaps=SensMaps[:,:,:,:nCh]
        SensMaps=tf.constant(SensMaps)
        IdxS=tf.random_uniform([1],minval=0,maxval=SensMaps.shape[0],dtype=tf.int32)
        Sens=tf.squeeze(tf.slice(SensMaps,[IdxS[0],0,0,0],[1,-1,-1,-1]),axis=0)

        Sens = tf.image.random_flip_left_right(Sens)
        Sens = tf.image.random_flip_up_down(Sens)
        uS=tf.random_uniform([1])
        Sens=tf.cond(uS[0]<0.5, lambda: tf.identity(Sens), lambda: tf.image.rot90(Sens))
        SensMsk=tf.cast(GT.TF_addDim(tf.reduce_sum(tf.abs(Sens),axis=2)>0),tf.complex64)
        TFMsk=SensMsk
        
        nToLoad=myParams.myDict['nToLoad']
        LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
        if LoadAndRunOnData:
            nToLoad=3

        print('loading images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        GREBaseP='/media/a/DATA/GREforBen/'
        SFN=GREBaseP+'All_Orientation-0x.mat'
        f = h5py.File(SFN, 'r')
        I=f['CurSetAll'][0:nToLoad]
        print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        TFI = tf.constant(np.int16(I))
        Idx=tf.random_uniform([1],minval=0,maxval=I.shape[0],dtype=tf.int32)
        
        Data4=tf.squeeze(tf.slice(I,[Idx[0],0,0,0],[1,-1,-1,-1]),axis=0)
        Data4 = tf.image.random_flip_left_right(Data4)
        Data4 = tf.image.random_flip_up_down(Data4)
        
        u1=tf.random_uniform([1])
        Data4=tf.cond(u1[0]<0.5, lambda: tf.identity(Data4), lambda: tf.image.rot90(Data4))

        Data4 = tf.random_crop(Data4, [H, W, 4])

        def TFexpix(X): return tf.exp(tf.complex(tf.zeros_like(X),X))

        M=tf.slice(Data4,[0,0,0],[-1,-1,1])
        Ph=tf.slice(Data4,[0,0,1],[-1,-1,1])
        feature=tf.cast(M,tf.complex64)*TFexpix(Ph)

        feature=feature*TFMsk
        
        T2S_ms=tf.slice(Data4,[0,0,2],[-1,-1,1])
        T2S_ms = tf.where( T2S_ms<1.5, 10000 * tf.ones_like( T2S_ms ), T2S_ms )
        
        B0_Hz=tf.slice(Data4,[0,0,3],[-1,-1,1])
        
        T2S_ms = tf.where( tf.is_nan(T2S_ms), 10000 * tf.ones_like( T2S_ms ), T2S_ms )
        B0_Hz = tf.where( tf.is_nan(B0_Hz), tf.zeros_like( B0_Hz ), B0_Hz )
        
        urand_ms=tf.random_uniform([1])*12
        urand_sec=(tf.random_uniform([1])*2-1)*3/1000

        feature=feature*tf.cast(tf.exp(-urand_ms/T2S_ms),tf.complex64)
        feature=feature*TFexpix(2*np.pi*B0_Hz*urand_sec)
        
        mx=tf.reduce_max(M)
        mx=tf.maximum(mx,1)
        mx=tf.cast(mx,tf.complex64)
        
        feature=feature/mx
        
        CurIWithPhase=feature
        
        TSCM=tf.exp(-TimePoints_ms/T2S_ms)
        TSCP=tf.exp(1j*2*np.pi*tf.cast(B0_Hz*TimePoints_ms/1000,tf.complex64))
        TSC=tf.cast(TSCM,tf.complex64)*TSCP

        TSC5=GT.TF_addDim(GT.TF_addDim(TSC)) # H,W,nTS,1,1
        Sens5=GT.TF_addDim(tf.stack([tf.cast(Sens,tf.complex64)],axis=2)) # H,W,1,nCh,1

#         LFac=myParams.myDict['RandomPhaseLinearFac']
#         QFac=myParams.myDict['RandomPhaseQuadraticFac']
#         SFac=myParams.myDict['RandomPhaseScaleFac']
#         Q=GT.TFGenerateRandomSinPhase(H, W,LFac,QFac,SFac) # (nx=100,ny=120,LFac=5,QFac=0.1,SFac=2):
#         CurIWithPhase=feature*tf.reshape(Q,[H,W,1])

#         label=tf.concat([tf.real(CurIWithPhase),tf.imag(CurIWithPhase)],axis=2)

#         CurIWithPhaseRI=GT.ConcatRIOn2(CurIWithPhase)
#         label=ConcatRIOn2(CurIWithPhase)
        
        ITSbase=CurIWithPhase*TSC # ITS is H,W,nTSC
        
#         RszFac=8
#         ITSsmallR=tf.image.resize_images(tf.real(ITS),[H/RszFac,W/RszFac],method=tf.image.ResizeMethod.BICUBIC)
#         ITSsmallI=tf.image.resize_images(tf.imag(ITS),[H/RszFac,W/RszFac],method=tf.image.ResizeMethod.BICUBIC)
#         ITSblurredR=tf.image.resize_images(ITSsmallR,[H,W])
#         ITSblurredI=tf.image.resize_images(ITSsmallI,[H,W])
#         ITSblurred=tf.complex(ITSblurredR,ITSblurredI)
# #         TFMontage(ITSblurred)
        
        ITS_P=tf.transpose(GT.TF_addDim(ITSbase),[3,0,1,2])
        AHA_ITS=GT.TS_NUFFT_OPHOP_ITS(ITS_P,Sens5,H,W,1,paddingsY,nTSC,nCh,fftkernc5)
        
        AHA_ITS_sum=tf.reduce_sum(tf.squeeze(AHA_ITS,axis=0)*tf.conj(TSC),axis=2)
#         ITSb=tf.tile(GT.TF_addDim(AHA_ITS_sum),[1,1,nTSC])

        
#         def ToepFunc(A,X): return toep(X,TKern,H,W)
        def TF_ToepFunc(A,X): return TF_toep(X,TKern,H,W)
    
        
        HamPow=GT.getparam('HamPow')
        Ham=np.roll(np.hamming(H),np.int32(H/2))
        Ham=np.power(Ham,HamPow)
        HamX=np.reshape(Ham,(1,H,1))
        HamY=np.reshape(Ham,(1,1,W))
        
        TSCP=tf.transpose(TSC,[2,0,1])
        TSCPF=tf.fft2d(TSCP)
        TSCPF=TSCPF*HamX
        TSCPF=TSCPF*HamY
        TSCPI=tf.ifft2d(TSCPF)
        TSCb=tf.transpose(TSCPI,[1,2,0])
#         TFMontage(TSCb)
        ITSb=GT.TF_addDim(AHA_ITS_sum)*TSCb

#         ITSP=tf.transpose(ITSbase,[2,0,1])
#         ITSPF=tf.fft2d(ITSP)
#         ITSPF=ITSPF*HamX
#         ITSPF=ITSPF*HamY
#         ITSPI=tf.ifft2d(ITSPF)
#         ITSb=tf.transpose(ITSPI,[1,2,0])
#         TFMontage(ITSb)
        
        ITS=tf.transpose(ITSbase,[0,2,1]) # H, nTSC, W
        ITS=tf.reshape(ITS,[H,W*nTSC,1])
        ITS_RI=GT.ConcatRIOn2(ITS)
        
        label=ITS_RI

#         TSCc=tf.constant(np.complex64(TSC))
#         TSC1D=ConcatCI(tf.reshape(TSCc,[-1,1,1]))
#         TSC1D=GT.ConcatRIOn0(tf.reshape(TSC,[-1,1,1]))
#         hide the TSC
#         TSC1D=tf.ones_like(TSC1D)
#         TSC1D=GT.ConcatRIOn0(tf.cast( tf.ones([H*W*nTSC,1,1]),tf.complex64))
    
        # Send sens
#         Sensc=tf.constant(np.complex64(Sens))
        Sensc=Sens
        Sens1D=GT.ConcatRIOn0(tf.reshape(Sensc,[-1,1,1]))
        feature=Sens1D
        
        if SendSig:
            SNc,paddings,sp_R,sp_I,TSBFX,sp_C=GT.TF_TSNUFFT_Prepare2(SN,Sens,TSC,TSBF,Kd,P)
                
            GT.setparam('Kd',Kd)
            GT.setparam('nTraj',nTraj)
            GT.setparam('TSBF',TSBF)
            GT.setparam('SN',SN)
            GT.setparam('sp_C',sp_C)

            Sig=GT.TF_TSNUFFT_Run(CurIWithPhase,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX)
            Sig=tf.reshape(tf.transpose(Sig, perm=[1,0]),[nTraj*nCh,1,1])
            Sig1D=GT.ConcatRIOn0(Sig)
            feature=tf.concat([feature,Sig1D],axis=0)
        else: # Send warm start ITSb
            ITSb1D=GT.ConcatRIOn0(tf.reshape(ITSb,[-1,1,1]))
            feature=tf.concat([feature,ITSb1D],axis=0)
            GT.setparam('paddingsY',paddingsY)
            GT.setparam('fftkernc5',fftkernc5)
        
#         send AHA_ITS
        AHA_ITS_1D=GT.ConcatRIOn0(tf.reshape(AHA_ITS,[-1,1,1]))
        feature=tf.concat([feature,AHA_ITS_1D],axis=0)

        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        return features, labels
    
    if myParams.myDict['InputMode'] == 'NUFT_B0_I_Try1':
        BaseTSDataP=myParams.myDict['BaseTSDataP']
        BaseNUFTDataP=myParams.myDict['BaseNUFTDataP']

        B0Data=scipy.io.loadmat(BaseTSDataP + 'B0TS.mat')
        # Sens=B0Data['Sens']
        TSBF=B0Data['TSBF']
        TSC=B0Data['TSC']

        SensCC=scipy.io.loadmat(BaseTSDataP + 'SensCC1.mat')
        Sens=SensCC['SensCC']
        SensMsk=SensCC['SensMsk']

        SensMsk=np.reshape(SensMsk,(SensMsk.shape[0],SensMsk.shape[1],1))

        TFMsk = tf.constant(np.complex64(SensMsk))
        
        print('loading images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        # I=scipy.io.loadmat('/media/a/H1/First3kIm256x256Magint16.mat')
        # I=I['First3kIm256x256Magint16']
        DatasetMatFN=myParams.myDict['DatasetMatFN']
        # f = h5py.File('/media/a/H1/HCPData_256x256_int16.mat', 'r')
        f = h5py.File(DatasetMatFN, 'r')
        
        # nToLoad=10000
        nToLoad=myParams.myDict['nToLoad']
        LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
        if LoadAndRunOnData:
            nToLoad=3

        I=f['HCPData'][1:nToLoad]
        print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        H=LabelsH
        W=LabelsW

        TFI = tf.constant(np.int16(I))
        Idx=tf.random_uniform([1],minval=0,maxval=I.shape[0],dtype=tf.int32)
        feature=tf.slice(TFI,[Idx[0],0,0],[1,-1,-1])

        feature=tf.transpose(feature, perm=[1,2,0])

        feature = tf.image.random_flip_left_right(feature)
        feature = tf.image.random_flip_up_down(feature)
        # u1 = tf.distributions.Uniform(low=0.0, high=1.0)
        u1=tf.random_uniform([1])
        feature=tf.cond(u1[0]<0.5, lambda: tf.identity(feature), lambda: tf.image.rot90(feature))

        feature = tf.random_crop(feature, [H, W, 1])

        feature = tf.cast(feature, tf.int32)

        mx=tf.reduce_max(feature)
        mx=tf.maximum(mx,1)

        feature = tf.cast(feature/mx, tf.complex64)

        feature=tf.multiply(feature,TFMsk)

        Q=GT.TFGenerateRandomSinPhase(H, W)
        CurIWithPhase=feature*tf.reshape(Q,[H,W,1])
        label=tf.cast(feature,tf.float32)
        # label=tf.concat([tf.real(CurIWithPhase),tf.imag(CurIWithPhase)],axis=2)

        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Kd=NUFTData['Kd']
        P=NUFTData['P']
        SN=NUFTData['SN']
        Trajm2=NUFTData['Trajm2']

        nTraj=Trajm2.shape[1]
        nCh=Sens.shape[2]
        nTSC=TSC.shape[2]

        SNc,paddings,sp_R,sp_I,TSBFX=GT.TF_TSNUFFT_Prepare(SN,Sens,TSC,TSBF,Kd,P)

        def ConcatCOnDim(X,dim): return tf.cast(tf.concat([tf.real(X),tf.imag(X)],axis=dim),tf.float32)
        
        featureC=tf.reshape(tf.transpose(GT.TF_TSNUFFT_Run(CurIWithPhase,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX), perm=[1,0]),[nTraj*nCh,1,1])
        aDataH=myParams.myDict['aDataH']
        aDataW=myParams.myDict['aDataW']
        kMax=myParams.myDict['kMax']

        nccInData=myParams.myDict['nccInData']
        ncc=myParams.myDict['nccToUse']
        nNeighbors=myParams.myDict['nNeighbors']
        
        achannelsIn=ncc*nNeighbors*2
        
        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Traj=NUFTData['Trajm2'][0:2,:]

        NMapCX=GT.GenerateNeighborsMapC(Traj,kMax,aDataH,nccInData,ncc,nNeighbors)
        NMapCX = tf.constant(NMapCX)
        
        featureA=tf.gather(featureC,NMapCX,validate_indices=None,name=None) # After 131,131,192,16
        featureA=tf.reshape(featureA,featureA.shape[0:3])
        
        G=np.mgrid[-63:65,-63:65]
        RR=np.sqrt(np.power(G[0,:,:],2)+np.power(G[1,:,:],2))
        RR3=np.reshape(RR,(128,128,1))
        RR3=np.repeat(RR3, 96,axis=2)

        RR3T=tf.constant(RR3<54,dtype=tf.complex64)
        A4=tf.multiply(featureA,RR3T)
        # feature=GT.gfft_TFOn3D(GT.gfft_TFOn3D(A4,H,dim=1),H,dim=0)
        feature=GT.gifft_TFOn3D(GT.gifft_TFOn3D(A4,H,dim=1),H,dim=0)
        
        feature=tf.concat([feature,SNc],axis=2)
        
        feature=ConcatCOnDim(feature,2)
        
        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == 'I2I_ApplySens':
        print('I2I loading labels ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        DatasetMatFN=myParams.myDict['LabelsMatFN']
        f = h5py.File(DatasetMatFN, 'r')
        
        nToLoad=myParams.myDict['nToLoad']
        LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
        if LoadAndRunOnData:
            nToLoad=3
        labels=f['Data'][1:nToLoad]
        print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        SensFN='/media/a/H2/home/a/gUM/ESensCC128.mat'
        SensCC=scipy.io.loadmat(SensFN)
        Sens=SensCC['ESensCC128']
        SensMsk=SensCC['MskS']

        SensMsk=np.reshape(SensMsk,(SensMsk.shape[0],SensMsk.shape[1],1))
        
        def ConcatCOnDim(X,dim): return tf.cast(tf.concat([tf.real(X),tf.imag(X)],axis=dim),tf.float32)

        def myrot90(X): return tf.transpose(X, perm=[1,0,2])
        
        with tf.device('/gpu:0'):
            TFL = tf.constant(np.int32(labels))
            Idx=tf.random_uniform([1],minval=0,maxval=TFL.shape[0],dtype=tf.int32)
            labelR=tf.slice(TFL,[Idx[0],0,0,0],[1,-1,-1,1])
            labelI=tf.slice(TFL,[Idx[0],0,0,1],[1,-1,-1,1])
            
            labelR=tf.cast(labelR,tf.complex64)
            labelI=tf.cast(labelI,tf.complex64)
            label=tf.cast((labelR + 1j*labelI)/30000.0, tf.complex64)
            
            myParams.myDict['channelsOut']=1
            myParams.myDict['LabelsH']=labels.shape[1]
            myParams.myDict['LabelsW']=labels.shape[2]
            myParams.myDict['DataH']=labels.shape[1]
            myParams.myDict['DataW']=labels.shape[2]

            label = tf.reshape(label, [LabelsH, LabelsW, 1])

            label = tf.image.random_flip_left_right(label)
            label = tf.image.random_flip_up_down(label)
            u1=tf.random_uniform([1])
            label=tf.cond(u1[0]<0.5, lambda: tf.identity(label), lambda: myrot90(label))

        
            TFMsk = tf.constant(np.complex64(SensMsk))
            TFSens = tf.constant(np.complex64(Sens))
        
        
            label=tf.multiply(label,TFMsk)
            feature=label

            # label=ConcatCOnDim(label,2)
            label = tf.cast(tf.abs(label),tf.float32)

            feature=tf.multiply(feature,TFSens)
            feature=ConcatCOnDim(feature,2)
        
        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == 'I2I_B0':
        print('I2I loading labels ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        DatasetMatFN=myParams.myDict['LabelsMatFN']
        f = h5py.File(DatasetMatFN, 'r')
        
        nToLoad=myParams.myDict['nToLoad']
        LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
        if LoadAndRunOnData:
            nToLoad=3
        labels=f['Data'][1:nToLoad]
        LMin=np.float32(f['Min'])
        LRange=np.float32(f['Range'])
        print('Min, Range: %f,%f' % (LMin,LRange))
        print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))

        print('I2I loading features ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        DatasetMatFN=myParams.myDict['FeaturesMatFN']
        f = h5py.File(DatasetMatFN, 'r')
        
        features=f['Data'][1:nToLoad]
        FMin=np.float32(f['Min'])
        FRange=np.float32(f['Range'])
        print('Min, Range: %f,%f' % (FMin,FRange))
        print('Loaded featuress ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        TFL = tf.constant(np.int16(labels))
        TFF = tf.constant(np.int16(features))
        Idx=tf.random_uniform([1],minval=0,maxval=TFL.shape[0],dtype=tf.int32)
        
        label=tf.slice(TFL,[Idx[0],0,0],[1,-1,-1,])
        feature=tf.slice(TFF,[Idx[0],0,0,0],[1,-1,-1,-1])

        label = tf.cast(label, tf.float32)
        feature = tf.cast(feature, tf.float32)

        label=(label*LRange/30000.0)+LMin
        feature=(feature*FRange/30000.0)+FMin

        if labels.ndim==4:
            label = tf.reshape(label, [LabelsH, LabelsW, TFL.shape[3]])
        else:
            label = tf.reshape(label, [LabelsH, LabelsW, 1])

        if features.ndim==4:
            feature = tf.reshape(feature, [LabelsH, LabelsW, TFF.shape[3]])
        else:
            feature = tf.reshape(feature, [LabelsH, LabelsW, 1])

        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == 'I2I':
        print('I2I loading labels ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        DatasetMatFN=myParams.myDict['LabelsMatFN']
        # DatasetMatFN='/media/a/H2/home/a/gUM/GRE_U1.4_Labels.mat'
        f = h5py.File(DatasetMatFN, 'r')
        
        nToLoad=myParams.myDict['nToLoad']
        LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
        if LoadAndRunOnData:
            nToLoad=3
        labels=f['labels'][1:nToLoad]
        print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))

        print('I2I loading features ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        DatasetMatFN=myParams.myDict['FeaturesMatFN']
        # DatasetMatFN='/media/a/H2/home/a/gUM/GRE_U1.4_Features.mat'
        f = h5py.File(DatasetMatFN, 'r')
        
        features=f['features'][1:nToLoad]
        print('Loaded featuress ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        TFL = tf.constant(np.int16(labels))
        TFF = tf.constant(np.int16(features))
        Idx=tf.random_uniform([1],minval=0,maxval=TFL.shape[0],dtype=tf.int32)
        # label=tf.slice(TFL,[Idx[0],0,0],[1,-1,-1])
        label=tf.slice(TFL,[Idx[0],0,0,0],[1,-1,-1,-1])
        feature=tf.slice(TFF,[Idx[0],0,0,0],[1,-1,-1,-1])

        label = tf.cast(label, tf.float32)
        feature = tf.cast(feature, tf.float32)

        if labels.ndim==4:
            label = tf.reshape(label, [LabelsH, LabelsW, TFL.shape[3]])
        else:
            label = tf.reshape(label, [LabelsH, LabelsW, 1])

        if features.ndim==4:
            feature = tf.reshape(feature, [LabelsH, LabelsW, TFF.shape[3]])
        else:
            feature = tf.reshape(feature, [LabelsH, LabelsW, 1])

        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == 'RegridTry3FMB':
        BaseTSDataP=myParams.myDict['BaseTSDataP']
        BaseNUFTDataP=myParams.myDict['BaseNUFTDataP']

        B0Data=scipy.io.loadmat(BaseTSDataP + 'B0TS.mat')
        TSBFA=B0Data['TSBFA']
        TSCA=B0Data['TSCA']
        TSBFB=B0Data['TSBFB']
        TSCB=B0Data['TSCB']

        SensCC=scipy.io.loadmat(BaseTSDataP + 'SensCC1.mat')
        SensA=SensCC['SensCCA']
        SensMskA=SensCC['SensMskA']
        SensB=SensCC['SensCCB']
        SensMskB=SensCC['SensMskB']

        SensMskA=np.reshape(SensMskA,(SensMskA.shape[0],SensMskA.shape[1],1))
        SensMskB=np.reshape(SensMskB,(SensMskB.shape[0],SensMskB.shape[1],1))

        TFMskA = tf.constant(np.complex64(SensMskA))
        TFMskB = tf.constant(np.complex64(SensMskB))
        
        print('loading images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        # f = h5py.File('/media/a/H1/HCPData_256x256_int16.mat', 'r')
        DatasetMatFN=myParams.myDict['DatasetMatFN']
        f = h5py.File(DatasetMatFN, 'r')
        
        nToLoad=myParams.myDict['nToLoad']
        # nToLoad=10000
        LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
        if LoadAndRunOnData:
            nToLoad=3

        I=f['HCPData'][1:nToLoad]
        print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        H=LabelsH
        W=LabelsW
        
        TFI = tf.constant(np.int16(I))
        IdxA=tf.random_uniform([1],minval=0,maxval=I.shape[0],dtype=tf.int32)
        IdxB=tf.random_uniform([1],minval=0,maxval=I.shape[0],dtype=tf.int32)
        featureA=tf.slice(TFI,[IdxA[0],0,0],[1,-1,-1])
        featureB=tf.slice(TFI,[IdxB[0],0,0],[1,-1,-1])

        featureA=tf.transpose(featureA, perm=[1,2,0])
        featureB=tf.transpose(featureB, perm=[1,2,0])

        featureA = tf.image.random_flip_left_right(featureA)
        featureA = tf.image.random_flip_up_down(featureA)
        u1=tf.random_uniform([1])
        featureA=tf.cond(u1[0]<0.5, lambda: tf.identity(featureA), lambda: tf.image.rot90(featureA))

        featureB = tf.image.random_flip_left_right(featureB)
        featureB = tf.image.random_flip_up_down(featureB)
        u1=tf.random_uniform([1])
        featureB=tf.cond(u1[0]<0.5, lambda: tf.identity(featureB), lambda: tf.image.rot90(featureB))
        
        featureA = tf.random_crop(featureA, [H, W, 1])
        featureB = tf.random_crop(featureB, [H, W, 1])

        featureA = tf.cast(featureA, tf.int32)
        featureB = tf.cast(featureB, tf.int32)

        mxA=tf.maximum(tf.reduce_max(featureA),1)
        mxB=tf.maximum(tf.reduce_max(featureB),1)

        featureA = tf.cast(featureA/mxA, tf.complex64)
        featureB = tf.cast(featureB/mxB, tf.complex64)

        featureA=tf.multiply(featureA,TFMskA)
        featureB=tf.multiply(featureB,TFMskB)

        LFac=myParams.myDict['RandomPhaseLinearFac']
        QFac=myParams.myDict['RandomPhaseQuadraticFac']
        SFac=myParams.myDict['RandomPhaseScaleFac']

        QA=GT.TFGenerateRandomSinPhase(H, W,LFac,QFac,SFac) # (nx=100,ny=120,LFac=5,QFac=0.1,SFac=2):
        QB=GT.TFGenerateRandomSinPhase(H, W,LFac,QFac,SFac)
        CurIWithPhaseA=featureA*tf.reshape(QA,[H,W,1])
        CurIWithPhaseB=featureB*tf.reshape(QB,[H,W,1])
        
        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Kd=NUFTData['Kd']
        P=NUFTData['P']
        SN=NUFTData['SN']
        Trajm2=NUFTData['Trajm2']

        nTraj=Trajm2.shape[1]
        nCh=SensA.shape[2]
        nTSC=TSCA.shape[2]

        # ggg Arrived till here. CAIPI supposed to be into TSB anyway
        SNcA,paddings,sp_R,sp_I,TSBFXA=GT.TF_TSNUFFT_Prepare(SN,SensA,TSCA,TSBFA,Kd,P)
        SNcB,paddings,sp_R,sp_I,TSBFXB=GT.TF_TSNUFFT_Prepare(SN,SensB,TSCB,TSBFB,Kd,P)

        def ConcatCI(X): return tf.concat([tf.real(X),tf.imag(X)],axis=0)
        def ConcatCIOn2(X): return tf.concat([tf.real(X),tf.imag(X)],axis=2)

        if myParams.myDict['BankSize']>0:
            BankSize=myParams.myDict['BankSize']
            BankK=myParams.myDict['BankK']
            label_indexes = tf.constant(np.int32(np.arange(0,BankSize)),dtype=tf.int32)
            BankK_indexes = tf.constant(np.int32(np.arange(0,BankSize*BankK)),dtype=tf.int32)

            Bankdataset = tf.data.Dataset.from_tensor_slices(label_indexes)
            Bankdataset = Bankdataset.repeat(count=None)
            Bankiter = Bankdataset.make_one_shot_iterator()
            label_index = Bankiter.get_next()
            label_index=tf.cast(label_index,tf.int32)
            label_index=label_index*2

            BankKdataset = tf.data.Dataset.from_tensor_slices(BankK_indexes)
            BankKdataset = BankKdataset.repeat(count=None)
            BankKiter = BankKdataset.make_one_shot_iterator()
            label_indexK = BankKiter.get_next()
            label_indexK=tf.cast(label_indexK,tf.int32)
            label_indexK=label_indexK*2

            IdxAX=tf.random_uniform([1],minval=0,maxval=BankSize,dtype=tf.int32)
            IdxBX=tf.random_uniform([1],minval=0,maxval=BankSize,dtype=tf.int32)

            with tf.device('/gpu:0'):
                OnlyTakeFromBank=tf.greater(label_indexK,label_index)

                with tf.variable_scope("aaa", reuse=True):
                    Bank=tf.get_variable("Bank",dtype=tf.float32)
                    LBank=tf.get_variable("LBank",dtype=tf.float32)

                def f2(): return tf.scatter_nd_update(Bank,[[label_index],[label_index+1]], [ConcatCI(tf.reshape(tf.transpose(GT.TF_TSNUFFT_Run(CurIWithPhaseA,SNcA,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFXA), perm=[1,0]),[nTraj*nCh,1,1])),ConcatCI(tf.reshape(tf.transpose(GT.TF_TSNUFFT_Run(CurIWithPhaseB,SNcB,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFXB), perm=[1,0]),[nTraj*nCh,1,1]))])
                def f2L(): return tf.scatter_nd_update(LBank,[[label_index],[label_index+1]], [ConcatCIOn2(CurIWithPhaseA),ConcatCIOn2(CurIWithPhaseB)])

                Bank = tf.cond(OnlyTakeFromBank, lambda: tf.identity(Bank), f2)
                LBank = tf.cond(OnlyTakeFromBank, lambda: tf.identity(LBank), f2L)
                
                
                IdxAF = tf.cond(OnlyTakeFromBank, lambda: tf.identity(IdxAX[0]*2), lambda: tf.identity(label_index))
                IdxBF = tf.cond(OnlyTakeFromBank, lambda: tf.identity(IdxBX[0]*2+1), lambda: tf.identity(label_index+1))
                
                # Take from bank in any case
                featureAX = tf.slice(Bank,[IdxAF,0,0,0],[1,-1,-1,-1])
                featureAX = tf.reshape(featureAX, [DataH, 1, 1])
                featureBX = tf.slice(Bank,[IdxBF,0,0,0],[1,-1,-1,-1])
                featureBX = tf.reshape(featureBX, [DataH, 1, 1])
                featureX=featureAX+featureBX # That's MB
                
                labelAX = tf.slice(LBank,[IdxAF,0,0,0],[1,-1,-1,-1])
                labelAX = tf.reshape(labelAX, [H, W, 2])
                labelBX = tf.slice(LBank,[IdxBF,0,0,0],[1,-1,-1,-1])
                labelBX = tf.reshape(labelBX, [H, W, 2])
                labelX  = tf.concat([labelAX,labelBX],axis=1);
 

            features, labels = tf.train.batch([featureX, labelX],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        else:
            featureA=GT.TF_TSNUFFT_Run(CurIWithPhaseA,SNcA,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFXA)
            featureB=GT.TF_TSNUFFT_Run(CurIWithPhaseB,SNcB,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFXB)
            feature=featureA+featureB # That's MB
            feature=tf.transpose(feature, perm=[1,0])
            F=tf.reshape(feature,[nTraj*nCh,1,1])
            feature=ConcatCI(F)

            CurIWithPhase=tf.concat([CurIWithPhaseA,CurIWithPhaseB],axis=1);
            label=tf.concat([tf.real(CurIWithPhase),tf.imag(CurIWithPhase)],axis=2)

            features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')


        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == 'RegridTry3F_B0T2S':
        AcqTime_ms=GT.getparam('AcqTime_ms')
        
        def tfrm(X): return tf.reduce_mean(tf.abs(X))
        
        BaseNUFTDataP=GT.getparam('BaseNUFTDataP')
        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Kd=NUFTData['Kd']
        P=NUFTData['P']
        SN=NUFTData['SN']
        Trajm2=NUFTData['Trajm2']
        nTraj=Trajm2.shape[1]
        nTSC=GT.getparam('nTimeSegments')

        TimePoints_ms=np.reshape(np.linspace(0.0,AcqTime_ms,nTSC),(1,1,nTSC))
        
        TSBF=GT.GetTSCoeffsByLinear(nTraj,nTSC)
        TSBF=np.transpose(TSBF,(1,0))

        BaseTSDataP=myParams.myDict['BaseTSDataP']
        SensCC=scipy.io.loadmat(BaseTSDataP + 'SensCC1.mat')
        Sens=SensCC['SensCC']
        SensMsk=SensCC['SensMsk']
        nCh=Sens.shape[2]

        SensMsk=np.reshape(SensMsk,(SensMsk.shape[0],SensMsk.shape[1],1))

        TFMsk = tf.constant(np.complex64(SensMsk))
        
        # nToLoad=10000
        nToLoad=myParams.myDict['nToLoad']
        LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
        if LoadAndRunOnData:
            nToLoad=3

        print('loading images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        GREBaseP='/media/a/DATA/GREforBen/'
        SFN=GREBaseP+'All_Orientation-0x.mat'
        f = h5py.File(SFN, 'r')
        I=f['CurSetAll'][0:nToLoad]
        print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        H=LabelsH
        W=LabelsW
        
        TFI = tf.constant(np.int16(I))
        Idx=tf.random_uniform([1],minval=0,maxval=I.shape[0],dtype=tf.int32)
        
        Data4=tf.squeeze(tf.slice(I,[Idx[0],0,0,0],[1,-1,-1,-1]),axis=0)
        Data4 = tf.image.random_flip_left_right(Data4)
        Data4 = tf.image.random_flip_up_down(Data4)
        
        u1=tf.random_uniform([1])
        Data4=tf.cond(u1[0]<0.5, lambda: tf.identity(Data4), lambda: tf.image.rot90(Data4))

        Data4 = tf.random_crop(Data4, [H, W, 4])

        def TFexpix(X): return tf.exp(tf.complex(tf.zeros_like(X),X))

        # M=tf.squeeze(tf.slice(Data4,[0,0,0],[-1,-1,1]))
        M=tf.slice(Data4,[0,0,0],[-1,-1,1])
        Ph=tf.slice(Data4,[0,0,1],[-1,-1,1])
        # P=tf.squeeze(tf.slice(Data4,[0,0,1],[-1,-1,1]))
        feature=tf.cast(M,tf.complex64)*TFexpix(Ph)

        feature=feature*TFMsk
        
        T2S_ms=tf.slice(Data4,[0,0,2],[-1,-1,1])
        T2S_ms = tf.where( T2S_ms<1.5, 10000 * tf.ones_like( T2S_ms ), T2S_ms )
#         T2S_ms=tf.maximum(T2S_ms,4)
        
        # B0_Hz=tf.squeeze(tf.slice(Data4,[0,0,3],[-1,-1,1]))
        B0_Hz=tf.slice(Data4,[0,0,3],[-1,-1,1])
        
        T2S_ms = tf.where( tf.is_nan(T2S_ms), 10000 * tf.ones_like( T2S_ms ), T2S_ms )
        B0_Hz = tf.where( tf.is_nan(B0_Hz), tf.zeros_like( B0_Hz ), B0_Hz )

#         T2S_ms=tf.Print(T2S_ms,[tfrm(T2S_ms)],'T2S_ms ')
#         B0_Hz=tf.Print(B0_Hz,[tfrm(B0_Hz)],'B0_Hz ')
        
        urand_ms=tf.random_uniform([1])*12
        urand_sec=(tf.random_uniform([1])*2-1)*3/1000

        feature=feature*tf.cast(tf.exp(-urand_ms/T2S_ms),tf.complex64)
        feature=feature*TFexpix(2*np.pi*B0_Hz*urand_sec)
        
        mx=tf.reduce_max(M)
        mx=tf.maximum(mx,1)
        mx=tf.cast(mx,tf.complex64)
        
        feature=feature/mx
        
        CurIWithPhase=feature
        
        TSCM=tf.exp(-TimePoints_ms/T2S_ms)
        TSCP=tf.exp(1j*2*np.pi*tf.cast(B0_Hz*TimePoints_ms/1000,tf.complex64))
        TSC=tf.cast(TSCM,tf.complex64)*TSCP

#         LFac=myParams.myDict['RandomPhaseLinearFac']
#         QFac=myParams.myDict['RandomPhaseQuadraticFac']
#         SFac=myParams.myDict['RandomPhaseScaleFac']
#         Q=GT.TFGenerateRandomSinPhase(H, W,LFac,QFac,SFac) # (nx=100,ny=120,LFac=5,QFac=0.1,SFac=2):
#         CurIWithPhase=feature*tf.reshape(Q,[H,W,1])


#         label=tf.concat([tf.real(CurIWithPhase),tf.imag(CurIWithPhase)],axis=2)

        CurIWithPhaseRI=GT.ConcatRIOn2(CurIWithPhase)
#         label=ConcatRIOn2(CurIWithPhase)
    
        TSCrep=tf.slice(TSC,[0,0,1],[-1,-1,1])
        TSCrep=TSCrep*TFMsk
        TSCrepRI=GT.ConcatRIOn2(TSCrep)
        
#         TSCrepRI=tf.Print(TSCrepRI,[tfrm(TSCrepRI)],'TSCrepRI ')
#         CurIWithPhaseRI=tf.Print(CurIWithPhaseRI,[tfrm(CurIWithPhaseRI)],'CurIWithPhaseRI ')
        
        label=tf.concat([CurIWithPhaseRI,TSCrepRI],axis=1)

        SNc,paddings,sp_R,sp_I,TSBFX,sp_C=GT.TF_TSNUFFT_Prepare2(SN,Sens,TSC,TSBF,Kd,P)
        
#         sp_C=tf.complex(sp_R,sp_I)
        
        GT.setparam('Kd',Kd)
        GT.setparam('nTraj',nTraj)
        GT.setparam('TSBF',TSBF)
#         GT.setparam('SNc',SNc)
        GT.setparam('SN',SN)
        GT.setparam('sp_C',sp_C)

        feature=GT.ConcatRIOn0(tf.reshape(tf.transpose(GT.TF_TSNUFFT_Run(CurIWithPhase,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX), perm=[1,0]),[nTraj*nCh,1,1]))

#         TSCc=tf.constant(np.complex64(TSC))
#         TSC1D=ConcatCI(tf.reshape(TSCc,[-1,1,1]))
        TSC1D=GT.ConcatRIOn0(tf.reshape(TSC,[-1,1,1]))
#         hide the TSC
        TSC1D=tf.ones_like(TSC1D)
    
        Sensc=tf.constant(np.complex64(Sens))
        Sens1D=GT.ConcatRIOn0(tf.reshape(Sensc,[-1,1,1]))

        feature=tf.concat([feature,TSC1D,Sens1D],axis=0)

        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        return features, labels
    
    if myParams.myDict['InputMode'] == 'RegridTry3F':

        BaseTSDataP=myParams.myDict['BaseTSDataP']
        BaseNUFTDataP=myParams.myDict['BaseNUFTDataP']

        B0Data=scipy.io.loadmat(BaseTSDataP + 'B0TS.mat')
        # Sens=B0Data['Sens']
        TSBF=B0Data['TSBF']
        print(TSBF.shape)
        
        RandomB0=myParams.myDict['RandomB0']>0
        
        nTSC=TSBF.shape[0]
        if RandomB0:
            print('Random B0,TSC!!')
            TimePoints=tf.constant(np.float32(np.reshape(np.linspace(0.0,12.5,nTSC)*2*np.pi/1000.0,[1,1,-1])))

            Z=GT.TFRandSumGeneralizedGaussians([128,128],200,5)
            Z3=tf.stack([Z],axis=2)
            Rads=tf.multiply(TimePoints,Z3)
            TSC=tf.exp(1j*tf.cast(Rads,tf.complex64))
        else:
            TSC=B0Data['TSC']
        

        SensCC=scipy.io.loadmat(BaseTSDataP + 'SensCC1.mat')
        Sens=SensCC['SensCC']
        SensMsk=SensCC['SensMsk']

        SensMsk=np.reshape(SensMsk,(SensMsk.shape[0],SensMsk.shape[1],1))

        TFMsk = tf.constant(np.complex64(SensMsk))
        
        print('loading images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        # I=scipy.io.loadmat('/media/a/H1/First3kIm256x256Magint16.mat')
        # I=I['First3kIm256x256Magint16']
        DatasetMatFN=myParams.myDict['DatasetMatFN']
        # f = h5py.File('/media/a/H1/HCPData_256x256_int16.mat', 'r')
        f = h5py.File(DatasetMatFN, 'r')
        
        # nToLoad=10000
        nToLoad=myParams.myDict['nToLoad']
        LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
        if LoadAndRunOnData:
            nToLoad=3

        I=f['HCPData'][1:nToLoad]
        print('Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # I=scipy.io.loadmat('/media/a/H1/First1kIm256x256Magint16.mat')
        # I=I['First1kIm256x256Magint16']

        H=LabelsH
        W=LabelsW
        
        TFI = tf.constant(np.int16(I))
        Idx=tf.random_uniform([1],minval=0,maxval=I.shape[0],dtype=tf.int32)
        feature=tf.slice(TFI,[Idx[0],0,0],[1,-1,-1])

        feature=tf.transpose(feature, perm=[1,2,0])

        feature = tf.image.random_flip_left_right(feature)
        feature = tf.image.random_flip_up_down(feature)
        # u1 = tf.distributions.Uniform(low=0.0, high=1.0)
        u1=tf.random_uniform([1])
        feature=tf.cond(u1[0]<0.5, lambda: tf.identity(feature), lambda: tf.image.rot90(feature))
        # tf.image.rot90(    image,    k=1,    name=None)

        # MYGlobalStep = tf.Variable(0, trainable=False, name='Myglobal_step')
        # MYGlobalStep = MYGlobalStep+1

        # feature=tf.cond(MYGlobalStep>0, lambda: tf.identity(feature), lambda: tf.identity(feature))
        # feature = tf.Print(feature,[MYGlobalStep,],message='MYGlobalStep:')

        # image = tf.image.random_saturation(image, .95, 1.05)
        # image = tf.image.random_brightness(image, .05)
        #image = tf.image.random_contrast(image, .95, 1.05)

        feature = tf.random_crop(feature, [H, W, 1])

        feature = tf.cast(feature, tf.int32)

        mx=tf.reduce_max(feature)
        mx=tf.maximum(mx,1)

        feature = tf.cast(feature/mx, tf.complex64)

        feature=tf.multiply(feature,TFMsk)

        LFac=myParams.myDict['RandomPhaseLinearFac']
        QFac=myParams.myDict['RandomPhaseQuadraticFac']
        SFac=myParams.myDict['RandomPhaseScaleFac']

        Q=GT.TFGenerateRandomSinPhase(H, W,LFac,QFac,SFac) # (nx=100,ny=120,LFac=5,QFac=0.1,SFac=2):
        
        #Q=GT.TFGenerateRandomSinPhase(H, W)
        CurIWithPhase=feature*tf.reshape(Q,[H,W,1])
        label=tf.concat([tf.real(CurIWithPhase),tf.imag(CurIWithPhase)],axis=2)

        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Kd=NUFTData['Kd']
        P=NUFTData['P']
        SN=NUFTData['SN']
        Trajm2=NUFTData['Trajm2']

        nTraj=Trajm2.shape[1]
        nCh=Sens.shape[2]

        if RandomB0:
            SNc,paddings,sp_R,sp_I,TSBFX=GT.TF_TSNUFFT_Prepare2(SN,Sens,TSC,TSBF,Kd,P)
        else:
            SNc,paddings,sp_R,sp_I,TSBFX=GT.TF_TSNUFFT_Prepare(SN,Sens,TSC,TSBF,Kd,P)

        print('SNc')
        print(SNc.shape)
        
        # feature=GT.TF_TSNUFFT_Run(CurIWithPhase,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX)
        # feature=tf.transpose(feature, perm=[1,0])

        # F=tf.reshape(feature,[nTraj*nCh,1,1])
        
        # feature=tf.concat([tf.real(F),tf.imag(F)],axis=0)
        def ConcatCI(X): return tf.concat([tf.real(X),tf.imag(X)],axis=0)
        # feature=ConcatCI(F)

        # feature=ConcatCI(tf.reshape(tf.transpose(GT.TF_TSNUFFT_Run(CurIWithPhase,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX), perm=[1,0]),[nTraj*nCh,1,1]))

        # ggg Signal Bank stuff:
        if myParams.myDict['BankSize']>0:
            BankSize=myParams.myDict['BankSize']
            BankK=myParams.myDict['BankK']
            label_indexes = tf.constant(np.int32(np.arange(0,BankSize)),dtype=tf.int32)
            BankK_indexes = tf.constant(np.int32(np.arange(0,BankSize*BankK)),dtype=tf.int32)

            Bankdataset = tf.data.Dataset.from_tensor_slices(label_indexes)
            Bankdataset = Bankdataset.repeat(count=None)
            Bankiter = Bankdataset.make_one_shot_iterator()
            label_index = Bankiter.get_next()
            label_index=tf.cast(label_index,tf.int32)

            BankKdataset = tf.data.Dataset.from_tensor_slices(BankK_indexes)
            BankKdataset = BankKdataset.repeat(count=None)
            BankKiter = BankKdataset.make_one_shot_iterator()
            label_indexK = BankKiter.get_next()
            label_indexK=tf.cast(label_indexK,tf.int32)

            with tf.device('/gpu:0'):
                OnlyTakeFromBank=tf.greater(label_indexK,label_index)

                with tf.variable_scope("aaa", reuse=True):
                    Bank=tf.get_variable("Bank",dtype=tf.float32)
                    LBank=tf.get_variable("LBank",dtype=tf.float32)

                def f2(): return tf.scatter_nd_update(Bank,[[label_index]], [ConcatCI(tf.reshape(tf.transpose(GT.TF_TSNUFFT_Run(CurIWithPhase,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX), perm=[1,0]),[nTraj*nCh,1,1]))])
                def f2L(): return tf.scatter_nd_update(LBank,[[label_index]], [label])

                Bank = tf.cond(OnlyTakeFromBank, lambda: tf.identity(Bank), f2)
                LBank = tf.cond(OnlyTakeFromBank, lambda: tf.identity(LBank), f2L)
                
                
                # Take from bank in any case
                featureX = tf.slice(Bank,[label_index,0,0,0],[1,-1,-1,-1])
                featureX = tf.reshape(featureX, [DataH, 1, 1])
                # featureX = tf.Print(featureX,[label_index,label_indexK],message='Taking from bank:')
                labelX = tf.slice(LBank,[label_index,0,0,0],[1,-1,-1,-1])
                labelX = tf.reshape(labelX, [H, W, 2])

            features, labels = tf.train.batch([featureX, labelX],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
            # feature = tf.cond(TakeFromBank, lambda: tf.identity(Bfeature), lambda: tf.identity(Afeature))
            # label = tf.cond(TakeFromBank, lambda: tf.identity(Blabel), lambda: tf.identity(Alabel))
        else:
            feature=ConcatCI(tf.reshape(tf.transpose(GT.TF_TSNUFFT_Run(CurIWithPhase,SNc,paddings,nTraj,nTSC,nCh,sp_R,sp_I,TSBFX), perm=[1,0]),[nTraj*nCh,1,1]))
            
            
            # TSCX=tf.stack([TSC],axis=3)
            # SensP=np.transpose(np.reshape(Sens,np.concatenate((Sens.shape,[1]),axis=0)),(0,1,3,2))
            # SensPT=tf.constant(np.complex64(SensP))
            TSCc=tf.constant(np.complex64(TSC))
            TSC1D=ConcatCI(tf.reshape(TSCc,[-1,1,1]))
            Sensc=tf.constant(np.complex64(Sens))
            Sens1D=ConcatCI(tf.reshape(Sensc,[-1,1,1]))
#             print('TSC')
#             print(TSC.shape) # 12
#             print('Sens')
#             print(Sens.shape) # 13
            
            feature=tf.concat([feature,TSC1D,Sens1D],axis=0)
#             asd
            
            features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        # ggg end Signal Bank stuff:

        tf.train.start_queue_runners(sess=sess)
        return features, labels



    if myParams.myDict['InputMode'] == 'RegridTry3M':
        Msk=scipy.io.loadmat('/media/a/DATA/meas_MID244_gBP_VD11_U19_G35S155_4min_FID22439/Sli08/Msk.mat')
        Msk=Msk['Msk']
        TFMsk = tf.constant(Msk)

        FN='/media/a/H1/meas_MID244_gBP_VD11_U19_G35S155_4min_FID22439/AllData_Sli8_6k.mat'
        if TestStuff:
            print('setup_inputs Test')
            ChunkSize=100
            ChunkSizeL=400
            FN='/media/a/H1/meas_MID244_gBP_VD11_U19_G35S155_4min_FID22439/AllData_Sli8_100.mat'
        else:
            print('setup_inputs Train')
            ChunkSize=1000
            ChunkSizeL=4000

        f = h5py.File(FN, 'r')
        print('loading Data ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        I=f['AllDatax'][:]
        print('Loaded labels ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        f.close()
        I=I.astype(np.float32)

        f = h5py.File('/media/a/H1/AllImWithPhaseComplexSingle_h5.mat', 'r')
        print('Loading labels ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        L=f['AllLh5'][0:(ChunkSizeL)]
        print('Loaded labels ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        f.close()
        L=L.astype(np.float32)

        TFI = tf.constant(I[0:ChunkSize])
        TFIb = tf.constant(I[(ChunkSize):(2*ChunkSize)])
        TFIc = tf.constant(I[(2*ChunkSize):(3*ChunkSize)])
        TFId = tf.constant(I[(3*ChunkSize):(4*ChunkSize)])

        TFL = tf.constant(L)

        # place = tf.placeholder(tf.float32, shape=(DataH, DataW, channelsIn))
        # placeL = tf.placeholder(tf.float32, shape=(LabelsH, LabelsW, channelsOut))

        Idx=tf.random_uniform([1],minval=0,maxval=ChunkSizeL,dtype=tf.int32)

        def f1(): return tf.cond(Idx[0]<ChunkSize, lambda: tf.slice(TFI,[Idx[0],0],[1,-1]), lambda: tf.slice(TFIb,[Idx[0]-ChunkSize,0],[1,-1]))
        def f2(): return tf.cond(Idx[0]<(3*ChunkSize), lambda: tf.slice(TFIc,[Idx[0]-2*ChunkSize,0],[1,-1]), lambda: tf.slice(TFId,[Idx[0]-3*ChunkSize,0],[1,-1]))
        feature=tf.cond(Idx[0]<(2*ChunkSize), f1, f2)
        # feature=tf.cond(Idx[0]<ChunkSize, lambda: tf.slice(TFI,[Idx[0],0],[1,-1]), lambda: tf.slice(TFIb,[Idx[0]-ChunkSize,0],[1,-1]))
        # feature=tf.slice(TFI,[Idx[0],0],[1,-1])


        # feature = tmp.assign(place)

        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature = tf.cast(feature, tf.float32)

        labels = tf.slice(TFL,[Idx[0],0,0,0],[1,-1,-1,-1])        
        # feature = tmpL.assign(placeL)
        labels = tf.reshape(labels, [LabelsH, LabelsW, channelsOut])
        label = tf.cast(labels, tf.float32)
        label=tf.multiply(label,TFMsk)

        # Using asynchronous queues
        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels





    if myParams.myDict['InputMode'] == 'SPEN_Local':

        SR=scipy.io.loadmat('/media/a/H1/SR.mat')
        SR=SR['SR']

        SR=np.reshape(SR,[DataH,DataH,1])
        SR=np.transpose(SR, (2,0,1))
        SR_TF=tf.constant(SR)
        
        # I=scipy.io.loadmat('/media/a/H1/First1kIm256x256Magint16.mat')
        # I=I['First1kIm256x256Magint16']
        I=scipy.io.loadmat('/media/a/H1/First3kIm256x256Magint16.mat')
        I=I['First3kIm256x256Magint16']

        TFI = tf.constant(np.float32(I))
        Idx=tf.random_uniform([1],minval=0,maxval=3000,dtype=tf.int32)
        feature=tf.slice(TFI,[Idx[0],0,0],[1,-1,-1])

        feature=tf.transpose(feature, perm=[1,2,0])

        feature = tf.random_crop(feature, [DataH, DataW, 1])

        mx=tf.reduce_max(feature)
        mx=tf.maximum(mx,1)

        feature = tf.cast(feature/mx, tf.complex64)

        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        CurIWithPhase=feature*tf.reshape(Q,[DataH,DataW,1])
        
        label=tf.concat([tf.real(CurIWithPhase),tf.imag(CurIWithPhase)],axis=2)
        
        P=tf.transpose(CurIWithPhase, perm=[2,1,0])
        F=tf.matmul(P,SR_TF)
        F=tf.transpose(F, perm=[2,1,0])
        
        SPENLocalFactor=myParams.myDict['SPENLocalFactor']
        F=GT.ExpandWithCopiesOn2(F,DataH,SPENLocalFactor)

        feature=tf.concat([tf.real(F),tf.imag(F)],axis=2)

        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == 'SPEN_FC':

        SR=scipy.io.loadmat('/media/a/H1/SR.mat')
        SR=SR['SR']

        SR=np.reshape(SR,[DataH,DataH,1])
        SR=np.transpose(SR, (2,0,1))
        SR_TF=tf.constant(SR)
        
        I=scipy.io.loadmat('/media/a/H1/First1kIm256x256Magint16.mat')
        I=I['First1kIm256x256Magint16']

        TFI = tf.constant(np.float32(I))
        Idx=tf.random_uniform([1],minval=0,maxval=1000,dtype=tf.int32)
        feature=tf.slice(TFI,[Idx[0],0,0],[1,-1,-1])

        feature=tf.transpose(feature, perm=[1,2,0])

        feature = tf.random_crop(feature, [DataH, DataW, 1])

        mx=tf.reduce_max(feature)
        mx=tf.maximum(mx,1)

        feature = tf.cast(feature/mx, tf.complex64)

        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        CurIWithPhase=feature*tf.reshape(Q,[DataH,DataW,1])
        
        label=tf.concat([tf.real(CurIWithPhase),tf.imag(CurIWithPhase)],axis=2)
        
        P=tf.transpose(CurIWithPhase, perm=[2,1,0])
        F=tf.matmul(P,SR_TF)
        F=tf.transpose(F, perm=[2,1,0])
        
        feature=tf.concat([tf.real(F),tf.imag(F)],axis=2)

        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == 'SMASH1DFTxyC':
        I=scipy.io.loadmat('/media/a/H1/First3kIm128x128MagSinglex.mat')
        I=I['First3kIm128x128MagSingle']

        Maps=scipy.io.loadmat('/media/a/H1/maps128x128x8.mat')
        Mask=Maps['Msk']
        Maps=Maps['maps']
        nChannels=8

        Mask=np.reshape(Mask,[128, 128, 1])

        Maps = tf.constant(Maps)
        Mask = tf.constant(np.float32(Mask))
        # Maps = tf.constant(np.float32(Maps))

        TFI = tf.constant(np.float32(I))
        Idx=tf.random_uniform([1],minval=0,maxval=3000,dtype=tf.int32)
        feature=tf.slice(TFI,[Idx[0],0,0],[1,-1,-1])

        feature = tf.reshape(feature, [128, 128, 1])

        feature = tf.multiply(feature,Mask)        

        feature = tf.cast(feature, tf.complex64)

        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        CurIWithPhase=feature*tf.reshape(Q,[DataH,DataW,1])

        WithPhaseAndMaps=tf.multiply(CurIWithPhase,Maps)
        
        
        label=tf.concat([tf.real(CurIWithPhase),tf.imag(CurIWithPhase)],axis=2)
        
        F=GT.gfft_TFOn3D(WithPhaseAndMaps,DataH,0)
        F=GT.gfft_TFOn3D(F,DataW,1)
        
        # now subsample 2
        F = tf.reshape(F, [64,2, 128, nChannels])
        F=tf.slice(F,[0,0,0,0],[-1,1,-1,-1])
        F = tf.reshape(F, [64, 128, nChannels])

        feature=tf.concat([tf.real(F),tf.imag(F)],axis=2)

        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == '1DFTxyCMaps':
        I=scipy.io.loadmat('/media/a/H1/First3kIm128x128MagSinglex.mat')
        I=I['First3kIm128x128MagSingle']

        Maps=scipy.io.loadmat('/media/a/H1/maps128x128x8.mat')
        Mask=Maps['Msk']
        Maps=Maps['maps']
        nChannels=8

        Mask=np.reshape(Mask,[128, 128, 1])

        Maps = tf.constant(Maps)
        Mask = tf.constant(np.float32(Mask))
        # Maps = tf.constant(np.float32(Maps))

        TFI = tf.constant(np.float32(I))
        Idx=tf.random_uniform([1],minval=0,maxval=3000,dtype=tf.int32)
        feature=tf.slice(TFI,[Idx[0],0,0],[1,-1,-1])

        feature = tf.reshape(feature, [128, 128, 1])

        feature = tf.multiply(feature,Mask)        

        feature = tf.cast(feature, tf.complex64)

        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        CurIWithPhase=feature*tf.reshape(Q,[DataH,DataW,1])

        WithPhaseAndMaps=tf.multiply(CurIWithPhase,Maps)
        
        
        label=tf.concat([tf.real(CurIWithPhase),tf.imag(CurIWithPhase)],axis=2)
        
        F=GT.gfft_TFOn3D(WithPhaseAndMaps,DataH,0)
        F=GT.gfft_TFOn3D(F,DataW,1)
        
        feature=tf.concat([tf.real(F),tf.imag(F)],axis=2)

        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == 'M2DFT':
        I=scipy.io.loadmat('/media/a/H1/First3kIm128x128MagSinglex.mat')
        I=I['First3kIm128x128MagSingle']

        TFI = tf.constant(np.float32(I))
        Idx=tf.random_uniform([1],minval=0,maxval=3000,dtype=tf.int32)
        feature=tf.slice(TFI,[Idx[0],0,0],[1,-1,-1])

        feature = tf.reshape(feature, [128, 128, 1])
        feature = tf.random_crop(feature, [DataH, DataW, 1])

        feature = tf.cast(feature, tf.complex64)
        
        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        IQ=feature*tf.reshape(Q,[DataH,DataW,1])
        
        label=tf.concat([tf.real(IQ),tf.imag(IQ)],axis=2)
        
        IQ2=tf.reshape(IQ,IQ.shape[0:2])

        IQ2=GT.gfft_TF(IQ2,DataH,0)
        IQ2=GT.gfft_TF(IQ2,DataW,1)
        feature=tf.reshape(IQ2,[DataH*DataW,1,1])
        
        feature=tf.concat([tf.real(feature),tf.imag(feature)],axis=2)

        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == 'M1DFTxy':
        I=scipy.io.loadmat('/media/a/H1/First3kIm128x128MagSinglex.mat')
        I=I['First3kIm128x128MagSingle']

        TFI = tf.constant(np.float32(I))
        Idx=tf.random_uniform([1],minval=0,maxval=3000,dtype=tf.int32)
        feature=tf.slice(TFI,[Idx[0],0,0],[1,-1,-1])

        feature = tf.reshape(feature, [128, 128, 1])
        feature = tf.random_crop(feature, [DataH, DataW, 1])

        feature = tf.cast(feature, tf.complex64)
        
        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        IQ=feature*tf.reshape(Q,[DataH,DataW,1])
        
        label=tf.concat([tf.real(IQ),tf.imag(IQ)],axis=2)
        
        IQ2=tf.reshape(IQ,IQ.shape[0:2])

        IQ2=GT.gfft_TF(IQ2,DataH,0)
        IQ2=GT.gfft_TF(IQ2,DataW,1)
        feature=tf.reshape(IQ2,[DataH,DataW,1])
        
        feature=tf.concat([tf.real(feature),tf.imag(feature)],axis=2)

        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == 'M1DFTx':
        I=scipy.io.loadmat('/media/a/H1/First3kIm128x128MagSinglex.mat')
        I=I['First3kIm128x128MagSingle']

        TFI = tf.constant(np.float32(I))
        Idx=tf.random_uniform([1],minval=0,maxval=3000,dtype=tf.int32)
        feature=tf.slice(TFI,[Idx[0],0,0],[1,-1,-1])

        feature = tf.reshape(feature, [DataH, DataW, 1])

        feature = tf.cast(feature, tf.complex64)
        
        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        IQ=feature*tf.reshape(Q,[DataH,DataW,1])
        
        label=tf.concat([tf.real(IQ),tf.imag(IQ)],axis=2)
        
        IQ2=tf.reshape(IQ,IQ.shape[0:2])

        IQ2=GT.gfft_TF(IQ2,DataW,1)
        feature=tf.reshape(IQ2,[DataH,DataW,1])
        
        feature=tf.concat([tf.real(feature),tf.imag(feature)],axis=2)

        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        tf.train.start_queue_runners(sess=sess)
        return features, labels

    if myParams.myDict['InputMode'] == 'M1DFTy':
        I=scipy.io.loadmat('/media/a/H1/First3kIm128x128MagSinglex.mat')
        I=I['First3kIm128x128MagSingle']

        TFI = tf.constant(np.float32(I))
        Idx=tf.random_uniform([1],minval=0,maxval=3000,dtype=tf.int32)
        feature=tf.slice(TFI,[Idx[0],0,0],[1,-1,-1])

        feature = tf.reshape(feature, [DataH, DataW, 1])

        feature = tf.cast(feature, tf.complex64)
        
        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        IQ=feature*tf.reshape(Q,[DataH,DataW,1])
        
        label=tf.concat([tf.real(IQ),tf.imag(IQ)],axis=2)
        
        IQ2=tf.reshape(IQ,IQ.shape[0:2])

        IQ2=GT.gfft_TF(IQ2,DataH,0)
        feature=tf.reshape(IQ2,[DataH,DataW,1])
        
        feature=tf.concat([tf.real(feature),tf.imag(feature)],axis=2)

        features, labels = tf.train.batch([feature, label],batch_size=batch_size,num_threads=4,capacity = capacity_factor*batch_size,name='labels_and_features')
        tf.train.start_queue_runners(sess=sess)
        return features, labels

    #if image_size is None:
    #    image_size = FLAGS.sample_size
    
    #pdb.set_trace()

    
    reader = tf.TFRecordReader()

    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    
    AlsoLabel=True
    kKick= myParams.myDict['InputMode'] == 'kKick'
    if kKick or myParams.myDict['InputMode'] == '1DFTx' or myParams.myDict['InputMode'] == '1DFTy' or myParams.myDict['InputMode'] == '2DFT':
        AlsoLabel=False

    
    
    if myParams.myDict['InputMode'] == 'AAA':
        #filename_queue = tf.Print(filename_queue,[filename_queue,],message='ZZZZZZZZZ:')
        keyX=key
        value = tf.Print(value,[keyX,],message='QQQ:')

        featuresA = tf.parse_single_example(
            value,
            features={
                'CurIs': tf.FixedLenFeature([], tf.string),
                'Labels': tf.FixedLenFeature([], tf.string)
            })
        feature = tf.decode_raw(featuresA['Labels'], tf.float32)
        CurIs = tf.decode_raw(featuresA['CurIs'], tf.float32)
        CurIs = tf.cast(CurIs, tf.int64)

        mx=CurIs
        # mx='qwe'+
        feature = tf.Print(feature,[keyX,mx],message='QQQ:')
        feature = tf.Print(feature,[keyX,mx],message='QQQ:')
        feature = tf.Print(feature,[keyX,mx],message='QQQ:')
        feature = tf.Print(feature,[keyX,mx],message='QQQ:')
        feature = tf.Print(feature,[keyX,mx],message='QQQ:')

        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature = tf.cast(feature, tf.float32)

        label=feature

        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels

        
    #image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")

    #print('1')
    if AlsoLabel:
        featuresA = tf.parse_single_example(
            value,
            features={
                'DataH': tf.FixedLenFeature([], tf.int64),
                'DataW': tf.FixedLenFeature([], tf.int64),
                'channelsIn': tf.FixedLenFeature([], tf.int64),
                'LabelsH': tf.FixedLenFeature([], tf.int64),
                'LabelsW': tf.FixedLenFeature([], tf.int64),
                'channelsOut': tf.FixedLenFeature([], tf.int64),
                'data_raw': tf.FixedLenFeature([], tf.string),
                'labels_raw': tf.FixedLenFeature([], tf.string)
            })
        labels = tf.decode_raw(featuresA['labels_raw'], tf.float32)
    else:
        featuresA = tf.parse_single_example(
            value,
            features={
                'DataH': tf.FixedLenFeature([], tf.int64),
                'DataW': tf.FixedLenFeature([], tf.int64),
                'channelsIn': tf.FixedLenFeature([], tf.int64),
                'data_raw': tf.FixedLenFeature([], tf.string)
            })
    feature = tf.decode_raw(featuresA['data_raw'], tf.float32)

    print('setup_inputs')
    print('Data   H,W,#ch: %d,%d,%d -> Labels H,W,#ch %d,%d,%d' % (DataH,DataW,channelsIn,LabelsH,LabelsW,channelsOut))
    print('------------------')
    
    if myParams.myDict['InputMode'] == '1DFTy':
        feature = tf.reshape(feature, [256, 256, 1])
        feature = tf.random_crop(feature, [DataH, DataW, channelsIn])
        
        mm=tf.reduce_mean(feature)
        mx=tf.reduce_max(feature)
        mx=tf.maximum(mx,1)

        #feature = tf.Print(feature,[mm,mx],message='QQQ:')        
        #assert_op = tf.Assert(tf.greater(mx, 0), [mx])
        #with tf.control_dependencies([assert_op]):

        feature = tf.cast(feature/mx, tf.complex64)
        
        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        IQ=feature*tf.reshape(Q,[DataH,DataW,channelsIn])
        
        label=tf.concat([tf.real(IQ),tf.imag(IQ)],axis=2)
        feature=label

        HalfDataW=DataW/2

        Id=np.hstack([np.arange(HalfDataW,DataW), np.arange(0,HalfDataW)])
        Id=Id.astype(int)

        IQ2=tf.reshape(IQ,IQ.shape[0:2])
        feature=tf.fft(IQ2)
        feature = tf.gather(feature,Id,axis=1)
        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature=tf.concat([tf.real(feature),tf.imag(feature)],axis=2)

        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels

    if myParams.myDict['InputMode'] == '1DFTx':
        feature = tf.reshape(feature, [256, 256, 1])
        feature = tf.random_crop(feature, [DataH, DataW, channelsIn])
        
        mm=tf.reduce_mean(feature)
        mx=tf.reduce_max(feature)
        mx=tf.maximum(mx,1)

        #feature = tf.Print(feature,[mm,mx],message='QQQ:')        
        #assert_op = tf.Assert(tf.greater(mx, 0), [mx])
        #with tf.control_dependencies([assert_op]):

        feature = tf.cast(feature/mx, tf.complex64)
        
        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        IQ=feature*tf.reshape(Q,[DataH,DataW,channelsIn])
        
        label=tf.concat([tf.real(IQ),tf.imag(IQ)],axis=2)
        feature=label

        HalfDataH=DataH/2

        Id=np.hstack([np.arange(HalfDataH,DataH), np.arange(0,HalfDataH)])
        Id=Id.astype(int)

        IQ2=tf.reshape(IQ,IQ.shape[0:2])
        IQ2 = tf.transpose(IQ2, perm=[1, 0])
        feature=tf.fft(IQ2)
        feature = tf.gather(feature,Id,axis=1)
        feature = tf.transpose(feature, perm=[1,0])
        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature=tf.concat([tf.real(feature),tf.imag(feature)],axis=2)

        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels

    if myParams.myDict['InputMode'] == '2DFT':
        feature = tf.reshape(feature, [256, 256, 1])
        feature = tf.random_crop(feature, [DataH, DataW, channelsIn])
        
        mm=tf.reduce_mean(feature)
        mx=tf.reduce_max(feature)
        mx=tf.maximum(mx,1)

        #feature = tf.Print(feature,[mm,mx],message='QQQ:')        
        #assert_op = tf.Assert(tf.greater(mx, 0), [mx])
        #with tf.control_dependencies([assert_op]):

        feature = tf.cast(feature/mx, tf.complex64)
        
        Q=GT.TFGenerateRandomSinPhase(DataH, DataW)
        IQ=feature*tf.reshape(Q,[DataH,DataW,channelsIn])
        
        label=tf.concat([tf.real(IQ),tf.imag(IQ)],axis=2)
        feature=label

        HalfDataH=DataH/2
        HalfDataW=DataW/2

        IdH=np.hstack([np.arange(HalfDataH,DataH), np.arange(0,HalfDataH)])
        IdH=IdH.astype(int)

        IdW=np.hstack([np.arange(HalfDataW,DataW), np.arange(0,HalfDataW)])
        IdW=IdW.astype(int)

        IQ2=tf.reshape(IQ,IQ.shape[0:2])

        IQ2=tf.fft(IQ2)
        IQ2=tf.gather(IQ2,IdW,axis=1)

        IQ2 = tf.transpose(IQ2, perm=[1, 0])
        feature=tf.fft(IQ2)
        feature = tf.gather(feature,IdH,axis=1)
        feature = tf.transpose(feature, perm=[1,0])
        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature=tf.concat([tf.real(feature),tf.imag(feature)],axis=2)

        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels

    if kKick:
        filename_queue2 = tf.train.string_input_producer(filenames)
        key2, value2 = reader.read(filename_queue2)
        featuresA2 = tf.parse_single_example(
            value2,
            features={
                'DataH': tf.FixedLenFeature([], tf.int64),
                'DataW': tf.FixedLenFeature([], tf.int64),
                'channelsIn': tf.FixedLenFeature([], tf.int64),
                'data_raw': tf.FixedLenFeature([], tf.string)
            })
        feature2 = tf.decode_raw(featuresA2['data_raw'], tf.float32)

        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature2 = tf.reshape(feature2, [DataH, DataW, channelsIn])


        feature.set_shape([None, None, channelsIn])
        feature2.set_shape([None, None, channelsIn])

        feature = tf.cast(feature, tf.float32)/tf.reduce_max(feature)
        feature2 = tf.cast(feature2, tf.float32)/tf.reduce_max(feature)
        
        feature= tf.concat([feature,feature*0,feature2,feature2*0], 2)
        label=feature

        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels

    if myParams.myDict['InputMode'] == 'RegridTry3':
        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature = tf.cast(feature, tf.float32)
        
        labels = tf.reshape(labels, [LabelsH, LabelsW, channelsOut])
        label = tf.cast(labels, tf.float32)

        # Using asynchronous queues
        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels


    if myParams.myDict['InputMode'] == 'RegridTry2':
        FullData=scipy.io.loadmat(myParams.myDict['NMAP_FN'])
        
        NMapCR=FullData['NMapCR']
        NMapCR = tf.constant(NMapCR)

        feature=tf.gather(feature,NMapCR,validate_indices=None,name=None)

        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature = tf.cast(feature, tf.float32)
        
        labels = tf.reshape(labels, [128, 128, channelsOut])

        # scipy.misc.imresize(arr, size, interp='bilinear', mode=None)
        labels = tf.image.resize_images(labels,[LabelsH, LabelsW]) #,method=tf.ResizeMethod.BICUBIC,align_corners=False) # or BILINEAR

        label = tf.cast(labels, tf.float32)

        # Using asynchronous queues
        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels

    if myParams.myDict['InputMode'] == 'RegridTry1':
        # FullData=scipy.io.loadmat('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/NMapIndTesta.mat')
        FullData=scipy.io.loadmat(myParams.myDict['NMAP_FN'])
        
        NMapCR=FullData['NMapCR']
        NMapCR = tf.constant(NMapCR)

        feature=tf.gather(feature,NMapCR,validate_indices=None,name=None)

        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature = tf.cast(feature, tf.float32)
        
        labels = tf.reshape(labels, [LabelsH, LabelsW, channelsOut])
        label = tf.cast(labels, tf.float32)

        # Using asynchronous queues
        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels

    if myParams.myDict['InputMode'] == 'SMASHTry1':
        feature = tf.reshape(feature, [DataH, DataW, channelsIn])
        feature = tf.cast(feature, tf.float32)
        
        labels = tf.reshape(labels, [LabelsH, LabelsW, channelsOut])
        label = tf.cast(labels, tf.float32)

        # Using asynchronous queues
        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels

    """if myParams.myDict['Mode'] == 'RegridTry1C2':
        FullData=scipy.io.loadmat('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/NMapIndC.mat')
        NMapCR=FullData['NMapCRC']
        NMapCR = tf.constant(NMapCR)

        feature=tf.gather(feature,NMapCR,validate_indices=None,name=None)

        feature = tf.reshape(feature, [DataH, DataW, channelsIn,2])
        feature = tf.cast(feature, tf.float32)
        
        labels = tf.reshape(labels, [LabelsH, LabelsW, channelsOut])
        label = tf.cast(labels, tf.float32)

        # Using asynchronous queues
        features, labels = tf.train.batch([feature, label],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity = capacity_factor*batch_size,
                                          name='labels_and_features')

        tf.train.start_queue_runners(sess=sess)
        
        return features, labels"""



    feature = tf.reshape(feature, [DataH, DataW, channelsIn])
    labels = tf.reshape(labels, [LabelsH, LabelsW, channelsOut])
    
    #print('44')
    #example.ParseFromString(serialized_example)
    #x_1 = np.array(example.features.feature['X'].float_list.value)

    # Convert from [depth, height, width] to [height, width, depth].
    #result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    feature.set_shape([None, None, channelsIn])
    labels.set_shape([None, None, channelsOut])

    

    # Crop and other random augmentations
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, .95, 1.05)
    #image = tf.image.random_brightness(image, .05)
    #image = tf.image.random_contrast(image, .95, 1.05)

    #print('55')
    #wiggle = 8
    #off_x, off_y = 25-wiggle, 60-wiggle
    #crop_size = 128
    #crop_size_plus = crop_size + 2*wiggle
    #print('56')
    #image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)
    #print('57')
    #image = tf.image.crop_to_bounding_box(image, 1, 2, crop_size, crop_size)
    #image = tf.random_crop(image, [crop_size, crop_size, 3])

    feature = tf.reshape(feature, [DataH, DataW, channelsIn])
    feature = tf.cast(feature, tf.float32) #/255.0

    
    labels = tf.reshape(labels, [LabelsH, LabelsW, channelsOut])
    label = tf.cast(labels, tf.float32) #/255.0


    #if crop_size != image_size:
    #    image = tf.image.resize_area(image, [image_size, image_size])

    # The feature is simply a Kx downscaled version
    #K = 1
    #downsampled = tf.image.resize_area(image, [image_size//K, image_size//K])

    #feature = tf.reshape(downsampled, [image_size//K, image_size//K, 3])
    #feature = tf.reshape(downsampled, [image_size//K, image_size//K, 3])
    #label   = tf.reshape(image,       [image_size,   image_size,     3])

    #feature = tf.reshape(image,     [image_size,    image_size,     channelsIn])
    #feature = tf.reshape(image,     [1, image_size*image_size*2,     channelsIn])
    #label   = tf.reshape(labels,    [image_size,    image_size,     channelsOut])

    # Using asynchronous queues
    features, labels = tf.train.batch([feature, label],
                                      batch_size=batch_size,
                                      num_threads=4,
                                      capacity = capacity_factor*batch_size,
                                      name='labels_and_features')

    tf.train.start_queue_runners(sess=sess)
    
    return features, labels
