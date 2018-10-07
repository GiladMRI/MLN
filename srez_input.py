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
    DataH=myParams.myDict['DataH']
    DataW=myParams.myDict['DataW']
    LabelsH=myParams.myDict['LabelsH']
    LabelsW=myParams.myDict['LabelsW']

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

    if myParams.myDict['InputMode'] == 'RegridTry3F':

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

        Q=GT.TFGenerateRandomSinPhase(H, W)
        CurIWithPhase=feature*tf.reshape(Q,[H,W,1])
        label=tf.concat([tf.real(CurIWithPhase),tf.imag(CurIWithPhase)],axis=2)

        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Kd=NUFTData['Kd']
        P=NUFTData['P']
        SN=NUFTData['SN']
        Trajm2=NUFTData['Trajm2']

        nTraj=Trajm2.shape[1]
        nCh=Sens.shape[2]
        nTSC=TSC.shape[2]

        SNc,paddings,sp_R,sp_I,TSBFX=GT.TF_TSNUFFT_Prepare(SN,Sens,TSC,TSBF,Kd,P)

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
