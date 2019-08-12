import os
import sys

# sys.path.insert(0, '/media/a/H2/home/a/TF/')
sys.path.insert(0, '/opt/data/TF/')

import time

import GTools as GT

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# HomeA='/home/deni/'
HomeA=GT.getHome()

sys.path.insert(0, HomeA + 'TF/')

# DatasetsBase='/home/deni/'
DatasetsBase=GT.getDatasetsBase()


os.chdir(HomeA + 'TF/srezN')
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pdb

# import srez_demo
import srez_input
import srez_model
import srez_train

import os.path
import random
import numpy as np
import numpy.random
import scipy.io
import h5py
import imageio

import sys

import datetime

from shutil import copyfile

import srez_modelBase

import myParams


import myParams
myParams.init()
# defaults
myParams.myDict['DataH']=1
myParams.myDict['MapSize']=3
myParams.myDict['learning_rate_half_life']=5000
myParams.myDict['learning_rate_start']=0.002
myParams.myDict['dataset']='xxx'
myParams.myDict['batch_size']=-8
myParams.myDict['WL1_Lambda']=0.0001
myParams.myDict['WL2_Lambda']=0.0001
myParams.myDict['QuickFailureTimeM']=3
myParams.myDict['QuickFailureTthresh']=0.3
myParams.myDict['summary_period']=0.5 # in minutes
myParams.myDict['checkpoint_period']=20 # in minutes

myParams.myDict['DiscStartMinute']=5 # in minutes

myParams.myDict['Mode']='kKick'

myParams.myDict['noise_level']=0.0

myParams.myDict['ShowRealData'] = 0
myParams.myDict['CmplxBias'] = 0

myParams.myDict['WL1_Lambda'] = 0
myParams.myDict['WL2_Lambda'] = 0
myParams.myDict['WPhaseOnly'] = 0

myParams.myDict['BankSize']=0

# ParamFN=HomeA +'TF/Params.txt'
# ParamsD = {}
# with open(ParamFN) as f:
#     for line in f:
#         if len(line)<3:
#             continue
#         # print(line)
#         #print(line.replace("\n",""))
#         (key, val) = line.split()
#         valx=getParam(val)
#         ParamsD[key] = valx
#         myParams.myDict[key]=ParamsD[key]
#         print(key + " : " + str(val) + " " + type(valx).__name__)

ParamFN=sys.argv[1]
GT.readParamsTxt(ParamFN)

LArgs=int(len(sys.argv)/2)

FNExtra=''
for ArgI in range(1, LArgs):
    GT.setparam(sys.argv[int(ArgI*2)],GT.getParam_tmpF(sys.argv[int(ArgI*2+1)]))
    FNExtra=FNExtra+'_'+sys.argv[int(ArgI*2)]+'_'+sys.argv[int(ArgI*2+1)]


if GT.getparam('CUDA_VISIBLE_DEVICES')==0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if GT.getparam('CUDA_VISIBLE_DEVICES')==1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if GT.getparam('CUDA_VISIBLE_DEVICES')==2:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
if GT.getparam('CUDA_VISIBLE_DEVICES')==3:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#     else:
#         os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

# Configuration (alphabetically)
tf.app.flags.DEFINE_integer('batch_size', -7,"Number of samples per batch.")

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint', "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 10000,   "Number of batches in between checkpoints")

tf.app.flags.DEFINE_string('dataset', 'dataset1', "Path to the dataset directory.")

tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_string('run', 'train',"Which operation to run. [demo|train]")

tf.app.flags.DEFINE_float('gene_l1_factor', 1  , "Multiplier for generator L1 loss term")
#tf.app.flags.DEFINE_float('gene_l1_factor', .90, "Multiplier for generator L1 loss term")

tf.app.flags.DEFINE_float('learning_beta1', 0.9, #0.5,
                          "Beta1 parameter used for AdamOptimizer")

# tf.app.flags.DEFINE_float('learning_rate_start', 0.0002,"Starting learning rate used for AdamOptimizer")
tf.app.flags.DEFINE_float('learning_rate_start', 0.002,"Starting learning rate used for AdamOptimizer")
#tf.app.flags.DEFINE_float('learning_rate_start', 0.00001, #0.00020,"Starting learning rate used for AdamOptimizer")

tf.app.flags.DEFINE_integer('learning_rate_half_life', 5000,
                            "Number of batches until learning rate is halved")

tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")

tf.app.flags.DEFINE_integer('sample_size', 64,
                            "Image sample size in pixels. Range [64,128]")

tf.app.flags.DEFINE_integer('summary_period', 200, "Number of batches between summary data dumps")

tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")

#tf.app.flags.DEFINE_integer('test_vectors', 16,"""Number of features to use for testing""")
                            
tf.app.flags.DEFINE_string('train_dir', 'train',"Output folder where training logs are dumped.")

#tf.app.flags.DEFINE_integer('train_time', 20,"Time in minutes to train the model")
tf.app.flags.DEFINE_integer('train_time', 180,"Time in minutes to train the model")

tf.app.flags.DEFINE_integer('DataH', 64*64,"DataH")
tf.app.flags.DEFINE_integer('DataW', 64*64,"DataW")
tf.app.flags.DEFINE_integer('channelsIn', 64*64,"channelsIn")
tf.app.flags.DEFINE_integer('LabelsH', 64*64,"LabelsH")
tf.app.flags.DEFINE_integer('LabelsW', 64*64,"LabelsW")
tf.app.flags.DEFINE_integer('channelsOut', 64*64,"channelsOut")

tf.app.flags.DEFINE_string('SessionName', 'SessionName', "Which operation to run. [demo|train]")

# def getParam(s):
#     try:
#         return int(s)
#     except ValueError:
#         try:
#             return float(s)
#         except ValueError:
#             try:
#                 return np.array(list(map(int, s.split(','))))
#             except ValueError:
#                 return s



#pdb.set_trace()
#FLAGS.nChosen=2045
#FLAGS.random_seed=0
#tf.app.flags.random_seed=0
# FullData=scipy.io.loadmat('/home/a/TF/ImgsMC1.mat')
# FullData=scipy.io.loadmat('/home/a/TF/CurChunk.mat')
# Data=FullData['Data']
# Labels=FullData['Labels']
# #pdb.set_trace()
# nSamples=Data.shape[0]
# DataH = Data.shape[1]
# if Data.ndim<3 :
#     DataW = 1
# else:
#     DataW = Data.shape[2]

# if Data.ndim<4 :
#     channelsIn=1
# else :
#     channelsIn = Data.shape[3]

# LabelsH = Labels.shape[1]
# LabelsW = Labels.shape[2]
# if Labels.ndim<4 :
#     channelsOut=1
# else :
#     channelsOut = Labels.shape[3]

# FLAGS.DataH=DataH
# FLAGS.DataW=DataW
# FLAGS.channelsIn=channelsIn
# FLAGS.LabelsH=LabelsH
# FLAGS.LabelsW=LabelsW
# FLAGS.channelsOut=channelsOut

# del FullData
# del Data
# del Labels


# pdb.set_trace()

# MatlabParams=scipy.io.loadmat('/home/a/TF/MatlabParams.mat')

#FLAGS.dataset='dataKnee'
#FLAGS.dataset='dataFaceP4'

# SessionNameBase=MatlabParams['SessionName'][0];
SessionNameBase= myParams.myDict['SessionNameBase']

#SessionName=MatlabParams['SessionName'][0] + '__' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

SessionName= SessionNameBase + '_'+myParams.myDict['dataset'] + '__' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# myParams.myDict['Full_dataset']=DatasetsBase+ParamsD['dataset']
myParams.myDict['Full_dataset']=myParams.myDict['dataset']

# FLAGS.checkpoint_dir=SessionName+'_checkpoint'
# FLAGS.train_dir=SessionName+'_train'
# FLAGS.SessionName=SessionName

myParams.myDict['SessionName']=SessionName
# myParams.myDict['train_dir']=SessionName+'_train'
# myParams.myDict['checkpoint_dir']=SessionName+'_checkpoint'
myParams.myDict['train_dir']=     SessionNameBase + '__' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +'_train'
myParams.myDict['checkpoint_dir']=SessionNameBase + '__' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +'_checkpoint'

remaining_args = FLAGS([sys.argv[0]] + [flag for flag in sys.argv if flag.startswith("--")])
assert(remaining_args == [sys.argv[0]])

def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(myParams.myDict['checkpoint_dir']):
        tf.gfile.MakeDirs(myParams.myDict['checkpoint_dir'])
    
    # Cleanup train dir
    if delete_train_dir:
        if tf.gfile.Exists(myParams.myDict['train_dir']):
            tf.gfile.DeleteRecursively(myParams.myDict['train_dir'])
        tf.gfile.MakeDirs(myParams.myDict['train_dir'])

    # ggg
    copyfile(ParamFN, myParams.myDict['train_dir'] + '/ParamsUsed' + FNExtra + '.txt')

    # Return names of training files
    if myParams.myDict['Full_dataset']=='xxx' or myParams.myDict['InputMode'] == 'RegridTry3F' or myParams.myDict['InputMode'] == 'RegridTry3FMB' or myParams.myDict['InputMode'] == 'I2I':
        filenames=''
    else:
        if not tf.gfile.Exists(myParams.myDict['Full_dataset']) or \
           not tf.gfile.IsDirectory(myParams.myDict['Full_dataset']):
            raise FileNotFoundError("Could not find folder `%s'" % (myParams.myDict['Full_dataset'],))

        filenames = tf.gfile.ListDirectory(myParams.myDict['Full_dataset'])
        filenames = sorted(filenames)
        random.shuffle(filenames)
        filenames = [os.path.join(myParams.myDict['Full_dataset'], f) for f in filenames]

    return filenames


def setup_tensorflow():
    print("setup_tensorflow")
    # Create session
    #config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    # config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=False)
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
    summary_writer = tf.summary.FileWriter(myParams.myDict['train_dir'], sess.graph)
    print("setup_tensorflow end")
    return sess, summary_writer

# def _demo():
#     # Load checkpoint
#     if not tf.gfile.IsDirectory(myParams.myDict['checkpoint_dir']):
#         raise FileNotFoundError("Could not find folder `%s'" % (myParams.myDict['checkpoint_dir'],))

#     # Setup global tensorflow state
#     sess, summary_writer = setup_tensorflow()

#     # Prepare directories
#     filenames = prepare_dirs(delete_train_dir=False)

#     # Setup async input queues
#     features, labels = srez_input.setup_inputs(sess, filenames)

#     # Create and initialize model
#     [gene_minput, gene_moutput,
#      gene_output, gene_var_list,
#      disc_real_output, disc_fake_output, disc_var_list] = \
#             srez_modelBase.create_model(sess, features, labels)

#     # Restore variables from checkpoint
#     saver = tf.train.Saver()
#     filename = 'checkpoint_new.txt'
#     filename = os.path.join(myParams.myDict['checkpoint_dir'], filename)
#     saver.restore(sess, filename)

#     # Execute demo
#     srez_demo.demo1(sess)

class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def _train():

    # LoadAndRunOnData=False
    LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
    if LoadAndRunOnData:
        # Setup global tensorflow state
        sess, summary_writer = setup_tensorflow()

        # Prepare directories
        # filenames = prepare_dirs(delete_train_dir=False)

        # Setup async input queues
        # features, labels = srez_input.setup_inputs(sess, filenames)
        features, labels = srez_input.setup_inputs(sess, 1)

        # Create and initialize model
        [gene_minput, gene_moutput,
         gene_output, gene_var_list,
         disc_real_output, disc_fake_output, disc_var_list] = \
                srez_modelBase.create_model(sess, features, labels)

        # Restore variables from checkpoint
        print("Adding to saver:")
        var_listX=gene_var_list
        var_listX = [v for v in var_listX if "Bank" not in v.name]
        for line in var_listX: print("Adding " +line.name+'           ' + str(line.shape.as_list()))
        print("Saver var list end")

        saver = tf.train.Saver(var_listX)
        # saver = tf.train.Saver()
        filename = 'checkpoint_new'
        # filename = os.path.join('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/srez/RegridTry1C2_TS2_dataNeighborhoodRCB0__2018-06-08_16-17-56_checkpoint', filename)
        # filename = os.path.join('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/srez/RegridTry1C2_TS2_dataNeighborhoodRCB0__2018-06-09_19-44-17_checkpoint', filename)
        # filename = os.path.join('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/srez/RegridTry1C2_TS__2018-06-29_10-39-13_checkpoint', filename)
        checkpointP=myParams.myDict['LoadAndRunOnData_checkpointP']
        filename = os.path.join(checkpointP, filename)
        
        saver.restore(sess, filename)

        if myParams.myDict['Mode'] == 'RegridTry1' or myParams.myDict['Mode'] == 'RegridTry1C' or myParams.myDict['Mode'] == 'RegridTry1C2' or myParams.myDict['Mode'] == 'RegridTry1C2_TS' or myParams.myDict['Mode'] == 'RegridTry1C2_TS2':
            FullData=scipy.io.loadmat(myParams.myDict['NMAP_FN'])
            NMapCR=FullData['NMapCR']

        for r in range(1,myParams.myDict['HowManyToRun']):
            # ifilename='/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/srez/RealData/b_Ben14May_Sli5_r' +  f'{r:02}' + '.mat'
            # ifilename='/media/a/DATA/14May18/Ben/meas_MID109_gBP_VD11_U19_4min_FID17944/RealData/Sli11_r' +  f'{r:02}' + '.mat'
            if myParams.myDict['InputMode'] == 'Cart3SB':
                # print('Loaded SensMaps, Shape %d %d %d %d' % (SensMapsSz[0],SensMapsSz[1],SensMapsSz[2],SensMapsSz[3]))
                print('Cart3SB running on real data %d' % r)
                # feature  shape: (1048576, 1, 1) <dtype: 'float32'>
                batch_size=myParams.myDict['batch_size']
                # RealData=np.zeros((batch_size,1048576, 1, 1),np.float32)
                # RealData=np.zeros((batch_size,640000, 1, 1),np.float32)

                # Simulating RealData from ITS, Sens
                MB=GT.getparam('MB')
                TimePoints_ms=GT.getparam('TimePoints_ms')
                nTSC=TimePoints_ms.shape[0]
                nCh=GT.getparam('nccToUse')

                LabelsH=myParams.myDict['LabelsH']
                LabelsW=myParams.myDict['LabelsW']

                H=LabelsH
                W=LabelsW

                SnsFN='/opt/data/CCSensMaps.mat'
                fS = h5py.File(SnsFN, 'r')
                SensMaps=fS['SensCC']
                SensMaps=SensMaps['real']+1j*SensMaps['imag']
                SensMapsSz=SensMaps.shape
                print('r Loaded SensMaps, Shape %d %d %d %d' % (SensMapsSz[0],SensMapsSz[1],SensMapsSz[2],SensMapsSz[3]))
                SensMaps=SensMaps[:,:,:,:nCh]

                NumSensMapsInFile=SensMaps.shape[0]
                # IdxS=15
                for b in range(0, MB):
                    # if b==1:
                    #     IdxB2=tf.random_uniform([1],minval=12,maxval=19,dtype=tf.int32)
                    #     IdxS=IdxS+IdxB2[0]
                    #     IdxS=tf.cond(IdxS[0]>=NumSensMapsInFile, lambda: IdxS-NumSensMapsInFile, lambda: IdxS)
                    
                    # Sens=np.squeeze(SensMaps[IdxS,:,:,:],axis=0)
                    Sens=(SensMaps[15,:,:,:])

                    Sens=Sens[:H,:W,:nCh]

                    # Sens = tf.image.random_flip_left_right(Sens)
                    # Sens = tf.image.random_flip_up_down(Sens)
                    # uS=tf.random_uniform([1])
                    # Sens=tf.cond(uS[0]<0.5, lambda: tf.identity(Sens), lambda: tf.image.rot90(Sens))
                    SensMsk=GT.NP_addDim(np.sum(np.abs(Sens),axis=2)>0).astype(np.complex64)
                    Sens=GT.NP_addDim(Sens)

                    if b==0:
                        SensMB=Sens
                        SensMskMB=SensMsk

                    # else:
                    #     SensMB=tf.concat([SensMB,Sens],axis=3) #     SensMB H W nCh MB
                    #     SensMskMB=tf.concat([SensMskMB,SensMsk],axis=2) #     SensMskMB H W MB
                
                # nToLoad=myParams.myDict['nToLoad']
                # LoadAndRunOnData=myParams.myDict['LoadAndRunOnData']>0
                # if LoadAndRunOnData:
                nToLoad=300

                print('r loading images ' + time.strftime("%Y-%m-%d %H:%M:%S"))
                GREBaseP='/opt/data/'
                SFN=GREBaseP+'All_Orientation-0x.mat'
                f = h5py.File(SFN, 'r')
                I=f['CurSetAll'][0:nToLoad]
                print('r Loaded images ' + time.strftime("%Y-%m-%d %H:%M:%S"))

                SendTSCest=GT.getparam('SendTSCest')>0
                HamPow=GT.getparam('HamPow')
                # def TFexpix(X): return tf.exp(tf.complex(tf.zeros_like(X),X))
                def NPexpix(X): return np.exp(1j*X)
                for b in range(0, MB):
                    # TFI = tf.constant(np.int16(I))
                    # Idx=tf.random_uniform([1],minval=0,maxval=I.shape[0],dtype=tf.int32)
                    Idx=133

                    Data4=(I[Idx,:,:,:])
                    # Data4=tf.squeeze(tf.slice(I,[Idx[0],0,0,0],[1,-1,-1,-1]),axis=0)
                    # Data4 = tf.image.random_flip_left_right(Data4)
                    # Data4 = tf.image.random_flip_up_down(Data4)

                    # u1=tf.random_uniform([1])
                    # Data4=tf.cond(u1[0]<0.5, lambda: tf.identity(Data4), lambda: tf.image.rot90(Data4))

                    # Data4 = tf.random_crop(Data4, [H, W, 4])
                    # Data4 = tf.random_crop(Data4, [:H, :W, 4])
                    Data4=Data4[:H,:W,:]

                    # M=tf.slice(Data4,[0,0,0],[-1,-1,1])
                    # Ph=tf.slice(Data4,[0,0,1],[-1,-1,1])
                    # feature=tf.cast(M,tf.complex64)*TFexpix(Ph)
                    M=Data4[:,:,0]
                    Ph=Data4[:,:,1]
                    feature=M.astype(np.complex64)*NPexpix(Ph)

                    feature=GT.NP_addDim(feature)*SensMskMB[:,:,b:b+1]

                    T2S_ms=Data4[:,:,2]
                    # T2S_ms = tf.where( T2S_ms<1.5, 10000 * tf.ones_like( T2S_ms ), T2S_ms )
                    T2S_ms[T2S_ms<1.5]=10000

                    B0_Hz=Data4[:,:,3]
                    # B0_Hz=M*0

                    # T2S_ms = tf.where( tf.is_nan(T2S_ms), 10000 * tf.ones_like( T2S_ms ), T2S_ms )
                    T2S_ms[np.isnan(T2S_ms)]=10000
                    # B0_Hz = tf.where( tf.is_nan(B0_Hz), tf.zeros_like( B0_Hz ), B0_Hz )
                    B0_Hz[np.isnan(B0_Hz)]=0

                    if SendTSCest:
                        # HamPowA=10
                        HamPowA=HamPow
                        HamA=np.roll(np.hamming(H),np.int32(H/2))
                        HamA=np.power(HamA,HamPowA)
                        HamXA=np.reshape(HamA,(1,H,1))
                        HamYA=np.reshape(HamA,(1,1,W))

                        B0_Hz_Smoothed=np.transpose(GT.NP_addDim(B0_Hz.astype(np.complex64)),(2,0,1))
                        B0_Hz_Smoothed=np.fft.fft2(B0_Hz_Smoothed)
                        B0_Hz_Smoothed=B0_Hz_Smoothed*HamXA
                        B0_Hz_Smoothed=B0_Hz_Smoothed*HamYA
                        B0_Hz_Smoothed=np.fft.ifft2(B0_Hz_Smoothed)
                        B0_Hz_Smoothed=np.transpose(B0_Hz_Smoothed,(1,2,0))
                        B0_Hz_Smoothed=np.real(B0_Hz_Smoothed)
                        
                        TSCest=np.exp(1j*2*np.pi*(B0_Hz_Smoothed*TimePoints_ms/1000).astype(np.complex64))
                        # TSCest=np.ones(TSCest.shape).astype(np.complex64)
                        print('TSCest shape: ' + str(TSCest.shape))
                        # TSCest=TSCest*0+1
                        # print('TSCest shape: ' + str(TSCest.shape))
                        # print('reducing B0')
                        # print('B0_Hz shape: ' + str(B0_Hz.shape))
                        # print('B0_Hz_Smoothed shape: ' + str(B0_Hz_Smoothed.shape))
                        # B0_Hz=B0_Hz-np.squeeze(B0_Hz_Smoothed)
                        # print('B0_Hz shape: ' + str(B0_Hz.shape))


                    # urand_ms=tf.random_uniform([1])*12
                    # urand_sec=(tf.random_uniform([1])*2-1)*3/1000

                    # feature=feature*tf.cast(tf.exp(-urand_ms/T2S_ms),tf.complex64)
                    # feature=feature*TFexpix(2*np.pi*B0_Hz*urand_sec)

                    mx=M.max()
                    mx=np.maximum(mx,1)
                    mx=mx.astype(np.complex64)

                    feature=feature/mx

                    CurIWithPhase=feature

                    TSCM=np.exp(-TimePoints_ms/GT.NP_addDim(T2S_ms))
                    TSCP=np.exp(1j*2*np.pi*(GT.NP_addDim(B0_Hz)*TimePoints_ms/1000).astype(np.complex64))
                    TSC=TSCM.astype(np.complex64)*TSCP
                    
                    ITSbase=CurIWithPhase*TSC # ITSbase is H,W,nTSC

                    TSC=GT.NP_addDim(TSC)
                    ITSbase=GT.NP_addDim(ITSbase)
                    if b==0:
                        CurIWithPhaseMB=CurIWithPhase
                        TSCMB=TSC
                        ITSbaseMB=ITSbase                
                        if SendTSCest:
                            TSCest=GT.NP_addDim(TSCest)
                            TSCMBest=TSCest
                    # else:
                    #     CurIWithPhaseMB=tf.concat([CurIWithPhaseMB,CurIWithPhase],axis=2) #     CurIWithPhaseMB H W MB
                    #     TSCMB=tf.concat([TSCMB,TSC],axis=3) #     TSCMB H W nTSC MB
                    #     ITSbaseMB=tf.concat([ITSbaseMB,ITSbase],axis=3) #     ITSbaseMB H W nTSC MB
                    #     if SendTSCest:
                    #         TSCMBest=tf.stack([TSCMBest,TSCest],axis=3)
                print('r ok 2')
                ITS_P=np.transpose(GT.NP_addDim(ITSbaseMB),(4,0,1,2,3)) # /batch_size/,H,W,nTSC,MB

                Msk3=np.zeros((H,W,nTSC,1,1,1))

                PEShifts=GT.getparam('PEShifts')
                PEJump=GT.getparam('PEJump')
                print('r Using PEShifts')
                for i in range(nTSC):
                    Msk3[PEShifts[i]::PEJump,:,i,:,:,:]=1

                Msk3=np.complex64(Msk3)
                
                # GT.setparam('CartMask',Msk3)

                Sens6=SensMB[:,:,np.newaxis,:,:,np.newaxis] # H,W,/nTS/,nCh,MB,/batch_size/

                # AHA_ITS=GT.Cartesian_OPHOP_ITS_MB(ITS_P,Sens6,Msk3)

                ITS=np.transpose(ITSbaseMB,(0,3,2,1)) # H, nTSC, W
                ITS=np.reshape(ITS,(H,W*nTSC*MB,1))
                ITS_RI=GT.NP_ConcatRIOn2(ITS)

                Sensc=SensMB
                Sens1D=GT.NP_ConcatRIOn0(np.reshape(Sensc,(-1,1,1)))
                feature=Sens1D
                    
                AHA_ITS=GT.NP_Cartesian_OPHOP_ITS_MB(ITS_P,Sens6,Msk3)
                # new simpler approach
                if SendTSCest:
                    TSCMBest_P=np.transpose(GT.NP_addDim(TSCMBest),(4,0,1,2,3)) # /batch_size/,H,W,nTSC,MB
                    AHA_ITS=AHA_ITS*np.conj(TSCMBest_P)

                #         send AHA_ITS
                AHA_ITS_1D=GT.NP_ConcatRIOn0(np.reshape(AHA_ITS,(-1,1,1)))
                feature=np.concatenate((feature,AHA_ITS_1D),axis=0)

                if SendTSCest:
                    TSCest1D=GT.NP_ConcatRIOn0(np.reshape(TSCMBest_P,(-1,1,1)))
                    feature=np.concatenate((feature,TSCest1D),axis=0)

                RealData=np.tile(feature, (batch_size,1,1,1))

                # End simulating RealData
                Real_feature=RealData
            else:
                ifilenamePrefix=myParams.myDict['LoadAndRunOnData_Prefix']
    #             ifilename=ifilenamePrefix +  f'{r:02}' + '.mat'
                ifilename=ifilenamePrefix+'%02d.mat' % (r)
                RealData=scipy.io.loadmat(ifilename)
                RealData=RealData['Data']
                
                if RealData.ndim==2:
                    RealData=RealData.reshape((RealData.shape[0],RealData.shape[1],1,1))
                if RealData.ndim==3:
                    RealData=RealData.reshape((RealData.shape[0],RealData.shape[1],RealData.shape[2],1))
                
                Real_feature=RealData

                # if myParams.myDict['Mode'] == 'RegridTry1' or myParams.myDict['Mode'] == 'RegridTry1C' or myParams.myDict['Mode'] == 'RegridTry1C2' or myParams.myDict['Mode'] == 'RegridTry1C2_TS' or myParams.myDict['Mode'] == 'RegridTry1C2_TS2':
                #     batch_size=myParams.myDict['batch_size']

                #     Real_feature=np.reshape(RealData[0],[RealData.shape[1]])
                #     Real_feature=np.take(Real_feature,NMapCR)
                #     Real_feature=np.tile(Real_feature, (batch_size,1,1,1))

            if myParams.myDict['InputMode'] == 'RegridTry1' or myParams.myDict['InputMode'] == 'RegridTry2':
                # FullData=scipy.io.loadmat('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/NMapIndTesta.mat')
                FullData=scipy.io.loadmat(myParams.myDict['NMAP_FN'])
                NMapCR=FullData['NMapCR']

                batch_size=myParams.myDict['batch_size']

                Real_feature=np.reshape(RealData[0],[RealData.shape[1]])
                Real_feature=np.take(Real_feature,NMapCR)
                Real_feature=np.tile(Real_feature, (batch_size,1,1,1))

            Real_dictOut = {gene_minput: Real_feature}


            gene_RealOutput = sess.run(gene_moutput, feed_dict=Real_dictOut)

            OnRealData={}
            OnRealDataM=gene_RealOutput
#             filenamex = 'OnRealData' + f'{r:02}' + '.mat'
            filenamexBase = 'OnRealData'+'%02d' % (r)
            filenamex= filenamexBase + '.mat'
            
            LoadAndRunOnData_OutP=myParams.myDict['LoadAndRunOnData_OutP']
            filename = os.path.join(LoadAndRunOnData_OutP, filenamex)
            OnRealData['x']=OnRealDataM
            scipy.io.savemat(filename,OnRealData)

            image=np.sqrt(np.square(OnRealDataM[0,-H:,:(W*3),0])+np.square(OnRealDataM[0,-H:,:(W*3),1]))
            filenamep = filenamexBase + '.png'
            filename = os.path.join(LoadAndRunOnData_OutP, filenamep)
            imageio.imwrite(filename,image)

        print('Saved recon of real data')
        exit()


    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    all_filenames = prepare_dirs(delete_train_dir=True)

    # Separate training and test sets
    #train_filenames = all_filenames[:-FLAGS.test_vectors]
    train_filenames = all_filenames
    #test_filenames  = all_filenames[-FLAGS.test_vectors:]

    # TBD: Maybe download dataset here

    #pdb.set_trace()

    # ggg Signal Bank stuff:
    if myParams.myDict['BankSize']>0:
        if myParams.myDict['InputMode'] == 'RegridTry3FMB':
            BankSize=myParams.myDict['BankSize']*2

            # BankInit=np.zeros([BankSize,myParams.myDict['DataH'],1,1])
            # LBankInit=np.zeros([BankSize,myParams.myDict['LabelsH'],myParams.myDict['LabelsW'], 2])
            with tf.variable_scope("aaa"):
                Bank=tf.get_variable("Bank",shape=[BankSize,myParams.myDict['DataH'],1,1],dtype=tf.float32,trainable=False)
                LBank=tf.get_variable("LBank",shape=[BankSize,myParams.myDict['LabelsH'],myParams.myDict['LabelsW'], 2],dtype=tf.float32,trainable=False)
                # LBank=tf.get_variable("LBank",initializer=tf.cast(LBankInit, tf.float32),dtype=tf.float32,trainable=False)
        else:
            BankSize=myParams.myDict['BankSize']

            BankInit=np.zeros([BankSize,myParams.myDict['DataH'],1,1])
            LBankInit=np.zeros([BankSize,myParams.myDict['LabelsH'],myParams.myDict['LabelsW'], 2])
            with tf.variable_scope("aaa"):
                # Bank=tf.get_variable("Bank",initializer=tf.cast(BankInit, tf.float32),dtype=tf.float32)
                Bank=tf.get_variable("Bank",shape=[BankSize,myParams.myDict['DataH'],1,1],dtype=tf.float32,trainable=False)
                LBank=tf.get_variable("LBank",shape=[BankSize,myParams.myDict['LabelsH'],myParams.myDict['LabelsW'], 2],dtype=tf.float32,trainable=False)
                # LBank=tf.get_variable("LBank",initializer=tf.cast(LBankInit, tf.float32),dtype=tf.float32)

        init_new_vars_op = tf.variables_initializer([Bank,LBank])
        sess.run(init_new_vars_op)
    # ggg end Signal Bank stuff:

    # Setup async input queues
    train_features, train_labels = srez_input.setup_inputs(sess, train_filenames)
    # test_features, test_labels = srez_input.setup_inputs(sess, train_filenames,TestStuff=True)
    test_features=train_features
    test_labels=train_labels
    #test_features,  test_labels  = srez_input.setup_inputs(sess, test_filenames)

    print('starting' + time.strftime("%Y-%m-%d %H:%M:%S"))
    print('train_features %s' % (train_features))
    print('train_labels %s' % (train_labels))


    # Add some noise during training (think denoising autoencoders)
    noise_level=myParams.myDict['noise_level']
    AddNoise=noise_level>0.0
    if AddNoise:
        noisy_train_features = train_features + tf.random_normal(train_features.get_shape(), stddev=noise_level)
    else:
        noisy_train_features = train_features

    
    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_modelBase.create_model(sess, noisy_train_features, train_labels)
    
    # gene_VarNamesL=[];
    # for line in gene_var_list: gene_VarNamesL.append(line.name+'           ' + str(line.shape.as_list()))
    # gene_VarNamesL.sort()

    # for line in gene_VarNamesL: print(line)
    # # var_23 = [v for v in tf.global_variables() if v.name == "gene/GEN_L020/C2D_weight:0"][0]

    # for line in sess.graph.get_operations(): print(line)
    # Gen3_ops=[]
    # for line in sess.graph.get_operations():
    #     if 'GEN_L003' in line.name:
    #         Gen3_ops.append(line)

    #     # LL=QQQ.outputs[0]
        
    # for x in Gen3_ops: print(x.name +'           ' + str(x.outputs[0].shape))

    # GenC2D_ops= [v for v in sess.graph.get_operations()]

    # GenC2D_ops= [v for v in tf.get_operations() if "weight" in v.name]
    # GenC2D_ops= [v for v in GenC2D_ops if "C2D" in v.name]
    # for x in GenC2D_ops: print(x.name +'           ' + str(x.outputs[0].shape))

    # for x in GenC2D_ops: print(x.name)

    AEops = [v for v in sess.graph.get_operations() if "AE" in v.name and not ("_1/" in v.name)]
    # AEops = [v for v in td.sess.graph.get_operations() if "Pixel" in v.name and not ("_1/" in v.name) and not ("opti" in v.name) and not ("Assign" in v.name) and not ("read" in v.name) and not ("Adam" in v.name)]
    AEouts = [v.outputs[0] for v in AEops]
    varsForL1=AEouts
    # varsForL1=AEouts[0:-1]
    # varsForL1=AEouts[1:]

    # for line in sess.graph.get_operations():
    #     if 'GEN_L003' in line.name:
    #         Gen3_ops.append(line)

    #     # LL=QQQ.outputs[0]
        
    # for x in Gen3_ops: print(x.name +'           ' + str(x.outputs[0].shape))

    print("Vars for l2 loss:")
    varws = [v for v in tf.global_variables() if (("weight" in v.name) or ("ConvNet" in v.name))  ]
    varsForL2 = [v for v in varws if "C2D" in v.name]
    varsForL2 = [v for v in varws if "disc" not in v.name]
    varsForL2 = [v for v in varws if "bias" not in v.name]
    for line in varsForL2: print(line.name+'           ' + str(line.shape.as_list()))


    print("Vars for Phase-only loss:")
    varws = [v for v in tf.global_variables() if "weight" in v.name]
    varsForPhaseOnly = [v for v in varws if "SharedOverFeat" in v.name]
    for line in varsForPhaseOnly: print(line.name+'           ' + str(line.shape.as_list()))

    # pdb.set_trace()

    gene_loss, MoreOut, MoreOut2, MoreOut3 = srez_modelBase.create_generator_loss(disc_fake_output, gene_output, train_features,train_labels,varsForL1,varsForL2,varsForPhaseOnly)
    disc_real_loss, disc_fake_loss = \
                     srez_modelBase.create_discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
    
    (global_step, learning_rate, gene_minimize, disc_minimize) = \
            srez_modelBase.create_optimizers(gene_loss, gene_var_list, disc_loss, disc_var_list)

    # Train model
    train_data = TrainData(locals())
    
    #pdb.set_trace()
    # ggg: to restore session
    RestoreSession=False
    if RestoreSession:
        saver = tf.train.Saver()
        filename = 'checkpoint_new'
        filename = os.path.join(myParams.myDict['checkpoint_dir'], filename)
        saver.restore(sess, filename)

    srez_train.train_model(train_data)

def main(argv=None):
    print("aaa")
    _train()
    # Training or showing off?
    #_train()
    #if FLAGS.run == 'demo':
    #    _demo()
    #elif FLAGS.run == 'train':
    #    _train()

if __name__ == '__main__':
  tf.app.run()

#print("asd40")
_train()
#setup_tensorflow()
#tf.app.run()

#print("asd5")
