import sys
import numpy as np
import tensorflow as tf
import scipy.io
import GTools as GT

FLAGS = tf.app.flags.FLAGS

import copy

import myParams

from srez_modelBase import Model
import srez_modelBase

def ConstConvKernel(K1,K2,FCOut):
    W=np.zeros([K1,K2,K1,K2,FCOut,FCOut])
    for i in range(0,K1):
        for j in range(0,K2):
            for t in range(0,FCOut):
                W[i,j,i,j,t,t]=1
    W=np.reshape(W,[K1,K2,K1*K2*FCOut,FCOut])
    return W

def _generator_model(sess, features, labels, channels):
    # Upside-down all-convolutional resnet

    mapsize = 3
    mapsize = myParams.myDict['MapSize']
    res_units  = [256, 128, 96]

    old_vars = tf.global_variables()

    # See Arxiv 1603.05027
    model = Model('GEN', features)

    # H=FLAGS.LabelsH;
    # W=FLAGS.LabelsW;
    H=myParams.myDict['LabelsH']
    W=myParams.myDict['LabelsW']
    channelsOut=myParams.myDict['channelsOut']

    batch_size=myParams.myDict['batch_size']

    DataH=myParams.myDict['DataH']

    print("_generator_model")
    print("%d %d %d" % (H, W,channels))

    if myParams.myDict['NetMode'] == 'SPEN_Local':
        print("SPEN_Local mode")

        model.add_Split4thDim(2) # now (16, H, W, HNeighbors, 2)
        model.add_PixelwiseMultC(1) #,NamePrefix='MapsForMat')
        
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'SPEN_FC':
        print("SPEN_FC mode")
        model.add_5thDim()
        model.add_Permute45()
        
        model.add_Mult2DMCxC(H,1)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'SMASH1DFTxyC_YCC':
        print("SMASH1DFTxyC_YCC mode")

        # model.print_size('AAA') # (16, 64, 128, 16)
        model.add_Split4thDim(2) # now (16, 64, 128, 8, 2)

        DFTM=GT.DFT_matrix(H)
        IDFTM=GT.IDFT_matrix(H)

        DFTM_Half=GT.DFT_matrix(64)
        IDFTM_Half=GT.IDFT_matrix(64)

        # back to image space on RO
        model.add_Mult2DMCyCSharedOverFeat(W,1,Trainable=False,InitC=IDFTM)
        # YCC: also on PE
        model.add_Mult2DMCxCSharedOverFeat(64,1,Trainable=False,InitC=IDFTM_Half)
        
        # CC part
        ncc=myParams.myDict['CC_channels']
        # CC: model.add_conv2dC(ncc,mapsize=1) # now (16, 64, 128, ncc, 2)
        model.add_einsumC('abcd,bcdx->abcx',[64,128,8, ncc])

        # back to k-space space on RO
        model.add_Mult2DMCyCSharedOverFeat(W,1,Trainable=False,InitC=DFTM)
        # YCC: also on PE
        model.add_Mult2DMCxCSharedOverFeat(64,1,Trainable=False,InitC=DFTM_Half)

        # now conv, from 8 to 2
        model.add_conv2dC(2,mapsize=3) # now (16, 64, 128, 2, 2)
        # now put combine the 2 with the 64
        model.add_Permute([0, 1, 3, 2, 4]) # now (16, 64, 2, 128, 2)
        model.add_Reshape([16, 128, 128, 1,2])
        
        
        # model.add_Mult2DMCxCSharedOverFeat(H,1,NamePrefix='MapsForMat')
        model.add_Mult2DMCxCSharedOverFeat(H,1,Trainable=False,InitC=IDFTM)
        model.add_Mult2DMCyCSharedOverFeat(W,1,Trainable=False,InitC=IDFTM)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'SMASH1DFTxyC_XCC':
        print("SMASH1DFTxyC_XCC mode")

        # model.print_size('AAA') # (16, 64, 128, 16)
        model.add_Split4thDim(2) # now (16, 64, 128, 8, 2)

        DFTM=GT.DFT_matrix(H)
        IDFTM=GT.IDFT_matrix(H)

        # back to image space on RO
        model.add_Mult2DMCyCSharedOverFeat(W,1,Trainable=False,InitC=IDFTM)
        
        # CC part
        ncc=myParams.myDict['CC_channels']
        # CC: model.add_conv2dC(ncc,mapsize=1) # now (16, 64, 128, ncc, 2)
        model.add_einsumC('abcd,bcdx->abcx',[64,128,8, ncc])

        # back to k-space space on RO
        model.add_Mult2DMCyCSharedOverFeat(W,1,Trainable=False,InitC=DFTM)

        # now conv, from 8 to 2
        model.add_conv2dC(2,mapsize=3) # now (16, 64, 128, 2, 2)
        # now put combine the 2 with the 64
        model.add_Permute([0, 1, 3, 2, 4]) # now (16, 64, 2, 128, 2)
        model.add_Reshape([16, 128, 128, 1,2])
        
        
        # model.add_Mult2DMCxCSharedOverFeat(H,1,NamePrefix='MapsForMat')
        model.add_Mult2DMCxCSharedOverFeat(H,1,Trainable=False,InitC=IDFTM)
        model.add_Mult2DMCyCSharedOverFeat(W,1,Trainable=False,InitC=IDFTM)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'SMASH1DFTxyC_GCC':
        print("SMASH1DFTxyC_GCC mode")

        # model.print_size('AAA') # (16, 64, 128, 16)
        model.add_Split4thDim(2) # now (16, 64, 128, 8, 2)

        DFTM=GT.DFT_matrix(H)
        IDFTM=GT.IDFT_matrix(H)

        # back to image space on RO
        model.add_Mult2DMCyCSharedOverFeat(W,1,Trainable=False,InitC=IDFTM)
        
        # CC part
        ncc=myParams.myDict['CC_channels']
        # CC: model.add_conv2dC(ncc,mapsize=1) # now (16, 64, 128, ncc, 2)
        model.add_einsumC('abcd,cdx->abcx',[128,8, ncc])

        # back to k-space space on RO
        model.add_Mult2DMCyCSharedOverFeat(W,1,Trainable=False,InitC=DFTM)

        # now conv, from 8 to 2
        model.add_conv2dC(2,mapsize=3) # now (16, 64, 128, 2, 2)
        # now put combine the 2 with the 64
        model.add_Permute([0, 1, 3, 2, 4]) # now (16, 64, 2, 128, 2)
        model.add_Reshape([16, 128, 128, 1,2])
        
        
        # model.add_Mult2DMCxCSharedOverFeat(H,1,NamePrefix='MapsForMat')
        model.add_Mult2DMCxCSharedOverFeat(H,1,Trainable=False,InitC=IDFTM)
        model.add_Mult2DMCyCSharedOverFeat(W,1,Trainable=False,InitC=IDFTM)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'SMASH1DFTxyC_SCC':
        print("SMASH1DFTxyC_SCC mode")

        # model.print_size('AAA') # (16, 64, 128, 16)
        model.add_Split4thDim(2) # now (16, 64, 128, 8, 2)

        DFTM=GT.DFT_matrix(H)
        IDFTM=GT.IDFT_matrix(H)

        # back to image space on RO
        model.add_Mult2DMCyCSharedOverFeat(W,1,Trainable=False,InitC=IDFTM)
        
        # CC part
        ncc=myParams.myDict['CC_channels']
        model.add_conv2dC(ncc,mapsize=1) # now (16, 64, 128, ncc, 2)

        # back to k-space space on RO
        model.add_Mult2DMCyCSharedOverFeat(W,1,Trainable=False,InitC=DFTM)

        # now conv, from 8 to 2
        model.add_conv2dC(2,mapsize=3) # now (16, 64, 128, 2, 2)
        # now put combine the 2 with the 64
        model.add_Permute([0, 1, 3, 2, 4]) # now (16, 64, 2, 128, 2)
        model.add_Reshape([16, 128, 128, 1,2])
        
        
        # model.add_Mult2DMCxCSharedOverFeat(H,1,NamePrefix='MapsForMat')
        model.add_Mult2DMCxCSharedOverFeat(H,1,Trainable=False,InitC=IDFTM)
        model.add_Mult2DMCyCSharedOverFeat(W,1,Trainable=False,InitC=IDFTM)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'SMASH1DFTxyC':
        print("1DFTxyCMaps mode")

        # model.print_size('AAA') # (16, 64, 128, 16)
        model.add_Split4thDim(2) # now (16, 64, 128, 8, 2)
        # now conv, from 8 to 2
        model.add_conv2dC(2,mapsize=3) # now (16, 64, 128, 2, 2)
        # now put combine the 2 with the 64
        model.add_Permute([0, 1, 3, 2, 4]) # now (16, 64, 2, 128, 2)
        model.add_Reshape([16, 128, 128, 1,2])
        
        IDFTM=GT.IDFT_matrix(H)

        # model.add_Mult2DMCxCSharedOverFeat(H,1,NamePrefix='MapsForMat')
        model.add_Mult2DMCxCSharedOverFeat(H,1,Trainable=False,InitC=IDFTM)
        model.add_Mult2DMCyCSharedOverFeat(W,1,Trainable=False,InitC=IDFTM)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == '1DFTxyCMaps':
        print("1DFTxyCMaps mode")

        # model.print_size('AAA') # (16, 128, 128, 16)
        model.add_Split4thDim(2) # now (16, 128, 128, 8, 2)
        # model.print_size('CCC')
        # model.add_Permute45()
        
        model.add_Mult2DMCxCSharedOverFeat(H,1,NamePrefix='MapsForMat')
        model.add_Mult2DMCyCSharedOverFeat(W,1)
        model.add_PixelwiseMultC(1) #,NamePrefix='MapsForMat')
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == '2DFTC':
        print("2DFTC mode")
        model.add_5thDim()
        model.add_Permute45()
        
        model.add_Mult2DMCxC(H*W,1)
        model.remove_5thDim()
        model.add_reshapeTo4D(H,W)
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == '1DFTxyC':
        print("1DFTxyC mode")
        model.add_5thDim()
        model.add_Permute45()
        
        model.add_Mult2DMCxC(H,1)
        model.add_Mult2DMCyC(W,1)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == '1DFTxC':
        print("1DFTxC mode")
        model.add_5thDim()
        model.add_Permute45()
        
        model.add_Mult2DMCxC(H,1)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == '1DFTyC':
        print("1DFTyC mode")
        model.add_5thDim()
        model.add_Permute45()
        
        model.add_Mult2DMCyC(W,1)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == '1DFTy':
        print("1DFTy mode")
        model.add_Mult2DMCy(W,channelsOut)
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == '1DFTx':
        print("1DFTx mode")
        model.add_Mult2DMCx(H,channelsOut)
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == '2DFT':
        print("2DFT mode")
        model.add_Mult2DMCy(W,channelsOut)
        model.add_Mult2DMCx(H,channelsOut)
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    # if myParams.myDict['NetMode'] == 'RegridTry1':
    #     print("RegridTry1 mode")
    #     model.add_PixelwiseMult(2, stddev_factor=1.0)
    #     model.add_Mult2DMCy(W,channelsOut)
    #     model.add_Mult2DMCx(H,channelsOut)
        
    #     new_vars  = tf.global_variables()
    #     gene_vars = list(set(new_vars) - set(old_vars))
    #     return model.get_output(), gene_vars

    # if myParams.myDict['NetMode'] == 'RegridTry1C':
    #     print("RegridTry1C mode")
    #     addBias=myParams.myDict['CmplxBias']>0
    #     if addBias:
    #         print("with bias")
    #     else:
    #         print("without bias")
    #     model.add_PixelwiseMult(2, stddev_factor=1.0)
    #     model.add_5thDim()
    #     model.add_Permute45()
    #     model.add_Mult2DMCyC(W,1,add_bias=addBias)
    #     model.add_Mult2DMCxC(H,1,add_bias=addBias)
    #     model.remove_5thDim()
        
    #     new_vars  = tf.global_variables()
    #     gene_vars = list(set(new_vars) - set(old_vars))
    #     return model.get_output(), gene_vars

    # if myParams.myDict['NetMode'] == 'RegridTry1C2':
    #     print("RegridTry1C2 mode")
    #     addBias=myParams.myDict['CmplxBias']>0
    #     if addBias:
    #         print("with bias")
    #     else:
    #         print("without bias")
    #     model.add_Split4thDim(2)
    #     model.add_PixelwiseMultC(1, stddev_factor=1.0)
    #     model.add_Mult2DMCyC(W,1,add_bias=addBias)
    #     model.add_Mult2DMCxC(H,1,add_bias=addBias)
    #     model.remove_5thDim()
        
    #     new_vars  = tf.global_variables()
    #     gene_vars = list(set(new_vars) - set(old_vars))
    #     return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'RegridTry3C2_TS_WithTSB':
        print("RegridTry3C2_TS_WithTSB mode")
        FullData=scipy.io.loadmat(myParams.myDict['NMAP_FN'])
        NMapCR=FullData['NMapCR']
        NMapCR = tf.constant(NMapCR)

        aDataH=myParams.myDict['aDataH']
        aDataW=myParams.myDict['aDataW']
        achannelsIn=myParams.myDict['achannelsIn']
        nTS=myParams.myDict['nTimeSegments']
        nccInData=myParams.myDict['nccInData']
        nTraj=myParams.myDict['nTraj']
        HalfDataH=np.int32(DataH/2)
        # 133068/2 =  66534
        # model.print_shape('Start') # now 16,133068,1,1
        # model.add_Permute([0,2,3,1])
        # model.add_Split4thDim(2) # now 16,1,1,133068/2,2C
        model.add_Reshape([batch_size,1,1,HalfDataH,2])
        # Now do TSB
        model.add_Permute([0,3,2,1,4]) # now 16,133068/2,1,1,2C
        model.add_Reshape([batch_size,nTraj,nccInData,1,2])
        model.add_Permute([0,2,1,3,4]) # now 16 13 5118 1 2
        model.add_Reshape([batch_size*nccInData,1,nTraj,1,2])
        model.add_PixelwiseMultC(nTS, stddev_factor=1.0) # This is TSB. After: 16*13,1,5118,nTS,2
        model.add_Reshape([batch_size,nccInData,nTraj,nTS,2])
        model.add_Permute([2,1,4,0,3]) # now 5118 13 2 16 nTS
        model.add_Reshape([nTraj*nccInData*2,batch_size*nTS,1,1])


        # model.add_Permute([1,0,2,3])
        # model.print_shape()

        feature=model.get_output();

        feature=tf.gather(feature,NMapCR,validate_indices=None,name=None)
        # feature = tf.reshape(feature, [aDataH, aDataW, achannelsIn])
        model.add_PutInOutput(feature) # After we're 131,131,192,16*nTS

        model.add_Permute([3,0,1,2,4,5]) # Now 16*nTS,131,131,192,1,1

        # model.add_Reshape([batch_size,nTS,aDataH,aDataW,2,96])

        # model.print_shape()
        model.add_Reshape([batch_size*nTS,aDataH,aDataW,achannelsIn]) # Now 16*nTS,131,131,192

        UseSharedWightesInRelaxedFT=myParams.myDict['UseSharedWightesInRelaxedFT']>0
        addBias=myParams.myDict['CmplxBias']>0
        if addBias:
            print("with bias")
        else:
            print("without bias")
            
        model.add_Split4thDim(2) # Now we're batch_size*nTS, kH,kW, Neighbors(12)*Channels(8),2C

        model.add_PixelwiseMultC(1, stddev_factor=1.0) # After we're batch_size*nTS,kH,kW,1,2C

        # AfterRegrid_ForOut = tf.identity(model.get_output(), name="AfterRegrid_ForOut")
        # model.add_PutInOutput(AfterRegrid_ForOut)

        model.add_Reshape([batch_size,nTS,aDataH,aDataW,2])
        model.add_Permute([0,2,3,1,4]) # After we're batch_size,kH,kW,nTS, 2C

        # AfterRegridP_ForOut = tf.identity(model.get_output(), name="AfterRegridP_ForOut")
        # model.add_PutInOutput(AfterRegridP_ForOut)
        # Now continuing as with no TSB

        MM=GT.gDFT_matrix(np.linspace(-50,50,aDataH),H)
        MM=np.transpose(MM,axes=[1,0])
        
        if UseSharedWightesInRelaxedFT:
            model.add_Mult2DMCyCSharedOverFeat(W,1,add_bias=addBias,Trainable=False,InitC=MM,NamePrefix='FTy')
            model.add_Mult2DMCxCSharedOverFeat(H,1,add_bias=addBias,Trainable=False,InitC=MM,NamePrefix='FTx')
        else:
            model.add_Mult2DMCyC(W,1,add_bias=addBias)
            model.add_Mult2DMCxC(H,1,add_bias=addBias)

        # AfterFT_ForOut = tf.identity(model.get_output(), name="AfterFT_ForOut")
        # model.add_PutInOutput(AfterFT_ForOut)

            # now supposedly batch_size,H,W,nTS
        model.add_PixelwiseMultC(1, stddev_factor=1.0,NamePrefix='TSC') # This collecting the different TS to the final image. 

        # AfterTSC_ForOut = tf.identity(model.get_output(), name="AfterTSC_ForOut")
        # model.add_PutInOutput(AfterTSC_ForOut)

        model.remove_5thDim()

        # EndForOut = tf.identity(model.get_output(), name="EndForOut")
        # model.add_PutInOutput(EndForOut)
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'RegridTry3C2FT_TS':
        print("RegridTry3C2_TS mode")
        
        aDataH=myParams.myDict['aDataH']
        kMax=myParams.myDict['kMax']
        aDataW=myParams.myDict['aDataW']
        # achannelsIn=myParams.myDict['achannelsIn']

        nTS=myParams.myDict['nTimeSegments']
        nTSI=myParams.myDict['nTimeSegmentsI']
        
        UseSharedWightesInRelaxedFT=myParams.myDict['UseSharedWightesInRelaxedFT']>0
        RelaxedFT=myParams.myDict['RelaxedFT']>0
        addBias=myParams.myDict['CmplxBias']>0

        # FullData=scipy.io.loadmat(myParams.myDict['NMAP_FN'])
        # NMapCR=FullData['NMapCR']
        # NMapCR = tf.constant(NMapCR)
        nccInData=myParams.myDict['nccInData']
        # ncc=8
        ncc=myParams.myDict['nccToUse']
        
        nNeighbors=myParams.myDict['nNeighbors']

        achannelsIn=ncc*nNeighbors*2
        
        BaseNUFTDataP=myParams.myDict['BaseNUFTDataP']
        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Traj=NUFTData['Trajm2'][0:2,:]

        NMapCR=GT.GenerateNeighborsMap(Traj,kMax,aDataH,nccInData,ncc,nNeighbors)
        NMapCR = tf.constant(NMapCR)

        tmp=model.get_output();
        tmp1=tf.slice(tmp,[0,0,0,0],[batch_size,132964,1,1])
        TSCR=tf.slice(tmp,[0,132964,0,0],[batch_size,128*128*12*1,1,1])
        TSCI=tf.slice(tmp,[0,132964+128*128*12*1,0,0],[batch_size,128*128*12*1,1,1])
        Sens1DR=tf.slice(tmp,[0,132964+128*128*12*2,0,0],[batch_size,128*128*13*1,1,1])
        Sens1DI=tf.slice(tmp,[0,132964+128*128*12*2+128*128*13,0,0],[batch_size,128*128*13*1,1,1])
        TSCR=tf.reshape(TSCR,[batch_size,128,128,12,1])
        TSCI=tf.reshape(TSCI,[batch_size,128,128,12,1])
        Sens1DR=tf.reshape(Sens1DR,[batch_size,128,128,13,1])
        Sens1DI=tf.reshape(Sens1DI,[batch_size,128,128,13,1])
#         TSCRI=tf.concat([tf.stack([TSCR],axis=4),tf.stack([TSCI],axis=4)],axis=4)
#         SensRI=tf.concat([tf.stack([Sens1DR],axis=4),tf.stack([Sens1DI],axis=4)],axis=4)
        TSCRI=tf.concat([TSCR,TSCI],axis=4)
        SensRI=tf.concat([Sens1DR,Sens1DI],axis=4)
        model.add_PutInOutput(tmp1)
        
        model.add_Permute([1,0,2,3]) # now we're 133068,16,1,1
        # model.print_shape()

        feature=model.get_output();

        feature=tf.gather(feature,NMapCR,validate_indices=None,name=None) # After 131,131,192,16
        # feature = tf.reshape(feature, [aDataH, aDataW, achannelsIn])
        model.add_PutInOutput(feature)

        model.add_Permute([3,0,1,2,4,5]) # After 16,131,131,192,1,1

        # model.print_shape()
        model.add_Reshape([batch_size,aDataH,aDataW,achannelsIn]) # After 16,131,131,192

        model.add_Split4thDim(2) # Now we're kH,kW, Neighbors(12)*Channels(8),2

        # model.add_PixelwiseMultC(nTS, stddev_factor=1.0) # After we're batch_size,kH,kW,nTS
        InitForRC=[]
        if myParams.myDict['InitForRFN'] != 'None':
            InitForRM=scipy.io.loadmat(myParams.myDict['InitForRFN'])
            InitForRR=InitForRM['gene_GEN_L007_PixelwiseMultC_weightR_0']
            InitForRI=InitForRM['gene_GEN_L007_PixelwiseMultC_weightI_0']
            InitForRC=InitForRR + 1j * InitForRI
        model.add_PixelwiseMultC(nTS, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=InitForRC)

        # model.print_shape('aaa')
        model.add_CombineFeaturesAndC()
        # or remove_5thDim
        
        # k side here
        model.add_ConvNetFromList(myParams.myDict['kSideNet'])
        #model.add_conv2d(64, mapsize=mapsize, stride=1, stddev_factor=2.)
        #model.add_elu()
        #model.add_conv2dWithName(32, name="ggg", mapsize=1, stride=1, stddev_factor=2.)
        #model.add_elu()
        #model.add_conv2d(nTS*2, mapsize=5, stride=1, stddev_factor=2.)  
        
        # model.print_shape('bbb')
        model.add_Split4thDim(2)
        # model.print_shape('ccc')
        
        model.add_2DFT()
        
        # First I-side here
        model.add_CombineFeaturesAndC()
        
        model.add_ConvNetFromList(myParams.myDict['ISide1Net'])
        #model.add_conv2d(64, mapsize=mapsize, stride=1, stddev_factor=2.)
        #model.add_elu()
        #model.add_conv2dWithName(32, name="ggg", mapsize=1, stride=1, stddev_factor=2.)
        #model.add_elu()
        #model.add_conv2d(nTS*2, mapsize=5, stride=1, stddev_factor=2.)  
        
        model.add_Split4thDim(2)
        
        # Voxelwise I-side:
        InitForLC=[]
        if myParams.myDict['InitForLFN'] != 'None':
            InitForLM=scipy.io.loadmat(myParams.myDict['InitForLFN'])
            InitForLR=InitForLM['gene_GEN_L010_PixelwiseMultC_weightR_0']
            InitForLI=InitForLM['gene_GEN_L010_PixelwiseMultC_weightI_0']
            InitForLC=InitForLR + 1j * InitForLI
        # model.add_PixelwiseMultC(1, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=InitForLC) # This collecting the different TS to the final image. 
        
        if nTSI>0:
            model.add_PixelwiseMultC(nTSI, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=InitForLC) # This collecting the different TS to the final image. 
        
#         ConcatSensTSC=True
        ConcatSensTSC=False
        if ConcatSensTSC:
            # Concat TSC and sens:
            tmp=model.get_output() # 5D, [batch,H,W,Features,2]
    #         tmp = tf.Print(tmp,[],message='tmp shape: '+str(tmp.get_shape())+' '+str(tmp.dtype))
    #         tmp = tf.Print(tmp,[],message='TSCRI shape: '+str(TSCRI.get_shape())+' '+str(TSCRI.dtype))
    #         tmp = tf.Print(tmp,[],message='SensRI shape: '+str(SensRI.get_shape())+' '+str(SensRI.dtype))
            tmp=tf.concat([tmp,TSCRI,SensRI],axis=3)
    #         tmp = tf.Print(tmp,[],message='tmpx shape: '+' '+str(tmp.get_shape()))
            model.add_PutInOutput(tmp)
        
        tmp=model.get_output() # 5D, [batch,H,W,Features,2]
#         tmp = tf.Print(tmp,[],message='tmp shape: '+str(tmp.get_shape())+' '+str(tmp.dtype))
        tmpR=tf.slice(tmp,[0,0,0,0,0],[batch_size,128,128,12,1])
        tmpI=tf.slice(tmp,[0,0,0,0,1],[batch_size,128,128,12,1])
        RR=tf.multiply(tmpR,TSCR)
        RI=tf.multiply(tmpR,TSCI)
        IR=tf.multiply(tmpI,TSCR)
        II=tf.multiply(tmpI,TSCI)
        R=RR+II
        I=-RI+IR
        tmp=tf.concat([R,I],axis=4)
        tmp=tf.reduce_sum(tmp,axis=3)
        model.add_PutInOutput(tmp)
        
#         tmp=model.get_output() # 5D, [batch,H,W,Features,2]
#         tmp2=tf.squeeze(tf.slice(TSCRI,[0,0,0,8,0],[batch_size,128,128,1,2]))
#         tmp2=tf.squeeze(tf.slice(SensRI,[0,0,0,8,0],[batch_size,128,128,1,2]))
#         tmp=tmp*0+tmp2
#         model.add_PutInOutput(tmp)
        
        # Final I-side here
        model.add_CombineFeaturesAndC()
        
        model.add_ConvNetFromList(myParams.myDict['ISide2Net'])
        #model.add_conv2d(64, mapsize=mapsize, stride=1, stddev_factor=2.)
        #model.add_elu()
        #model.add_conv2dWithName(32, name="ggg", mapsize=1, stride=1, stddev_factor=2.)
        #model.add_elu()
        #model.add_conv2d(2, mapsize=5, stride=1, stddev_factor=2.)  
        
        # model.add_Split4thDim(2)
        
        # model.remove_5thDim()
        
#         if myParams.myDict['ISide2Net'][-2]==4:
#             print("MB")
#             model.add_Split4thDim(2)
#             model.add_Permute34()
#             model.add_Combine34(True)
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars
    
    if myParams.myDict['NetMode'] == 'GenRegridCNN':
        print("GenRegridCNN mode")
        
        aDataH=myParams.myDict['aDataH']
        kMax=myParams.myDict['kMax']
        aDataW=myParams.myDict['aDataW']
        nTS=myParams.myDict['nTimeSegments']
        nTSI=myParams.myDict['nTimeSegmentsI']
        UseSharedWightesInRelaxedFT=myParams.myDict['UseSharedWightesInRelaxedFT']>0
        RelaxedFT=myParams.myDict['RelaxedFT']>0
        addBias=myParams.myDict['CmplxBias']>0
        nccInData=myParams.myDict['nccInData']
        ncc=myParams.myDict['nccToUse']
        nNeighbors=myParams.myDict['nNeighbors']

        DataH=myParams.myDict['DataH']
        DataW=myParams.myDict['DataW']
        LabelsH=myParams.myDict['LabelsH']
        LabelsW=myParams.myDict['LabelsW']
        H=LabelsH
        W=LabelsW
    
        achannelsIn=ncc*nNeighbors*2
        
        BaseNUFTDataP=myParams.myDict['BaseNUFTDataP']
        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Traj=NUFTData['Trajm2'][0:2,:]
        Kd=NUFTData['Kd']
        P=NUFTData['P']
        SN=NUFTData['SN']
        
#         NMap,DMap=GT.GenerateNeighborsMapBaseExt(Traj,kMax,aDataH,nNeighbors)
        NMap,DMap=GT.GenerateNeighborsMapBaseExt(Traj,64,aDataH,nNeighbors)
        NMap=tf.constant(NMap)
        # DMap H,W,nNeighbors,2
        DMapx=np.reshape(DMap,(1,H,W,nNeighbors*2))
        DMapx=np.tile(DMapx,(batch_size,1,1,1))/H
        print('DMapx')
        print(str(DMapx.shape))
        
        DMapx=tf.constant(DMapx)
        
        X=np.arange(-63,65)
        X=np.tile(X,(H,1))
        X=X.astype(np.float32)
        X=np.reshape(X,(1,H,W))
        Y=np.transpose(X,(0,2,1))
        XY=np.stack((X,Y),axis=3)
        XY=np.tile(XY,(batch_size,1,1,1))
        XY=XY/H
        XY=tf.constant(XY)
        
#         print('XY')
#         print(str(XY.shape))
        
        
        
        nTraj=Traj.shape[1]
        nCh=nccInData
        
        tmp=model.get_output()
        tmp1R=tf.slice(tmp,[0,0,0,0],[batch_size,nTraj*nCh,1,1])
        tmp1I=tf.slice(tmp,[0,nTraj*nCh,0,0],[batch_size,nTraj*nCh,1,1])
        SigC=tf.complex(tf.reshape(tmp1R,[batch_size,nCh,nTraj]),tf.reshape(tmp1I,[batch_size,nCh,nTraj]))
        SigC=tf.transpose(SigC,[0,2,1])
        
#         Traj.shape[0]
        nTSC=12
    
#         Sens1DR=tf.slice(tmp,[0,132964+H*W*nTSC*2,0,0],[batch_size,H*W*nCh*1,1,1])
#         Sens1DI=tf.slice(tmp,[0,132964+H*W*nTSC*2+H*W*nCh,0,0],[batch_size,H*W*nCh*1,1,1])
#         Sens1DR=tf.reshape(Sens1DR,[batch_size,H,W,nCh])
#         Sens1DI=tf.reshape(Sens1DI,[batch_size,H,W,nCh])
#         SensC=tf.complex(Sens1DR,Sens1DI) # batch_size, H, W, nCh

        Reg=tf.gather(SigC,NMap,validate_indices=None,name=None,axis=1) #batch_size, H, W, nNeighbors, nCh
        Reg=tf.reshape(Reg,(batch_size, H, W, nNeighbors*nCh))
        RegRI=GT.ConcatCOnDim(Reg,3)
        model.add_PutInOutput(RegRI)
        model.add_concat(DMapx)
        model.add_concat(XY)
        
        UseBN=GT.getparam('UseBN')
        
        model.add_ConvNetFromListWithNameAndScope( myParams.myDict['ISide1Net'],name='Net',scope='ConvNet',UseBN=UseBN)
        
        F_RI=model.get_output()
        
        def RItoCon4(X): return tf.squeeze(tf.complex(tf.slice(X,[0,0,0,0],[batch_size,H,W,1]),tf.slice(X,[0,0,0,1],[batch_size,H,W,1])))
        
        F_C=RItoCon4(F_RI)
#         F_C=tf.squeeze( tf.complex(tf.slice(F_RI,[0,0,0,0],[batch_size,H,W,1]),tf.slice(F_RI,[0,0,0,1],[batch_size,H,W,1])) )
        
#         F_C=tf.transpose(F_C,[0,2,1])
        AfterFT=tf.ifft2d(F_C)
#         AfterFT=F_C
        
        initR   = GT._glorot_initializer_g((1,H,W), stddev_factor=2)
        initI   = GT._glorot_initializer_g((1,H,W), stddev_factor=2)
        
        weightR = tf.get_variable("AfterFTR", initializer=initR)
        weightI = tf.get_variable("AfterFTI", initializer=initI)
        
        weightC=tf.complex(weightR,weightI)
        
        AfterFT=tf.multiply(AfterFT,weightC) # batch_size, H, W
        
#         AfterFT=tf.transpose(AfterFT,[0,2,3,1]) # so batch_size,H,W,nCh
        
        FinalRI=GT.ConcatCOnDimWithStack(AfterFT,3)
        
        model.add_PutInOutput(FinalRI)
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars
    
    if myParams.myDict['NetMode'] == 'PerChannelKer':
        print("96 mode")
        
        aDataH=myParams.myDict['aDataH']
        kMax=myParams.myDict['kMax']
        aDataW=myParams.myDict['aDataW']
        nTS=myParams.myDict['nTimeSegments']
        nTSI=myParams.myDict['nTimeSegmentsI']
        UseSharedWightesInRelaxedFT=myParams.myDict['UseSharedWightesInRelaxedFT']>0
        RelaxedFT=myParams.myDict['RelaxedFT']>0
        addBias=myParams.myDict['CmplxBias']>0
        nccInData=myParams.myDict['nccInData']
        ncc=myParams.myDict['nccToUse']
        nNeighbors=myParams.myDict['nNeighbors']

        DataH=myParams.myDict['DataH']
        DataW=myParams.myDict['DataW']
        LabelsH=myParams.myDict['LabelsH']
        LabelsW=myParams.myDict['LabelsW']
        H=LabelsH
        W=LabelsW
    
        achannelsIn=ncc*nNeighbors*2
        
        BaseNUFTDataP=myParams.myDict['BaseNUFTDataP']
        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Traj=NUFTData['Trajm2'][0:2,:]
        Kd=NUFTData['Kd']
        P=NUFTData['P']
        SN=NUFTData['SN']
        
        NMap=GT.GenerateNeighborsMapBase(Traj,kMax,aDataH,nNeighbors)
        NMap=tf.constant(NMap)
        
        nTraj=Traj.shape[1]
        nCh=nccInData
        
        X=np.arange(-63,65)
        X=np.tile(X,(H,1))
        X=X.astype(np.float32)
        X=np.reshape(X,(1,H,W))
        Y=np.transpose(X,(0,2,1))
        XY=np.stack((X,Y),axis=3)
        XY=np.tile(XY,(batch_size,1,1,1))
        XY=XY/H
        XY=tf.constant(XY)
        
        tmp=model.get_output()
        tmp1R=tf.slice(tmp,[0,0,0,0],[batch_size,nTraj*nCh,1,1])
        tmp1I=tf.slice(tmp,[0,nTraj*nCh,0,0],[batch_size,nTraj*nCh,1,1])
        SigC=tf.complex(tf.reshape(tmp1R,[batch_size,nCh,nTraj]),tf.reshape(tmp1I,[batch_size,nCh,nTraj]))
        SigC=tf.transpose(SigC,[0,2,1])
        
#         Traj.shape[0]
        nTSC=12
    
        Sens1DR=tf.slice(tmp,[0,132964+H*W*nTSC*2,0,0],[batch_size,H*W*nCh*1,1,1])
        Sens1DI=tf.slice(tmp,[0,132964+H*W*nTSC*2+H*W*nCh,0,0],[batch_size,H*W*nCh*1,1,1])
        Sens1DR=tf.reshape(Sens1DR,[batch_size,H,W,nCh])
        Sens1DI=tf.reshape(Sens1DI,[batch_size,H,W,nCh])
        SensC=tf.complex(Sens1DR,Sens1DI) # batch_size, H, W, nCh
        SensRI=tf.concat([Sens1DR,Sens1DI],axis=3)
        

        Reg=tf.gather(SigC,NMap,validate_indices=None,name=None,axis=1) #batch_size, H, W, nNeighbors, nCh
        
        initR   = GT._glorot_initializer_g((1,H,W,nNeighbors,1), stddev_factor=2)
        initI   = GT._glorot_initializer_g((1,H,W,nNeighbors,1), stddev_factor=2)
        
        weightR = tf.get_variable("PerChannelKernelR", initializer=initR)
        weightI = tf.get_variable("PerChannelKernelI", initializer=initI)
        
        weightC=tf.complex(weightR,weightI)
        
        Res=tf.reduce_sum(tf.multiply(Reg,weightC) ,axis=3) # batch_size, H, W, nCh
        
        HalfH=H/2
        HalfW=W/2
            
        IdH=tf.concat([tf.range(HalfH,H), tf.range(0,HalfH)],axis=0)
        IdH=tf.cast(IdH,tf.int32)
        
        IdW=tf.concat([tf.range(HalfW,W), tf.range(0,HalfW)],axis=0)
        IdW=tf.cast(IdW,tf.int32)
        
        C = tf.gather(Res,IdH,axis=1)
        C = tf.gather(C,IdW,axis=2)
        
#         C = tf.Print(C,[],message=message + ' C shape: '+str(C.get_shape())+' '+str(C.dtype))
        
        BeforeFT=tf.transpose(C,[0,3,1,2]) # so batch_size,nCh,H,W
        
        BeforeFT=tf.transpose(BeforeFT,[0,1,3,2]) # so batch_size,nCh,W,H
        
        AfterFT=tf.ifft2d(BeforeFT)
        AfterFT=tf.transpose(AfterFT,[0,2,3,1]) # so batch_size,H,W,nCh
        
        AfterFTRI=GT.ConcatCOnDim(AfterFT,3)
        
        model.add_PutInOutput(AfterFTRI)
        model.add_concat(XY)
        model.add_concat(SensRI)
        
        UseBN=GT.getparam('UseBN')
        
        model.add_ConvNetFromListWithNameAndScope( myParams.myDict['ISide1Net'],name='Net',scope='ConvNet',UseBN=UseBN)
        
#         WithSens=tf.reduce_sum(tf.multiply(AfterFT,tf.conj(SensC)),axis=3) # batch_size, H, W
#         FinalRI=GT.ConcatCOnDimWithStack(WithSens,3)
#         model.add_PutInOutput(FinalRI)
            
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars
    
    
    if myParams.myDict['NetMode'] == 'Cart_ISTA_ITS_MB':
        print("Cart_ISTA_ITS_MB mode")
        
        nTS=GT.getparam('nTimeSegments')
        nTSI=GT.getparam('nTimeSegmentsI')
        UseSharedWightesInRelaxedFT=GT.getparam('UseSharedWightesInRelaxedFT')>0
        nccInData=GT.getparam('nccInData')
        ncc=GT.getparam('nccToUse')
        nTSC=GT.getparam('nTimeSegments')
        LabelsH=GT.getparam('LabelsH')
        LabelsW=GT.getparam('LabelsW')
        H=LabelsH
        W=LabelsW
        nCh=ncc

        kMax=H/2
#         kMax=GT.getparam('kMax')
        MB=GT.getparam('MB')
#         MB=1
        
        X=np.arange(-kMax+1,kMax+1)
        X=np.tile(X,(H,1))
        X=X.astype(np.float32)
        X=np.reshape(X,(1,H,W))
        Y=np.transpose(X,(0,2,1))
        XY=np.stack((X,Y),axis=3)
        XY=np.tile(XY,(batch_size,1,1,1))
        XY=XY/H
        XY=tf.constant(XY)
        
        CurLoc=0
        
        tmp=model.get_output()
        
        def ReadCFrom1D(Data,Loc,Sz):
            ProdSz=np.prod(Sz)
            tmp1R=Data[:,Loc:(Loc+ProdSz),:1,:1]
            tmp1I=Data[:,Loc+ProdSz:Loc+2*ProdSz,:1,:1]
            NewSz=np.concatenate(([-1],Sz),axis=0)
            C=tf.complex(tf.reshape(tmp1R,NewSz),tf.reshape(tmp1I,NewSz))
            AddedLoc=ProdSz*2
            return C,AddedLoc
            
        SensC6,ToAddToLoc=ReadCFrom1D(tmp,CurLoc,(H,W,nCh,MB)) # batch_size, H, W, nCh,1
        CurLoc=CurLoc+ToAddToLoc
        
        AHA_ITS,ToAddToLoc=ReadCFrom1D(tmp,CurLoc,(H,W,nTSC*MB)) # batch_size, H, W, nTSC*MB
        CurLoc=CurLoc+ToAddToLoc

        SendTSCest=GT.getparam('SendTSCest')>0
        if SendTSCest:
            TSCest,ToAddToLoc=ReadCFrom1D(tmp,CurLoc,(H,W,nTSC,MB)) # batch_size, H,W,nTS,MB
            CurLoc=CurLoc+ToAddToLoc

        AHA_ITSRI=GT.ConcatCOnDim(AHA_ITS,3) # batch_size, H, W, nTSC*MB*RI

        SendWarmStart=GT.getparam('SendWarmStart')>0
        if SendWarmStart:
            WarmStart,ToAddToLoc=ReadCFrom1D(tmp,CurLoc,(H,W,nTSC*MB)) # batch_size, H,W,nTS,MB
            CurLoc=CurLoc+ToAddToLoc

            y0RI=GT.ConcatCOnDim(WarmStart,3) # batch_size, H, W, nTSC*MB*RI
            # ShapeA=AHA_ITSRI.get_shape()
            # ShapeB=y0RI.get_shape()
            # y0RI=tf.Print(y0RI,[],'AHA_ITSRI shape '+str(ShapeA))
            # y0RI=tf.Print(y0RI,[],'y0RI shape '+str(ShapeB))
        else:
            y0RI=AHA_ITSRI*0

#         y0RI=GT.ConcatCOnDim(y0,3) # batch_size, H, W, nTSC*MB*RI

        # SensC6 is batch_size, H, W, nCh,MB
        # SensC6 should be H,W,/nTSC/,nCh,MB,batch_size
        SensC6=tf.transpose(GT.TF_addDim(SensC6),[1,2,5,3,4,0])
        
        
        model.add_PutInOutput(y0RI)
        
#         NetList=GT.getparam('ISide1Net')
        NetList=copy.deepcopy(GT.getparam('ISide1Net'))
        nExtraFeatures=NetList[-2]
#         nExtraFeatures=0
#         ExtraFeatures=tf.constant(tf.zeros([batch_size,H,W,nExtraFeatures]),tf.float32)
        ExtraFeatures=tf.zeros([batch_size,H,W,nExtraFeatures],tf.float32)
        model.add_concat(ExtraFeatures)
        NetList[-2]=nTSC*MB*2+nExtraFeatures
        
#         model.add_concat(XY)
#         model.add_concat(SensRI)

        UseBN=GT.getparam('UseBN')
#         model.add_ConvNetFromListWithNameAndScope(myParams.myDict['ISide1Net'],name='Net',scope='ConvNet',UseBN=UseBN,AddDirectConnection=True)
        
        Iters=GT.getparam('Iterations')
        nIter=Iters.shape[0]
#         nIter=1

#         AllItersRes=tf.Variable(tf.zeros([batch_size,H*(nIter+1),W*nTS*MB,2],tf.float32))
        ResList=[]
    
        def RItoCon5(X): return tf.squeeze(tf.complex(tf.slice(X,[0,0,0,0,0],[-1,-1,-1,-1,1]),tf.slice(X,[0,0,0,0,1],[-1,-1,-1,-1,1])),axis=4)
        def RItoCon6(X): return tf.squeeze(tf.complex(tf.slice(X,[0,0,0,0,0,0],[-1,-1,-1,-1,-1,1]),tf.slice(X,[0,0,0,0,0,1],[-1,-1,-1,-1,-1,1])),axis=5)
        
        CartMask=GT.getparam('CartMask')
        
        for Iter in range(0, nIter):
            CurEstRIa=model.get_output() # batch_size,H,W,2*nTS*MB
            Cury0FeatRI=CurEstRIa[:,:,:,:2*nTS*MB]
            ExtraFeatures=CurEstRIa[:,:,:,2*nTS*MB:]
            CurEstRIa=tf.reshape(Cury0FeatRI,[batch_size,H,W,2,nTS,MB])
            CurEstRI=tf.transpose(CurEstRIa,[0,1,2,4,5,3])
            CurEst=RItoCon6(CurEstRI) #  batch_size,H,W,nTSC,MB
            
            # new simpler approach
            if SendTSCest:
                CurEstForAHA=CurEst*TSCest
            else:
                CurEstForAHA=CurEst


#             if SendTSCest:
#                 CurEstForAHA=CurEst*TSCest
#                 Finaln_ITS_RI=GT.ConcatCOnDimWithStack(CurEstForAHA,5) #  batch_size,H,W,nTSC,MB, RI
#                 Finaln_ITS_RI=tf.transpose(Finaln_ITS_RI,[0,1,4,3,2,5]) #  batch_size,H,MB,nTSC,W,RI
#             else:
            # new simpler approach removed this:
            # CurEstForAHA=CurEst
#             TSCest0=tf.Print(TSCest0,[tfrm(TSCest0)],'TSCest0 ')

#             print_op = tf.print("tensors:", CurEstForAHA.get_shape(),output_stream=sys.stderr)
#             with tf.control_dependencies([print_op]):
#               CurEstForAHA = CurEstForAHA * 1

            Finaln_ITS_RI=tf.transpose(CurEstRIa,[0,1,5,4,2,3]) #  batch_size,H,MB,nTSC,W,RI
            Finaln_ITS_RI=tf.reshape(Finaln_ITS_RI,[batch_size,H,W*nTSC*MB,2])
            ResList.append(Finaln_ITS_RI)
#             AllItersRes[:,Iter*H:(Iter+1)*H,:,:]=Finaln_ITS_RI
#             AllItersRes=tf.assign(AllItersRes[:,Iter*H:(Iter+1)*H,:,:],Finaln_ITS_RI)
#             AllItersRes = AllItersRes[:,Iter*H:(Iter+1)*H,:,:].assign(Finaln_ITS_RI)

            # InImage is batch_size,H,W,nTSC,MB
#             AHA_CurEst=GT.TS_NUFFT_OPHOP_ITS_MB(CurEstForAHA,SensC6,H,W,batch_size,paddingsYMB,nTSC,nCh,fftkernc7)
            AHA_CurEst=GT.Cartesian_OPHOP_ITS_MB(CurEstForAHA,SensC6,CartMask)
            # AHA_CurEst = tf.Print(AHA_CurEst,[],message='AAAAAAA')
            # new simpler approach
            if SendTSCest:
                # print('Applying TSCest')
                # AHA_CurEst = tf.Print(AHA_CurEst,[],message='xx Applying TSCest')
                AHA_CurEst=AHA_CurEst*tf.conj(TSCest)

            # batch_size,H,W,nTSC,MB?
            AHA_CurEst=tf.reshape(AHA_CurEst,[batch_size,H,W,nTS*MB])
            AHA_CurEstRI=GT.ConcatCOnDim(AHA_CurEst,3)
            
            model.add_PutInOutput(Cury0FeatRI)
            model.add_concat(AHA_ITSRI)
            model.add_concat(AHA_CurEstRI)
            
            model.add_concat(ExtraFeatures)
        
            NetName='Net'+str(Iters[Iter])
            model.add_ConvNetFromListWithNameAndScope(NetList,name=NetName,scope='ConvNet',UseBN=UseBN,AddDirectConnection=True) # , stddev_factor=0.3
        
        # test to show warm start
#         t0= tf.get_variable('t0', initializer=tf.cast(1.0,tf.float32))
#         model.add_PutInOutput(y0RI+t0*0)
        Iter=Iter+1

        CurEstRIa=model.get_output() # batch_size,H,W,2*nTS*MB
        CurEstRIa=CurEstRIa[:,:,:,:2*nTS*MB]
        CurEstRIa=tf.reshape(CurEstRIa,[batch_size,H,W,2,nTS,MB])
        
        CurEstRI=tf.transpose(CurEstRIa,[0,1,2,4,5,3])
        CurEst=RItoCon6(CurEstRI) #  batch_size,H,W,nTSC,MB

#         if SendTSCest:
#             CurEstForAHA=CurEst*TSCest
#             Finaln_ITS_RI=GT.ConcatCOnDimWithStack(CurEstForAHA,5) #  batch_size,H,W,nTSC,MB, RI
#             Finaln_ITS_RI=tf.transpose(Finaln_ITS_RI,[0,1,4,3,2,5]) #  batch_size,H,MB,nTSC,W,RI
#         else:
        # CurEstForAHA=CurEst
        Finaln_ITS_RI=tf.transpose(CurEstRIa,[0,1,5,4,2,3]) #  batch_size,H,MB,nTSC,W,RI

#         Finaln_ITS_RI=model.get_output() # batch_size,H,W,2*nTS*MB
#         Finaln_ITS_RI=Finaln_ITS_RI[:,:,:,:2*nTS*MB]
#         Finaln_ITS_RI=tf.reshape(Finaln_ITS_RI,[batch_size,H,W,2,nTS,MB])
#         Finaln_ITS_RI=tf.transpose(Finaln_ITS_RI,[0,1,5,4,2,3]) #  batch_size,H,W,nTSC,MB,RI
        Finaln_ITS_RI=tf.reshape(Finaln_ITS_RI,[batch_size,H,W*nTSC*MB,2])
#         AllItersRes[:,Iter*H:(Iter+1)*H,:,:]=Finaln_ITS_RI
#         AllItersRes=tf.assign(AllItersRes[:,Iter*H:(Iter+1)*H,:,:],Finaln_ITS_RI)
#         AllItersRes = AllItersRes[:,Iter*H:(Iter+1)*H,:,:].assign(Finaln_ITS_RI)
        ResList.append(Finaln_ITS_RI)
        AllItersRes=tf.concat(ResList,axis=1)
#         model.add_PutInOutput(Finaln_ITS_RI)
        model.add_PutInOutput(AllItersRes)
    
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars
    
    
    
    
    
    if myParams.myDict['NetMode'] == 'ISTA_ITS_MB':
        print("ISTA_ITS_MB mode")
        
        aDataH=GT.getparam('aDataH')
        kMax=GT.getparam('kMax')
        aDataW=GT.getparam('aDataW')
        nTS=GT.getparam('nTimeSegments')
        nTSI=GT.getparam('nTimeSegmentsI')
        UseSharedWightesInRelaxedFT=GT.getparam('UseSharedWightesInRelaxedFT')>0
        RelaxedFT=GT.getparam('RelaxedFT')>0
        addBias=GT.getparam('CmplxBias')>0
        nccInData=GT.getparam('nccInData')
        ncc=GT.getparam('nccToUse')
        nNeighbors=GT.getparam('nNeighbors')
        nTSC=GT.getparam('nTimeSegments')
        LabelsH=GT.getparam('LabelsH')
        LabelsW=GT.getparam('LabelsW')
        H=LabelsH
        W=LabelsW
        nCh=ncc

        MB=GT.getparam('MB')
        
        X=np.arange(-kMax+1,kMax+1)
        X=np.tile(X,(H,1))
        X=X.astype(np.float32)
        X=np.reshape(X,(1,H,W))
        Y=np.transpose(X,(0,2,1))
        XY=np.stack((X,Y),axis=3)
        XY=np.tile(XY,(batch_size,1,1,1))
        XY=XY/H
        XY=tf.constant(XY)
        
        CurLoc=0
        
        tmp=model.get_output()
        
        def ReadCFrom1D(Data,Loc,Sz):
            ProdSz=np.prod(Sz)
            tmp1R=Data[:,Loc:(Loc+ProdSz),:1,:1]
            tmp1I=Data[:,Loc+ProdSz:Loc+2*ProdSz,:1,:1]
            NewSz=np.concatenate(([-1],Sz),axis=0)
            C=tf.complex(tf.reshape(tmp1R,NewSz),tf.reshape(tmp1I,NewSz))
            AddedLoc=ProdSz*2
            return C,AddedLoc
            
        SensC6,ToAddToLoc=ReadCFrom1D(tmp,CurLoc,(H,W,nCh,MB)) # batch_size, H, W, nCh,1
        CurLoc=CurLoc+ToAddToLoc
        
        SendTSCest=GT.getparam('SendTSCest')>0
        if SendTSCest:
            TSCest,ToAddToLoc=ReadCFrom1D(tmp,CurLoc,(H,W,nTSC,MB)) # batch_size, H,W,nTS,MB
            CurLoc=CurLoc+ToAddToLoc
        
        SendSig=False
        if SendSig:
            NMap=GT.getparam('NMap')
            Kd=GT.getparam('Kd')
            nTraj=GT.getparam('nTraj')
            TSBF=GT.getparam('TSBF')
            SN=GT.getparam('SN')
            sp_C=GT.getparam('sp_C')
            
            SigC,ToAddToLoc=ReadCFrom1D(tmp,CurLoc,(nCh,nTraj))
            CurLoc=CurLoc+ToAddToLoc
            SigC=tf.transpose(SigC,[0,2,1])
            
            Reg=tf.gather(SigC,NMap,validate_indices=None,name=None,axis=1) #batch_size, H, W, nNeighbors, nCh
        
            initR   = GT._glorot_initializer_g((1,H,W,nNeighbors,1,nTS), stddev_factor=2)
            initI   = GT._glorot_initializer_g((1,H,W,nNeighbors,1,nTS), stddev_factor=2)

            weightR = tf.get_variable("PerChannelKernelR", initializer=initR)
            weightI = tf.get_variable("PerChannelKernelI", initializer=initI)

            weightC=tf.complex(weightR,weightI)

            Reg=GT.TF_5d_to_6d(Reg)
            Res=tf.reduce_sum(tf.multiply(Reg,weightC) ,axis=3) # batch_size, H, W, nCh, nTS

            HalfH=H/2
            HalfW=W/2

            IdH=tf.concat([tf.range(HalfH,H), tf.range(0,HalfH)],axis=0)
            IdH=tf.cast(IdH,tf.int32)

            IdW=tf.concat([tf.range(HalfW,W), tf.range(0,HalfW)],axis=0)
            IdW=tf.cast(IdW,tf.int32)

            C = tf.gather(Res,IdH,axis=1)
            C = tf.gather(C,IdW,axis=2)

            BeforeFT=tf.transpose(C,[0,3,4,1,2]) # so batch_size,nCh,nTS,H,W

            BeforeFT=tf.transpose(BeforeFT,[0,1,2,4,3]) # so batch_size,nCh,nTS,W,H

            AfterFT=tf.ifft2d(BeforeFT)
            AfterFT=tf.transpose(AfterFT,[0,3,4,1,2]) # so batch_size,H,W,nCh,nTS

            y0=tf.reduce_sum( tf.multiply(AfterFT, tf.conj(SensC5)),axis=3) # batch_size,H,W,nTS
        else:
            y0,ToAddToLoc=ReadCFrom1D(tmp,CurLoc,(H,W,nTSC*MB)) # batch_size, H, W, nTSC*MB
            CurLoc=CurLoc+ToAddToLoc
        
        paddingsYMB=GT.getparam('paddingsYMB')
        fftkernc7=GT.getparam('fftkernc7')
        
        AHA_ITS,ToAddToLoc=ReadCFrom1D(tmp,CurLoc,(H,W,nTSC*MB)) # batch_size, H, W, nTSC*MB
        CurLoc=CurLoc+ToAddToLoc
        
        AHA_ITSRI=GT.ConcatCOnDim(AHA_ITS,3) # batch_size, H, W, nTSC*MB*RI

        y0RI=GT.ConcatCOnDim(y0,3) # batch_size, H, W, nTSC*MB*RI

        # SensC6 is batch_size, H, W, nCh,MB
        # SensC6 should be H,W,/nTSC/,nCh,MB,batch_size
        SensC6=tf.transpose(GT.TF_addDim(SensC6),[1,2,5,3,4,0])
        
        
        model.add_PutInOutput(y0RI)
        
#         NetList=GT.getparam('ISide1Net')
        NetList=copy.deepcopy(GT.getparam('ISide1Net'))
        nExtraFeatures=NetList[-2]
#         nExtraFeatures=0
#         ExtraFeatures=tf.constant(tf.zeros([batch_size,H,W,nExtraFeatures]),tf.float32)
        ExtraFeatures=tf.zeros([batch_size,H,W,nExtraFeatures],tf.float32)
        model.add_concat(ExtraFeatures)
        NetList[-2]=nTSC*MB*2+nExtraFeatures
        
#         model.add_concat(XY)
#         model.add_concat(SensRI)
        
        UseBN=GT.getparam('UseBN')
#         model.add_ConvNetFromListWithNameAndScope(myParams.myDict['ISide1Net'],name='Net',scope='ConvNet',UseBN=UseBN,AddDirectConnection=True)
        
        Iters=GT.getparam('Iterations')
        nIter=Iters.shape[0]
#         nIter=1

#         AllItersRes=tf.Variable(tf.zeros([batch_size,H*(nIter+1),W*nTS*MB,2],tf.float32))
        ResList=[]
    
        def RItoCon5(X): return tf.squeeze(tf.complex(tf.slice(X,[0,0,0,0,0],[-1,-1,-1,-1,1]),tf.slice(X,[0,0,0,0,1],[-1,-1,-1,-1,1])),axis=4)
        def RItoCon6(X): return tf.squeeze(tf.complex(tf.slice(X,[0,0,0,0,0,0],[-1,-1,-1,-1,-1,1]),tf.slice(X,[0,0,0,0,0,1],[-1,-1,-1,-1,-1,1])),axis=5)
        
        for Iter in range(0, nIter):
            CurEstRIa=model.get_output() # batch_size,H,W,2*nTS*MB
            Cury0FeatRI=CurEstRIa[:,:,:,:2*nTS*MB]
            ExtraFeatures=CurEstRIa[:,:,:,2*nTS*MB:]
            CurEstRIa=tf.reshape(Cury0FeatRI,[batch_size,H,W,2,nTS,MB])
            CurEstRI=tf.transpose(CurEstRIa,[0,1,2,4,5,3])
            CurEst=RItoCon6(CurEstRI) #  batch_size,H,W,nTSC,MB
            
            if SendTSCest:
                CurEstForAHA=CurEst*TSCest
                Finaln_ITS_RI=GT.ConcatCOnDimWithStack(CurEstForAHA,5) #  batch_size,H,W,nTSC,MB, RI
                Finaln_ITS_RI=tf.transpose(Finaln_ITS_RI,[0,1,4,3,2,5]) #  batch_size,H,MB,nTSC,W,RI
            else:
                CurEstForAHA=CurEst
                Finaln_ITS_RI=tf.transpose(CurEstRIa,[0,1,5,4,2,3]) #  batch_size,H,MB,nTSC,W,RI
            
            Finaln_ITS_RI=tf.reshape(Finaln_ITS_RI,[batch_size,H,W*nTSC*MB,2])
            ResList.append(Finaln_ITS_RI)
#             AllItersRes[:,Iter*H:(Iter+1)*H,:,:]=Finaln_ITS_RI
#             AllItersRes=tf.assign(AllItersRes[:,Iter*H:(Iter+1)*H,:,:],Finaln_ITS_RI)
#             AllItersRes = AllItersRes[:,Iter*H:(Iter+1)*H,:,:].assign(Finaln_ITS_RI)

            # InImage is batch_size,H,W,nTSC,MB
            AHA_CurEst=GT.TS_NUFFT_OPHOP_ITS_MB(CurEstForAHA,SensC6,H,W,batch_size,paddingsYMB,nTSC,nCh,fftkernc7)
            # batch_size,H,W,nTSC,MB?aux?
            AHA_CurEst=tf.reshape(AHA_CurEst,[batch_size,H,W,nTS*MB])
            AHA_CurEstRI=GT.ConcatCOnDim(AHA_CurEst,3)
            
            model.add_PutInOutput(Cury0FeatRI)
            model.add_concat(AHA_ITSRI)
            model.add_concat(AHA_CurEstRI)
            
            model.add_concat(ExtraFeatures)
        
            NetName='Net'+str(Iters[Iter])
            model.add_ConvNetFromListWithNameAndScope(NetList,name=NetName,scope='ConvNet',UseBN=UseBN,AddDirectConnection=True) # , stddev_factor=0.3
        
        # test to show warm start
#         t0= tf.get_variable('t0', initializer=tf.cast(1.0,tf.float32))
#         model.add_PutInOutput(y0RI+t0*0)
        Iter=Iter+1

        CurEstRIa=model.get_output() # batch_size,H,W,2*nTS*MB
        CurEstRIa=CurEstRIa[:,:,:,:2*nTS*MB]
        CurEstRIa=tf.reshape(CurEstRIa,[batch_size,H,W,2,nTS,MB])
        
        CurEstRI=tf.transpose(CurEstRIa,[0,1,2,4,5,3])
        CurEst=RItoCon6(CurEstRI) #  batch_size,H,W,nTSC,MB

        if SendTSCest:
            CurEstForAHA=CurEst*TSCest
            Finaln_ITS_RI=GT.ConcatCOnDimWithStack(CurEstForAHA,5) #  batch_size,H,W,nTSC,MB, RI
            Finaln_ITS_RI=tf.transpose(Finaln_ITS_RI,[0,1,4,3,2,5]) #  batch_size,H,MB,nTSC,W,RI
        else:
            CurEstForAHA=CurEst
            Finaln_ITS_RI=tf.transpose(CurEstRIa,[0,1,5,4,2,3]) #  batch_size,H,MB,nTSC,W,RI

#         Finaln_ITS_RI=model.get_output() # batch_size,H,W,2*nTS*MB
#         Finaln_ITS_RI=Finaln_ITS_RI[:,:,:,:2*nTS*MB]
#         Finaln_ITS_RI=tf.reshape(Finaln_ITS_RI,[batch_size,H,W,2,nTS,MB])
#         Finaln_ITS_RI=tf.transpose(Finaln_ITS_RI,[0,1,5,4,2,3]) #  batch_size,H,W,nTSC,MB,RI
        Finaln_ITS_RI=tf.reshape(Finaln_ITS_RI,[batch_size,H,W*nTSC*MB,2])
#         AllItersRes[:,Iter*H:(Iter+1)*H,:,:]=Finaln_ITS_RI
#         AllItersRes=tf.assign(AllItersRes[:,Iter*H:(Iter+1)*H,:,:],Finaln_ITS_RI)
#         AllItersRes = AllItersRes[:,Iter*H:(Iter+1)*H,:,:].assign(Finaln_ITS_RI)
        ResList.append(Finaln_ITS_RI)
        AllItersRes=tf.concat(ResList,axis=1)
#         model.add_PutInOutput(Finaln_ITS_RI)
        model.add_PutInOutput(AllItersRes)
    
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars
    
    if myParams.myDict['NetMode'] == 'ISTA_ITS':
        print("ISTA_ITS mode")
        
        aDataH=myParams.myDict['aDataH']
        kMax=myParams.myDict['kMax']
        aDataW=myParams.myDict['aDataW']
        nTS=myParams.myDict['nTimeSegments']
        nTSI=myParams.myDict['nTimeSegmentsI']
        UseSharedWightesInRelaxedFT=myParams.myDict['UseSharedWightesInRelaxedFT']>0
        RelaxedFT=myParams.myDict['RelaxedFT']>0
        addBias=myParams.myDict['CmplxBias']>0
        nccInData=myParams.myDict['nccInData']
        ncc=myParams.myDict['nccToUse']
        nNeighbors=myParams.myDict['nNeighbors']
        nTSC=GT.getparam('nTimeSegments')
#         DataH=myParams.myDict['DataH']
#         DataW=myParams.myDict['DataW']
        LabelsH=myParams.myDict['LabelsH']
        LabelsW=myParams.myDict['LabelsW']
        H=LabelsH
        W=LabelsW
#         nCh=nccInData
        nCh=ncc

        X=np.arange(-63,65)
        X=np.tile(X,(H,1))
        X=X.astype(np.float32)
        X=np.reshape(X,(1,H,W))
        Y=np.transpose(X,(0,2,1))
        XY=np.stack((X,Y),axis=3)
        XY=np.tile(XY,(batch_size,1,1,1))
        XY=XY/H
        XY=tf.constant(XY)
        
        
        tmp=model.get_output()
        CurLoc=0
        
        def ReadCFrom1D(Data,Loc,Sz):
            ProdSz=np.prod(Sz)
            tmp1R=tf.slice(Data,[0,Loc,0,0],[batch_size,ProdSz,1,1])
            tmp1I=tf.slice(Data,[0,Loc+ProdSz,0,0],[batch_size,ProdSz,1,1])
            NewSz=np.concatenate(([-1],Sz),axis=0)
            C=tf.complex(tf.reshape(tmp1R,NewSz),tf.reshape(tmp1I,NewSz))
            AddedLoc=ProdSz*2
            return C,AddedLoc
            
        
#         Sens1DR=tf.slice(tmp,[0, CurLoc,0,0],[batch_size,H*W*nCh*1,1,1])
#         Sens1DI=tf.slice(tmp,[0, CurLoc+H*W*nCh,0,0],[batch_size,H*W*nCh*1,1,1])
#         Sens1DR=tf.reshape(Sens1DR,[batch_size,H,W,nCh])
#         Sens1DI=tf.reshape(Sens1DI,[batch_size,H,W,nCh])
#         SensC=tf.complex(Sens1DR,Sens1DI) # batch_size, H, W, nCh
#         SensC5=GT.TF_4d_to_5d(SensC) # batch_size, H, W, nCh,1
        SensC5,ToAddToLoc=ReadCFrom1D(tmp,CurLoc,(H,W,nCh,1)) # batch_size, H, W, nCh,1
        CurLoc=CurLoc+ToAddToLoc
#         SensRI=tf.concat([Sens1DR,Sens1DI],axis=3)
#         CurLoc=CurLoc+H*W*nCh*2
        
        SendSig=False
        if SendSig:
            NMap=GT.getparam('NMap')
            Kd=GT.getparam('Kd')
            nTraj=GT.getparam('nTraj')
            TSBF=GT.getparam('TSBF')
            SN=GT.getparam('SN')
            sp_C=GT.getparam('sp_C')
            
#             tmp1R=tf.slice(tmp,[0,CurLoc,0,0],[batch_size,nTraj*nCh,1,1])
#             tmp1I=tf.slice(tmp,[0,CurLoc+nTraj*nCh,0,0],[batch_size,nTraj*nCh,1,1])
#             SigC=tf.complex(tf.reshape(tmp1R,[batch_size,nCh,nTraj]),tf.reshape(tmp1I,[batch_size,nCh,nTraj]))
#             SigC=tf.transpose(SigC,[0,2,1])
#             CurLoc=CurLoc+nTraj*nCh*2
            SigC,ToAddToLoc=ReadCFrom1D(tmp,CurLoc,(nCh,nTraj))
            CurLoc=CurLoc+ToAddToLoc
            SigC=tf.transpose(SigC,[0,2,1])
            
            Reg=tf.gather(SigC,NMap,validate_indices=None,name=None,axis=1) #batch_size, H, W, nNeighbors, nCh
        
            initR   = GT._glorot_initializer_g((1,H,W,nNeighbors,1,nTS), stddev_factor=2)
            initI   = GT._glorot_initializer_g((1,H,W,nNeighbors,1,nTS), stddev_factor=2)

            weightR = tf.get_variable("PerChannelKernelR", initializer=initR)
            weightI = tf.get_variable("PerChannelKernelI", initializer=initI)

            weightC=tf.complex(weightR,weightI)

            Reg=GT.TF_5d_to_6d(Reg)
            Res=tf.reduce_sum(tf.multiply(Reg,weightC) ,axis=3) # batch_size, H, W, nCh, nTS

            HalfH=H/2
            HalfW=W/2

            IdH=tf.concat([tf.range(HalfH,H), tf.range(0,HalfH)],axis=0)
            IdH=tf.cast(IdH,tf.int32)

            IdW=tf.concat([tf.range(HalfW,W), tf.range(0,HalfW)],axis=0)
            IdW=tf.cast(IdW,tf.int32)

            C = tf.gather(Res,IdH,axis=1)
            C = tf.gather(C,IdW,axis=2)

            BeforeFT=tf.transpose(C,[0,3,4,1,2]) # so batch_size,nCh,nTS,H,W

            BeforeFT=tf.transpose(BeforeFT,[0,1,2,4,3]) # so batch_size,nCh,nTS,W,H

            AfterFT=tf.ifft2d(BeforeFT)
            AfterFT=tf.transpose(AfterFT,[0,3,4,1,2]) # so batch_size,H,W,nCh,nTS

            y0=tf.reduce_sum( tf.multiply(AfterFT, tf.conj(SensC5)),axis=3) # batch_size,H,W,nTS
        else:
#             y01DR=tf.slice(tmp,[0, CurLoc,0,0],[batch_size,H*W*nTSC*1,1,1])
#             y01DI=tf.slice(tmp,[0, CurLoc+H*W*nTSC,0,0],[batch_size,H*W*nTSC*1,1,1])
#             y0R=tf.reshape(y01DR,[batch_size,H,W,nTSC])
#             y0I=tf.reshape(y01DI,[batch_size,H,W,nTSC])
#             y0=tf.complex(y0R,y0I) # batch_size, H, W, nTSC
#             CurLoc=CurLoc+H*W*nTSC*2
            y0,ToAddToLoc=ReadCFrom1D(tmp,CurLoc,(H,W,nTSC))
            CurLoc=CurLoc+ToAddToLoc
        
        paddingsY=GT.getparam('paddingsY')
        fftkernc5=GT.getparam('fftkernc5')
        
#         AHA_ITS_1DR=tf.slice(tmp,[0, CurLoc,0,0],[batch_size,H*W*nTSC*1,1,1])
#         AHA_ITS_1DI=tf.slice(tmp,[0, CurLoc+H*W*nTSC,0,0],[batch_size,H*W*nTSC*1,1,1])
#         AHA_ITSR=tf.reshape(AHA_ITS_1DR,[batch_size,H,W,nTSC])
#         AHA_ITSI=tf.reshape(AHA_ITS_1DI,[batch_size,H,W,nTSC])
#         AHA_ITS=tf.complex(AHA_ITSR,AHA_ITSI) # batch_size, H, W, nTSC
#         CurLoc=CurLoc+H*W*nTSC*2
        AHA_ITS,ToAddToLoc=ReadCFrom1D(tmp,CurLoc,(H,W,nTSC))
        CurLoc=CurLoc+ToAddToLoc
        
        AHA_ITSRI=GT.ConcatCOnDim(AHA_ITS,3)
        
        y0RI=GT.ConcatCOnDim(y0,3)
        
        # Sens5 is batch_size, H, W, nCh,1
        # Sens5 should be H,W,1,nCh,batch_size
        SensC5=tf.transpose(SensC5,[1,2,4,3,0])
        
        
        model.add_PutInOutput(y0RI)
        
        
#         model.add_concat(XY)
#         model.add_concat(SensRI)
        
        UseBN=GT.getparam('UseBN')
#         model.add_ConvNetFromListWithNameAndScope(myParams.myDict['ISide1Net'],name='Net',scope='ConvNet',UseBN=UseBN,AddDirectConnection=True)
        
        Iters=GT.getparam('Iterations')
        nIter=Iters.shape[0]
#         nIter=1
        
        def RItoCon5(X): return tf.squeeze(tf.complex(tf.slice(X,[0,0,0,0,0],[-1,-1,-1,-1,1]),tf.slice(X,[0,0,0,0,1],[-1,-1,-1,-1,1])),axis=4)
        
        for Iter in range(0, nIter):
            CurEstRI=model.get_output() # batch_size,H,W,2*nTS
            CurEstRI=tf.reshape(CurEstRI,[batch_size,H,W,2,nTS])
            CurEstRI=tf.transpose(CurEstRI,[0,1,2,4,3])
            CurEst=RItoCon5(CurEstRI) #  batch_size,H,W,nTSC
            
            # InImage is batch_size,H,W,nTSC
            AHA_CurEst=GT.TS_NUFFT_OPHOP_ITS(CurEst,SensC5,H,W,1,paddingsY,nTSC,nCh,fftkernc5)
            AHA_CurEstRI=GT.ConcatCOnDim(AHA_CurEst,3)
            
            model.add_concat(AHA_ITSRI)
            model.add_concat(AHA_CurEstRI)
    
            model.add_ConvNetFromListWithNameAndScope( myParams.myDict['ISide1Net'],name='Net'+str(Iters[Iter]),scope='ConvNet',UseBN=UseBN,AddDirectConnection=True) # , stddev_factor=0.3
        
#             model.add_ConvNetFromListWithNameAndScope( myParams.myDict['ISide1Net'],name='Net',scope='ConvNet',UseBN=UseBN,AddDirectConnection=True) # , stddev_factor=0.3
        
        # test to show warm start
#         t0= tf.get_variable('t0', initializer=tf.cast(1.0,tf.float32))
#         model.add_PutInOutput(y0RI+t0*0)
        
        
        # now reshape images
        Finaln_ITS_RI=model.get_output() # batch_size,H,W,2*nTS
        Finaln_ITS_RI=tf.transpose(Finaln_ITS_RI,[0,1,3,2]) # batch_size,H,2*nTS,W
        Finaln_ITS=tf.reshape(Finaln_ITS_RI,[batch_size,H,2,W*nTS])
        Finaln_ITS=tf.transpose(Finaln_ITS,[0,1,3,2]) # batch_size,H,W*nTS,2
        
        model.add_PutInOutput(Finaln_ITS)
    
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars
    
    if myParams.myDict['NetMode'] == 'ISTA_WithB0T2S_Try1':
        print("ISTA_WithB0T2S_Try1 mode")
        def tfrm(X): return tf.reduce_mean(tf.abs(X))
        
        UseBN=GT.getparam('UseBN')
        Iters=GT.getparam('Iterations')
        nIter=Iters.shape[0]
        
        aDataH=myParams.myDict['aDataH']
        kMax=myParams.myDict['kMax']
        aDataW=myParams.myDict['aDataW']
        nTS=myParams.myDict['nTimeSegments']
        nTSI=myParams.myDict['nTimeSegmentsI']
        UseSharedWightesInRelaxedFT=myParams.myDict['UseSharedWightesInRelaxedFT']>0
        RelaxedFT=myParams.myDict['RelaxedFT']>0
        addBias=myParams.myDict['CmplxBias']>0
        nccInData=myParams.myDict['nccInData']
        ncc=myParams.myDict['nccToUse']
        nNeighbors=myParams.myDict['nNeighbors']

        DataH=myParams.myDict['DataH']
        DataW=myParams.myDict['DataW']
        LabelsH=myParams.myDict['LabelsH']
        LabelsW=myParams.myDict['LabelsW']
        H=LabelsH
        W=LabelsW
    
        achannelsIn=ncc*nNeighbors*2
        
#         BaseNUFTDataP=myParams.myDict['BaseNUFTDataP']
#         NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
#         Traj=NUFTData['Trajm2'][0:2,:]
#         Kd=NUFTData['Kd']
#         P=NUFTData['P']
#         SN=NUFTData['SN']
        
#         NMapCR=GT.GenerateNeighborsMap(Traj,kMax,aDataH,nccInData,ncc,nNeighbors)
#         NMapCR = tf.constant(NMapCR)

        
#         nTraj=Traj.shape[1]
        nCh=nccInData
        
        nTSC=GT.getparam('nTimeSegments')
        
        Kd=GT.getparam('Kd')
        nTraj=GT.getparam('nTraj')
        TSBF=GT.getparam('TSBF')
        SN=GT.getparam('SN')
        sp_C=GT.getparam('sp_C')

        ToPad=[Kd[0,0]-H,Kd[0,1]-W]
    
        paddings = tf.constant([[0, ToPad[0]], [0, ToPad[1]],[0,0]])
    
        paddingsX=tf.gather(paddings,[0,1,2],axis=0)
        paddingsY=tf.gather(paddings,[0,1,2,2,2],axis=0)
        
        SNc=tf.stack([tf.stack([tf.stack([tf.constant(SN,dtype=tf.complex64)],axis=2)],axis=3)],axis=4)

#         Idx=scipy.sparse.find(P)
#         I2=np.vstack([Idx[0],Idx[1]]).T
#         I2=tf.constant(np.int64(I2))
#         ValC=tf.constant(np.complex64(Idx[2]))
#         sp_C = tf.SparseTensor(I2, ValC, [P.shape[0],P.shape[1]])

#         BaseTSDataP=GT.getparam('BaseTSDataP')
#         B0Data=scipy.io.loadmat(BaseTSDataP + 'B0TS.mat')
#         TSBF=B0Data['TSBF']

        TSBFX=np.transpose(np.reshape(TSBF,(nTSC,1,nTraj)),axes=(2,0,1))
        TSBFX=tf.constant(np.complex64(TSBFX))
        TSBFXc=tf.stack([TSBFX],axis=3)

        tmp=model.get_output()
#         tmp1=tf.slice(tmp,[0,0,0,0],[batch_size,132964,1,1])
        TSCR=tf.slice(tmp,[0,nTraj*nCh*2,0,0],[batch_size,H*W*nTSC*1,1,1])
        TSCI=tf.slice(tmp,[0,nTraj*nCh*2+H*W*nTSC*1,0,0],[batch_size,H*W*nTSC*1,1,1])
        SensR=tf.slice(tmp,[0,nTraj*nCh*2+H*W*nTSC*2,0,0],[batch_size,H*W*nCh*1,1,1])
        SensI=tf.slice(tmp,[0,nTraj*nCh*2+H*W*nTSC*2+H*W*nCh,0,0],[batch_size,H*W*nCh*1,1,1])
        TSCR4=tf.reshape(TSCR,[batch_size,H,W,nTSC])
        TSCI4=tf.reshape(TSCI,[batch_size,H,W,nTSC])
        TSCR=tf.stack([TSCR4],axis=4)
        TSCI=tf.stack([TSCI4],axis=4)
#         TSCR=tf.reshape(TSCR,[batch_size,H,W,nTSC,1])
#         TSCI=tf.reshape(TSCI,[batch_size,H,W,nTSC,1])
        SensR=tf.reshape(SensR,[batch_size,H,W,nCh,1])
        SensI=tf.reshape(SensI,[batch_size,H,W,nCh,1])
        TSCRI=tf.concat([TSCR,TSCI],axis=4)
        SensRI=tf.concat([SensR,SensI],axis=4)
        
        TSCRI4=tf.concat([TSCR4,TSCI4],axis=3)
        
        TSCc=tf.complex(TSCR,TSCI)
        Sensc=tf.transpose(tf.complex(SensR,SensI),[0,1,2,4,3])

        tmp1R=tf.slice(tmp,[0,0,0,0],[batch_size,nTraj*nCh,1,1])
        tmp1I=tf.slice(tmp,[0,nTraj*nCh,0,0],[batch_size,nTraj*nCh,1,1])
        SigC=tf.complex(tf.reshape(tmp1R,[batch_size,nCh,nTraj]),tf.reshape(tmp1I,[batch_size,nCh,nTraj]))
        SigC=tf.transpose(SigC,[0,2,1])

#         model.add_PutInOutput(tmp1)
        
#         model.add_Permute([1,0,2,3]) # now we're 133068,16,1,1
        
#         TSCrepEst0=tf.ones_like(y0,dtype=tf.complex64)
        TSCrepEst0=tf.ones([batch_size,H,W,1,1],dtype=tf.complex64)
        
        PowN=tf.reshape(tf.cast(np.arange(0,nTSC),tf.complex64),[1,1,1,nTSC,1]) # [batch_size,H,W,nTSC,nCh]
#         PowN4=tf.reshape(tf.cast(np.arange(0,nTSC),tf.complex64),[1,1,1,nTSC]) # [batch_size,H,W,nTSC,nCh]
        
        TSCest0=tf.pow(TSCrepEst0,PowN)
        
        
#         TSCest0=tf.Print(TSCest0,[tfrm(TSCest0)],'TSCest0 ')
        
#         TSCSens=tf.multiply(TSCc,Sensc)
#         TSCSens=tf.transpose(TSCSens,[1,2,3,4,0])
        TSCSens=tf.multiply(TSCest0,Sensc)
        TSCSens=tf.transpose(TSCSens,[1,2,3,4,0])
        
        
        
        
#         y0=GT.TS_NUFFT_OP_H(SigC,Sens,TSC,TSB,sp_C)
#         SigC=tf.Print(SigC,[tfrm(SigC)],'SigC ')
#         TSCSens=tf.Print(TSCSens,[tfrm(TSCSens)],'TSCSens ')
        SigH=GT.TS_NUFFT_OP_H(SigC,TSCSens,SNc,H,W,batch_size,paddingsX,nTraj,nTSC,nCh,sp_C,TSBFXc, False)
    
#         SigH=tf.Print(SigH,[tfrm(SigH)],'SigH ')
#         print('SigH : ' + str(SigH.shape)) # [H,W, nTSC, nCh, batch_size]
        TSC0=tf.reduce_sum(SigH,axis=3) # [H,W, nTSC, batch_size]
#         print('TSC0a : ' + str(TSC0.shape)) # [H,W, nTSC, nCh, batch_size]
        TSC0=tf.transpose(TSC0,[3,0,1,2]) # # [batch_size, H, W, nTSC]
#         print('TSC0a : ' + str(TSC0.shape)) # [H,W, nTSC, nCh, batch_size]
        y0=tf.reduce_sum(TSC0,axis=3) # [batch_size, H, W]
#         print('y0a : ' + str(y0.shape)) # [H,W, nTSC, nCh, batch_size]
#         print(y0.shape)
#         y0=tf.Print(y0,[],'y0: ' + str(y0.shape))
        
        t0= tf.get_variable('t0', initializer=tf.cast(1.0,tf.float32))
        
        
#         y0=tf.Print(y0,[tfrm(y0)],'y0 ')
#         y0=tf.multiply(tf.cast(t0,tf.complex64),y0)
        
#         def ConcatCOnDim(X,dim): return tf.cast(tf.concat([tf.real(X),tf.imag(X)],axis=dim),tf.float32)
        def ConcatCOnDimWithStack(X,dim): return tf.cast(tf.concat([tf.stack([tf.real(X)],axis=dim),tf.stack([tf.imag(X)],axis=dim)],axis=dim),tf.float32)
#         y0RI=tf.concat([tf.stack([tf.real(y0)],axis=3),tf.stack([tf.imag(y0)],axis=3)],axis=3)
        y0RI=ConcatCOnDimWithStack(y0,3)
#         y0RI=tf.Print(y0RI,[tfrm(y0RI)],'y0RIa ')
        
        y0RI=tf.multiply(t0,y0RI)
#         y0RI=tf.Print(y0RI,[tfrm(y0RI)],'y0RIb ')
        
        TSC0RI=GT.ConcatRIOn3(TSC0)
        TSC0RI=TSC0RI*t0
        
#         y0RI=tf.Print(y0RI,[tfrm(y0RI)],'y0RI ')
#         TSC0RI=tf.Print(TSC0RI,[tfrm(TSC0RI)],'TSC0RI ')
        
        model.add_PutInOutput(y0RI)
        model.add_concat(TSC0RI)
        
#         print(y0RI.shape)
#         print(TSCRI4.shape)
        
#         model.add_concat(TSCRI4)
                
        model.add_ConvNetFromListWithNameAndScope(myParams.myDict['ISide1Net'],name='Net'+str(Iters[0]),scope='ConvNet',UseBN=UseBN)
            
#         yNewDiff=model.get_output()
#         y=y0RI+yNewDiff
#         model.add_PutInOutput(y)
        IAndTSC1x=model.get_output()
        Ix=tf.slice(IAndTSC1x,[0,0,0,0],[-1,-1,-1,2])
        TSC1x=tf.slice(IAndTSC1x,[0,0,0,2],[-1,-1,-1,2])
        
#         Ix=tf.Print(Ix,[tfrm(Ix)],'Ix ')
#         TSC1x=tf.Print(TSC1x,[tfrm(TSC1x)],'TSC1x ')
        
#         model.add_sum(y0RI)
        model.add_PutInOutput(y0RI+Ix)
        model.add_concat( (tf.cast(1,tf.float32)+TSC1x) /100.0 )
        
        
        

        fftkernTSF=scipy.io.loadmat('/media/a/H2/home/a/gUM/fftkernTS.mat')
        fftkernTS=fftkernTSF['fftkernTS']

        fftkernTS=tf.constant(fftkernTS)
        fftkernc=tf.cast(fftkernTS,tf.complex64)
        fftkernc5D=GT.TF_3d_to_5d(fftkernc)

#         ConcatInsteadOfAdd=False
        ConcatInsteadOfAdd=True
        
        
        def RItoCon4(X): return tf.squeeze(tf.complex(tf.slice(X,[0,0,0,0],[batch_size,H,W,1]),tf.slice(X,[0,0,0,1],[batch_size,H,W,1])))
        
        for Iter in range(1, nIter):
#             y1=model.get_output()
            IAndTSC11=model.get_output()
    
#             IAndTSC11=tf.Print(IAndTSC11,[tfrm(IAndTSC11)],'IAndTSC11 ')
    
            y1=tf.slice(IAndTSC11,[0,0,0,0],[-1,-1,-1,2])
            TSC11=tf.slice(IAndTSC11,[0,0,0,2],[-1,-1,-1,2])
#             y=tf.squeeze(tf.complex(tf.slice(y1,[0,0,0,0],[batch_size,H,W,1]),tf.slice(y1,[0,0,0,1],[batch_size,H,W,1])))
#             y=RItoCon4(y1)
            y=RItoCon4(y1)
    
#             y=tf.Print(y,[tfrm(y)],'y ')
    
            TSC11c=RItoCon4(TSC11)
        
#             TSC11c=tf.Print(TSC11c,[tfrm(TSC11c)],'TSC11c ')
            
            def TFexpix(X): return tf.exp(tf.complex(tf.zeros_like(X),X))
            
            TSC11cMag=tf.abs(TSC11c)
            TSC11cPhi=tf.angle(TSC11c)
            TSC11cMag=tf.minimum(TSC11cMag,1.0)
            TSC11c=tf.cast(TSC11cMag,tf.complex64)*TFexpix(TSC11cPhi)
        
            TSCest1=tf.pow(GT.TF_3d_to_5d(TSC11c),PowN)
    
#             TSCest1=tf.Print(TSCest1,[tfrm(TSCest1)],'TSCest1 ')
        
            TSCSens=tf.multiply(TSCest1,Sensc)
            TSCSens=tf.transpose(TSCSens,[1,2,3,4,0])
            
#             TSCSens=tf.Print(TSCSens,[tfrm(TSCSens)],'TSCSens ')
    
#             y = tf.Print(y,[tfrm(y)],message='y ')
#             SigCur=GT.TS_NUFFT_OP(y,Sens,TSC,TSB,sp_C)
#             SigCur=GT.TS_NUFFT_OP(y,TSCSens,SNc,H,W,batch_size,paddingsX,nTraj,nTSC,nCh,sp_C,TSBFXc)
# #             yPrime=GT.TS_NUFFT_OP_H(SigCur,Sens,TSC,TSB,sp_C)
#             yPrime=GT.TS_NUFFT_OP_H(SigCur,TSCSens,SNc,H,W,batch_size,paddingsX,nTraj,nTSC,nCh,sp_C,TSBFXc)
    
            yPrime=GT.TS_NUFFT_OPHOP(y,TSCSens,H,W,batch_size,paddingsY,nTSC,nCh,fftkernc5D,SumOver=False) 
            yPrime=yPrime/(H*W*2*2)
#             yPrime=tf.Print(yPrime,[tfrm(yPrime)],'yPrimea ')
            # yPrime is [H,W,nTSC,nCh,batch_size]
            yPrime=tf.reduce_sum(yPrime,axis=3) # [H,W,nTSC,batch_size]
            TSCPrime=tf.transpose(yPrime,[3,0,1,2]) # [batch_size,H,W,nTSC]
            yPrime=tf.reduce_sum(TSCPrime,axis=3) # [batch_size,H,W]
            
#             yPrime=tf.Print(yPrime,[tfrm(yPrime)],'yPrime ')
            
#             yPrime = tf.Print(yPrime,[tfrm(yPrime)],message='yPrime ')
#             yPrime=y
    
            if ConcatInsteadOfAdd:
                yPrimeRI=ConcatCOnDimWithStack(yPrime,3)
                model.add_concat(yPrimeRI)
                model.add_concat(y0RI)
                yRI=y1
                TSCPrimeRI=GT.ConcatCOnDim(TSCPrime,3)
                TSCest1RI=GT.ConcatCOnDim(tf.squeeze(TSCest1,axis=4),3)
                model.add_concat(TSCPrimeRI/100.0)
                model.add_concat(TSCest1RI/100.0)
            else:
                with tf.variable_scope('tScope', reuse=tf.AUTO_REUSE):
                    tCur= tf.get_variable('t_Iter'+str(Iters[Iter]), initializer=tf.cast(-2.0,tf.float32))
                Diff=yPrime-y0
                DiffTSC=TSCPrime-TSC0
                y=y+tf.multiply(tf.cast(tCur,tf.complex64),Diff)
                TSCupdated=TSCest1+GT.TF_4d_to_5d(tf.multiply(tf.cast(tCur,tf.complex64),DiffTSC))
                yRI=ConcatCOnDimWithStack(y,3)
                TSCupdatedRI=GT.ConcatRIOn3(TSCupdated)
                TSCupdatedRI=tf.squeeze(TSCupdatedRI,axis=4)
                model.add_PutInOutput(yRI)
#                 print('yRI : ' + str(yRI.shape)) # [H,W, nTSC, nCh, batch_size]
#                 print('TSCupdatedRI : ' + str(TSCupdatedRI.shape)) # [H,W, nTSC, nCh, batch_size]
                model.add_concat(TSCupdatedRI)
            
#             model.add_concat(TSCRI4)
            model.add_ConvNetFromListWithNameAndScope( myParams.myDict['ISide1Net'],name='Net'+str(Iters[Iter]),scope='ConvNet',UseBN=UseBN) # , stddev_factor=0.3
#             model.add_sum(yRI)
            IAndTSC12=model.get_output()
            y2=tf.slice(IAndTSC12,[0,0,0,0],[-1,-1,-1,2])
            TSC12=tf.slice(IAndTSC12,[0,0,0,2],[-1,-1,-1,2])
            model.add_PutInOutput(y2+yRI)
#             model.add_concat(TSCupdatedRI+TSC12)
#             model.add_concat(TSC11+TSC12)
            model.add_concat(TSC12)
        
        IAndTSC1F=model.get_output()
        yF=tf.slice(IAndTSC1F,[0,0,0,0],[-1,-1,-1,2])
        TSC1F=tf.slice(IAndTSC1F,[0,0,0,2],[-1,-1,-1,2])
        
        ResF=tf.concat([yF,TSC1F],axis=2)
        model.add_PutInOutput(ResF)
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars
    
    if myParams.myDict['NetMode'] == 'ISTA_Try1':
        print("ISTA_Try1 mode")
        
        aDataH=myParams.myDict['aDataH']
        kMax=myParams.myDict['kMax']
        aDataW=myParams.myDict['aDataW']
        nTS=myParams.myDict['nTimeSegments']
        nTSI=myParams.myDict['nTimeSegmentsI']
        UseSharedWightesInRelaxedFT=myParams.myDict['UseSharedWightesInRelaxedFT']>0
        RelaxedFT=myParams.myDict['RelaxedFT']>0
        addBias=myParams.myDict['CmplxBias']>0
        nccInData=myParams.myDict['nccInData']
        ncc=myParams.myDict['nccToUse']
        nNeighbors=myParams.myDict['nNeighbors']

        DataH=myParams.myDict['DataH']
        DataW=myParams.myDict['DataW']
        LabelsH=myParams.myDict['LabelsH']
        LabelsW=myParams.myDict['LabelsW']
        H=LabelsH
        W=LabelsW
    
        achannelsIn=ncc*nNeighbors*2
        
        BaseNUFTDataP=myParams.myDict['BaseNUFTDataP']
        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Traj=NUFTData['Trajm2'][0:2,:]
        Kd=NUFTData['Kd']
        P=NUFTData['P']
        SN=NUFTData['SN']
        
        NMapCR=GT.GenerateNeighborsMap(Traj,kMax,aDataH,nccInData,ncc,nNeighbors)
        NMapCR = tf.constant(NMapCR)

        
        nTraj=Traj.shape[1]
        nCh=nccInData
        nTSC=12

        ToPad=[Kd[0,0]-H,Kd[0,1]-W]
    
        paddings = tf.constant([[0, ToPad[0]], [0, ToPad[1]],[0,0]])
    
        paddingsX=tf.gather(paddings,[0,1,2],axis=0)
        paddingsY=tf.gather(paddings,[0,1,2,2,2],axis=0)
        
        SNc=tf.stack([tf.stack([tf.stack([tf.constant(SN,dtype=tf.complex64)],axis=2)],axis=3)],axis=4)

        Idx=scipy.sparse.find(P)
        I2=np.vstack([Idx[0],Idx[1]]).T
        I2=tf.constant(np.int64(I2))
        ValC=tf.constant(np.complex64(Idx[2]))
        sp_C = tf.SparseTensor(I2, ValC, [P.shape[0],P.shape[1]])

        BaseTSDataP=GT.getparam('BaseTSDataP')
        B0Data=scipy.io.loadmat(BaseTSDataP + 'B0TS.mat')
        TSBF=B0Data['TSBF']

        TSBFX=np.transpose(np.reshape(TSBF,(nTSC,1,nTraj)),axes=(2,0,1))
        TSBFX=tf.constant(np.complex64(TSBFX))
        TSBFXc=tf.stack([TSBFX],axis=3)

        tmp=model.get_output()
#         tmp1=tf.slice(tmp,[0,0,0,0],[batch_size,132964,1,1])
        TSCR=tf.slice(tmp,[0,nTraj*nCh*2,0,0],[batch_size,H*W*nTSC*1,1,1])
        TSCI=tf.slice(tmp,[0,nTraj*nCh*2+H*W*nTSC*1,0,0],[batch_size,H*W*nTSC*1,1,1])
        SensR=tf.slice(tmp,[0,nTraj*nCh*2+H*W*nTSC*2,0,0],[batch_size,H*W*nCh*1,1,1])
        SensI=tf.slice(tmp,[0,nTraj*nCh*2+H*W*nTSC*2+H*W*nCh,0,0],[batch_size,H*W*nCh*1,1,1])
        TSCR4=tf.reshape(TSCR,[batch_size,H,W,nTSC])
        TSCI4=tf.reshape(TSCI,[batch_size,H,W,nTSC])
        TSCR=tf.stack([TSCR4],axis=4)
        TSCI=tf.stack([TSCI4],axis=4)
#         TSCR=tf.reshape(TSCR,[batch_size,H,W,nTSC,1])
#         TSCI=tf.reshape(TSCI,[batch_size,H,W,nTSC,1])
        SensR=tf.reshape(SensR,[batch_size,H,W,nCh,1])
        SensI=tf.reshape(SensI,[batch_size,H,W,nCh,1])
        TSCRI=tf.concat([TSCR,TSCI],axis=4)
        SensRI=tf.concat([SensR,SensI],axis=4)
        
        TSCRI4=tf.concat([TSCR4,TSCI4],axis=3)
        
        TSCc=tf.complex(TSCR,TSCI)
        Sensc=tf.transpose(tf.complex(SensR,SensI),[0,1,2,4,3])
        TSCSens=tf.multiply(TSCc,Sensc)
        TSCSens=tf.transpose(TSCSens,[1,2,3,4,0])

        tmp1R=tf.slice(tmp,[0,0,0,0],[batch_size,nTraj*nCh,1,1])
        tmp1I=tf.slice(tmp,[0,nTraj*nCh,0,0],[batch_size,nTraj*nCh,1,1])
        SigC=tf.complex(tf.reshape(tmp1R,[batch_size,nCh,nTraj]),tf.reshape(tmp1I,[batch_size,nCh,nTraj]))
        SigC=tf.transpose(SigC,[0,2,1])

#         model.add_PutInOutput(tmp1)
        
#         model.add_Permute([1,0,2,3]) # now we're 133068,16,1,1
        
        y0=GT.TS_NUFFT_OP_H(SigC,Sens,TSC,TSB,sp_C)
        y0=GT.TS_NUFFT_OP_H(SigC,TSCSens,SNc,H,W,batch_size,paddingsX,nTraj,nTSC,nCh,sp_C,TSBFXc)
    
        t0= tf.get_variable('t0', initializer=tf.cast(1.0,tf.float32))
        
#         y0=tf.multiply(tf.cast(t0,tf.complex64),y0)
        
        def ConcatCOnDim(X,dim): return tf.cast(tf.concat([tf.real(X),tf.imag(X)],axis=dim),tf.float32)
        def ConcatCOnDimWithStack(X,dim): return tf.cast(tf.concat([tf.stack([tf.real(X)],axis=dim),tf.stack([tf.imag(X)],axis=dim)],axis=dim),tf.float32)
#         y0RI=tf.concat([tf.stack([tf.real(y0)],axis=3),tf.stack([tf.imag(y0)],axis=3)],axis=3)
        y0RI=ConcatCOnDimWithStack(y0,3)
        
        y0RI=tf.multiply(t0,y0RI)
        
        model.add_PutInOutput(y0RI)
        
        model.add_concat(TSCRI4)
        
#         model.print_shape(message="y0 ")

        UseBN=GT.getparam('UseBN')
#         UseBN=False
        
        Iters=GT.getparam('Iterations')
        nIter=Iters.shape[0]
        
#         UseSameNet=True
#         if UseSameNet:
        model.add_ConvNetFromListWithNameAndScope(myParams.myDict['ISide1Net'],name='Net'+str(Iters[0]),scope='ConvNet',UseBN=UseBN)
#         else:
#             model.add_ConvNetFromList(myParams.myDict['ISide1Net'],UseBN=UseBN)
            
#         yNewDiff=model.get_output()
#         y=y0RI+yNewDiff
#         model.add_PutInOutput(y)
        model.add_sum(y0RI)
#         add_batch_norm
        
        
#         nIter=GT.getparam('nIterations')
#         Iter=0
#         UseSamet=True
#         UseSamet=False
#         if UseSamet:
#         with tf.variable_scope('tScope', reuse=tf.AUTO_REUSE):
#             tCur= tf.get_variable('t'+str(Iters[0]), initializer=tf.cast(-2.0,tf.float32))
        fftkernTSF=scipy.io.loadmat('/media/a/H2/home/a/gUM/fftkernTS.mat')
        fftkernTS=fftkernTSF['fftkernTS']
#         print('asasd')
#         print(np.mean(np.abs(fftkernTS)))

        fftkernTS=tf.constant(fftkernTS)
        fftkernc=tf.cast(fftkernTS,tf.complex64)
        fftkernc5D=GT.TF_3d_to_5d(fftkernc)

        ConcatInsteadOfAdd=False
        
        def tfrm(X): return tf.reduce_mean(tf.abs(X))
        
        def RItoCon4(X): return tf.squeeze(tf.complex(tf.slice(X,[0,0,0,0],[batch_size,H,W,1]),tf.slice(X,[0,0,0,1],[batch_size,H,W,1])))
        
        for Iter in range(1, nIter):
            y1=model.get_output()
#             y=tf.squeeze(tf.complex(tf.slice(y1,[0,0,0,0],[batch_size,H,W,1]),tf.slice(y1,[0,0,0,1],[batch_size,H,W,1])))
            y=RItoCon4(y1)
    
#             y = tf.Print(y,[tfrm(y)],message='y ')
#             SigCur=GT.TS_NUFFT_OP(y,Sens,TSC,TSB,sp_C)
#             SigCur=GT.TS_NUFFT_OP(y,TSCSens,SNc,H,W,batch_size,paddingsX,nTraj,nTSC,nCh,sp_C,TSBFXc)
# #             yPrime=GT.TS_NUFFT_OP_H(SigCur,Sens,TSC,TSB,sp_C)
#             yPrime=GT.TS_NUFFT_OP_H(SigCur,TSCSens,SNc,H,W,batch_size,paddingsX,nTraj,nTSC,nCh,sp_C,TSBFXc)
    
            yPrime=GT.TS_NUFFT_OPHOP(y,TSCSens,H,W,batch_size,paddingsY,nTSC,nCh,fftkernc5D)
            yPrime=yPrime/(H*W*2*2)
#             yPrime = tf.Print(yPrime,[tfrm(yPrime)],message='yPrime ')
#             yPrime=y
    
            if ConcatInsteadOfAdd:
                yPrimeRI=ConcatCOnDimWithStack(yPrime,3)
                model.add_concat(yPrimeRI)
                model.add_concat(y0RI)
                yRI=y1
            else:
                with tf.variable_scope('tScope', reuse=tf.AUTO_REUSE):
                    tCur= tf.get_variable('t_Iter'+str(Iters[Iter]), initializer=tf.cast(-2.0,tf.float32))
                Diff=yPrime-y0
                y=y+tf.multiply(tf.cast(tCur,tf.complex64),Diff)
                yRI=ConcatCOnDimWithStack(y,3)
                model.add_PutInOutput(yRI)
            
            model.add_concat(TSCRI4)
            model.add_ConvNetFromListWithNameAndScope( myParams.myDict['ISide1Net'],name='Net'+str(Iters[Iter]),scope='ConvNet',UseBN=UseBN) # , stddev_factor=0.3
            model.add_sum(yRI)
            
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars
        
    if myParams.myDict['NetMode'] == 'RegridTry3C2_TS':
        print("RegridTry3C2_TS mode")
        
        aDataH=myParams.myDict['aDataH']
        kMax=myParams.myDict['kMax']
        aDataW=myParams.myDict['aDataW']
        # achannelsIn=myParams.myDict['achannelsIn']

        nTS=myParams.myDict['nTimeSegments']
        UseSharedWightesInRelaxedFT=myParams.myDict['UseSharedWightesInRelaxedFT']>0
        RelaxedFT=myParams.myDict['RelaxedFT']>0
        addBias=myParams.myDict['CmplxBias']>0

        # FullData=scipy.io.loadmat(myParams.myDict['NMAP_FN'])
        # NMapCR=FullData['NMapCR']
        # NMapCR = tf.constant(NMapCR)
        nccInData=myParams.myDict['nccInData']
        # ncc=8
        ncc=myParams.myDict['nccToUse']
        
        nNeighbors=myParams.myDict['nNeighbors']

        achannelsIn=ncc*nNeighbors*2
        # T=scipy.io.loadmat('/media/a/DATA/180628_AK/meas_MID244_gBP_VD11_U19_G35S155_4min_FID22439/Traj.mat')
        # Traj=T['Traj'][0:2,:]

        # BaseNUFTDataP='/media/a/DATA/13May18/Me/meas_MID409_gBP_VD11_U19_7ADCs_FID17798/'
        # BaseNUFTDataP='/media/a/DATA/11Jul18/RL/meas_MID149_gBP_VD11_U19_G35S155_FID23846/'
        BaseNUFTDataP=myParams.myDict['BaseNUFTDataP']
        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Traj=NUFTData['Trajm2'][0:2,:]

        # T=scipy.io.loadmat('/media/a/DATA/13May18/Me/meas_MID409_gBP_VD11_U19_7ADCs_FID17798/TrajForNUFT.mat')
        # Traj=T['Trajm2'][0:2,:]
        
        NMapCR=GT.GenerateNeighborsMap(Traj,kMax,aDataH,nccInData,ncc,nNeighbors)
        NMapCR = tf.constant(NMapCR)

        
        # nNeighbors=myParams.myDict['nNeighbors']
        # nccInData=myParams.myDict['nccInData']
        
        # CurBartTraj=scipy.io.loadmat('/media/a/DATA/180628_AK/meas_MID244_gBP_VD11_U19_G35S155_4min_FID22439/Traj.mat')
        # CurBartTraj=CurBartTraj['BARTTrajMS'][0:2,2:]
        
        # osN=aDataH
        # nNeighbors=nNeighbors
        # NMap=np.zeros([osN,osN,nNeighbors],dtype='int32')

        # C=GT.linspaceWithHalfStep(-kMax,kMax,osN)

        # nChToUseInNN=nccInData
        # ncc=nccInData
        # nTrajAct=CurBartTraj.shape[1]

        # for i in np.arange(0,osN):
        #     for j in np.arange(0,osN):

        #         CurLoc=np.vstack([C[i], C[j]])

        #         D=CurBartTraj-CurLoc
        #         R=np.linalg.norm(D,ord=2,axis=0)/np.sqrt(2)
        #         Idx=np.argsort(R)

        #         NMap[i,j,:]=Idx[0:nNeighbors]
                
        # a=np.reshape(np.arange(0,nChToUseInNN)*nTrajAct,(1,1,1,nChToUseInNN))
        # NMapC=np.reshape(NMap,(NMap.shape[0],NMap.shape[1],NMap.shape[2],1))+a
        # NMapC=np.transpose(NMapC,(0,1,2,3))
        # NMapCX=np.reshape(NMapC,(osN,osN,nNeighbors*nChToUseInNN))
        # NMapCR=np.concatenate((NMapCX,NMapCX+nTrajAct*ncc),axis=2)

        
        

        # model.print_shape()

        # model.add_Reshape([16*133068])
        model.add_Permute([1,0,2,3]) # now we're 133068,16,1,1
        # model.print_shape()

        feature=model.get_output();

        feature=tf.gather(feature,NMapCR,validate_indices=None,name=None) # After 131,131,192,16
        # feature = tf.reshape(feature, [aDataH, aDataW, achannelsIn])
        model.add_PutInOutput(feature)

        model.add_Permute([3,0,1,2,4,5]) # After 16,131,131,192,1,1

        # model.print_shape()
        model.add_Reshape([batch_size,aDataH,aDataW,achannelsIn]) # After 16,131,131,192

        model.add_Split4thDim(2) # Now we're kH,kW, Neighbors(12)*Channels(8),2

        # model.add_PixelwiseMultC(nTS, stddev_factor=1.0) # After we're batch_size,kH,kW,nTS
        InitForRC=[]
        if myParams.myDict['InitForRFN'] != 'None':
            InitForRM=scipy.io.loadmat(myParams.myDict['InitForRFN'])
            InitForRR=InitForRM['gene_GEN_L007_PixelwiseMultC_weightR_0']
            InitForRI=InitForRM['gene_GEN_L007_PixelwiseMultC_weightI_0']
            InitForRC=InitForRR + 1j * InitForRI
        model.add_PixelwiseMultC(nTS, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=InitForRC)

        # model.print_shape('aaa')
        
        model.add_2DFT()
        
        # MM=GT.gDFT_matrix(np.linspace(-kMax,kMax,aDataH),H)
        # MM=np.transpose(MM,axes=[1,0])
        
        # if UseSharedWightesInRelaxedFT:
        #     model.add_Mult2DMCyCSharedOverFeat(W,1,add_bias=addBias,Trainable=RelaxedFT,InitC=MM)
        #     model.add_Mult2DMCxCSharedOverFeat(H,1,add_bias=addBias,Trainable=RelaxedFT,InitC=MM)
        # else:
        #     model.add_Mult2DMCyC(W,1,add_bias=addBias)
        #     model.add_Mult2DMCxC(H,1,add_bias=addBias)

            # now supposedly batch_size,H,W,nTS
        # model.add_PixelwiseMultC(1, stddev_factor=1.0) # This collecting the different TS to the final image. 
        InitForLC=[]
        if myParams.myDict['InitForLFN'] != 'None':
            InitForLM=scipy.io.loadmat(myParams.myDict['InitForLFN'])
            InitForLR=InitForLM['gene_GEN_L010_PixelwiseMultC_weightR_0']
            InitForLI=InitForLM['gene_GEN_L010_PixelwiseMultC_weightI_0']
            InitForLC=InitForLR + 1j * InitForLI
        model.add_PixelwiseMultC(1, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=InitForLC) # This collecting the different TS to the final image. 





        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'RegridTry3C2_TSME':
        print("RegridTry3C2_TSME mode")

        # WhichEchosToRec=GT.getparam('WhichEchosToRec')
        # nEchos=WhichEchosToRec.shape[0]
        TimePointsForRec_ms=GT.getparam('TimePointsForRec_ms')
        nEchos=TimePointsForRec_ms.shape[0]
                
        aDataH=myParams.myDict['aDataH']
        kMax=myParams.myDict['kMax']
        aDataW=myParams.myDict['aDataW']
        # achannelsIn=myParams.myDict['achannelsIn']

        nTS=myParams.myDict['nTimeSegments']
        UseSharedWightesInRelaxedFT=myParams.myDict['UseSharedWightesInRelaxedFT']>0
        RelaxedFT=myParams.myDict['RelaxedFT']>0
        addBias=myParams.myDict['CmplxBias']>0

        nccInData=myParams.myDict['nccInData']
        # ncc=8
        ncc=myParams.myDict['nccToUse']
        
        nNeighbors=myParams.myDict['nNeighbors']

        achannelsIn=ncc*nNeighbors*2

        BaseNUFTDataP=myParams.myDict['BaseNUFTDataP']
        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Traj=NUFTData['Trajm2'][0:2,:]

        NMapCR=GT.GenerateNeighborsMap(Traj,kMax,aDataH,nccInData,ncc,nNeighbors)
        NMapCR = tf.constant(NMapCR)

        model.add_Permute([1,0,2,3]) # now we're 133068,16,1,1
        # model.print_shape()

        feature=model.get_output();

        feature=tf.gather(feature,NMapCR,validate_indices=None,name=None) # After 131,131,192,16
        # feature = tf.reshape(feature, [aDataH, aDataW, achannelsIn])
        model.add_PutInOutput(feature)

        model.add_Permute([3,0,1,2,4,5]) # After 16,131,131,192,1,1

        # model.print_shape()
        model.add_Reshape([batch_size,aDataH,aDataW,achannelsIn]) # After 16,131,131,192

        model.add_Split4thDim(2) # Now we're kH,kW, Neighbors(12)*Channels(8),2

        # model.add_PixelwiseMultC(nTS, stddev_factor=1.0) # After we're batch_size,kH,kW,nTS
        InitForRC=[]
        if myParams.myDict['InitForRFN'] != 'None':
            InitForRM=scipy.io.loadmat(myParams.myDict['InitForRFN'])
            InitForRR=InitForRM['gene_GEN_L007_PixelwiseMultC_weightR_0']
            InitForRI=InitForRM['gene_GEN_L007_PixelwiseMultC_weightI_0']
            InitForRC=InitForRR + 1j * InitForRI
        model.add_PixelwiseMultC(nTS, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=InitForRC)
        
        model.add_2DFT()
        
        InitForLC=[]
        if myParams.myDict['InitForLFN'] != 'None':
            InitForLM=scipy.io.loadmat(myParams.myDict['InitForLFN'])
            InitForLR=InitForLM['gene_GEN_L010_PixelwiseMultC_weightR_0']
            InitForLI=InitForLM['gene_GEN_L010_PixelwiseMultC_weightI_0']
            InitForLC=InitForLR + 1j * InitForLI

        # model.add_PixelwiseMultC(1, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=InitForLC) # This collecting the different TS to the final image. 
        model.add_PixelwiseMultC(nEchos, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=InitForLC) # This collects the different TS to the final image. 

        # model.print_shape('AfterL')

        model.add_Permute34()
        model.add_Combine34(True)

        # model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars
    
    
    if myParams.myDict['NetMode'] == 'RegridTry3C2_TS_MB':
        print("RegridTry3C2_TS_MB mode")
        
        aDataH=myParams.myDict['aDataH']
        kMax=myParams.myDict['kMax']
        aDataW=myParams.myDict['aDataW']
        # achannelsIn=myParams.myDict['achannelsIn']

        nTS=myParams.myDict['nTimeSegments']
        UseSharedWightesInRelaxedFT=myParams.myDict['UseSharedWightesInRelaxedFT']>0
        RelaxedFT=myParams.myDict['RelaxedFT']>0
        addBias=myParams.myDict['CmplxBias']>0

        
        nccInData=myParams.myDict['nccInData']
        
        ncc=myParams.myDict['nccToUse']
        
        nNeighbors=myParams.myDict['nNeighbors']

        achannelsIn=ncc*nNeighbors*2
        
        BaseNUFTDataP=myParams.myDict['BaseNUFTDataP']
        NUFTData=scipy.io.loadmat(BaseNUFTDataP + 'TrajForNUFT.mat')
        Traj=NUFTData['Trajm2'][0:2,:]

        NMapCR=GT.GenerateNeighborsMap(Traj,kMax,aDataH,nccInData,ncc,nNeighbors)
        NMapCR = tf.constant(NMapCR)

        model.add_Permute([1,0,2,3]) # now we're 133068,16,1,1
        # model.print_shape()

        feature=model.get_output();

        feature=tf.gather(feature,NMapCR,validate_indices=None,name=None) # After 131,131,192,16
        # feature = tf.reshape(feature, [aDataH, aDataW, achannelsIn])
        model.add_PutInOutput(feature)

        model.add_Permute([3,0,1,2,4,5]) # After 16,131,131,192,1,1

        # model.print_shape()
        model.add_Reshape([batch_size,aDataH,aDataW,achannelsIn]) # After 16,131,131,192

        if addBias:
            print("with bias")
        else:
            print("without bias")

        model.add_Split4thDim(2) # Now we're kH,kW, Neighbors(12)*Channels(8),2

        # model.add_PixelwiseMultC(nTS, stddev_factor=1.0) # After we're batch_size,kH,kW,nTS
        InitForRC=[]
        if myParams.myDict['InitForRFN'] != 'None':
            InitForRM=scipy.io.loadmat(myParams.myDict['InitForRFN'])
            InitForRR=InitForRM['gene_GEN_L007_PixelwiseMultC_weightR_0']
            InitForRI=InitForRM['gene_GEN_L007_PixelwiseMultC_weightI_0']
            InitForRC=InitForRR + 1j * InitForRI
        model.add_PixelwiseMultC(nTS, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=InitForRC)

        MM=GT.gDFT_matrix(np.linspace(-kMax,kMax,aDataH),H)
        MM=np.transpose(MM,axes=[1,0])
        
        if UseSharedWightesInRelaxedFT:
            model.add_Mult2DMCyCSharedOverFeat(W,1,add_bias=addBias,Trainable=RelaxedFT,InitC=MM)
            model.add_Mult2DMCxCSharedOverFeat(H,1,add_bias=addBias,Trainable=RelaxedFT,InitC=MM)
        else:
            model.add_Mult2DMCyC(W,1,add_bias=addBias)
            model.add_Mult2DMCxC(H,1,add_bias=addBias)

            # now supposedly batch_size,H,W,nTS
        # ggg: 2 here is MB
        # model.add_PixelwiseMultC(2, stddev_factor=1.0) # This collecting the different TS to the final image. 
        # model.print_shape('BeforeL')

        InitForLC=[]
        if myParams.myDict['InitForLFN'] != 'None':
            InitForLM=scipy.io.loadmat(myParams.myDict['InitForLFN'])
            InitForLR=InitForLM['gene_GEN_L010_PixelwiseMultC_weightR_0']
            InitForLI=InitForLM['gene_GEN_L010_PixelwiseMultC_weightI_0']
            InitForLC=InitForLR + 1j * InitForLI
        model.add_PixelwiseMultC(2, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=InitForLC) # This collects the different TS to the final image. 

        # model.print_shape('AfterL')

        model.add_Permute34()
        model.add_Combine34(True)

        # model.print_shape('After Combine34')

        # model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'RegridTry1C2_TS':
        print("RegridTry1C2_TS mode")
        aDataH=myParams.myDict['aDataH']
        kMax=myParams.myDict['kMax']

        nTS=myParams.myDict['nTimeSegments']
        UseSharedWightesInRelaxedFT=myParams.myDict['UseSharedWightesInRelaxedFT']>0
        RelaxedFT=myParams.myDict['RelaxedFT']>0
        addBias=myParams.myDict['CmplxBias']>0
        model.add_Split4thDim(2)
        # model.add_PixelwiseMultC(nTS, stddev_factor=1.0) # After we're batch_size,kH,kW,nTS
        InitForRC=[]
        print("InitForRC...")
        print(myParams.myDict['InitForRFN'])
        if myParams.myDict['InitForRFN'] != 'None':
            print("InitForRC From file")
            InitForRM=scipy.io.loadmat(myParams.myDict['InitForRFN'])
            InitForRR=InitForRM['gene_GEN_L007_PixelwiseMultC_weightR_0']
            InitForRI=InitForRM['gene_GEN_L007_PixelwiseMultC_weightI_0']
            InitForRC=InitForRR + 1j * InitForRI
        print("InitForRC...")
        model.add_PixelwiseMultC(nTS, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=InitForRC)

        MM=GT.gDFT_matrix(np.linspace(-kMax,kMax,aDataH),H)
        MM=np.transpose(MM,axes=[1,0])
        
        if UseSharedWightesInRelaxedFT:
            model.add_Mult2DMCyCSharedOverFeat(W,1,add_bias=addBias,Trainable=RelaxedFT,InitC=MM)
            model.add_Mult2DMCxCSharedOverFeat(H,1,add_bias=addBias,Trainable=RelaxedFT,InitC=MM)
        else:
            model.add_Mult2DMCyC(W,1,add_bias=addBias)
            model.add_Mult2DMCxC(H,1,add_bias=addBias)

        # now supposedly batch_size,H,W,nTS

        # model.add_PixelwiseMultC(1, stddev_factor=1.0) # This collecting the different TS to the final image. 
        # add_PixelwiseMultC(self, numOutChannels, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=[]):
        InitForLC=[]
        if myParams.myDict['InitForLFN'] != 'None':
            InitForLM=scipy.io.loadmat(myParams.myDict['InitForLFN'])
            InitForLR=InitForLM['gene_GEN_L010_PixelwiseMultC_weightR_0']
            InitForLI=InitForLM['gene_GEN_L010_PixelwiseMultC_weightI_0']
            InitForLC=InitForLR + 1j * InitForLI
        model.add_PixelwiseMultC(1, stddev_factor=1.0,NamePrefix='',Trainable=True,InitC=InitForLC) # This collecting the different TS to the final image. 


        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    # if myParams.myDict['NetMode'] == 'RegridTry1C2_TS2': # Shared features in relaxed FT
    #     print("RegridTry1C2_TS mode")
    #     addBias=myParams.myDict['CmplxBias']>0
    #     if addBias:
    #         print("with bias")
    #     else:
    #         print("without bias")
    #     nTS=7
    #     model.add_Split4thDim(2)
    #     model.add_PixelwiseMultC(nTS, stddev_factor=1.0)
    #     model.add_Mult2DMCyCSharedOverFeat(W,1,add_bias=addBias)
    #     model.add_Mult2DMCxCSharedOverFeat(H,1,add_bias=addBias)
    #     model.add_PixelwiseMultC(1, stddev_factor=1.0)
    #     model.remove_5thDim()
        
    #     new_vars  = tf.global_variables()
    #     gene_vars = list(set(new_vars) - set(old_vars))
    #     return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'SMASHTry1':
        print("SMASHTry1 mode")
        addBias=myParams.myDict['CmplxBias']>0
    
        model.add_PixelwiseMultC(2, stddev_factor=1.0)
        model.add_Combine34()
        model.add_Mult2DMCyC(W,1,add_bias=addBias)
        model.add_Mult2DMCxC(H,1,add_bias=addBias)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'SMASHTry1_CC':
        print("SMASHTry1_CC mode")
        addBias=myParams.myDict['CmplxBias']>0
        # we're [Batch, kH,kW,AllChannels*Neighbors*RI]
        model.add_Split4thDim(2) # Now [Batch, kH,kW,AllChannels*Neighbors,RI]

        model.add_Mult2DMCxCSharedOverFeat(DataH, 1) # Now [Batch, H,kW,AllChannels*Neighbors,RI]
        model.add_Split4thDim(6) # Now [Batch, H,kW,AllChannels,Neighbors,RI]

        ncc=4
        model.add_einsumC('abcde,dx->abcxe',[8, ncc])

        model.add_Combine45(squeeze=True) # Now [Batch, H,kW,CompressedChannels*Neighbors,RI]
        model.add_Mult2DMCxCSharedOverFeat(DataH, 1) # Now [Batch, kH,kW,CompressedChannels*Neighbors,RI]

        model.add_PixelwiseMultC(2, stddev_factor=1.0)
        model.add_Combine34()
        model.add_Mult2DMCyC(W,1,add_bias=addBias)
        model.add_Mult2DMCxC(H,1,add_bias=addBias)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars



    if myParams.myDict['NetMode'] == 'SMASHTry1_GCC':
        print("SMASHTry1_GCC mode")
        addBias=myParams.myDict['CmplxBias']>0
        # we're [Batch, kH,kW,AllChannels*Neighbors*RI]
        model.add_Split4thDim(2) # Now [Batch, kH,kW,AllChannels*Neighbors,RI]

        model.add_Mult2DMCxCSharedOverFeat(DataH, 1) # Now [Batch, H,kW,AllChannels*Neighbors,RI]
        model.add_Split4thDim(6) # Now [Batch, H,kW,AllChannels,Neighbors,RI]

        ncc=4
        model.add_einsumC('abcde,bdx->abcxe',[DataH,8, ncc])

        model.add_Combine45(squeeze=True) # Now [Batch, H,kW,CompressedChannels*Neighbors,RI]
        model.add_Mult2DMCxCSharedOverFeat(DataH, 1) # Now [Batch, kH,kW,CompressedChannels*Neighbors,RI]

        model.add_PixelwiseMultC(2, stddev_factor=1.0)
        model.add_Combine34()
        model.add_Mult2DMCyC(W,1,add_bias=addBias)
        model.add_Mult2DMCxC(H,1,add_bias=addBias)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'SMASHTry1_GCCF':
        print("SMASHTry1_GCCF mode")
        addBias=myParams.myDict['CmplxBias']>0
        # we're [Batch, kH,kW,AllChannels*Neighbors*RI]
        model.add_Split4thDim(2) # Now [Batch, H,kW,AllChannels*Neighbors,RI]

        model.add_Split4thDim(6) # Now [Batch, H,kW,AllChannels,Neighbors,RI]

        ncc=4
        model.add_einsumC('abcde,bdx->abcxe',[DataH,8, ncc])

        model.add_Combine45(squeeze=True) # Now [Batch, H,kW,CompressedChannels*Neighbors,RI]

        DFTM=DFT_matrix(DataH)
        model.add_Mult2DMCxCSharedOverFeat(DataH, 1,add_bias=addBias,Trainable=False,InitC=DFTM) # Now [Batch, kH,kW,CompressedChannels*Neighbors,RI]

        model.add_PixelwiseMultC(2, stddev_factor=1.0)
        model.add_Combine34()
        model.add_Mult2DMCyC(W,1,add_bias=addBias)
        model.add_Mult2DMCxC(H,1,add_bias=addBias)
        model.remove_5thDim()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'Conv_3Layers':
        print("Conv_3Layers mode")
        # model.print_shape()
        model.add_conv2d(64, mapsize=mapsize, stride=1, stddev_factor=2.)
        model.add_elu()
        model.add_conv2dWithName(32, name="ggg", mapsize=1, stride=1, stddev_factor=2.)
        model.add_elu()
        model.add_conv2d(channelsOut, mapsize=5, stride=1, stddev_factor=2.)  

        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'Unet_v1_ForB0':
        b=np.array([[64,0,0,128,128,0,0,64],[128,0,0,256,256,0,0,128],[512,0,0,0,0,0,0,512]])
        model.add_UnetKsteps(b, mapsize=mapsize, stride=2, stddev_factor=1e-3)

        # OutChannels=labels.shape[3]

        model.add_conv2d(channelsOut, mapsize=mapsize, stride=1, stddev_factor=2.)  

        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))

        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'Conv_v1_ForB0':
        print("Conv_v1_ForB0 mode")
        # model.print_shape()
        model.add_conv2d(64, mapsize=mapsize, stride=1, stddev_factor=2.)
        model.add_elu()
        model.add_conv2dWithName(128, name="ggg", mapsize=mapsize, stride=1, stddev_factor=2.)
        model.add_elu()
        model.add_conv2d(128, mapsize=mapsize, stride=1, stddev_factor=2.)   
        model.add_elu()
        model.add_conv2d(channelsOut, mapsize=mapsize, stride=1, stddev_factor=2.)  
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars

    if myParams.myDict['NetMode'] == 'Conv_v1':
        print("Conv_v1 mode")
        # model.print_shape()
        model.add_conv2d(64, mapsize=mapsize, stride=1, stddev_factor=2.)
        model.add_elu()
        model.add_conv2dWithName(128, name="ggg", mapsize=mapsize, stride=1, stddev_factor=2.)
        model.add_elu()
        model.add_conv2d(128, mapsize=mapsize, stride=1, stddev_factor=2.)   
        model.add_elu()
        model.add_conv2d(1, mapsize=mapsize, stride=1, stddev_factor=2.)  

    # SAE
    SAE = myParams.myDict['NetMode'] == 'SAE'
    if SAE:
        model.add_conv2d(64, mapsize=mapsize, stride=1, stddev_factor=2.)
        model.add_elu()
        model.add_conv2dWithName(128, name="AE", mapsize=mapsize, stride=1, stddev_factor=2.)

        model.add_conv2d(64, mapsize=mapsize, stride=1, stddev_factor=2.)   
        model.add_elu()
        model.add_conv2d(channels, mapsize=7, stride=1, stddev_factor=2.)  
        model.add_sigmoid()
        # model.add_tanh()

    # kKick:
    kKick= myParams.myDict['NetMode'] == 'kKick'
    if kKick:
        model.add_conv2d(64, mapsize=1, stride=1, stddev_factor=2.)   
        model.add_elu()
        b=np.array([[64,0,0,128,128,0,0,64],[128,0,0,256,256,0,0,128],[512,0,0,0,0,0,0,512]])
        model.add_UnetKsteps(b, mapsize=mapsize, stride=2, stddev_factor=1e-3)
        model.add_conv2dWithName(50, name="AE", mapsize=3, stride=1, stddev_factor=2.)
        model.add_elu()
        model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=2.)    

        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))
        return model.get_output(), gene_vars



    # AUTOMAP
    # AUTOMAP_units  = [64, 64, channels]
    # AUTOMAP_mapsize  = [5, 5, 7]

    # ggg option 1: FC
    # model.add_flatten() # FC1
    # model.add_dense(num_units=H*W*2)
    # model.add_reshapeTo4D(H,W)

    TSRECON = myParams.myDict['NetMode'] == 'TSRECON'
    if TSRECON:
        # ggg option 2: FC per channel, and then dot multiplication per pixel, then conv
        ChannelsPerCoil=myParams.myDict['NumFeatPerChannel']
        NumTotalFeat=myParams.myDict['NumTotalFeat']
        model.add_Mult2DMC(H*W,ChannelsPerCoil)
        model.add_reshapeTo4D(H, W)
        model.add_PixelwiseMult(NumTotalFeat, stddev_factor=1.0)
        model.add_elu()


        #model.add_denseFromM('piMDR')
        #model.add_reshapeTo4D(FLAGS.LabelsH,FLAGS.LabelsW)
        # #model.add_tanh() # FC2

        #model.add_Unet1Step(128, mapsize=5, stride=2, num_layers=2, stddev_factor=1e-3)
        #model.add_conv2d(channels, mapsize=5, stride=1, stddev_factor=2.)

        b=np.array([[64,0,0,128,128,0,0,64],[128,0,0,256,256,0,0,128],[512,0,0,0,0,0,0,512]])
        #b=np.array([[64,0,0,128,128,0,0,64],[128,0,0,256,256,0,0,128]])
        #b=np.array([[64,0,0,0,0,0,0,64]])

        model.add_UnetKsteps(b, mapsize=mapsize, stride=2, stddev_factor=1e-3)
        # model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=2.)

        # ggg: Autoencode
        model.add_conv2dWithName(50, name="AE", mapsize=3, stride=1, stddev_factor=2.)
        model.add_elu()

        # ggg: Finish
        model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=2.)    

        # #model.add_flatten()
        # #model.add_dense(num_units=H*W*1)
        # model.add_reshapeTo4D(FLAGS.LabelsH,FLAGS.LabelsW)
        # #model.add_batch_norm()
        # #model.add_tanh() # TC3

        # # model.add_conv2d(AUTOMAP_units[0], mapsize=AUTOMAP_mapsize[0], stride=1, stddev_factor=2.)
        # # model.add_batch_norm()
        # #model.add_relu()

        # #model.add_conv2d(AUTOMAP_units[1], mapsize=AUTOMAP_mapsize[1], stride=1, stddev_factor=2.)
        # # model.add_batch_norm()
        # #model.add_relu()

        # #model.add_conv2d(AUTOMAP_units[2], mapsize=AUTOMAP_mapsize[2], stride=1, stddev_factor=2.)
        # # model.add_conv2d(AUTOMAP_units[2], mapsize=1, stride=1, stddev_factor=2.)
        # # model.add_relu()


        #model.add_constMatMul()
        #for ru in range(len(res_units)-1):
        #    nunits  = res_units[ru]

        #    for j in range(2):
        #        model.add_residual_block(nunits, mapsize=mapsize)

            # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
            # and transposed convolution
        #    model.add_upscale()
            
        #    model.add_batch_norm()
        #    model.add_relu()
        #    model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

        # model.add_flatten()
        # model.add_dense(num_units=H*W*4)
        # model.add_reshapeTo4D(FLAGS.LabelsH,FLAGS.LabelsW)

        # #model.add_Mult2D()
        # #model.add_Mult3DComplexRI()

    SrezOrigImagePartModel=False
    if SrezOrigImagePartModel:
        nunits  = res_units[0]
        for j in range(2):
            model.add_residual_block(nunits, mapsize=mapsize)
        #model.add_upscale()
        model.add_batch_norm()
        model.add_relu()
        model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

        nunits  = res_units[1]
        for j in range(2):
            model.add_residual_block(nunits, mapsize=mapsize)
        #model.add_upscale()
        model.add_batch_norm()
        model.add_relu()
        model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

        # Finalization a la "all convolutional net"
        nunits = res_units[-1]
        model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()
        model.add_relu()

        model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()
        model.add_relu()

        # Last layer is sigmoid with no batch normalization
        model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)
        model.add_sigmoid()
    
    new_vars  = tf.global_variables()
    gene_vars = list(set(new_vars) - set(old_vars))

    # ggg = tf.identity(model.get_output(), name="ggg")

    return model.get_output(), gene_vars
