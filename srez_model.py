import numpy as np
import tensorflow as tf
import scipy.io
import GTools as GT

FLAGS = tf.app.flags.FLAGS

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
