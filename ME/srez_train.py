import numpy as np
import os.path
import scipy.misc
import imageio
import tensorflow as tf
import time
import scipy.io
import pdb
import myParams
import GTools as GT

from tensorflow.python.client import timeline

FLAGS = tf.app.flags.FLAGS

def np3DCToImg(label,FN):
    # label=label+0.37;
    LabelA=np.sqrt(np.power(label[:,:,:,0],2)+np.power(label[:,:,:,1],2))
    LabelP=np.arctan2(label[:,:,:,1],label[:,:,:,0])/(2*np.pi)+0.5;

    image   = np.concatenate([label[:,:,:,0], label[:,:,:,1],LabelA,LabelP], axis=2)
    image   = np.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
    image   = np.concatenate([image,image,image], axis=3)

    image = image[:,:,:,:]
    
    image=np.reshape(image,[image.shape[1], image.shape[2],image.shape[3]])

    filename = '%s.png' % (FN)
    filename = os.path.join(myParams.myDict['train_dir'], filename)
    # scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    imageio.imwrite(filename,image)
    return

def _summarize_progress(train_data, label, gene_output, batch, suffix, max_samples=8):
    td = train_data

    size = [label.shape[1], label.shape[2]]

    if max_samples>myParams.myDict['batch_size']:
        max_samples=myParams.myDict['batch_size']

    # AEops = [v for v in td.sess.graph.get_operations() if "ForPNG" in v.name and not ("_1" in v.name)]
    # AEouts = [v.outputs[0] for v in AEops]
    # label=td.sess.run(AEouts[0])

    # filename = 'WV%06d_%s' % (batch, suffix)
    # np3DCToImg(label,filename)

    if myParams.myDict['ImgMode'] == 'Cmplx_MB':
        Matops = [v for v in td.sess.graph.get_operations() if "ForMat" in v.name and not (("_1" in v.name) or ("ApplyAdam" in v.name) or ("Assign" in v.name) or ("read" in v.name) or ("optimizer" in v.name))]
        Matouts = [v.outputs[0] for v in Matops]

        VLen=Matouts.__len__()
        if VLen>1:
            for x in Matops: print(x.name +'           ' + str(x.outputs[0].shape))

            # pdb.set_trace()
            # save all:
            TrainSummary={}
            filenamex = 'ForMat_%06d.mat' % (batch)
            filename = os.path.join(myParams.myDict['train_dir'], filenamex)
            var_list=[]
            for i in range(0, VLen): 
                var_list.append(Matops[i].name);
                tmp=td.sess.run(Matouts[i])
                s1=Matops[i].name
                print("Saving  %s" % (s1))
                s1=s1.replace(':','_')
                s1=s1.replace('/','_')
                TrainSummary[s1]=tmp
            
            scipy.io.savemat(filename,TrainSummary)

        Matops = [v for v in td.sess.graph.get_operations() if "ForOut" in v.name and not (("_1" in v.name) or ("ApplyAdam" in v.name) or ("Assign" in v.name) or ("read" in v.name) or ("optimizer" in v.name))]
        Matouts = [v.outputs[0] for v in Matops]

        VLen=Matouts.__len__()
        if VLen>1:
            for x in Matops: print(x.name +'           ' + str(x.outputs[0].shape))

            # pdb.set_trace()
            # save all:
            TrainSummary={}
            filenamex = 'ForOut_%06d.mat' % (batch)
            filename = os.path.join(myParams.myDict['train_dir'], filenamex)
            var_list=[]
            for i in range(0, VLen): 
                var_list.append(Matops[i].name);
                tmp=td.sess.run(Matouts[i])
                s1=Matops[i].name
                print("Saving  %s" % (s1))
                s1=s1.replace(':','_')
                s1=s1.replace('/','_')
                TrainSummary[s1]=tmp
            
            scipy.io.savemat(filename,TrainSummary)        

        LabelA=np.sqrt(np.power(label[:,:,:,0],2)+np.power(label[:,:,:,1],2))
        LabelP=np.arctan2(label[:,:,:,1],label[:,:,:,0])/(2*np.pi)+0.5;

        GeneA=np.sqrt(np.power(gene_output[:,:,:,0],2)+np.power(gene_output[:,:,:,1],2))
        GeneA=np.maximum(np.minimum(GeneA, 1.0), 0.0)
        GeneP=np.arctan2(gene_output[:,:,:,1],gene_output[:,:,:,0])/(2*np.pi)+0.5;

        gene_output=(gene_output+1)/2
        label=(label+1)/2

        # clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)
        clipped = np.maximum(np.minimum(gene_output, 1.0), 0.0)

        # image   = tf.concat([clipped[:,:,:,0], label[:,:,:,0], clipped[:,:,:,1], label[:,:,:,1],LabelA,GeneA,LabelP,GeneP], 2)
        image=np.concatenate((clipped[:,:,:,0], label[:,:,:,0], clipped[:,:,:,1], label[:,:,:,1],LabelA,GeneA,LabelP,GeneP), axis=2)
        

        # image   = tf.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
        image   = np.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
        # image   = tf.concat([image,image,image], 3)
        image   = np.concatenate((image,image,image), axis=3)

        image = image[0:max_samples,:,:,:]
        # image = tf.concat([image[i,:,:,:] for i in range(int(max_samples))], 0)
        # image = np.concatenate(((image[i,:,:,:] for i in ra/nge(int(max_samples)))), axis=0)
        image = np.reshape(image,[image.shape[0]*image.shape[1],image.shape[2],image.shape[3]])
        # image = td.sess.run(image)

        filename = 'batch%06d_%s.png' % (batch, suffix)
        filename = os.path.join(myParams.myDict['train_dir'], filename)
        # scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
        imageio.imwrite(filename,image)

        print("    Saved %s" % (filename,))
        return

    if myParams.myDict['ImgMode'] == 'Cmplx_AP':
        Matops = [v for v in td.sess.graph.get_operations() if "ForMat" in v.name and not (("_1" in v.name) or ("ApplyAdam" in v.name) or ("Assign" in v.name) or ("read" in v.name) or ("optimizer" in v.name))]
        Matouts = [v.outputs[0] for v in Matops]

        VLen=Matouts.__len__()
        if VLen>1:
            for x in Matops: print(x.name +'           ' + str(x.outputs[0].shape))

            # pdb.set_trace()
            # save all:
            TrainSummary={}
            filenamex = 'ForMat_%06d.mat' % (batch)
            filename = os.path.join(myParams.myDict['train_dir'], filenamex)
            var_list=[]
            for i in range(0, VLen): 
                var_list.append(Matops[i].name);
                tmp=td.sess.run(Matouts[i])
                s1=Matops[i].name
                print("Saving  %s" % (s1))
                s1=s1.replace(':','_')
                s1=s1.replace('/','_')
                TrainSummary[s1]=tmp
            
            scipy.io.savemat(filename,TrainSummary)

        Matops = [v for v in td.sess.graph.get_operations() if "ForOut" in v.name and not (("_1" in v.name) or ("ApplyAdam" in v.name) or ("Assign" in v.name) or ("read" in v.name) or ("optimizer" in v.name))]
        Matouts = [v.outputs[0] for v in Matops]

        VLen=Matouts.__len__()
        if VLen>1:
            for x in Matops: print(x.name +'           ' + str(x.outputs[0].shape))

            # pdb.set_trace()
            # save all:
            TrainSummary={}
            filenamex = 'ForOut_%06d.mat' % (batch)
            filename = os.path.join(myParams.myDict['train_dir'], filenamex)
            var_list=[]
            for i in range(0, VLen): 
                var_list.append(Matops[i].name);
                tmp=td.sess.run(Matouts[i])
                s1=Matops[i].name
                print("Saving  %s" % (s1))
                s1=s1.replace(':','_')
                s1=s1.replace('/','_')
                TrainSummary[s1]=tmp
            
            scipy.io.savemat(filename,TrainSummary)        

        LabelA=np.sqrt(np.power(label[:,:,:,0],2)+np.power(label[:,:,:,1],2))
        LabelP=np.arctan2(label[:,:,:,1],label[:,:,:,0])/(2*np.pi)+0.5;

        GeneA=np.sqrt(np.power(gene_output[:,:,:,0],2)+np.power(gene_output[:,:,:,1],2))
        GeneA=np.maximum(np.minimum(GeneA, 1.0), 0.0)
        GeneP=np.arctan2(gene_output[:,:,:,1],gene_output[:,:,:,0])/(2*np.pi)+0.5;

        gene_output=(gene_output+1)/2
        label=(label+1)/2

        # clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)
        clipped = np.maximum(np.minimum(gene_output, 1.0), 0.0)

        # image   = tf.concat([clipped[:,:,:,0], label[:,:,:,0], clipped[:,:,:,1], label[:,:,:,1],LabelA,GeneA,LabelP,GeneP], 2)
#         image=np.concatenate((clipped[:,:,:,0], label[:,:,:,0], clipped[:,:,:,1], label[:,:,:,1],LabelA,GeneA,LabelP,GeneP), axis=2)
        image=np.concatenate((LabelA,GeneA,LabelP,GeneP), axis=2)
        

        # image   = tf.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
        image   = np.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
        # image   = tf.concat([image,image,image], 3)
        image   = np.concatenate((image,image,image), axis=3)

        image = image[0:max_samples,:,:,:]
        # image = tf.concat([image[i,:,:,:] for i in range(int(max_samples))], 0)
        # image = np.concatenate(((image[i,:,:,:] for i in ra/nge(int(max_samples)))), axis=0)
        # image = np.concatenate((image[0,:,:,:],image[1,:,:,:],image[2,:,:,:],image[3,:,:,:],image[4,:,:,:],image[5,:,:,:],image[6,:,:,:],image[7,:,:,:]), axis=0)
        # image = np.concatenate((image[0,:,:,:],image[1,:,:,:],image[2,:,:,:],image[3,:,:,:],image[4,:,:,:],image[5,:,:,:]), axis=0)
        # image = np.concatenate((image[0,:,:,:],image[1,:,:,:],image[2,:,:,:],image[3,:,:,:]), axis=0)
        image = np.reshape(image,[image.shape[0]*image.shape[1],image.shape[2],image.shape[3]])
        # image = td.sess.run(image)

        filename = 'batch%06d_%s.png' % (batch, suffix)
        filename = os.path.join(myParams.myDict['train_dir'], filename)
        # scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
        imageio.imwrite(filename,image)

        print("    Saved %s" % (filename,))
        return
    
    if myParams.myDict['ImgMode'] == 'Cmplx':
        Matops = [v for v in td.sess.graph.get_operations() if "ForMat" in v.name and not (("_1" in v.name) or ("ApplyAdam" in v.name) or ("Assign" in v.name) or ("read" in v.name) or ("optimizer" in v.name))]
        Matouts = [v.outputs[0] for v in Matops]

        VLen=Matouts.__len__()
        if VLen>1:
            for x in Matops: print(x.name +'           ' + str(x.outputs[0].shape))

            # pdb.set_trace()
            # save all:
            TrainSummary={}
            filenamex = 'ForMat_%06d.mat' % (batch)
            filename = os.path.join(myParams.myDict['train_dir'], filenamex)
            var_list=[]
            for i in range(0, VLen): 
                var_list.append(Matops[i].name);
                tmp=td.sess.run(Matouts[i])
                s1=Matops[i].name
                print("Saving  %s" % (s1))
                s1=s1.replace(':','_')
                s1=s1.replace('/','_')
                TrainSummary[s1]=tmp
            
            scipy.io.savemat(filename,TrainSummary)

        Matops = [v for v in td.sess.graph.get_operations() if "ForOut" in v.name and not (("_1" in v.name) or ("ApplyAdam" in v.name) or ("Assign" in v.name) or ("read" in v.name) or ("optimizer" in v.name))]
        Matouts = [v.outputs[0] for v in Matops]

        VLen=Matouts.__len__()
        if VLen>1:
            for x in Matops: print(x.name +'           ' + str(x.outputs[0].shape))

            # pdb.set_trace()
            # save all:
            TrainSummary={}
            filenamex = 'ForOut_%06d.mat' % (batch)
            filename = os.path.join(myParams.myDict['train_dir'], filenamex)
            var_list=[]
            for i in range(0, VLen): 
                var_list.append(Matops[i].name);
                tmp=td.sess.run(Matouts[i])
                s1=Matops[i].name
                print("Saving  %s" % (s1))
                s1=s1.replace(':','_')
                s1=s1.replace('/','_')
                TrainSummary[s1]=tmp
            
            scipy.io.savemat(filename,TrainSummary)        

        LabelA=np.sqrt(np.power(label[:,:,:,0],2)+np.power(label[:,:,:,1],2))
        LabelP=np.arctan2(label[:,:,:,1],label[:,:,:,0])/(2*np.pi)+0.5;

        GeneA=np.sqrt(np.power(gene_output[:,:,:,0],2)+np.power(gene_output[:,:,:,1],2))
        GeneA=np.maximum(np.minimum(GeneA, 1.0), 0.0)
        GeneP=np.arctan2(gene_output[:,:,:,1],gene_output[:,:,:,0])/(2*np.pi)+0.5;

        gene_output=(gene_output+1)/2
        label=(label+1)/2

        # clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)
        clipped = np.maximum(np.minimum(gene_output, 1.0), 0.0)

        # image   = tf.concat([clipped[:,:,:,0], label[:,:,:,0], clipped[:,:,:,1], label[:,:,:,1],LabelA,GeneA,LabelP,GeneP], 2)
        image=np.concatenate((clipped[:,:,:,0], label[:,:,:,0], clipped[:,:,:,1], label[:,:,:,1],LabelA,GeneA,LabelP,GeneP), axis=2)
        

        # image   = tf.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
        image   = np.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
        # image   = tf.concat([image,image,image], 3)
        image   = np.concatenate((image,image,image), axis=3)

        image = image[0:max_samples,:,:,:]
        # image = tf.concat([image[i,:,:,:] for i in range(int(max_samples))], 0)
        # image = np.concatenate(((image[i,:,:,:] for i in ra/nge(int(max_samples)))), axis=0)
        # image = np.concatenate((image[0,:,:,:],image[1,:,:,:],image[2,:,:,:],image[3,:,:,:],image[4,:,:,:],image[5,:,:,:],image[6,:,:,:],image[7,:,:,:]), axis=0)
        # image = np.concatenate((image[0,:,:,:],image[1,:,:,:],image[2,:,:,:],image[3,:,:,:],image[4,:,:,:],image[5,:,:,:]), axis=0)
        # image = np.concatenate((image[0,:,:,:],image[1,:,:,:],image[2,:,:,:],image[3,:,:,:]), axis=0)
        image = np.reshape(image,[image.shape[0]*image.shape[1],image.shape[2],image.shape[3]])
        # image = td.sess.run(image)

        filename = 'batch%06d_%s.png' % (batch, suffix)
        filename = os.path.join(myParams.myDict['train_dir'], filename)
        # scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
        imageio.imwrite(filename,image)

        print("    Saved %s" % (filename,))
        return

    if myParams.myDict['ImgMode'] == 'SimpleImg2':
        clipped = np.maximum(np.minimum(gene_output, 1.0), 0.0)

        image=np.concatenate((clipped[:,:,:,0],label[:,:,:,0],clipped[:,:,:,1],label[:,:,:,1]), axis=2)
        image = np.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
        image = np.concatenate((image,image,image), axis=3)
        image = image[0:max_samples,:,:,:]
        image = np.reshape(image,[image.shape[0]*image.shape[1],image.shape[2],image.shape[3]])

        filename = 'batch%06d_%s.png' % (batch, suffix)
        filename = os.path.join(myParams.myDict['train_dir'], filename)
        # scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
        imageio.imwrite(filename,image)

        print("    Saved %s" % (filename,))
        return

    if myParams.myDict['ImgMode'] == 'SimpleImg':
        image=np.concatenate((gene_output[:,:,:,0],label[:,:,:,0]), axis=2)
        # image=(image+1)/2
        image = np.maximum(np.minimum(image, 1.0), 0.0)
        image = np.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
        image = np.concatenate((image,image,image), axis=3)
        image = image[0:max_samples,:,:,:]
        image = np.reshape(image,[image.shape[0]*image.shape[1],image.shape[2],image.shape[3]])

        filename = 'batch%06d_%s.png' % (batch, suffix)
        filename = os.path.join(myParams.myDict['train_dir'], filename)
        # scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
        imageio.imwrite(filename,image)

        print("    Saved %s" % (filename,))
        return

    SAE = myParams.myDict['ImgMode'] == 'SAE'
    if SAE:
        clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

        # pdb.set_trace()
        image   = tf.concat([clipped[:,:,:,0],label[:,:,:,0]], 2)

        image=tf.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
        image   = tf.concat([image,image,image], 3)

        image = image[0:max_samples,:,:,:]
        image = tf.concat([image[i,:,:,:] for i in range(int(max_samples))], 0)
        image = td.sess.run(image)

        filename = 'batch%06d_%s.png' % (batch, suffix)
        filename = os.path.join(myParams.myDict['train_dir'], filename)
        # scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
        imageio.imwrite(filename,image)

        print("    Saved %s" % (filename,))
        return

    kKick= myParams.myDict['ImgMode'] == 'kKick'
    if kKick:
        clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

        image   = tf.concat([clipped[:,:,:,0], label[:,:,:,0], clipped[:,:,:,1], label[:,:,:,1]], 2)

        image=tf.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
        image   = tf.concat([image,image,image], 3)

        image = image[0:max_samples,:,:,:]
        image = tf.concat([image[i,:,:,:] for i in range(int(max_samples))], 0)
        image = td.sess.run(image)

        filename = 'batch%06d_%s.png' % (batch, suffix)
        filename = os.path.join(myParams.myDict['train_dir'], filename)
        # scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
        imageio.imwrite(filename,image)

        print("    Saved %s" % (filename,))
        return

    #nearest = tf.image.resize_nearest_neighbor(feature, size)
    #nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    #bicubic = tf.image.resize_bicubic(feature, size)
    #bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

    gene_outputx=np.sqrt(np.power(gene_output[:,:,:,0],2)+np.power(gene_output[:,:,:,1],2))
    Theta=np.arctan2(gene_output[:,:,:,1],gene_output[:,:,:,0])/(2*np.pi)+0.5;
    labelx=np.sqrt(np.power(label[:,:,:,0],2)+np.power(label[:,:,:,1],2))
    labelx[0]=Theta[0];

    clipped = tf.maximum(tf.minimum(gene_outputx, 1.0), 0.0)

    #image   = tf.concat([nearest, bicubic, clipped, label], 2)
    image   = tf.concat([clipped, labelx], 2)

    image=tf.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
    # pdb.set_trace()
    image   = tf.concat([image,image,image], 3)

    image = image[0:max_samples,:,:,:]
    #image = tf.concat([image[i,:,:,:] for i in range(max_samples)], 0)
    image1 = tf.concat([image[i,:,:,:] for i in range(int(max_samples/2))], 0)
    image2 = tf.concat([image[i,:,:,:] for i in range(int(max_samples/2),max_samples)], 0)
    image  = tf.concat([image1, image2], 1)
    image = td.sess.run(image)

    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(myParams.myDict['train_dir'], filename)
    # scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    imageio.imwrite(filename,image)

    OnRealData={}
    OnRealDataM=gene_output[0]
    filenamex = 'OnRealData.mat'
    filename = os.path.join(myParams.myDict['train_dir'], filenamex)
    OnRealData['x']=OnRealDataM
    scipy.io.savemat(filename,OnRealData)

    print("    Saved %s" % (filename,))

def _save_checkpoint(train_data, batch,G_LossV,saver):
    td = train_data

    oldname = 'checkpoint_old'
    newname = 'checkpoint_new'

    oldname = os.path.join(myParams.myDict['checkpoint_dir'], oldname)
    newname = os.path.join(myParams.myDict['checkpoint_dir'], newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # pdb.set_trace()
    # Generate new checkpoint
    # vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gen")
    # saver = tf.train.Saver()
    saver.save(td.sess, newname,write_meta_graph=False)

    print("    Checkpoint saved")

    # save all:
    TrainSummary={}
    filenamex = 'TrainSummary_%06d.mat' % (batch)
    filename = os.path.join(myParams.myDict['train_dir'], filenamex)
    VLen=td.gene_var_list.__len__()
    var_list=[]
    for i in range(0, VLen): 
        var_list.append(td.gene_var_list[i].name);
        tmp=td.sess.run(td.gene_var_list[i])
        s1=td.gene_var_list[i].name
        print("Saving  %s" % (s1))
        s1=s1.replace(':','_')
        s1=s1.replace('/','_')
        TrainSummary[s1]=tmp
    
    TrainSummary['var_list']=var_list
    TrainSummary['G_LossV']=G_LossV

    scipy.io.savemat(filename,TrainSummary)

    print("saved to %s" % (filename))

def train_model(train_data):
    td = train_data

    summaries = tf.summary.merge_all()
    RestoreSession=False
    if not RestoreSession:
        td.sess.run(tf.global_variables_initializer())

    # lrval       = FLAGS.learning_rate_start
    learning_rate_start=myParams.myDict['learning_rate_start']
    lrval       = myParams.myDict['learning_rate_start']
    start_time  = time.time()
    last_summary_time  = time.time()
    last_checkpoint_time  = time.time()
    done  = False
    batch = 0

    print("lrval %f" % (lrval))

    # assert FLAGS.learning_rate_half_life % 10 == 0

    # Cache test features and labels (they are small)
    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])
    # test_label = td.sess.run([td.test_features, td.test_labels])

    G_LossV=np.zeros((1000000), dtype=np.float32)
    filename = os.path.join(myParams.myDict['train_dir'], 'TrainSummary.mat')
    
    feed_dictOut = {td.gene_minput: test_feature}
    gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)
    # _summarize_progress(td, test_label, gene_output, batch, 'out')

    feed_dict = {td.learning_rate : lrval}
    opsx = [td.gene_minimize, td.gene_loss]
    _, gene_loss = td.sess.run(opsx, feed_dict=feed_dict)

    # opsy = [td.gene_loss]
    # gene_loss = td.sess.run(opsy, feed_dict=feed_dict)

    # ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.disc_real_loss, td.disc_fake_loss]
    # _, _, gene_loss, disc_real_loss, disc_fake_loss = td.sess.run(ops, feed_dict=feed_dict)

    batch += 1

    # run_metadata = tf.RunMetadata()
    gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)
    # gene_output = td.sess.run(td.gene_moutput, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True), feed_dict=feed_dictOut,run_metadata=run_metadata)
    # _summarize_progress(td, test_label, gene_output, batch, 'out')
    # with open("/tmp/run2.txt", "w") as out:
    #     out.write(str(run_metadata))

    # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    # chrome_trace = fetched_timeline.generate_chrome_trace_format()
    # with open('timeline_01.json', 'w') as f:
    #     f.write(chrome_trace)

    # tl = timeline.Timeline(run_metadata.step_stats)
    # print(tl.generate_chrome_trace_format(show_memory=True))
    # trace_file = tf.gfile.Open(name='timeline', mode='w')
    # trace_file.write(tl.generate_chrome_trace_format(show_memory=True))

    feed_dict = {td.learning_rate : lrval}
    # ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.disc_real_loss, td.disc_fake_loss]

    opsx = [td.gene_minimize, td.gene_loss]
    _, gene_loss = td.sess.run(opsx, feed_dict=feed_dict)

    batch += 1

    gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)
    # _summarize_progress(td, test_label, gene_output, batch, 'out')

    # load model
    #saver.restore(sess,tf.train.latest_checkpoint('./'))
    # running model on data:test_feature
    RunOnData=False
    if RunOnData:
        filenames = tf.gfile.ListDirectory('DataAfterpMat')
        filenames = sorted(filenames)
        #filenames = [os.path.join('DataAfterpMat', f) for f in filenames]
        Ni=len(filenames)
        OutBase=myParams.myDict['SessionName']+'_OutMat'
        tf.gfile.MakeDirs(OutBase)

        #pdb.set_trace()

        for index in range(Ni):
            print(index)
            print(filenames[index])
            CurData=scipy.io.loadmat(os.path.join('DataAfterpMat', filenames[index]))
            Data=CurData['CurData']
            Data=Data.reshape((1,64,64,1))
            test_feature=np.kron(np.ones((16,1,1,1)),Data)
            #test_feature = np.array(np.random.choice([0, 1], size=(16,64,64,1)), dtype='float32')


            feed_dictOut = {td.gene_minput: test_feature}
            gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)

            filenameOut=os.path.join(OutBase, filenames[index][:-4] + '_out.mat') 

            SOut={}
            SOut['X']=gene_output[0]
            scipy.io.savemat(filenameOut,SOut)

    # pdb.set_trace()

    #_summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')
    # to get value of var:
    # ww=td.sess.run(td.gene_var_list[1])
    
    if GT.getparam('ShowRealData')>0:
        # ifilename=os.path.join('RealData', 'b.mat')
        
        if GT.getparam('InputMode') == 'RegridTry3F_B0T2S_ITS_MB':
            MB=GT.getparam('MB')
            nCh=GT.getparam('nccToUse') 
            nTSC=GT.getparam('nTimeSegments')
            batch_size=myParams.myDict['batch_size']

            channelsIn=myParams.myDict['channelsIn']
            channelsOut=myParams.myDict['channelsOut']
            LabelsH=myParams.myDict['LabelsH']
            LabelsW=myParams.myDict['LabelsW']
            H=LabelsH
            W=LabelsW

            TimePoints_ms=GT.getparam('TimePoints_ms')
            SN=GT.getparam('SN') # H,W,f8
            P=GT.getparam('P') # H,W,f8
            nTraj=GT.getparam('nTraj')

            cCAIPIVecZ=GT.getparam('cCAIPIVecZ') # MB,nTraj complex128
            TSBF=GT.getparam('TSBF') # nTS,nTraj f64

            Rec0FN='/opt/data/RealDataMB/meas_MID426_gBP_2dSpiral_multiecho_ASL_2mm_iso_run1_FID34170_2Echos/Sli06_Rec0.mat'
            B0TSFFN='/opt/data/RealDataMB/meas_MID426_gBP_2dSpiral_multiecho_ASL_2mm_iso_run1_FID34170_2Echos/Sli06/B0TS.mat'
            SensDataFN='/opt/data/RealDataMB/meas_MID426_gBP_2dSpiral_multiecho_ASL_2mm_iso_run1_FID34170_2Echos/Sli06/SensCC1.mat'
            RealDtFN='/opt/data/RealDataMB/meas_MID426_gBP_2dSpiral_multiecho_ASL_2mm_iso_run1_FID34170_2Echos/Sli06/RealDataForNN.mat'
            
            Rec0F=scipy.io.loadmat(Rec0FN)
            Rec0=Rec0F['Rec0'] # (96, 192) complex128
            Rec0MB=np.transpose(np.reshape(Rec0,[H,MB,W]),(0,2,1)) # H,W,MB complex128

            B0TSF=scipy.io.loadmat(B0TSFFN)
            CurB0=B0TSF['CurB0'] # (96, 96, 2) f8, in Hz
            # B0TSF.keys()

            SensData=scipy.io.loadmat(SensDataFN)
            # SensData.keys()
            SensCCA=SensData['SensCCA'] # (96, 96, 16) complex128
            SensCCB=SensData['SensCCB'] # (96, 96, 16) complex128
            RealSensCCMB=np.stack((SensCCA,SensCCB),axis=3) # (96, 96, 16,2) complex128
            RealSensCCMB=RealSensCCMB[:,:,:nCh,:] # H,W,nCh,MB

            
            RealData=scipy.io.loadmat(RealDtFN)
            DataCC=RealData['DataCC'] # (10238, 16) (nTraj, ncc) complex64
            DataCC=DataCC[:,:nCh] # (nTraj, nCh) complex64
            
            
            CurB0_Hz=CurB0[:,:,np.newaxis,:]

            RealTSC0=np.complex64(np.exp(1j*2*np.pi*CurB0_Hz*GT.NP_addDim(TimePoints_ms)/1000)) # (96, 96, 7, 2)
            # Add T2* of 20
            T2S_est_ms=20
            RealTSC0=RealTSC0*np.exp(-GT.NP_addDim(TimePoints_ms)/T2S_est_ms)

            WarmStart_ITS=Rec0MB[:,:,np.newaxis,:]*RealTSC0 # H,W,nTS,MB
            
            TSBFCAIPI=np.transpose(cCAIPIVecZ[:,:,np.newaxis],(1,2,0))*np.transpose(TSBF[:,:,np.newaxis],(1,0,2)) # nTraj,nTS,MB c128
            cTSBFCAIPI=np.conj(np.transpose(TSBFCAIPI[:,:,:,np.newaxis,np.newaxis],(0,1,3,2,4))) # nTraj,nTS,/nCh/,MB,/batch_size/
            
            DataCCMB=DataCC[:,np.newaxis,:,np.newaxis,np.newaxis] # nTraj,/nTS/,nCh,/MB/,/batch_size/
            DataCCMB=DataCCMB*cTSBFCAIPI # nTraj,nTS,nCh,MB,/batch_size/
            
            # Padded=sps_x_dense_vecs(np.conj(np.transpose(P)),DataCCMB)
            Padded=np.conj(np.transpose(P))*np.reshape(DataCCMB,(nTraj,-1))
            Padded=np.reshape(Padded,(H*2,W*2,nTSC,nCh,MB,1))
            Padded=np.transpose(Padded,(2,3,4,5,0,1))
            # Padded=np.reshape(Padded,(nTSC,nCh,MB,1,H*2,W*2))
            IFPadded=np.fft.ifft2(Padded,axes=(-2, -1))
            
            Cropped=IFPadded[:,:,:,:,:H,:W] # nTS,nCh,MB,/batch_size/,H,W
            CroppedP=np.transpose(Cropped,(5,4,1,2,0,3)) # H,W,nCh,MB,nTS,/batch_size/
            Real_NUFFTHSig=np.sum(CroppedP*np.conj(RealSensCCMB[:,:,:,:,np.newaxis,np.newaxis]),axis=2) # H,W,MB,nTS,/batch_size/
            # RealSensCCMB=RealSensCCMB[:,:,:nCh,:] # H,W,nCh,MB
            Real_NUFFTHSig=np.transpose(Real_NUFFTHSig,(4,0,1,3,2))
            Real_NUFFTHSig=Real_NUFFTHSig*np.conj(SN[np.newaxis,:,:,np.newaxis,np.newaxis]) # /batch_size/,H,W,nTS,MB

            RealSensCCMB1D=GT.NP_ConcatRIOn0(np.reshape(RealSensCCMB,(-1,1,1)))
            Real_feature=RealSensCCMB1D
            RealTSC01D=GT.NP_ConcatRIOn0(np.reshape(RealTSC0,(-1,1,1)))
            Real_feature=np.concatenate((Real_feature,RealTSC01D),axis=0)
            WarmStart_ITS1D=GT.NP_ConcatRIOn0(np.reshape(WarmStart_ITS,(-1,1,1)))
            Real_feature=np.concatenate((Real_feature,WarmStart_ITS1D),axis=0)
            Real_NUFFTHSig1D=GT.NP_ConcatRIOn0(np.reshape(Real_NUFFTHSig,(-1,1,1)))
            Real_feature=np.concatenate((Real_feature,Real_NUFFTHSig1D),axis=0)

            Real_feature=np.tile(Real_feature, (batch_size,1,1,1))
            
        if myParams.myDict['InputMode'] == 'SPEN_FC':
            ifilename=myParams.myDict['RealDataFN']
            RealData=scipy.io.loadmat(ifilename)
            RealData=RealData['Data']
            # RealData=np.reshape(RealData,)
#             RealData=RealData
            Real_feature=RealData

        if myParams.myDict['InputMode'] == 'SPEN_Local':
            ifilename=myParams.myDict['RealDataFN']
            RealData=scipy.io.loadmat(ifilename)
            RealData=RealData['Data']
#             RealData=RealData
            Real_feature=RealData
        
        if False:
            if RealData.ndim==2:
                RealData=RealData.reshape((RealData.shape[0],RealData.shape[1],1,1))
            if RealData.ndim==3:
                RealData=RealData.reshape((RealData.shape[0],RealData.shape[1],RealData.shape[2],1))
        
        if myParams.myDict['InputMode'] == 'RegridTry1' or myParams.myDict['InputMode'] == 'RegridTry2':
            ifilename=myParams.myDict['RealDataFN']
            RealData=scipy.io.loadmat(ifilename)
            RealData=RealData['Data']
            # FullData=scipy.io.loadmat('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/NMapIndTesta.mat')
            FullData=scipy.io.loadmat(myParams.myDict['NMAP_FN'])
            NMapCR=FullData['NMapCR']

            batch_size=myParams.myDict['batch_size']

            Real_feature=np.reshape(RealData[0],[RealData.shape[1]])
            Real_feature=np.take(Real_feature,NMapCR)
            Real_feature=np.tile(Real_feature, (batch_size,1,1,1))

        if myParams.myDict['InputMode'] == 'RegridTry3' or myParams.myDict['InputMode'] == 'RegridTry3M' or myParams.myDict['InputMode'] == 'RegridTry3F' or myParams.myDict['InputMode'] == 'RegridTry3FMB' or myParams.myDict['InputMode'] == 'RegridTry3FME':
            ifilename=myParams.myDict['RealDataFN']
            RealData=scipy.io.loadmat(ifilename)
            RealData=RealData['Data']
            batch_size=myParams.myDict['batch_size']
            nTraj=myParams.myDict['nTraj']
            RealDatancc=myParams.myDict['RealDatancc']
            nccInData=myParams.myDict['nccInData']
            
            # RealData=RealData
            # RealData=np.reshape(RealData,[batch_size,RealDatancc,nTraj,2])
            # RealData=RealData[:,0:nccInData,:,:]
            # RealData=np.reshape(RealData,[batch_size,nTraj,RealDatancc,2])
            # RealData=RealData[:,:,0:nccInData,:]
            # RealData=np.reshape(RealData,[batch_size,-1])
            RealData=RealData[0,:]
            RealData=np.tile(RealData, (batch_size,1))
            Real_feature=np.reshape(RealData,[RealData.shape[0],RealData.shape[1],1,1])

        Real_dictOut = {td.gene_minput: Real_feature}

    # LearningDecayFactor=np.power(2,(-1/FLAGS.learning_rate_half_life))
    learning_rate_half_life=myParams.myDict['learning_rate_half_life']
    LearningDecayFactor=np.power(2,(-1/learning_rate_half_life))

    # train_time=FLAGS.train_time
    train_time=myParams.myDict['train_time']

    QuickFailureTimeM=myParams.myDict['QuickFailureTimeM']
    QuickFailureThresh=myParams.myDict['QuickFailureThresh']

    summary_period=myParams.myDict['summary_period'] # in Minutes
    checkpoint_period=myParams.myDict['checkpoint_period'] # in Minutes

    DiscStartMinute=myParams.myDict['DiscStartMinute']

    gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)
            
    if myParams.myDict['ShowRealData']>0:
        gene_RealOutput = td.sess.run(td.gene_moutput, feed_dict=Real_dictOut)
        gene_output[0]=gene_RealOutput[0]
    
    Asuffix = 'out_%06.4f' % (gene_loss)
    _summarize_progress(td, test_label, gene_output, batch, Asuffix)
    
    print("Adding to saver:")
    var_listX=td.gene_var_list
    var_listX = [v for v in var_listX if "Bank" not in v.name]
#     var_listX = tf.sort(var_listX)
    for line in var_listX: print("Adding " +line.name+'           ' + str(line.shape.as_list()))
    print("Saver var list end")

    saver = tf.train.Saver(var_listX)
    # _save_checkpoint(td, batch,G_LossV,saver)

    tf.get_default_graph().finalize()

    while not done:
        batch += 1
        gene_loss = disc_real_loss = disc_fake_loss = -1.234

        # elapsed = int(time.time() - start_time)/60
        CurTime=time.time()
        elapsed = (time.time() - start_time)/60
            
        # Update learning rate
        lrval*=LearningDecayFactor
        if(learning_rate_half_life<1000): # in minutes
            lrval=learning_rate_start*np.power(0.5,elapsed/learning_rate_half_life)

        

        #print("batch %d gene_l1_factor %f' " % (batch,FLAGS.gene_l1_factor))
        # if batch==200:
        if elapsed>DiscStartMinute:
            FLAGS.gene_l1_factor=0.9
        
        RunDiscriminator= FLAGS.gene_l1_factor < 0.999

        feed_dict = {td.learning_rate : lrval}
        if RunDiscriminator:
            ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.disc_real_loss, td.disc_fake_loss]
            _, _, gene_loss, disc_real_loss, disc_fake_loss = td.sess.run(ops, feed_dict=feed_dict)
        else:
            ops = [td.gene_minimize, td.gene_loss, td.MoreOut, td.MoreOut2, td.MoreOut3]
            _, gene_loss, MoreOut, MoreOut2, MoreOut3 = td.sess.run(ops, feed_dict=feed_dict)
        
        
        G_LossV[batch]=gene_loss



        # ggg: Force phase only var
        # VR = [v for v in tf.global_variables() if v.name == "gene/GEN_L004/add_Mult2DMCyCSharedOverFeat_weightR:0"][0]
        # VI = [v for v in tf.global_variables() if v.name == "gene/GEN_L004/add_Mult2DMCyCSharedOverFeat_weightI:0"][0]
        # VRX=td.sess.run(VR);
        # VIX=td.sess.run(VI);
        # VC=VRX+1J*VIX
        # Norm=np.abs(VC)
        # Norm[Norm == 0] = 0.00001
        # VRX=VRX/Norm
        # VIX=VIX/Norm
        # VR.load(VRX, td.sess)
        # VI.load(VIX, td.sess)

        # VR = [v for v in tf.global_variables() if v.name == "gene/GEN_L005/add_Mult2DMCxCSharedOverFeat_weightR:0"][0]
        # VI = [v for v in tf.global_variables() if v.name == "gene/GEN_L005/add_Mult2DMCxCSharedOverFeat_weightI:0"][0]
        # VRX=td.sess.run(VR);
        # VIX=td.sess.run(VI);
        # VC=VRX+1J*VIX
        # Norm=np.abs(VC)
        # Norm[Norm == 0] = 0.00001
        # VRX=VRX/Norm
        # VIX=VIX/Norm
        # VR.load(VRX, td.sess)
        # VI.load(VIX, td.sess)

        # VR = [v for v in tf.global_variables() if v.name == "gene/GEN_L004/einsum_weightR:0"][0]
        # VI = [v for v in tf.global_variables() if v.name == "gene/GEN_L004/einsum_weightI:0"][0]
        # VRX=td.sess.run(VR);
        # VIX=td.sess.run(VI);
        # HmngWnd=np.power(np.hamming(98),1)
        # HmngWnd=np.reshape(HmngWnd,[98,1,1])
        # VC=VRX +1j*VIX

        # FVC=GT.gfft(VC,dim=0)
        # FVC=FVC*HmngWnd
        # VC=GT.gifft(FVC,dim=0)
        # VYR=np.real(VC)
        # VYI=np.imag(VC)
        # VR.load(VYR, td.sess)
        # VI.load(VYI, td.sess)

            
        if batch % 10 == 0:

            # pdb.set_trace()

            # Show we are alive
            #print('Progress[%3d%%], ETA[%4dm], Batch [%4d], G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f]' %
            #      (int(100*elapsed/train_time), train_time - int(elapsed), batch, gene_loss, disc_real_loss, disc_fake_loss))

            print('Progress[%3d%%], ETA[%4dm], Batch [%4d], G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f], MoreOut[%3.3f, %3.3f, %3.3f]' %
                  (int(100*elapsed/train_time), train_time - int(elapsed), batch, gene_loss, disc_real_loss, disc_fake_loss, MoreOut, MoreOut2, MoreOut3))


            # VLen=td.gene_var_list.__len__()
            # for i in range(0, VLen): 
            #     print(td.gene_var_list[i].name);

            
            # print(VRX.dtype)
            # print(VRX)
            # exit()
            # var_23 = [v for v in tf.global_variables() if v.name == "gene/GEN_L020/C2D_weight:0"][0]
            # tmp=td.sess.run(td.gene_var_list[i])
            # v.load([2, 3], td.sess)


            if np.isnan(gene_loss):
                print('NAN!!')
                done = True

            # ggg: quick failure test
            if elapsed>QuickFailureTimeM :
                if gene_loss>QuickFailureThresh :
                    print('Quick failure!!')
                    done = True
                else:
                    QuickFailureTimeM=10000000

            # Finished?            
            current_progress = elapsed / train_time
            if current_progress >= 1.0:
                done = True
            

            StopFN='/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/stop.a'
            if os.path.isfile(StopFN):
                print('Stop file used!!')
                done = True
                try:
                    tf.gfile.Remove(StopFN)
                except:
                    pass


            # Update learning rate
            # if batch % FLAGS.learning_rate_half_life == 0:
            #     lrval *= .5

        # if batch % FLAGS.summary_period == 0:
        if (CurTime-last_summary_time)/60>summary_period:
            # Show progress with test features
            # feed_dict = {td.gene_minput: test_feature}
            gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)
            
            if myParams.myDict['ShowRealData']>0:
                gene_RealOutput = td.sess.run(td.gene_moutput, feed_dict=Real_dictOut)
                gene_output[0]=gene_RealOutput[0]
            
            Asuffix = 'out_%06.4f' % (gene_loss)
            _summarize_progress(td, test_label, gene_output, batch, Asuffix)

            last_summary_time  = time.time()
    
            
        # if batch % FLAGS.checkpoint_period == 0:
        SaveCheckpoint_ByTime=(CurTime-last_checkpoint_time)/60>checkpoint_period
        CheckpointFN='/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/save.a'
        SaveCheckPointByFile=os.path.isfile(CheckpointFN)
        if SaveCheckPointByFile:
            tf.gfile.Remove(CheckpointFN)

        if SaveCheckpoint_ByTime or SaveCheckPointByFile:
            last_checkpoint_time  = time.time()
            # Save checkpoint
            _save_checkpoint(td, batch,G_LossV,saver)

        RunOnAllFN='/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/RunOnAll.a'
        RunOnAllFNByFile=os.path.isfile(RunOnAllFN)
        if RunOnAllFNByFile:
            tf.gfile.Remove(RunOnAllFN)

            for r in range(1,81):
                ifilenamePrefix=myParams.myDict['LoadAndRunOnData_Prefix']
#                 ifilename=ifilenamePrefix +  f'{r:02}' + '.mat'
                ifilename=ifilenamePrefix+'%02d.mat' % (r)
                RealData=scipy.io.loadmat(ifilename)
                RealData=RealData['Data']
                
                if RealData.ndim==2:
                    RealData=RealData.reshape((RealData.shape[0],RealData.shape[1],1,1))
                if RealData.ndim==3:
                    RealData=RealData.reshape((RealData.shape[0],RealData.shape[1],RealData.shape[2],1))
                
                Real_feature=RealData

                if myParams.myDict['InputMode'] == 'RegridTry1' or myParams.myDict['InputMode'] == 'RegridTry2':
                    # FullData=scipy.io.loadmat('/media/a/f38a5baa-d293-4a00-9f21-ea97f318f647/home/a/TF/NMapIndTesta.mat')
                    FullData=scipy.io.loadmat(myParams.myDict['NMAP_FN'])
                    NMapCR=FullData['NMapCR']

                    batch_size=myParams.myDict['batch_size']

                    Real_feature=np.reshape(RealData[0],[RealData.shape[1]])
                    Real_feature=np.take(Real_feature,NMapCR)
                    Real_feature=np.tile(Real_feature, (batch_size,1,1,1))

                Real_dictOut = {td.gene_minput: Real_feature}


                gene_RealOutput = td.sess.run(td.gene_moutput, feed_dict=Real_dictOut)

                OnRealData={}
                OnRealDataM=gene_RealOutput
#                 filenamex = 'OnRealData' + f'{r:02}' + '.mat'
                filenamex = 'OnRealData'+'%02d.mat' % (r)
                
                LoadAndRunOnData_OutP=myParams.myDict['LoadAndRunOnData_OutP']
                filename = os.path.join(LoadAndRunOnData_OutP, filenamex)
                OnRealData['x']=OnRealDataM
                scipy.io.savemat(filename,OnRealData)

            print('Saved recon of real data')

    _save_checkpoint(td, batch,G_LossV,saver)
    
    print('Finished training!')