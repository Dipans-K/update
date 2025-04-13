# build command:  
# python3 cython_setup.py build_ext --inumpy.ace
# cython merging python + c
# cython version is faster 
import os
import os.path
import time
import sys
import numpy 
import numpy as np
import uuid
from dotenv import load_dotenv  # For dotENV-based configuration

from datetime import datetime
import json
import pickle
import shutil

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from scipy import signal
from sklearn.model_selection import train_test_split

from filelock import FileLock

import grpc
import dongle_pb2
import dongle_pb2_grpc

class WindowsServiceHelper():
    module_dir = ''
    ws = ''
    ws_flag = ''
    ws_ip = ''
    ws_port = ''
    def __init__(self, module_dir):
        self.module_dir = module_dir
        self.ws = self.__read_init_windows_service_param()
        self.ws_flag, self.ws_ip, self.ws_port = self.__function_wsl_ip()

    def __read_init_windows_service_param(self):
        init_windows_service_param = self.module_dir+'/init_windows_service_param.json'
        #print('init_windows_service_param: ', init_windows_service_param)
        if os.path.exists(init_windows_service_param):
            # read init file
            file_name_lock = init_windows_service_param+'.lock'
            lock_file_name = FileLock(file_name_lock)
            with lock_file_name:
                with open(init_windows_service_param, 'r') as myfile:
                    data=myfile.read()
            try:
                os.remove(file_name_lock)
            except:
                pass
            # parse file
            obj = json.loads(data)
            try:
                ws = str(obj['ws'])
            except:
                ws = ''
        else:
            ws = ''
        return ws

    def __function_wsl_ip(self):
        ws_flag = -1
        if self.ws=='YES':
            ws_flag=1
            try:
                with open(self.module_dir+'/windowsservice_config.json', 'r') as jsonFile:
                        data = jsonFile.read()
                        obj = json.loads(data)
                        ws_ip = str(obj['ws_ip'])
                        ws_port = int(obj['ws_port'])
            except:
                ws_ip = '0.0.0.0'
                ws_port = 55555
                ws_flag = -2
        else:
            ws_flag=0
            ws_ip = '0.0.0.0'
            ws_port = 55555
        return ws_flag, ws_ip, str(ws_port)

    def __is_licence_available(self, service_name): 

        dongle_connection_address = None
    
        dongle_connection_address = self.ws_ip+':'+self.ws_port
        #print('dongle_connection_address: ', dongle_connection_address)

        channel = grpc.insecure_channel(dongle_connection_address)
        dongle_stub = dongle_pb2_grpc.DongleStub(channel)
        
        req_dongle = dongle_pb2.DongleRequest(application = service_name)  
    
        licence_flag = None
        try:
            resp_dongle = dongle_stub.AskDongle(req_dongle)
            #print(resp_dongle)
            #print(resp_dongle.status)
            dongle_status = resp_dongle.status
            if dongle_status == 'true':
                licence_flag = True
            else:
                licence_flag = False
        except:
            licence_flag = False
        return licence_flag

    def IsLicenceAvailable(self, service_name):
        flag = self.__is_licence_available(service_name)
        return flag

def server_operation(server, WSH, service_name):
    try:                                                                               
        while True:
            if WSH.ws_flag==1:
                licence_flag = WSH.IsLicenceAvailable(service_name)
                print('licence_flag: ', licence_flag)
                if not licence_flag:
                    print('Licence unavailable, server going to stop ...')
                    server.stop(0)
                    print('Server stopped, now exiting the code ...')
                    exit()
                    #print('Exit successful ...')
                time.sleep(20)
            elif WSH.ws_flag==-2:
                server.stop(0)
                exit()
            else:
                time.sleep(86400)
    except KeyboardInterrupt:                                                           
        server.stop(0)


cpdef prepareFeatureStream_1D_c(module_dir,db_type,session,StatusData,status_datas_schema,Features,int NumFeatures):                       # frame type values: 1D , 2D (for example I/Q data)

    module = 'prepareFeatureStream_1D_c '

    cdef:
        int i,NumRemainElements,counter
        list listFeatures = []                                                                 
        list listFrames = []         

    if NumFeatures > 0:                                          
        NumRemainElements = len(Features)%(NumFeatures)      
        if NumRemainElements!=0:                                                     # if there are some extra elements in received buffer
            Features = Features[:-NumRemainElements]                                 # delete the extra elements
        NumFrames = int(len(Features)/(NumFeatures))
    else:
        NumRemainElements = 0
        NumFrames = 0

    logging_text = ' Length of features data buffer: '+str(len(Features))
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = ' No. of 1D frames: '+str(int(NumFrames))    
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = ' No. of remaining 1D features after the last frame: '+str(int(NumRemainElements))    
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    
    floatarray = numpy.asarray(Features) 
    arrayFrames = np.reshape(floatarray,(NumFrames,NumFeatures))
    logging_text = ' Shape of frames created: '+str(arrayFrames.shape)    
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    return arrayFrames                                                          # return a 3D- numpy array of frames

cpdef prepareFeatureStream_2D_c(module_dir,db_type,session,StatusData,status_datas_schema,Features,int NumFeatures):                       # frame type values: 1D , 2D (for example I/Q data)

    module = 'prepareFeatureStream_2D_c '

    cdef:
        int i,NumRemainElements,counter
        list listFeatures = []                                                   # define a list to store I & Q values in a single frame
        list listFrames = []    
                                                                                 # define a list for stroing IQ frames                   
    if NumFeatures > 0:                                          
        NumRemainElements = len(Features)%(NumFeatures*2)      
        if NumRemainElements!=0:                                                     # if there are some extra elements in received buffer
            Features = Features[:-NumRemainElements]                                 # delete the extra elements
        NumFrames = int(len(Features)/(NumFeatures*2))
    else:
        NumRemainElements = 0
        NumFrames = 0

    logging_text = ' Length of features data buffer: '+str(len(Features))
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = ' No. of 2D frames: '+str(int(NumFrames))    
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = ' No. of remaining 2D features after the last frame: '+str(int(NumRemainElements))    
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    floatarray = numpy.asarray(Features, dtype=numpy.float32) 
    arrayFrames = np.reshape(floatarray,(NumFrames,NumFeatures,2))
    logging_text = ' Shape of frames created: '+str(arrayFrames.shape)    
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    return arrayFrames                                                          # return a 3D- numpy array of frames

# normalize features columns to range [0,1] 
#cpdef normalize_columns_c(float[:,::1]X,float[::1] X_max,float[::1] X_min):
cpdef normalize_columns_c(X,X_max,X_min):
    # normalize features columns in range [0,1]
    cdef:
        int i,num_columns,j,num_rows
    num_rows = X.shape[0]
    num_columns = X.shape[1]
    for j in range (0,num_rows):
        for i in range (0,num_columns):
            if (X_max[i]-X_min[i])>0:
                X[j,i]=((X[j,i]-X_min[i])/(X_max[i]-X_min[i]))*1.0
            else:
                X[j,i]=1.0
    return X

cpdef train_single_classes_init_knn_c(module_dir,db_type,session,StatusData,status_datas_schema,
                                      X_train,y_train,ai_type):

    cdef:
        int i,num_classes

    module = 'train_single_classes_init_knn_c '

    unique,counts = np.unique(y_train, return_counts=True)

    num_classes = len(counts)
    classes = np.column_stack((unique))
    classes_array = classes[0]

    logging_text = 'Number of single classes: '+str(num_classes)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Single classes: '+str(classes_array)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    # read kNN init parameters parameters from JSON init file
    kNN,kNN_reject,reject_thres,max_num_results,reject_labelval_diff,pow_norm,feature_norm,softmax = \
    read_knn_init_param(module_dir,db_type,session,StatusData,status_datas_schema)


    # the number of kNN neigbours must be at least 3 for the single classes init
    # this is to get distances > 0 for within class distances
    if kNN_reject < 3:
        kNN_reject = 3

    # initialize kNN classifier/regressor for each class separately
    neighs = []
    y_trains_i = []
    X_trains_i = []
    if num_classes > 0:
        for i in range (0,num_classes):
            curr_class = classes_array[i]
            cond_i = y_train == curr_class
            y_train_i = y_train[cond_i]
            y_trains_i.append(y_train_i)
            X_train_i = X_train[cond_i]
            X_trains_i.append(X_train_i)
            if ai_type == 'kNN_Class':
                neigh_i = KNeighborsClassifier(n_neighbors=kNN_reject)  
            if ai_type == 'kNN_Reg':
                neigh_i = KNeighborsRegressor(n_neighbors=kNN_reject)           
            neigh_i.fit(X_train_i,y_train_i)
            neighs.append(neigh_i)
            logging_text = 'Train single class / Train number of records: '+str(y_train_i[0])+' / '+str(len(y_train_i))
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    return classes_array,neighs,X_trains_i,y_trains_i

cpdef test_single_classes_c(module_dir,db_type,session,StatusData,status_datas_schema,
                            X_test,y_test):

    cdef:
        int i,num_classes

    module = 'test_single_classes_c '

    unique,counts = np.unique(y_test, return_counts=True)

    num_classes = len(counts)
    classes = np.column_stack((unique))
    classes_array = classes[0]

    logging_text = 'Number of single classes: '+str(num_classes)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Single classes: '+str(classes_array)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    # separate test data per class
    y_tests_i = []
    X_tests_i = []
    if num_classes > 0:
        for i in range (0,num_classes):
            curr_class = classes_array[i]
            cond_i = y_test == curr_class
            y_test_i = y_test[cond_i]
            y_tests_i.append(y_test_i)
            X_test_i = X_test[cond_i]
            X_tests_i.append(X_test_i)
            logging_text = 'Test single class / Test number of records: '+str(y_test_i[0])+' / '+str(len(y_test_i))
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    return classes_array,X_tests_i,y_tests_i

cpdef calc_single_class_distances_c(module_dir,db_type,session,StatusData,status_datas_schema,
                                    train_classes_array,neighs,X_tests_i,y_tests_i):

    cdef:
        int i,j,num_classes

    module = 'calc_single_class_distances_c '

    num_classes = len(train_classes_array)
    max_dists_within_classes = np.empty(num_classes)
    max_dists_between_classes = np.empty(num_classes)

    t0 = time.time()

    for i in range (0,num_classes):
        max_dists_within_classes[i] = 0
        max_dists_between_classes[i] = 0
        curr_class = train_classes_array[i]
        neigh_i = neighs[i]
        for j in range (0,num_classes):
            #print('j,len(X_tests_i[j]),X_tests_i[j].shape:',j,len(X_tests_i[j]),X_tests_i[j].shape)
            #print('X_tests_i[j]:',X_tests_i[j])
            dist_i_j,dist_i_j_ind = neigh_i.kneighbors(X_tests_i[j])
            num_dist = len(dist_i_j[j])
            #print('i,j,dist_i_j.shape,num_dist:',i,j,dist_i_j.shape,num_dist)
            #print('dist_i_j:',dist_i_j)
            #print('X_tests_i[j]:',X_tests_i[j])
            max_dist_i_j = np.max(dist_i_j)
            #print('max_dist_i_j:',max_dist_i_j)
            if i == j:
                max_dists_within_classes[i] = max_dist_i_j          # max distance to the trained class
                if (num_classes == 1):
                    max_dists_between_classes[i] = max_dist_i_j     # this is needed for the case of num_classes = 1
            else:
                if max_dist_i_j > max_dists_between_classes[i]:
                    max_dists_between_classes[i] = max_dist_i_j      # max distance to non trained classes
        
    t1 = time.time() - t0
    logging_text = 'Time elapsed for single class distance calculations: '+ ' {:.4f}'.format(t1)+ ' s'
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    return max_dists_within_classes,max_dists_between_classes

cpdef softmax_single_class_distances_c(module_dir,db_type,session,StatusData,status_datas_schema,train_classes_array,neighs,
                                       x_test,max_dists_within_classes,max_dists_between_classes):
    cdef:

        int i,j,num_classes,num_in_vectors

    module = 'softmax_single_class_distances_c '

    num_classes = len(train_classes_array)
    num_in_vectors = len(x_test)
    norm_dists = np.zeros((num_in_vectors,num_classes))
    softmax_norm_dists = np.zeros((num_in_vectors,num_classes))

    t0 = time.time()

    logging_text = 'num_in_vectors,num_classes  : '+str(num_in_vectors)+' , '+str(num_classes)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'ADDLOG train_classes_array  : '+str(train_classes_array)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    #logging_text = 'ADDLOG x_test  : '+str(x_test)
    #if db_type == "File":
    #   write_log_api_file(module_dir,module,logging_text)
    #else:
    #    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'ADDLOG max_dists_within_classes  : '+str(max_dists_within_classes)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'ADDLOG max_dists_between_classes  : '+str(max_dists_between_classes)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    max_dists_within_classes_mv = np.mean(max_dists_within_classes)
    max_dists_between_classes_mv = np.mean(max_dists_between_classes)

    # calculation of inverse normalized distances - high values for close distances
    #print('num_classes:',num_classes)
    #print('num_in_vectors:',num_in_vectors)
    if (num_classes > 1):
        for i in range (0,num_classes):
            #curr_class = classes_array[i]
            neigh_i = neighs[i]
            #x_test_j = X_test[j].reshape(1, -1)
            dist_i_j,dist_i_j_ind = neigh_i.kneighbors(x_test)
            #print('dist_i_j.shape:',dist_i_j.shape)
            #print('max_dist_within_classes[i],np.max(dist_i_j[0]):',max_dists_within_classes[i],np.max(dist_i_j[0]))
            for j in range (0,num_in_vectors):
                """
                if dist_i_j[j][0] > 0.1*max_dists_within_classes[i]:   # to avoid to big normalized distances and overflow for soft_max_sum_j
                    norm_dists[j][i] = max_dists_within_classes[i]/dist_i_j[j][0]
                    print('dist_i_j[j][0]:',j,i,dist_i_j[j][0])
                    print('norm_dists[j][i]:',j,i,norm_dists[j][i])
                else:
                    norm_dists[j][i] = max_dists_between_classes[i]
                    print('dist_i_j[j][0] - else:',j,i,dist_i_j[j][0])
                    print('norm_dists[j][i] - else:',j,i,norm_dists[j][i])
                """
                dist_i_j_mv = np.mean(dist_i_j[j])  # mean value distance of knn next neighbours for class i
                if dist_i_j[j][0] > 0:   
                    #norm_dists[j][i] = max_dists_within_classes[i]/dist_i_j[j][0]
                    norm_dists[j][i] = max_dists_within_classes_mv/dist_i_j_mv
                    #print('dist_i_j[j][0]:',j,i,dist_i_j[j][0])
                    #print('norm_dists[j][i]:',j,i,norm_dists[j][i])
                else:
                    #norm_dists[j][i] = 100*max_dists_between_classes[i]
                    norm_dists[j][i] = 100*max_dists_between_classes_mv
                    #print('dist_i_j[j][0] - else:',j,i,dist_i_j[j][0])
                    #print('norm_dists[j][i] - else:',j,i,norm_dists[j][i])

        #print('norm_dists:',norm_dists)

        # softmax and normalization
        """
        for i in range (0,num_classes):            
            for j in range (0,num_in_vectors):
                fact = 1.0/(np.sum(norm_dists[j][:]))
                soft_max_sum_j = np.sum(np.exp(fact*norm_dists[j][:]))
                print('j,fact,soft_max_sum_j:',j,fact,soft_max_sum_j)
                softmax_norm_dists[j][i] = np.exp(fact*norm_dists[j][i])/soft_max_sum_j
                print('softmax_norm_dists[j][i]:',softmax_norm_dists[j][i])
        """

        # normalization - positive vactor values and summation over vector components = 1
        for i in range (0,num_classes):            
            for j in range (0,num_in_vectors):
                #min_norm_dists_j = np.min(norm_dists[j][:])
                #norm_dists[j][:] = norm_dists[j][:] - min_norm_dists_j  # shift to positive values
                sum_norm_dists_j = np.sum(norm_dists[j][:])
                if sum_norm_dists_j > 0:
                    softmax_norm_dists[j][i] = norm_dists[j][i]/sum_norm_dists_j
                else:
                    softmax_norm_dists[j][i] = 0
                #print('softmax_norm_dists[j][i]:',softmax_norm_dists[j][i])

    else:  # simple mormalization from [0,max_dists_within_classes] to [1,0]
        neigh_0 = neighs[0]
        #print('len(x_test),x_test.shape:',len(x_test),x_test.shape)
        #print('x_test:',x_test)
        dist_0_j,dist_0_j_ind = neigh_0.kneighbors(x_test)
        #print('dist_0_j:',dist_0_j)
        #print('dist_0_j.shape:',dist_0_j.shape)
        #print('dist_0_j[0][0]:',dist_0_j[0][0])
        max_dist_within_classes = max_dists_within_classes[0]
        #print('dist_0_j:',dist_0_j)
        #print('max_dist_within_classes,np.max(dist_0_j[0]):',max_dist_within_classes,np.max(dist_0_j))
        for j in range (0,num_in_vectors):
            #if (dist_0_j[j][0] > 10*max_dist_within_classes):         
            #    softmax_norm_dists[j][0] = 0       
            #else:
            #    softmax_norm_dists[j][0] = (10*max_dist_within_classes-dist_0_j[j][0])/(10*max_dist_within_classes)
            # changed on 28.8.23 because reject_thres = 0.5 should correcpond to dist_0_j[j][0] = max_dist_within_classes
            #print('j,dist_0_j[j][0],max_dist_within_classes : ',j,dist_0_j[j][0],max_dist_within_classes)
            if (dist_0_j[j][0] > 4*max_dist_within_classes):         
                softmax_norm_dists[j][0] = 0       
            else:
                softmax_norm_dists[j][0] = (4*max_dist_within_classes-dist_0_j[j][0])/(4*max_dist_within_classes)
        #print('softmax_norm_dists:',softmax_norm_dists)

    #for j in range (0,num_in_vectors):
        #print('softmax_norm_dists:',softmax_norm_dists[j][:],np.sum(softmax_norm_dists[j][:]))

    t1 = time.time() - t0
    logging_text = 'Time elapsed for single class softmax calculations: '+ ' {:.4f}'.format(t1)+ ' s'
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    return softmax_norm_dists

cpdef load_train_data_init_knn_c(module_dir,db_type,session,StatusData,status_datas_schema,
                               train_dir_knn,DataRefInfo,data_ref_infos_schema,
                               ai_type,int n_neighbors,userid,dataref_list,
                               softmax,feature_norm):

    from basic_db_functions_cython import get_ref_data_info

    cdef int i,NPZ_file_num

    ResOk = False
    module = 'load_train_data_init_knn_c '

    # check feature_norm and softmax
    if feature_norm != "YES":
        feature_norm = "NO"
    if softmax != "YES":
        softmax = "NO"

    cdef list NPZ_file_name_list_train = []

    if db_type == "File":
        ai_module = get_module_from_dir(module_dir)
        logging_text = 'ai_module:'+ai_module
        write_log_api_file(module_dir,module,logging_text) 
        train_dir_knn = set_train_dir(ai_module,'knn')
        logging_text = 'train_dir_knn:'+train_dir_knn
        write_log_api_file(module_dir,module,logging_text) 
        if len(dataref_list)>0:
            for i in range (0,len(dataref_list)):   # loop over data refs
                #check if subfolder dataref_list[i] exists
                train_data_subfolder = train_dir_knn+dataref_list[i]+'/'
                logging_text = 'train_data_subfolder:'+train_data_subfolder
                write_log_api_file(module_dir,module,logging_text) 
                try: 
                    #if os.exists(train_data_subfolder):
                    npz_file_list = os.listdir(train_data_subfolder)
                    logging_text = 'train_data_npz_file_list:'+str(npz_file_list)
                    write_log_api_file(module_dir,module,logging_text) 
                    for filename in npz_file_list:
                        logging_text = 'train_data_file:'+train_data_subfolder+filename
                        write_log_api_file(module_dir,module,logging_text) 
                        NPZ_file_name_list_train.append(train_data_subfolder+filename)
                except:
                    logging_text = 'data with data_ref '+dataref_list[i]+' for knn init not loaded sucessfully'
                    write_log_api_file(module_dir,module,logging_text)  
                    neigh_file_name = ''
                    train_labels_file_name = ''
                    train_min_file_name = ''
                    train_max_file_name = ''
                    softmax_file_name = ''
                    init_ref = ''
                    return ResOk,init_ref,neigh_file_name,train_labels_file_name,train_min_file_name,train_max_file_name,softmax_file_name     
    else:
        #logging_text = ' dataref_list '+str(dataref_list[i])
        #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        if len(dataref_list)>0:
            for i in range (0,len(dataref_list)):   # loop over data refs
                try:
                    # get ref_data for current data ref
                    GetRefDataInfoOK,ref_data = get_ref_data_info(session,DataRefInfo,data_ref_infos_schema,'ALL',dataref_list[i], 'ALL')
                    if GetRefDataInfoOK:
                        NPZ_file_name_list_train.append(ref_data[0]['dataref_filename'])
                    else:
                        logging_text = ' data with data_ref '+dataref_list[i]+' for knn init not loaded sucessfully'
                        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                        neigh_file_name = ''
                        train_labels_file_name = ''
                        train_min_file_name = ''
                        train_max_file_name = ''
                        softmax_file_name = ''
                        init_ref = ''
                        return ResOk,init_ref,neigh_file_name,train_labels_file_name,train_min_file_name,train_max_file_name,softmax_file_name    
                except:
                    logging_text = ' data with data_ref '+dataref_list[i]+' for knn init not loaded '
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                    neigh_file_name = ''
                    train_labels_file_name = ''
                    train_min_file_name = ''
                    train_max_file_name = ''
                    softmax_file_name = ''
                    init_ref = ''
                    return ResOk,init_ref,neigh_file_name,train_labels_file_name,train_min_file_name,train_max_file_name,softmax_file_name     
        else:
            logging_text = ' reference data info for knn init not available'
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            neigh_file_name = ''
            train_labels_file_name = ''
            train_min_file_name = ''
            train_max_file_name = ''
            softmax_file_name = ''
            init_ref = ''
            return ResOk,init_ref,neigh_file_name,train_labels_file_name,train_min_file_name,train_max_file_name,softmax_file_name
        
    logging_text = 'List of kNN train data files '+str(NPZ_file_name_list_train)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    t0 = time.time()
    cdef list data_labels_multiple_train = []
    cdef list data_features_multiple_train = []
    NPZ_file_num = 0
    for NPZ_file_name in NPZ_file_name_list_train:
        if ".npz" in NPZ_file_name:
            try:
                data_features_labels = numpy.load(NPZ_file_name,allow_pickle=True)
                features_has_nan = numpy.isnan(data_features_labels['f']).any()
                if features_has_nan:
                    NPZ_file_num = NPZ_file_num + 1
                    logging_text = 'kNN train data file '+NPZ_file_name+' contains nan values'
                    if db_type == "File":
                        write_log_api_file(module_dir,module,logging_text)  
                    else:
                        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                else:
                    data_features_multiple_train.append(data_features_labels['f'])
                    data_labels_multiple_train.append(data_features_labels['l'])
                    NPZ_file_num = NPZ_file_num + 1
                    logging_text = 'kNN train data file '+NPZ_file_name+' sucessfully loaded'
                    if db_type == "File":
                        write_log_api_file(module_dir,module,logging_text)  
                    else:
                        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            except:
                logging_text = ' file '+NPZ_file_name+' in kNN train data directory do not exist'
                if db_type == "File":
                    write_log_api_file(module_dir,module,logging_text)  
                else:
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    if NPZ_file_num == 0:
        logging_text = ' files in kNN train data directory do not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        neigh_file_name = ''
        train_labels_file_name = ''
        train_min_file_name = ''
        train_max_file_name = ''
        softmax_file_name = ''
        init_ref = ''
        return ResOk,init_ref,neigh_file_name,train_labels_file_name,train_min_file_name,train_max_file_name,softmax_file_name

    #Initialize kNN Classifier/Regressor from multiple training files
    #neigh = KNeighborsRegressor(n_neighbors=n_neighbors,weights = 'distance')
    if ai_type == 'kNN_Reg':
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
        # create concateneted numpy array for train labels 
        y_train = numpy.array(data_labels_multiple_train[0])
        for i in range (1,len(data_labels_multiple_train)):
            y_train_i = numpy.array(data_labels_multiple_train[i])
            y_train = numpy.concatenate((y_train,y_train_i),axis=0)
    if ai_type == 'kNN_Class':
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        # create concateneted numpy array for train labels 
        y_train = numpy.array(data_labels_multiple_train[0])
        for i in range (1,len(data_labels_multiple_train)):
            y_train_i = numpy.array(data_labels_multiple_train[i])
            y_train = numpy.concatenate((y_train,y_train_i),axis=0)
    #print('first y_train -labels ',y_train[0:10])
    #print('last y_train -labels ',y_train[len(y_train)-10:len(y_train)])

    # create concateneted numpy array for train features 
    X_train = numpy.array(data_features_multiple_train[0])
    for i in range (1,len(data_features_multiple_train)):
        X_train_i = numpy.array(data_features_multiple_train[i])
        X_train = numpy.concatenate((X_train,X_train_i),axis=0)

    # calculate minima and maxima from train data
    num_features = X_train.shape[1]
    if feature_norm == 'YES':
        X_train_min = numpy.empty(num_features)
        X_train_max = numpy.empty(num_features)
        for i in range (0,num_features):
            X_train_min[i] = X_train[:,i].min()
            X_train_max[i] = X_train[:,i].max()
        X_train = normalize_columns_c(X_train,X_train_max,X_train_min)
        #print('X_train_min:',X_train_min)
        #print('X_train_max:',X_train_max)
        #print('X_train:',X_train)
    logging_text = 'Normalization: '+feature_norm
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    # train the kNN classifier/regressor
    try:
        neigh.fit(X_train,y_train)
    except:
        neigh.fit(X_train,y_train)
        logging_text = 'Inconsinsistent training feature and label data : '+'shape features : '+str(X_train.shape)+'shape labels : '+str(y_train.shape)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        ResOk=False
        return ResOk,'','','','','',''

    t1 = time.time() - t0
    logging_text = 'Time elapsed for initializing kNN Classifier/Regressor: '+ ' {:.4f}'.format(t1)+ ' s'
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    #create file names for neigh and y_train
    #time.sleep(0.1) # Sleep for 0.1 seconds - to be sure that a new data ref is created
    #datetime_int = int(time.time()*1000)
    #init_ref = f"{datetime_int}"
    init_ref = str(uuid.uuid4())
    NPZ_file = 'kNN_Train'
    # set knn init file names
    neigh_file_name = train_dir_knn+'neigh_'+str(NPZ_file_num)+'_files_'+init_ref+'_'+NPZ_file+'.pickle'
    logging_text = 'Filename for neigh object: '+neigh_file_name
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    train_labels_file_name = train_dir_knn+'train_labels_'+str(NPZ_file_num)+'_files_'+init_ref+'_'+NPZ_file+'.pickle'
    logging_text = 'Filename for train labels object: '+ train_labels_file_name
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    if feature_norm == 'YES':
        train_min_file_name = train_dir_knn+'min_'+str(NPZ_file_num)+'_files_'+init_ref+'_'+NPZ_file+'.pickle'
        logging_text = 'Filename for train min object: '+ train_min_file_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        train_max_file_name = train_dir_knn+'max_'+str(NPZ_file_num)+'_files_'+init_ref+'_'+NPZ_file+'.pickle'
        logging_text = 'Filename for train max object: '+ train_max_file_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    
    # Save knn init objects as pickle files
    file_name_lock = neigh_file_name+'.lock'
    lock_file_name = FileLock(file_name_lock)
    with lock_file_name:
        with open(neigh_file_name,"wb") as pickle_out:
            pickle.dump(neigh, pickle_out)
    try:
        os.remove(lock_file_name)
    except:
        pass

    file_name_lock = train_labels_file_name+'.lock'
    lock_file_name = FileLock(file_name_lock)
    with lock_file_name:
        with open(train_labels_file_name,"wb") as pickle_out:
            pickle.dump(y_train, pickle_out)
    try:
        os.remove(lock_file_name)
    except:
        pass

    if feature_norm == 'YES':
        file_name_lock = train_min_file_name+'.lock'
        lock_file_name = FileLock(file_name_lock)
        with lock_file_name:
            with open(train_min_file_name,"wb") as pickle_out:
                pickle.dump(X_train_min, pickle_out)
        try:
            os.remove(lock_file_name)
        except:
            pass

        file_name_lock = train_max_file_name+'.lock'
        lock_file_name = FileLock(file_name_lock)
        with lock_file_name:
            with open(train_max_file_name,"wb") as pickle_out:
                pickle.dump(X_train_max, pickle_out)
        try:
            os.remove(lock_file_name)
        except:
            pass

    else:
        train_min_file_name = ''
        train_max_file_name = ''

    # create softmax statistics
    softmax_file_name = ''
    if softmax == "YES":
        # create train/test data sets for softmax statitics 
        #x_train_softmax, x_test_softmax, y_train_softmax, y_test_softmax = \
        #    train_test_split(X_train, y_train, test_size=0.75, random_state=123)
        # create single class training data sets and initialize single class knn classifiers
        #print('X_train.shape,y_train.shape:',X_train.shape,y_train.shape)
        classes_array_train,neighs,x_train_softmax_i,y_train_softmax_i = \
        train_single_classes_init_knn_c(module_dir,db_type,session,StatusData,status_datas_schema,
                                        X_train,y_train,ai_type)
        #print('x_train_softmax_i[0].shape,y_train_softmax_i[0].shape:',x_train_softmax_i[0].shape,y_train_softmax_i[0].shape)
        #train_single_classes_init_knn_c(module_dir,session,StatusData,status_datas_schema,
        #                                x_train_softmax,y_train_softmax)
        # create single class test data sets and initialize single class knn classifiers
        """
        classes_array_test,x_test_softmax_i,y_test_softmax_i = \
        test_single_classes_c(module_dir,db_type,session,StatusData,status_datas_schema,
                              X_train,y_train)
        """
        #print('x_test_softmax_i[0].shape,y_test_softmax_i[0].shape:',x_test_softmax_i[0].shape,y_test_softmax_i[0].shape)
        #test_single_classes_c(module_dir,session,StatusData,status_datas_schema,
        #                      x_test_softmax,y_test_softmax)
        # calculate maximum distances test/train within single classes 
        # calculate maximum distances test/train between single classes and the othe classes
        """
        max_dists_within_classes,max_dists_between_classes = \
        calc_single_class_distances_c(module_dir,db_type,session,StatusData,status_datas_schema,
                                      classes_array_train,neighs,
                                      x_test_softmax_i,y_test_softmax_i)
        """
        max_dists_within_classes,max_dists_between_classes = \
        calc_single_class_distances_c(module_dir,db_type,session,StatusData,status_datas_schema,
                                      classes_array_train,neighs,
                                      x_train_softmax_i,y_train_softmax_i)
        #print('load_train_data_init_knn_c/max_dists_within_classes:',max_dists_within_classes)
        # create file name for export of softmax data to npz file                              
        softmax_file_name = train_dir_knn+'softmax_'+init_ref+'_'+NPZ_file
        logging_text = 'Filename for softmax object: '+softmax_file_name+'.npz'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        # Save softmax objects as npz file
        np.savez_compressed(softmax_file_name,c=classes_array_train,n=neighs,w=max_dists_within_classes,b=max_dists_between_classes)

    ResOk=True
    return ResOk,init_ref,neigh_file_name,train_labels_file_name,train_min_file_name,train_max_file_name,softmax_file_name

cpdef load_test_data_multiple_c(module_dir,db_type,session,StatusData,status_datas_schema,test_data_dir):
    cdef int i
    LoadOK = False
    module = 'load_test_data_multiple_c '
    if os.path.exists(test_data_dir):
        logging_text = ' kNN test data directory: '+test_data_dir,' does exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    else:
        logging_text = ' kNN test data directory: '+test_data_dir,' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        X_test = numpy.zeros(1)
        return LoadOK,X_test
    #cdef list data_labels_multiple_test = []
    cdef list data_features_multiple_test = []
    NPZ_file_name_list_test = os.listdir(test_data_dir)
    NPZ_file_name_list_test.sort()
    t0 = time.time()
    for NPZ_file_name in NPZ_file_name_list_test:
        if ".npz" in NPZ_file_name:
            data_features_labels = numpy.load(test_data_dir+NPZ_file_name,allow_pickle=True)
            data_features_multiple_test.append(data_features_labels['f'])
            #data_labels_multiple_test.append(data_features_labels['l'])
            logging_text = 'kNN Test File '+str(NPZ_file_name)+' loaded'
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    
    #Initialize kNN Classifier/Regressor from multiple test files
    #create concateneted numpy array for test labels 
    """
    y_test = numpy.array(data_labels_multiple_test[0],dtype='f',order='C')
    for i in range (1,len(data_labels_multiple_test)):
        y_test_i = numpy.array(data_labels_multiple_test[i],dtype='f',order='C')
        y_test = numpy.concatenate((y_test,y_test_i),axis=0)
    """
    # create concateneted numpy array for test features 
    X_test = numpy.array(data_features_multiple_test[0])
    for i in range (1,len(data_features_multiple_test)):
        X_test_i = numpy.array(data_features_multiple_test[i])
        X_test = numpy.concatenate((X_test,X_test_i),axis=0)
    t1 = time.time() - t0
    logging_text = 'Time elapsed for loading kNN test files: '+ ' {:.4f}'.format(t1)+ ' s'
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    LoadOK = True
    return LoadOK,X_test


cpdef knn_check_reject_c(module_dir,db_type,session,StatusData,status_datas_schema,x_test,neigh,train_labels,ai_type_knn,kNN_reject,reject_thres,reject_labelval_diff):

    cdef:
        int i,j,num_test_samples,num_reject,num_kNN_LabelNotSame,labels_different
        float kNN_MinLabelVal,kNN_MaxLabelVal,kNN_LabelVal

    module = 'knn_check_reject_c '
    num_reject = 0
    num_test_samples =  len(x_test)   

    cdef:
        int[::1] rejects = numpy.zeros(num_test_samples,dtype=numpy.int32)
        float[::1] qvals = numpy.zeros(num_test_samples,dtype=numpy.float32)
        float[::1] diff_knn_dist_test = numpy.zeros(num_test_samples,dtype=numpy.float32)
        float[::1] diff_labelval_test = numpy.zeros(num_test_samples,dtype=numpy.float32)

    #rejects = numpy.zeros(num_test_samples)
    #qvals = numpy.zeros(num_test_samples)   #best distance in percentage to reject threshold

    # additional logging
    #x_test_log = np.asarray(x_test)
    #logging_text = 'x_test: '+str(x_test_log[0])
    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    
    t0 = time.time()

    dist_neigh,dist_neigh_ind = neigh.kneighbors(x_test,kNN_reject)

    # additional logging
    #logging_text = 'ADDLOG Nearest Distances: '+str(dist_neigh)
    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    #logging_text = 'ADDLOG Nearest Distance Indices: '+str(dist_neigh_ind)
    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    #t1 = time.time() - t0
    #logging_text = 'Time elapsed for neigh.kneighbors: '+' {:.4f}'.format(t1) + ' s for '+str(num_test_samples)+' test samples'
    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    logging_text = 'ai_type_knn: '+ai_type_knn
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    #t0 = time.time()

    num_reject_diff_val = 0
    if ai_type_knn == 'kNN_Reg':
        for i in range (0,num_test_samples):
            dist_neigh_i = dist_neigh[i][0]
            diff_knn_dist_test[i] = dist_neigh_i
            qvals[i] = 100-((dist_neigh_i)/reject_thres)*100
            if (dist_neigh_i>reject_thres):
                num_reject = num_reject+1    
                rejects[i] = 1
            else:
                #min/max labelled distance in kNN set
                kNN_MinLabelVal = 1000000.0
                kNN_MaxLabelVal = -1000000.0
                for j in range(0,kNN_reject):
                    kNN_LabelVal = train_labels[dist_neigh_ind[i][j]]
                    if (kNN_MinLabelVal > kNN_LabelVal):
                        kNN_MinLabelVal = kNN_LabelVal
                    if (kNN_MaxLabelVal < kNN_LabelVal):
                        kNN_MaxLabelVal = kNN_LabelVal
                diff_labelval_test[i] = (kNN_MaxLabelVal-kNN_MinLabelVal)
                if (diff_labelval_test[i]>reject_labelval_diff):
                    num_reject_diff_val = num_reject_diff_val+1
                    num_reject = num_reject+1    
                    rejects[i] = 1
        # additional logging
        #logging_text = 'Dist Rejects: '+str(num_reject-num_reject_diff_val)
        #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        #logging_text = 'Diff LabelVal Rejects: '+str(num_reject_diff_val)
        #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    
    num_kNN_LabelNotSame = 0
    if ai_type_knn == 'kNN_Class':
        for i in range (0,num_test_samples):
            dist_neigh_i = dist_neigh[i][0]
            diff_knn_dist_test[i] = dist_neigh_i
            qvals[i] = 100-((dist_neigh_i)/reject_thres)*100
            if (dist_neigh_i>reject_thres):
                num_reject = num_reject+1    
                rejects[i] = 1
            else:
                #min/max labelled distance in kNN set
                labels_different = 0
                kNN_LabelValLabel1 = train_labels[dist_neigh_ind[i][0]]
                for j in range(1,kNN_reject):
                    kNN_LabelValLabel2 = train_labels[dist_neigh_ind[i][j]]
                    if (kNN_LabelValLabel1 != kNN_LabelValLabel2):
                        labels_different = 1
                    break
                if (labels_different): # labels within train_labels[dist_neigh_ind[i]] are not the same
                    num_kNN_LabelNotSame = num_kNN_LabelNotSame+1
                    num_reject = num_reject+1    
                    rejects[i] = 1
        # additional logging
        #logging_text = 'Labels Not The Same Rejects: '+str(num_kNN_LabelNotSame)
        #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    t1 = time.time() - t0
    logging_text = 'Time elapsed for kNN reject predictions: '+' {:.4f}'.format(t1) + ' s for '+str(num_test_samples)+' test samples'
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Results including rejects on test data: '
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Maximum distance on test samples: '+' {:.4f}'.format(numpy.max(diff_knn_dist_test))
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Minimum distance on test samples: '+' {:.4f}'.format(numpy.min(diff_knn_dist_test))
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Avarage distance on test samples: '+' {:.4f}'.format((numpy.sum(diff_knn_dist_test)/len(diff_knn_dist_test)))
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Distance threshold rejects:'+' {:.4f}'.format(reject_thres)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Reject rate:'+' {:.4f}'.format(num_reject/num_test_samples)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    if ai_type_knn == 'kNN_Reg':
        logging_text = 'Maximum kNN label value difference on test samples: '+' {:.4f}'.format(numpy.max(diff_labelval_test))
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        logging_text = 'Minimum kNN label value difference on test samples: '+' {:.4f}'.format(numpy.min(diff_labelval_test))
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        logging_text = 'Average kNN label value difference on test samples: '+' {:.4f}'.format((numpy.sum(diff_labelval_test)/len(diff_labelval_test)))
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        logging_text = 'Label value difference threshold rejects::'+' {:.4f}'.format(reject_labelval_diff)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    if ai_type_knn == 'kNN_Class': 
        logging_text = 'Labels not the same rate:'+' {:.4f}'.format(num_kNN_LabelNotSame/num_test_samples)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    return rejects,qvals


# calculate mean value data on data_frames(num_frames_all,frames_dim):  
# calculate mean values of data frames over num_frames_ave frames in step_frames_ave steps
cpdef MeanValDataFrames_c(data_frames,int num_frames_ave,int step_frames_ave):
 
    cdef int i,num_frames_all,frames_dim,num_mean_val_frames

    num_frames_all = len(data_frames)
    frames_dim = len(data_frames[0])

    """
    # count first num_mean_frames for mean_cms_data_pow numpy.array allocation 
    num_mean_val_frames=0
    for i in range (0,num_frames_all-num_frames_ave+1,step_frames_ave):
        num_mean_val_frames=num_mean_val_frames+1

    # now calculated the mean values 
    mean_val_data_frames = numpy.zeros((num_mean_val_frames,frames_dim))
    num_mean_val_frames=0
    for i in range (0,len(data_frames)-num_frames_ave+1,step_frames_ave):
        mean_val_data_frames[num_mean_val_frames] = numpy.mean(data_frames[i:i+num_frames_ave],axis=0)
        num_mean_val_frames=num_mean_val_frames+1
    """
    if (step_frames_ave != 1):
        # 'step_frames_ave must have value = 1'
         # set result status with no result available
        result_status = 0   
        return result_status,data_frames

    # now calculated the mean values 
    mean_val_data_frames = numpy.zeros((num_frames_all,frames_dim))
    for i in range (0,num_frames_all):
        if i < num_frames_ave:
            mean_val_data_frames[i] = numpy.mean(data_frames[0:i+1],axis=0)                       #averaging first avaliable frames i < num_frames_ave
        else:
            mean_val_data_frames[i] = numpy.mean(data_frames[i+1-num_frames_ave:i+1],axis=0)      #averaging over num_frames_ave frames before i 

    # set result status with result available
    result_status = 1  

    return result_status,mean_val_data_frames

# calculate mean value data on data_frames(num_frames_all,frames_dim):  
# calculate mean values of data frames over num_frames_ave frames in step_frames_ave steps
# using file buffer which is depending on the task_id, the input for each call is only one frame
cpdef MeanValDataFrames_FileBuf_c(db_type,data_frames,int num_frames_ave,int step_frames_ave,task_id, \
                                  module_dir,session,StatusData,status_datas_schema):
 
    cdef int i,num_frames_all,frames_dim,num_mean_val_frames,result_status,buf_created

    module = 'MeanValDataFrames_FileBuf_c'

    # initialize result status with no result available
    result_status = 0   
    frames_dim = len(data_frames[0])

    if (step_frames_ave != 1):
        logging_text = 'step_frames_ave must have value = 1 for MeanValDataFrames_FileBuf_c'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        buf_created = False
        mean_val_data_frames = numpy.zeros((1,frames_dim))
        return result_status,mean_val_data_frames        

    # set pp_buf_file_path
    pp_buf_file_path = module_dir+"/ppbuf/"
    # set preprocessing file name based on PpBufId
    pp_buf_file_name = "pp_buf_"+str(task_id)
    pp_buf_file = os.path.join(pp_buf_file_path,pp_buf_file_name)
    pp_buf_file_npz = pp_buf_file+".npz"
    #print(pp_buf_file)

    # check if the pre processing buffer file exists, create  pre processing buffer file with current feature records if it does not exist
    buf_created = False
    if os.path.isfile(pp_buf_file_npz):
        logging_text = 'preprocessing buffer file '+pp_buf_file_npz+' does exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)         
    else:
        np.savez_compressed(pp_buf_file, f=data_frames) 
        logging_text = 'preprocessing buffer file '+pp_buf_file_npz+' was created'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        buf_created = True
        #mean_val_data_frames = numpy.zeros((1,frames_dim))  changed on 31.7.23 - average is now also calculated for num_feature_records < num_frames_ave
        #return result_status,mean_val_data_frames           changed on 31.7.23 - average is now also calculated for num_feature_records < num_frames_ave

    # try to load the pre processing buffer file
    buf_loaded = False
    try:
        pp_buf_features_load = np.load(pp_buf_file_npz,allow_pickle=True)
        pp_buf_features = pp_buf_features_load['f']
        #print('pp_buf_features.shape:',pp_buf_features.shape)
        buf_loaded = True
    except:
        logging_text = 'preprocessing buffer file '+pp_buf_file_npz+' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    # add the current feature vector frame to the file buffer (only if buffer was not created)
    if buf_loaded == True:

        # add current feature record to buffer if buffer did already exist and was loaded
        if buf_created == False:
            pp_buf_features = np.vstack([pp_buf_features, data_frames[0]])
        
        logging_text = 'preprocessing buffer shape :'+str(pp_buf_features.shape)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

        num_feature_records = len(pp_buf_features)
        logging_text = 'num feature records in preprocessing buffer file '+str(num_feature_records)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

        """  changed on 31.7.23 - average is now also calculated for num_feature_records < num_frames_ave
        if num_feature_records < num_frames_ave:   # not enough frames for preprocessing in file buffer
            #print('pp_buf_features.shape:',pp_buf_features.shape)
            np.savez_compressed(pp_buf_file, f=pp_buf_features) 
            mean_val_data_frames = numpy.zeros((1,frames_dim))
            return result_status,mean_val_data_frames
        else:
            mean_val_data_frames = np.mean(pp_buf_features,axis=0)
            mean_val_data_frames = mean_val_data_frames.reshape(1,frames_dim)
            #print(mean_val_data_frames.shape)
            #print(mean_val_data_frames)
            result_status = 1  
            # remome the first feature vector (frame) from the file buffer
            pp_buf_features_new = np.delete(pp_buf_features,0,0)
            np.savez_compressed(pp_buf_file, f=pp_buf_features_new) 
            #np.savez_compressed(pp_buf_file_sum, f=sum_val_data_frames_new)
        """
        mean_val_data_frames = np.mean(pp_buf_features,axis=0)
        mean_val_data_frames = mean_val_data_frames.reshape(1,frames_dim)
        #print('data_frames[0] :',data_frames[0])
        #print('mean_val_data_frames.shape :',mean_val_data_frames.shape)
        #print('mean_val_data_frames ',mean_val_data_frames)

        # remome the first feature vector (frame) from the file buffer
        if num_frames_ave <= num_feature_records:
            pp_buf_features_new = np.delete(pp_buf_features,0,0)
            np.savez_compressed(pp_buf_file, f=pp_buf_features_new) 
        else:
            np.savez_compressed(pp_buf_file, f=pp_buf_features) 
        #np.savez_compressed(pp_buf_file_sum, f=sum_val_data_frames_new)

        #set result status to sucessfull
        result_status = 1  

    else:
        #set result status to not sucessfull
        result_status = 0

    return result_status,mean_val_data_frames

# Resample data frames to dimension frames_dim_new starting from min_index to max_index (in data frames)
cpdef ReSampleDataFrames_c(data_frames,int frames_dim_new,int min_ind=0,int max_ind=0): 
 
    cdef int i,num_frames_all,frames_dim

    if frames_dim_new > 0:

        num_frames_all = len(data_frames)
        frames_dim = len(data_frames[0])
        if max_ind == 0:
            max_ind = frames_dim
        if max_ind > frames_dim:
            max_ind = frames_dim

        if max_ind-min_ind > 1:
            decimate_factor = int((max_ind-min_ind)/frames_dim_new)
        else:
            decimate_factor = 1

        # dimension of extracted and resampled power data - n_dim
        resampled_data_frames = np.zeros((num_frames_all,frames_dim_new))
        # resample maximun power data to frames_dim_new values
        if decimate_factor < 4:
            for i in range (0,num_frames_all):
                resampled_data_frames[i] = signal.resample(data_frames[i][min_ind:max_ind],frames_dim_new)
        else:
            for i in range (0,num_frames_all):
                #decimated_data_frame = signal.decimate(data_frames[i][min_ind:max_ind],decimate_factor)
                #resampled_data_frames[i] = signal.resample(decimated_data_frame,frames_dim_new)
                resampled_data_frames[i] = signal.resample(data_frames[i][min_ind:max_ind:decimate_factor],frames_dim_new)

        return resampled_data_frames

    else:

        return data_frames

# normalize data frames to mean value = 0 and divide by 2.5 standard deviation 
cpdef NormlizeDataFrames_c(data_frames): 
 
    cdef int i,num_frames_all,frames_dim
    cdef float mean_val,std_val

    num_frames_all = len(data_frames)
    frames_dim = len(data_frames[0])

    data_frames_n = numpy.zeros((num_frames_all,frames_dim))

    #  normalize data frames to mean value = 0 and divide by 2.5 standard deviation 
    for i in range (0,num_frames_all):
        mean_val = numpy.mean(data_frames[i][:])
        std_val = numpy.std(data_frames[i][:])
        data_frames_n[i][:] = data_frames[i][:]-mean_val
        data_frames_n[i][:] = data_frames_n[i][:]/(2.5*std_val)

    return data_frames_n

# calculate power in watts from dbm
cpdef DBmToPowWatt_c(data_frames_dbm): 
 
    cdef int i,num_frames_all,frames_dim

    num_frames_all = len(data_frames_dbm)
    frames_dim = len(data_frames_dbm[0])

    data_frames_watt = numpy.zeros((num_frames_all,frames_dim))

    # calculate power in watt from dbm: pow_watt = 10**((pow_dbm-30)/10)
    for i in range (0,num_frames_all):
        data_frames_watt[i][:] = 10**((data_frames_dbm[i][:]-30.0)/10.0)

    return data_frames_watt

# calculate dbm from power in watts
cpdef PowWattTodBm_c(data_frames_watt): 
 
    cdef int i,num_frames_all,frames_dim

    num_frames_all = len(data_frames_watt)
    frames_dim = len(data_frames_watt[0])

    data_frames_dbm = numpy.zeros((num_frames_all,frames_dim))

    # calculate power in dbm from watts: pow_dbm = 10*log(pow_watt) +30
    for i in range (0,num_frames_all):
        data_frames_dbm[i][:] = 10*numpy.log(data_frames_watt[i][:]) + 30.0

    return data_frames_dbm

# Integrate data frames within num_integration_intervalls, modes: 'integral' or 'average'
cpdef IntegrateDataFrames_c(data_frames,int num_integration_intervalls,mode='average'): 
 
    #from scipy.integrate import simps

    cdef int i,j,num_frames_all,frames_dim,frames_dim_integration

    if num_integration_intervalls > 0:

        num_frames_all = len(data_frames)
        frames_dim = len(data_frames[0])

        # calculate the frame dimension for the integration on the frame intervalls
        frames_dim_integration = int(frames_dim/num_integration_intervalls)
        #print('frames_dim_integration',frames_dim_integration)
        xp_red = numpy.linspace(0., 1.0, frames_dim_integration)
        data_frames_integrated = numpy.zeros((num_frames_all,frames_dim_integration))
        data_frames_interval = numpy.zeros((num_frames_all,frames_dim_integration))

        # calculate integrals (or averages) for all frame intervalls
        for i in range (0,num_frames_all):
            for j in range (0,frames_dim_integration):
                data_frames_interval = data_frames[i][j*num_integration_intervalls:(j+1)*num_integration_intervalls]
                #print('frame interval',data_frames_interval)
                if mode == 'integral':
                    #data_frames_integrated[i][j] = simps(data_frames_interval,x=xp_red)
                    data_frames_integrated[i][j] = numpy.average(data_frames_interval)
                else:
                    data_frames_integrated[i][j] = numpy.average(data_frames_interval)

        return data_frames_integrated

    else:

        return data_frames 

cpdef knn_init_c(module_dir,db_type,kNN_InfoStr,kNN_InfoDetails,
                 session,StatusData,status_datas_schema,
                 train_dir_knn,ai_type,KnnInit,knn_inits_schema,
                 DataRefInfo,data_ref_infos_schema,userid,
                 dataref_list,kNN,softmax,feature_norm,
                 knn_param_id,sv_param_id):

    from basic_db_functions_cython import set_knn_init_data

    module = 'knn_init_c ' 

    # read local kNN init parameters parameters from JSON init file

    #initiale kNN Regressor via loading multiple train data npz files
    ResOk,knninit_ref,neigh_file_name,train_labels_file_name, \
    train_min_file_name,train_max_file_name,softmax_file_name \
    = load_train_data_init_knn_c(module_dir,db_type,session,StatusData,status_datas_schema,
                                 train_dir_knn,DataRefInfo,data_ref_infos_schema,                                                                                                                               
                                 ai_type,kNN,userid,dataref_list,softmax,feature_norm)

    # check softmax file name if softmax == 'YES'
    if softmax == 'YES':
        if softmax_file_name == '':
            logging_text = 'softmax file does not exists'
            ResOk = False
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)     
            return ResOk,knninit_ref    

    # set softmax file name 
    if softmax_file_name != '':
        softmax_file_name = softmax_file_name+'.npz'
    # set knn init data 
    if db_type == "File":
        knninit_info = kNN_InfoStr
        knninit_info_details = kNN_InfoDetails
        SetKnnInitDataOk = set_knn_init_file(module_dir,knninit_ref,knninit_info,knninit_info_details,
                                             neigh_file_name,train_labels_file_name,train_min_file_name,
                                             train_max_file_name,softmax_file_name,dataref_list,userid,
                                             knn_param_id,sv_param_id)

    else:
        SetKnnInitDataOk = set_knn_init_data(session,KnnInit,neigh_file_name,
                                             train_labels_file_name,train_min_file_name,
                                             train_max_file_name,softmax_file_name,knninit_ref,userid,
                                             knn_param_id,sv_param_id)

    if SetKnnInitDataOk:
        logging_text = 'set_knn_init_data status: OK'
    else:
        logging_text = 'set_knn_init_data status: not OK'
        ResOk = False

    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    return ResOk,knninit_ref

cpdef knn_get_result_c(module_dir,db_type,session,StatusData,status_datas_schema,
                       ai_type_knn,KnnInitRef,x_test,KnnInit,knn_inits_schema,    
                       kNN,kNN_reject,reject_thres,max_num_results,
                       reject_labelval_diff,pow_norm,feature_norm,softmax,info_string):

    cdef int i,num_test_samples
    cdef list knn_results = []

    from basic_db_functions_cython import get_knn_init_data

    module = 'knn_get_result_c '

    # read AI init parameters from JSON init file
    ai_type,ai_type_dl,ai_type_knn,db_type,num_threads_gprc_server = read_init_param(module_dir)

    logging_text = 'AI init data: '+str(ai_type)+','+str(ai_type_dl)+','+str(ai_type_knn)+','+str(db_type)+','+str(num_threads_gprc_server)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    # check feature_norm and softmax
    if feature_norm != "YES":
        feature_norm = "NO"
    if softmax != "YES":
        softmax = "NO"

    # get knn init data
    t0 = time.time()
    if db_type == "File":
        #knninit_ref_in_list = []
        #knninit_ref_in_list.append(KnnInitRef)
        #GetKnnInitDataInfoOK,knn_init_data_info_list = get_knn_init_info_file(module_dir,knninit_ref_in_list,"ALL")
        GetKnnInitDataInfoOK,knn_init_data_info_list = get_knn_init_info_file(module_dir,KnnInitRef,"ALL")
        neigh_file_name = knn_init_data_info_list[0]['neigh_file_name']
        train_labels_file_name = knn_init_data_info_list[0]['labels_file_name']
        train_min_file_name = knn_init_data_info_list[0]['min_file_name']    
        train_max_file_name = knn_init_data_info_list[0]['max_file_name']  
        softmax_file_name = knn_init_data_info_list[0]['softmax_file_name']  
    else:
        #logging_text = 'get_knn_init_data input: '+str(KnnInitRef)+','+str(session)+','+str(KnnInit)+','+str(knn_inits_schema)
        #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        neigh_file_name,train_labels_file_name, \
        train_min_file_name,train_max_file_name,softmax_file_name = \
        get_knn_init_data(session,KnnInitRef, KnnInit,knn_inits_schema) 

    if neigh_file_name != '':

        # load neigh and train labels object from pickle files
        lock_file_name = neigh_file_name+'.lock'
        file_name_lock = FileLock(lock_file_name)
        with file_name_lock:
            with open(neigh_file_name,"rb") as pickle_in:
                neigh = pickle.load(pickle_in)
        try:
            os.remove(lock_file_name)
        except:
            pass

        lock_file_name = train_labels_file_name+'.lock'
        file_name_lock = FileLock(lock_file_name)
        with file_name_lock:
            with open(train_labels_file_name,"rb") as pickle_in:
                train_labels = pickle.load(pickle_in)
        try:
            os.remove(lock_file_name)
        except:
            pass

        if feature_norm == 'YES':

            lock_file_name = train_min_file_name+'.lock'
            file_name_lock = FileLock(lock_file_name)
            with file_name_lock:
                with open(train_min_file_name,"rb") as pickle_in:
                    train_min = pickle.load(pickle_in)
            try:
                os.remove(lock_file_name)
            except:
                pass

            lock_file_name = train_max_file_name+'.lock'
            file_name_lock = FileLock(lock_file_name)
            with file_name_lock:
                with open(train_max_file_name,"rb") as pickle_in:
                    train_max = pickle.load(pickle_in)
            try:
                os.remove(lock_file_name)
            except:
                pass

        t1 = time.time() - t0
        logging_text = 'kNN Neigh File: '+neigh_file_name+' loaded in '+' {:.4f}'.format(t1)+' s'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    else:
        logging_text = 'kNN Neigh File: was not loaded sucessfully'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

        knn_results = []
        knn_result = {
            'id': 0,
            'knn_result': 'REJ',
            'knn_qval': -1000,
            'knn_info':'kNN Neigh File: was not loaded sucessfully',
            'knn_qvals': str(-1000),
            }
        knn_results.append(knn_result)
        return knn_results

    # check max_num_results
    if max_num_results < 1:
        max_num_results = 1
    if max_num_results > 3:
        max_num_results = 3

    # check reject_thres
    if reject_thres <= 0:
        logging_text = 'reject_thres is '+str(reject_thres)+' but must be > 0'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        knn_results = []
        knn_result = {
            'id': 0,
            'knn_result': 'REJ',
            'knn_qval': -1000,
            'knn_info': logging_text,
            'knn_qvals': str(-1000),
            }
        knn_results.append(knn_result)
        return knn_results

    # additional logging
    #logging_text = 'x_test (before normalization): '+str(x_test)
    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    # normalize features columns in range [0,1]
    if feature_norm == 'YES':
        x_test = normalize_columns_c(x_test,train_max,train_min)
    logging_text = 'Feature normalization: '+feature_norm
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    
    # additional logging
    #logging_text = 'x_test (after normalization): '+str(x_test)
    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    #print('neigh.get_params : ',neigh.get_params())
    #print('x_test.shape : ',x_test.shape)

    # calculate predictions
    t0 = time.time()

    try: 
        pred_test = neigh.predict(x_test)
    except:
        logging_text = 'knn prediction failed'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        knn_results = []
        knn_result = {
            'id': 0,
            'knn_result': 'REJ',
            'knn_qval': -1000,
            'knn_info': logging_text,
            'knn_qvals': str(-1000),
            }
        knn_results.append(knn_result)
        return knn_results


    # additional logging
    logging_text = 'ADDLOG Predictions: '+str(pred_test)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    
    t1 = time.time() - t0
    num_test_samples = len(x_test)
    logging_text = 'Time elapsed for kNN predictions: '+' {:.4f}'.format(t1)+' s for '+str(num_test_samples)+' test samples'
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    
    # calculate softmax result vector
    if softmax == 'YES':

        if reject_thres >= 1:
            logging_text = 'reject_thres is '+str(reject_thres)+' but must be < 1 in softmax mode'
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            knn_results = []
            knn_result = {
                'id': 0,
                'knn_result': 'REJ',
                'knn_qval': -1000,
                'knn_info': logging_text,
                'knn_qvals': str(-1000),
                }
            knn_results.append(knn_result)
            return knn_results

        t0 = time.time()

        logging_text = 'Softmax file name: '+softmax_file_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        try:
            softmax_data = np.load(softmax_file_name,allow_pickle=True)
            train_classes_array = softmax_data['c']
            neighs = softmax_data['n']
            max_dists_within_classes = softmax_data['w']
            max_dists_between_classes = softmax_data['b']
            #print('knn_get_result_c/max_dists_within_classes:',max_dists_within_classes)
            #print('knn_get_result_c/max_dists_between_classes:',max_dists_between_classes)
        except:
            logging_text = 'Softmax file not loaded sucessfully'
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            knn_results = []
            knn_result = {
                'id': 0,
                'knn_result': 'REJ',
                'knn_qval': -1000,
                'knn_info': logging_text,
                'knn_qvals': str(-1000),
                }
            knn_results.append(knn_result)
            return knn_results 

        t1 = time.time() - t0
        logging_text = 'Time elapsed for loading softmax file: '+' {:.4f}'.format(t1)+' s'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

        softmax_res = \
        softmax_single_class_distances_c(module_dir,db_type,session,StatusData,status_datas_schema,
                                         train_classes_array,neighs,x_test,
                                         max_dists_within_classes,max_dists_between_classes)
        #print('softmax_res.shape:',softmax_res.shape)
        #print('softmax_res[0]:',softmax_res[0])
        #print('len(softmax_res):',len(softmax_res))

        t0 = time.time()
        #num_test_samples = len(softmax_res)
        for i in range (0,num_test_samples):  
            if ai_type_knn == 'kNN_Class':
                #additional logging
                logging_text = 'ADDLOG i,softmax_res[i]: '+str(i)+','+str(softmax_res[i])
                if db_type == "File":
                    write_log_api_file(module_dir,module,logging_text)  
                else:
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
                if max_num_results > 1:  # more than one result was requested
                    #print('test0')
                    cond_res = softmax_res[i] > reject_thres
                    num_results_i = cond_res.sum()
                    res_string = ''
                    qval_string = ''
                    #info_string = 'Success: '+str(num_results_i)+' multiple results avaliable'
                    max_qval = 0
                    res_string_list = []
                    res_qval_list = []
                    for j in range(0,len(softmax_res[i])):
                        if cond_res[j] == True:
                            #pred_min = np.abs(softmax_res[i] - softmax_res[i][j])   # identify index with prediction pred[i][j]
                            #res_string_list.append(train_classes_array[np.argmin(pred_min)])
                            res_string_list.append(train_classes_array[j])
                            res_qval_list.append(f'{100*softmax_res[i][j]:1.0f}')
                            if int(100*softmax_res[i][j]) > max_qval:
                                max_qval = int(100*(softmax_res[i][j])+0.5)
                    # sort res_string_list based on res_qval_list
                    res_string_list_s = list_sort(res_string_list, res_qval_list)
                    res_qval_list = sorted(res_qval_list,reverse=True)
                    # delete last num_del from sorted results if more than requested exist
                    if (len(res_string_list_s)) > max_num_results:
                        num_del = len(res_string_list_s) - max_num_results
                        del res_string_list_s[-num_del:]
                        del res_qval_list[-num_del:]
                    # create semicolon separated list
                    res_string =";".join(map(str,res_string_list_s))
                    qval_string =";".join(map(str,res_qval_list))
                    # remove last ; if necessery
                    if res_string.endswith(';'):
                        res_string = res_string[:-1]
                    if qval_string.endswith(';'):
                        qval_string = qval_string[:-1]
                    #logging_text = 'result string multiple decisions: '+res_string
                    #if db_type == "File":
                    #    write_log_api_file(module_dir,module,logging_text)  
                    #else:
                    #    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
                    #logging_text = 'qval string multiple decisions: '+qval_string
                    #if db_type == "File":
                    #    write_log_api_file(module_dir,module,logging_text)  
                    #else:
                    #    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
                    knn_result = {
                    'id': i,
                    'knn_result': res_string,
                    'knn_qval': max_qval,
                    'knn_info': info_string,
                    'knn_qvals': qval_string,
                    }
                    if num_results_i == 0:
                        knn_result = {
                        'id': i,
                        'knn_result': 'REJ',
                        'knn_qval': 0,
                        'knn_info':'Reject',
                        'knn_qvals': str(0),
                        }
                    knn_results.append(knn_result)
                else: # only one result was requested
                    #print('test1 : ',i,np.argmax(softmax_res[i]),softmax_res[i][np.argmax(softmax_res[i])],int(100*(softmax_res[i][np.argmax(softmax_res[i])])+0.5))
                    if softmax_res[i][np.argmax(softmax_res[i])] > reject_thres:
                        knn_result = {
                        'id': i,
                        'knn_result': str(train_classes_array[np.argmax(softmax_res[i])]),
                        'knn_qval': int(100*(softmax_res[i][np.argmax(softmax_res[i])])+0.5),
                        'knn_info': info_string,
                        'knn_qvals': str(int(100*(softmax_res[i][np.argmax(softmax_res[i])])+0.5)),
                        }
                    else:
                        knn_result = {
                        'id': i,
                        'knn_result': 'REJ',
                        'knn_qval': 0,
                        'knn_info': info_string,
                        'knn_qvals': str(0),
                        }
                    knn_results.append(knn_result)

            if ai_type_knn == 'kNN_Reg':
                #additional logging
                logging_text = 'ADDLOG softmax_res[i]: '+str(softmax_res[i])
                if db_type == "File":
                    write_log_api_file(module_dir,module,logging_text)  
                else:
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
                if max_num_results > 1:  # more than one result was requested
                    #print('test2')
                    cond_res = softmax_res[i] > reject_thres
                    num_results_i = cond_res.sum()
                    res_string = ''
                    qval_string = ''
                    #info_string = 'Success: Multiple results avaliable'
                    max_qval = 0
                    res_string_list = []
                    res_qval_list = []
                    for j in range(0,len(softmax_res[i])):
                        if cond_res[j] == True:
                            pred_min = np.abs(softmax_res[i] - softmax_res[i][j])   # identify index with prediction pred[i][j]
                            res_string_list.append(train_classes_array[np.argmin(pred_min)])
                            res_qval_list.append(f'{100*softmax_res[i][j]:1.0f}')
                            if int(100*softmax_res[i][j]) > max_qval:
                                max_qval = int(100*softmax_res[i][j]+0.5)
                    # sort res_string_list based on res_qval_list
                    res_string_list_s = list_sort(res_string_list, res_qval_list)
                    res_qval_list = sorted(res_qval_list,reverse=True)
                    # delete last num_del from sorted results if more than requested exist
                    if (len(res_string_list_s)) > max_num_results:
                        num_del = len(res_string_list_s) - max_num_results
                        del res_string_list_s[-num_del:]
                        del res_qval_list[-num_del:]
                    # create semicolon separated list
                    res_string =";".join(map(str,res_string_list_s))
                    qval_string =";".join(map(str,res_qval_list))
                    # remove last ; if necessery
                    if res_string.endswith(';'):
                        res_string = res_string[:-1]
                    if qval_string.endswith(';'):
                        qval_string = qval_string[:-1]
                    #logging_text = 'result string multiple decisions: '+res_string
                    #if db_type == "File":
                    #    write_log_api_file(module_dir,module,logging_text)  
                    #else:
                    #    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
                    #logging_text = 'qval string multiple decisions: '+qval_string
                    #if db_type == "File":
                    #    write_log_api_file(module_dir,module,logging_text)  
                    #else:
                    #    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
                    knn_result = {
                    'id': i,
                    'knn_result': res_string,
                    'knn_qval': max_qval,
                    'knn_info': info_string,
                    'knn_qvals': qval_string,
                    }
                    if num_results_i == 0:
                        knn_result = {
                        'id': i,
                        'knn_result': 'REJ',
                        'knn_qval': 0,
                        'knn_info': info_string,
                        'knn_qvals': str(0),
                        }
                    knn_results.append(knn_result)
                else: # only one result was requested
                    if softmax_res[i][np.argmax(softmax_res[i])] > reject_thres:
                        knn_result = {
                        'id': i,
                        'knn_result': str(f'{train_classes_array[np.argmax(softmax_res[i])]:1.6f}'),
                        'knn_qval': int(100*softmax_res[i][np.argmax(softmax_res[i])]),
                        'knn_info': info_string,
                        'knn_qvals': str(int(100*softmax_res[i][np.argmax(softmax_res[i])])),
                        }
                    else:
                        knn_result = {
                        'id': i,
                        'knn_result': 'REJ',
                        'knn_qval': 0,
                        'knn_info': info_string,
                        'knn_qvals': str(0),
                        }
                    knn_results.append(knn_result)

        t1 = time.time() - t0
        logging_text = 'Time elapsed for kNN softmax result preparation: '+' {:.4f}'.format(t1)+' s for '+str(num_test_samples)+' test samples'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

        logging_text = 'ADDLOG kNN softmax results: '+str(knn_results)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    else:

        # this does not yet support multiple resuls for each inference as in the softmax case !!!!
        # must be added later when needed

        #check rejects
        rejects,qvals = knn_check_reject_c(module_dir,db_type,session,StatusData,status_datas_schema,x_test,neigh,train_labels,ai_type_knn,kNN_reject,reject_thres,reject_labelval_diff)


        t0 = time.time()
        for i in range (0,num_test_samples):
            if (rejects[i] == 0):
                if ai_type_knn == 'kNN_Class':
                    knn_result = {
                    'id': i,
                    'knn_result': pred_test[i],
                    'knn_qval': int(qvals[i]),
                    'knn_info': info_string,
                    'knn_qvals': str(int(qvals[i])),
                    }
                if ai_type_knn == 'kNN_Reg':
                    knn_result = {
                    'id': i,
                    'knn_result':"{:.6f}".format(pred_test[i]),
                    'knn_qval': int(qvals[i]),
                    'knn_info': info_string,
                    'knn_qvals': str(int(qvals[i])),
                    }
            else:
                knn_result = {
                'id': i,
                'knn_result':'REJ',
                'knn_qval': int(qvals[i]),
                'knn_info': info_string,
                'knn_qvals': str(int(qvals[i])),
                }
            knn_results.append(knn_result)
        t1 = time.time() - t0
        logging_text = 'Time elapsed for kNN check reject result preparation: '+' {:.4f}'.format(t1)+' s for '+str(num_test_samples)+' test samples'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

        logging_text = 'ADDLOG kNN results: '+str(knn_results)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    return knn_results

# getting one result for multiple frame results for one class (if the input data below ll to one class)
cpdef knn_get_one_class_result_c(module_dir,db_type,session,StatusData,status_datas_schema,
                                 ai_type_knn,KnnInitRef,x_test,
                                 KnnInit,knn_inits_schema,num_single_results,reject_one_class_thres,
                                 kNN,kNN_reject,reject_thres,max_num_results,
                                 reject_labelval_diff,pow_norm,feature_norm,softmax,info_string):

    module = 'knn_get_one_class_result_c '

    cdef int i,num_frames,frames_dim,rej_count

    num_frames = len(x_test)
    frames_dim = len(x_test[0])

    logging_text = ' Number of frames: '+ str(num_frames)
    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    logging_text = ' frames_dim: '+ str(frames_dim)
    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    knn_results = knn_get_result_c(module_dir,db_type,session,StatusData,status_datas_schema,
                       ai_type_knn,KnnInitRef,x_test,KnnInit,knn_inits_schema,kNN,kNN_reject,reject_thres,
                       max_num_results,reject_labelval_diff,pow_norm,feature_norm,softmax,info_string)

    num_knn_results = len(knn_results)
    logging_text = ' number of knn_results: ' + str(num_knn_results)
    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    #logging_text = ' knn_results: ' + str(knn_results)
    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    if (num_single_results > num_knn_results):
        num_single_results = num_knn_results

    cdef list results_new = []
    if (num_single_results <= 0):  # parameters for one class result processing not loaded sucessfully 
        result_new = {
            'id': 0,
            'knn_result':'REJ',
            'knn_qval': 0,
            'knn_info':"REJECT - not enough single results for one class result processing ",
            'knn_qvals': str(0),
            }
        results_new.append(result_new)
        return results_new

    # calculate number of knn results within single result blocks
    num_knn_result_blocks = int(num_knn_results/num_single_results)
    logging_text = ' number of num_knn_result_blocks: ' + str(num_knn_result_blocks)
    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    if (num_knn_result_blocks <= 0): 
        result_new = {
            'id': 0,
            'knn_result':'REJ',
            'knn_qval': 0,
            'knn_info':"REJECT - not enough single results for one class result processing ",
            'knn_qvals': str(0),
            }
        results_new.append(result_new)
        return results_new

    cdef list results_list = []
    cdef list qvals_list = []

    # loop over single result blocks - the last results which are not within a full block anymore will be ignored
    for j in range(0,num_knn_result_blocks):   

        rej_count = 0
        res_count = 0
        results_list.clear()
        qvals_list.clear()

        for i in range(j*num_single_results,(j+1)*num_single_results): # loop within single result blocks
            if knn_results[i]['knn_result'] == 'REJ':
                rej_count = rej_count+1
                #qvals_list.append(float(knn_results[i]['knn_qval']))
            else:
                #logging_text = ' Result: '+ str(knn_results[i]['knn_result'])
                #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                #logging_text = ' QVal: '+ str(knn_results[i]['knn_qval'])
                #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                #print('knn_results[i]:',knn_results[i])
                if ai_type_knn == 'kNN_Class':
                    knn_i_result_list = knn_results[i]['knn_result'].split(';')
                    for li in range (0,len(knn_i_result_list)):
                        results_list.append(knn_i_result_list[li])
                if ai_type_knn == 'kNN_Reg':
                    knn_i_result_list = knn_results[i]['knn_result'].split(';')
                    #print('knn_i_result_list:',knn_i_result_list)
                    #print('len(knn_i_result_list):',len(knn_i_result_list))
                    for li in range (0,len(knn_i_result_list)):
                        results_list.append(float(knn_i_result_list[li]))
                knn_i_qval_list = knn_results[i]['knn_qvals'].split(';')
                #print('knn_i_qval_list:',knn_i_qval_list)
                #print('len(knn_i_qval_list):',len(knn_i_qval_list))
                for li in range (0,len(knn_i_qval_list)):
                    qvals_list.append(float(knn_i_qval_list[li]))
                    res_count = res_count+1
 
        # additional logging
        logging_text = 'ADDLOG results list: ' + str(results_list)
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        logging_text = 'ADDLOG  qvals list: ' + str(qvals_list)
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

        if ai_type_knn == 'kNN_Reg':
            
            # display information on result list
            logging_text = ' reject count for single results: ' + str(rej_count)
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            logging_text = ' reject threshold: ' + str(reject_one_class_thres)
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            logging_text = ' number of single results: ' + str(num_single_results)
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            reject_thres_count = (1.0-reject_one_class_thres)*num_single_results
            logging_text = ' reject_thres_count: '+str(reject_thres_count)
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            result_ok = True 
            if len(results_list) > 0:
                # qval weighted average
                result_ave = 0
                for i in range (0,len(results_list)):
                    result_ave = result_ave + results_list[i]*qvals_list[i]
                if sum(qvals_list) > 0:
                    result_ave = result_ave/sum(qvals_list)
                else:
                    result_ok = False
                logging_text = ' result_ave: '+str(result_ave)
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            else:
                result_ave = 0
            if len(qvals_list) > 0:
                qval_ave = sum(qvals_list)/len(qvals_list)
                logging_text = ' qval_ave: '+str(qval_ave)
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            else:
                qval_ave = 0

            if (rej_count < reject_thres_count)and(result_ok):
                result_new = {
                'id': j,
                'knn_result': "{:.6f}".format(result_ave),
                'knn_qval': int(qval_ave),
                'knn_info':"OK - One class from multiple frames",
                'knn_qvals': str(int(qval_ave)),
                }
            else:
                result_new = {
                'id': j,
                'knn_result':'REJ',
                'knn_qval': int(qval_ave),
                'knn_info':"REJECT - One class from multiple frames",
                'knn_qvals': str(int(qval_ave)),
                }
            results_new.append(result_new)

        if ai_type_knn == 'kNN_Class':
            # determine the different result classes and the corresponding counts
            unique,counts = np.unique(results_list,return_counts=True)
            logging_text = ' unique results and counts: ' + str(np.asarray((unique,counts)).T)
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            logging_text = ' number of unique results and counts: '+str(len(unique))
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            # display information on result list
            logging_text = ' reject count for single results: ' + str(rej_count)
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            #logging_text = ' result list for single results: '+str(results_list)
            #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            #logging_text = ' qval list for single results: '+str(qvals_list)
            #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            if len(qvals_list) > 0:
                qval_ave = sum(qvals_list)/len(qvals_list)
                logging_text = ' qval_ave: '+str(qval_ave)
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            else:
                qval_ave = 0
            count_decision_thres = reject_one_class_thres*num_single_results
            logging_text = ' count_decision_thres: '+str(count_decision_thres)
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

            count_ind = 0 
            for i in range (0,len(unique)):
                if ( int(counts[i]) > count_decision_thres): 
                    count_ind = count_ind + 1
                    result_new = {
                        'id': j,
                        'knn_result': str(unique[i]),
                        'knn_qval': int(qval_ave),
                        'knn_info':"OK - One class from multiple frames",
                        'knn_qvals': str(int(qval_ave)),
                        }
            if (count_ind == 0):
                result_new = {
                    'id': j,
                    'knn_result':'REJ',
                    'knn_qval': int(qval_ave),
                    'knn_info':"REJECT - One class from multiple frames",
                    'knn_qvals': str(int(qval_ave)),
                    }
            results_new.append(result_new)

    return results_new

# getting one result for multiple frame results for one class (if the input data below ll to one class)
cpdef dl_get_one_class_result_c(x_test_multiple,module_dir,db_type,session,StatusData,status_datas_schema,
                              DlInitInfo,dl_init_infos_schema,ip_address_tf,port_tf,ai_type_dl,DlInitRef,UserId,
                              max_num_results,reject_thres,oc_num_single_results,oc_reject_thres):

    module = 'dl_get_one_class_result_c '

    cdef int i,num_frames,frames_dim,rej_count

    num_frames = len(x_test_multiple)
    frames_dim = len(x_test_multiple[0])

    logging_text = ' Number of frames: '+ str(num_frames)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    logging_text = ' frames_dim: '+ str(frames_dim)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    logging_text = ' oc_num_single_results: '+ str(oc_num_single_results)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    logging_text = ' oc_reject_thres: '+ str(oc_reject_thres)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    logging_text = ' max_num_results: '+ str(max_num_results)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    logging_text = ' reject_thres: '+ str(reject_thres)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    dl_results = TensorServingGRPC_Client(x_test_multiple,module_dir,db_type,session,
                                          StatusData,status_datas_schema,
                                          DlInitInfo,dl_init_infos_schema,ip_address_tf,port_tf,
                                          ai_type_dl,DlInitRef,UserId,reject_thres,max_num_results)

    num_dl_results = len(dl_results)
    logging_text = ' number of dl_results: ' + str(num_dl_results)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    logging_text = 'ADDLOG dl_results: ' + str(dl_results)
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    if (oc_num_single_results >= num_dl_results):
        oc_num_single_results =num_dl_results-1

    cdef list results_new = []
    if (oc_num_single_results <= 0):  # parameters for one class result processing not loaded sucessfully 
        result_new = {
            'id': 0,
            'dl_result':'REJ',
            'dl_qval': 0,
            'dl_info':"REJECT - not enough single results for one class result processing ",
            'dl_qvals': str(0),
            }
        results_new.append(result_new)
        return results_new

    # calculate number of results within single result blocks
    num_result_blocks = int(num_dl_results/oc_num_single_results)

    if (num_result_blocks <= 0): 
        result_new = {
            'id': 0,
            'knn_result':'REJ',
            'knn_qval': 0,
            'knn_info':"REJECT - not enough single results for one class result processing ",
            'knn_qvals': str(0),
            }
        results_new.append(result_new)
        return results_new

    cdef list results_list = []
    cdef list qvals_list = []

    # loop over single result blocks - the last results which are not within a full block anymore will be ignored
    for j in range(0,num_result_blocks):   

        rej_count = 0
        res_count = 0
        results_list.clear()
        qvals_list.clear()

        for i in range(j*oc_num_single_results,(j+1)*oc_num_single_results): # loop within single result blocks

            if dl_results[i]['dl_result'] == 'REJ':
                rej_count = rej_count+1
                qvals_list.append(float(dl_results[i]['dl_qval']))
            else:
                #logging_text = ' Result: '+ str(dl_results[i]['dl_result'])
                #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                #logging_text = ' QVal: '+ str(dl_results[i]['dl_qval'])
                #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                if ai_type_dl == 'DL_Class':
                    results_list.append(dl_results[i]['dl_result'])
                if ai_type_dl == 'DL_Reg':
                    results_list.append(float(dl_results[i]['dl_result']))
                qvals_list.append(float(dl_results[i]['dl_qval']))

        if ai_type_dl == 'DL_Reg':
            # display information on result list
            logging_text = ' reject count for single results: ' + str(rej_count)
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            reject_thres_count = (1.0-oc_reject_thres)*oc_num_single_results
            logging_text = ' reject_thres_count: '+str(reject_thres_count)
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            result_ave = sum(results_list)/len(results_list)
            #logging_text = ' result list for single results: '+str(results_list)
            #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            #logging_text = ' qval list for single results: '+str(qvals_list)
            #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            if len(results_list) > 0:
                result_ave = sum(results_list)/len(results_list)
                logging_text = ' result_ave: '+str(result_ave)
                if db_type == "File":
                    write_log_api_file(module_dir,module,logging_text)  
                else:
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            else:
                result_ave = 0
            if len(qvals_list) > 0:
                qval_ave = sum(qvals_list)/len(qvals_list)
                logging_text = ' qval_ave: '+str(qval_ave)
                if db_type == "File":
                    write_log_api_file(module_dir,module,logging_text)  
                else:
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            else:
                qval_ave = 0

            if (rej_count < reject_thres_count):
                result_new = {
                'id': 0,
                'dl_result': "{:.6f}".format(result_ave),
                'dl_qval': int(qval_ave),
                'dl_info': "OK - One class from multiple frames",
                'dl_qvals': str(int(qval_ave)),
                }
            else:
                result_new = {
                'id': 0,
                'dl_result':'REJ',
                'dl_qval': int(qval_ave),
                'dl_info':"REJECT - One class from multiple frames",
                'dl_qvals': str(int(qval_ave)),
                }
            results_new.append(result_new)

        if ai_type_dl == 'DL_Class':
            # determine the different result classes and the corresponding counts
            unique,counts = np.unique(results_list,return_counts=True)
            logging_text = ' unique results and counts: ' + str(np.asarray((unique,counts)).T)
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            logging_text = ' number of unique results and counts: '+str(len(unique))
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            # display information on result list
            logging_text = ' reject count for single results: ' + str(rej_count)
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            #logging_text = ' result list for single results: '+str(results_list)
            #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            #logging_text = ' qval list for single results: '+str(qvals_list)
            #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            if len(qvals_list) > 0:
                qval_ave = sum(qvals_list)/len(qvals_list)
                logging_text = ' qval_ave: '+str(qval_ave)
                if db_type == "File":
                    write_log_api_file(module_dir,module,logging_text)  
                else:
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            else:
                qval_ave = 0
            count_decision_thres = oc_reject_thres*oc_num_single_results
            logging_text = ' count_decision_thres: '+str(count_decision_thres)
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

            count_ind = 0 
            for i in range (0,len(unique)):
                if ( int(counts[i]) > count_decision_thres): 
                    count_ind = count_ind + 1
                    result_new = {
                        'id': 0,
                        'dl_result': str(unique[i]),
                        'dl_qval': int(qval_ave),
                        'dl_info': "OK - One class from multiple frames",
                        'dl_qvals': str(int(qval_ave)),
                        }
            if (count_ind == 0):
                result_new = {
                    'id': 0,
                    'dl_result':'REJ',
                    'dl_qval': int(qval_ave),
                    'dl_info':"REJECT - One class from multiple frames",
                    'dl_qvals': str(int(qval_ave)),
                    }
            results_new.append(result_new)

    return results_new

# ------------------------------------------------------------------------------------------------------------------------------
# original python basic functions (without change)
# ------------------------------------------------------------------------------------------------------------------------------

# check if running within docker
def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )

# directory structure of AI Basic Library and AI services (which are named by "module")
#
# Basic Library
#
# ai_root_dir/basic/ai_basic/ 
# 
# any AI service with variable module name
#
# ai_root_dir/'+module+'/ai_'+module+'_service/api
#
# Notes: 
#  - ai_root_dir can be defined individually for a partucular user / host 
#  - ai_root_dir/basic/ is the director to which the AI Basic Library must be cloned from Bitbucket
#  - ai_root_dir/'+module+'/ is the director to which the AI services (which are named by "module") 
#    must be cloned from Bitbucket
#

def set_module_dir(module):    # examples for modules: aoa, savd, ...
    if is_docker():
        module_dir = '/data/ai_'+module+'_service/api'  
        print('Running in docker with module directory:',module_dir)
    else:
        # get current directory of AI service
        module_dir = os.getcwd()
        print('Not running in docker with module directory:',module_dir)
    return module_dir

def get_module_from_dir(module_dir):    # examples for modules: aoa, savd, ...
    ai_start = module_dir.find('ai_') + 3
    service_start = module_dir.find('_service')
    ai_module = module_dir[ai_start:service_start]
    return ai_module

def get_tfserving_models(module_dir,db_type,session,StatusData,status_datas_schema,module_desc):    # examples for module_desc: SC_ (Signal Classification),  ...
    module = 'get_tfserving_models'
    tfs_models_list = []
    tfs_models_list_red = []
    if is_docker():
        tfs_models_dir = '/models/'  
        print('Running in docker')
    else:
        # get current directory of AI service
        current_dir = os.getcwd()
        ai_module_api = '/'+module+'/ai_'+module+'_service/api'
        tfs_models_dir = current_dir.replace(ai_module_api,"")+'/tfserving/tfservingmodels/'
        print('Tensorflow serving models directory:',tfs_models_dir)
        print('Not running in docker')

    # check if the tfserving directory exists
    if os.path.exists(tfs_models_dir) == True:
        logging_text = ' Tensorflow Serving models directory '+tfs_models_dir+' does exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        for it in os.scandir(tfs_models_dir):
            if it.is_dir():
                model_name = it.path.replace(tfs_models_dir,"")
                if module_desc in model_name:
                    tfs_models_list.append(model_name)
                    logging_text = ' Tensorflow Serving Models '+model_name+' added to the list'
                    if db_type == "File":
                        write_log_api_file(module_dir,module,logging_text)  
                    else:
                        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    else:
        logging_text = ' Tensorflow Serving models directory '+tfs_models_dir+' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    # load models.config file from for tfs_models_dir
    tfs_model_config = tfs_models_dir+'/models.config'
    if os.path.exists(tfs_model_config):
        # read tfs_model_config file
        lock_file_name = tfs_model_config+'.lock'
        file_name_lock = FileLock(lock_file_name)
        with file_name_lock:
            with open(tfs_model_config, 'r') as myfile:
                data=myfile.read()
        try:
            os.remove(lock_file_name)
        except:
            pass
        logging_text = ' Tensorflow Serving model config:'+str(data)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        for i in range (0,len(tfs_models_list)):
            if tfs_models_list[i] in data:
                tfs_models_list_red.append(tfs_models_list[i])
                logging_text = ' Tensorflow Serving Models '+tfs_models_list[i]+' added to the list'
                if db_type == "File":
                    write_log_api_file(module_dir,module,logging_text)  
                else:
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    return tfs_models_list_red

def get_tfserving_model_versions(module_dir,db_type,session,StatusData,status_datas_schema,model_name):    
    module = 'get_tfserving_model_versions'
    version_list = []
    if is_docker():
        tfs_models_dir = '/models/'  
        print('Running in docker')
    else:
        # get current directory of AI service
        current_dir = os.getcwd()
        ai_module_api = '/'+module+'/ai_'+module+'_service/api'
        tfs_models_dir = current_dir.replace(ai_module_api,"")+'/tfserving/tfservingmodels/'
        print('Tensorflow serving models directory:',tfs_models_dir)
        print('Not running in docker')

    # set tfserving model name directory
    tfs_models_name_dir = tfs_models_dir+model_name+'/'
    # check if the tfserving model name directory exists
    if os.path.exists(tfs_models_name_dir) == True:
        logging_text = ' Tensorflow Serving model name directory '+tfs_models_name_dir+' does exist'
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        for it in os.scandir(tfs_models_name_dir):
            if it.is_dir():
                version = it.path.replace(tfs_models_name_dir,"")
                version_list.append(version)
                logging_text = ' Tensorflow Serving version '+version+' added to the list'
                if db_type == "File":
                    write_log_api_file(module_dir,module,logging_text)  
                else:
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    else:
        logging_text = ' Tensorflow Serving model name directory '+tfs_models_dir+' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    return version_list

# directory structure of training and test data for AI services (which are named by "module")
#
# any AI service with variable module name
#
# ai_root_dir/'+module+'/train_data/'+aitype+''   
# ai_root_dir/'+module+'/test_data/'+aitype+'' 
# 
# aitype = 'dl' for deep lerning training data 
# aitype = 'knn' for knn training data 
#
# Notes: 
#  - ai_root_dir can be defined individually for a particular user / host 
#  - ai_root_dir/basic/ is the director to which the AI Basic Library must be cloned from Bitbucket
#  - ai_root_dir/tfserving/tfservingmodels/ is the director of the tensorflow models
#  - ai_root_dir/'+module+'/ is the director to which the AI services (which are named by "module") 
#    must be cloned from Bitbucket
#

# set training directory depending on ai module and ai type
# values aitype: knn, dl, ...
def set_train_dir(module,aitype):
    if is_docker():
        train_base_dir = '/data/train_data'
        if aitype == 'knn':
            #train_dir = train_base_dir + '/knn/'
            train_dir = '/traindata/'  # using docker volumes for the training data to support doker swarm
        if aitype == 'dl':
            train_dir = train_base_dir + '/dl/'
    else:
        # get current directory of AI service
        current_dir = os.getcwd()
        ai_module_api = '/ai_'+module+'_service/api'
        train_base_dir = current_dir.replace(ai_module_api,"")+'/train_data' 
        if aitype == 'knn':
            train_dir = train_base_dir + '/knn/'
        if aitype == 'dl':
            train_dir = train_base_dir + '/dl/'
        #print('Training data directory:',train_dir)
        # check if training directory depending on ai module and ai type exists
        if (os.path.isdir(train_dir) == False):
            try:
                os.mkdir(train_dir, 0o777)
            except:
                print('Training data directory:',train_dir,' was not created sucessfully')
            else:
                print('Training data directory:',train_dir,' was created sucessfully')

    return train_dir

# set test directory depending on ai module and ai type
# values aitype: knn, dl, ...
def set_test_dir(module,aitype):
    if is_docker():
        test_base_dir = '/data/test_data'
        if aitype == 'knn':
            test_dir = test_base_dir + '/knn/'
        if aitype == 'dl':
            test_dir = test_base_dir + '/dl/'
    else:
        # get current directory of AI service
        current_dir = os.getcwd()
        ai_module_api = '/ai_'+module+'_service/api'
        test_base_dir = current_dir.replace(ai_module_api,"")+'/test_data' 
        if aitype == 'knn':
            test_dir = test_base_dir + '/knn/'
        if aitype == 'dl':
            test_dir = test_base_dir + '/dl/'
        print('Test data directory:',test_dir)
        # check if test directory depending on ai module and ai type exists
        if (os.path.isdir(test_dir) == False):
            try:
                os.mkdir(test_dir, 0o777)
            except:
                print('Test data directory:',test_dir,' was not created sucessfully')
            else:
                print('Test data directory:',test_dir,' was created sucessfully')

    return test_dir

def init_global_file_data(module_dir):

    # create or set data path 

    #data_file_path = module_dir+"/data/"
    if is_docker():
        data_file_path = "/filedata/"  #using docker volumes to support docker swarm
    else:
        data_file_path = module_dir+"/data/"

    if os.path.exists(data_file_path) == False:
        try:
            os.mkdir(data_file_path,0o777)
        except:
            print ("Creation of the data directory %s failed" % data_file_path)
        else:
            print ("Successfully created data directory %s " % data_file_path)

    #global data file name
    global_data_file =  os.path.join(data_file_path,'global_data.json')

    # storing all global data in dictionary
    global_data = {"description":"Service global data","logging":"OFF","addlogging":"OFF"}

    Error = False
    try: 
        lock_file_name = global_data_file+'.lock'
        lock_data_file = FileLock(lock_file_name)
        with lock_data_file:
            with open(global_data_file, 'w') as myfile:
                json.dump(global_data, myfile)
                print('Global data file '+global_data_file+' was created ')
        try:
            os.remove(lock_file_name)
        except:
            pass
    except:
        Error = True
        print('Global data file could not be created ')

    return Error

def get_global_file_data(module_dir):

    # set data path 
    #data_file_path = module_dir+"/data/"
    if is_docker():
        data_file_path = "/filedata/"  #using docker volumes to support docker swarm
    else:
        data_file_path = module_dir+"/data/"

    #read global data
    global_data_file =  os.path.join(data_file_path,'global_data.json')

    if os.path.exists(global_data_file):
        # read init file
        lock_file_name = global_data_file+'.lock'
        lock_data_file = FileLock(lock_file_name)
        with lock_data_file:   
            with open(global_data_file, 'r') as myfile:
                data=myfile.read()
        try:
            os.remove(lock_file_name)
        except:
            pass

        # parse file
        global_data = json.loads(data)

    return global_data,global_data_file

def set_global_file_data(module_dir,logging,addlogging):

    # storing all global data in dictionary
    global_data,global_data_file = get_global_file_data(module_dir)

    # set logging data
    Error = False
    if logging != '':
        try:
            global_data['logging'] = logging
        except:
            Error = True
    if addlogging != '':
        try:
            global_data['addlogging'] = addlogging
        except:
            Error = True

    if os.path.exists(global_data_file):
        # write init file
        try: 
            lock_file_name = global_data_file+'.lock'
            lock_data_file = FileLock(lock_file_name)
            with lock_data_file:
                with open(global_data_file, 'w') as myfile:            
                    json.dump(global_data, myfile)
            try:
                os.remove(lock_file_name)
            except:
                pass
        except:
            Error = True
            print('Global data file could not be created ')

    return Error

# create own file based logging
def write_log_api_file(module_dir,module,logging_text):

    # read global data
    global_data,global_data_file = get_global_file_data(module_dir)
    try:
        description = str(global_data['description'])  
    except:
        description = ''
    try:
        logging = str(global_data['logging'])  
    except:
        logging = ''  
    try:
        addlogging = str(global_data['addlogging'])  
    except:
        addlogging = ''   

    #print('description,logging:',description,logging)

    if (logging == 'ON'): # Status logging is swithed on
        log_file_path = module_dir+"/log/"
        log_file =  os.path.join(log_file_path,'logfile_file.txt')
        now = datetime.now()
        if "ADDLOG" not in logging_text: # log only text which does not include ADDLOG
            if os.path.exists(log_file_path) == False:
                try:
                    os.mkdir(log_file_path,0o777)
                except:
                    print ("Creation of the log directory %s failed" % log_file_path)
                else:
                    print ("Successfully created the log directory %s " % log_file_path)
            # check if log path exists
            if os.path.exists(log_file_path):
                original_stdout = sys.stdout # Save a reference to the original standard output
                file_lock = log_file+'.lock'
                lock_file = FileLock(file_lock)
                with lock_file:
                    try: 
                        with open(log_file, 'a') as f:
                            sys.stdout = f # Change the standard output to the file we created.
                            print(module,now.strftime("%Y-%m-%d %H:%M:%S"),logging_text)
                            sys.stdout = original_stdout # Reset the standard output to its original value
                    except:
                        pass
                try:
                    os.remove(file_lock)
                except:
                    pass

    if (addlogging == 'ON'): # Status logging is swithed on
        if "ADDLOG" in logging_text: # log only text which includes ADDLOG
            log_file_path = module_dir+"/log/"
            add_log_file =  os.path.join(log_file_path,'additional_logfile_file.txt')
            now = datetime.now()
            if os.path.exists(log_file_path) == False:
                try:
                    os.mkdir(log_file_path,0o777)
                except:
                    print ("Creation of the log directory %s failed" % log_file_path)
                else:
                    print ("Successfully created the log directory %s " % log_file_path)
            # check if log path exists
            if os.path.exists(log_file_path):
                original_stdout = sys.stdout # Save a reference to the original standard output
                file_lock = add_log_file+'.lock'
                lock_file = FileLock(file_lock)
                with lock_file:              
                    try: 
                        with open(add_log_file, 'a') as f:
                            sys.stdout = f # Change the standard output to the file we created.
                            print(module,now.strftime("%Y-%m-%d %H:%M:%S"),logging_text)
                            sys.stdout = original_stdout # Reset the standard output to its original value
                    except:
                        pass
                try:
                    os.remove(file_lock)
                except:
                    pass

# create own file based logging
def write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text):

    from basic_db_functions_cython import get_init_status

    # get current status data
    UserId = 'AllUsers'
    status_data = get_init_status(session,StatusData,status_datas_schema,UserId)

    if len(status_data) > 0: # Status logging is set

        if (status_data['log_status'] == 'ON'): # Status logging is swithed on

            log_file_path = module_dir+"/log/"
            log_file =  os.path.join(log_file_path,'logfile_db.txt')
            now = datetime.now()

            if os.path.exists(log_file_path) == False:
                try:
                    os.mkdir(log_file_path,0o777)
                except:
                    print ("Creation of the log directory %s failed" % log_file_path)
                else:
                    print ("Successfully created the log directory %s " % log_file_path)

            # check if log path exists
            if "ADDLOG" not in logging_text: # log only text which does not include ADDLOG
                if os.path.exists(log_file_path):
                    original_stdout = sys.stdout # Save a reference to the original standard output
                    file_lock = log_file+'.lock'
                    lock_file = FileLock(file_lock)
                    with lock_file:              
                        with open(log_file, 'a') as f:
                            sys.stdout = f # Change the standard output to the file we created.
                            print(module,now.strftime("%Y-%m-%d %H:%M:%S"),logging_text)
                            sys.stdout = original_stdout # Reset the standard output to its original value
                    try:
                        os.remove(file_lock)
                    except:
                        pass

            addlogging = 'OFF'
            if 'additional log status: ON' in status_data['log_status_info']:
                addlogging = 'ON'
            if (addlogging == 'ON'): # Status logging is swithed on
                if "ADDLOG" in logging_text: # log only text which includes ADDLOG
                    log_file_path = module_dir+"/log/"
                    add_log_file =  os.path.join(log_file_path,'additional_logfile_db.txt')
                    now = datetime.now()
                    if os.path.exists(log_file_path) == False:
                        try:
                            os.mkdir(log_file_path,0o777)
                        except:
                            print ("Creation of the log directory %s failed" % log_file_path)
                        else:
                            print ("Successfully created the log directory %s " % log_file_path)
                    # check if log path exists
                    if os.path.exists(log_file_path):
                        original_stdout = sys.stdout # Save a reference to the original standard output
                        file_lock = add_log_file+'.lock'
                        lock_file = FileLock(file_lock)
                        with lock_file:                                   
                            with open(add_log_file, 'a') as f:
                                sys.stdout = f # Change the standard output to the file we created.
                                print(module,now.strftime("%Y-%m-%d %H:%M:%S"),logging_text)
                                sys.stdout = original_stdout # Reset the standard output to its original value
                        try:
                            os.remove(file_lock)
                        except:
                            pass

def prepareFeatureStream(module_dir,db_type,session,StatusData,status_datas_schema,Features,NumFeatures,frametype):  

    module = 'prepareFeatureStream '
    listFeatures = []                                                                 # define a list to store I & Q values in a single frame
    listFrames = []                                                             # define a list for stroing IQ frames
    logging_text = ' Length of features data buffer: '+str(len(Features))
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    if (frametype =='1D'):       
        logging_text = ' Frame size of 1D dimensional frames: '+str(NumFeatures)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        if NumFeatures > 0:                                          
            NumRemainElements = len(Features)%(NumFeatures)      
            if NumRemainElements!=0:                                                     # if there are some extra elements in received buffer
                Features = Features[:-NumRemainElements]                                 # delete the extra elements
            NumFrames = int(len(Features)/(NumFeatures))
        else:
            NumRemainElements = 0
            NumFrames = 0
        logging_text = ' No. of 1D frames: '+str(int(NumFrames))    
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        logging_text = ' No. of remaining 1D features after the last frame: '+str(int(NumRemainElements))    
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        counter = 0                                                                 # define and initialize a counter
        for i in range (0, len(Features)):                                          # run a loop over received IQ-values (increment by 1)
            counter = counter+1                                                    # increment counter
            listFeatures.append(Features[i])                                            # store features
            if counter%NumFeatures == 0:                                            # if frame length reaches      
                frame = np.vstack(listFeatures)                                         # stack the list elements vertically, its a complete frame now (2D values)              
                listFrames.append(frame)                                          # append the frame into a list of frames (3D values)
                listFeatures.clear() 
        arrayFrames1D = np.asarray(listFrames, dtype=np.float32).flatten()
        arrayFrames = arrayFrames1D.reshape(len(listFrames),NumFeatures)
    if (frametype =='2D'):       
        logging_text = ' Frame size of 2D dimensional frames: '+str(NumFeatures)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        if NumFeatures > 0:                                          
            NumRemainElements = len(Features)%(NumFeatures*2)      
            if NumRemainElements!=0:                                                     # if there are some extra elements in received buffer
                Features = Features[:-NumRemainElements]                                 # delete the extra elements
            NumFrames = int(len(Features)/(NumFeatures*2))
        else:
            NumRemainElements = 0
            NumFrames = 0
        logging_text = ' No. of 2D frames: '+str(int(NumFrames))    
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        logging_text = ' No. of remaining 2D features after the last frame: '+str(int(NumRemainElements))    
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        counter = 0                                                                 # define and initialize a counter
        for i in range (0, len(Features), 2):                                       # run a loop over received IQ-values (increment by 2)
            counter = counter+1                                                    # increment counter
            listFeatures.append((Features[i], Features[i+1]))                             # store IQ-value pair
            if counter%NumFeatures == 0:                                            # if frame length reaches      
                frame = np.vstack(listFeatures)                                         # stack the list elements vertically, its a complete frame now (2D values)              
                listFrames.append(frame)                                          # append the frame into a list of frames (3D values)
                listFeatures.clear()                                                      # clear the list for stroing I& Q values in a single frame
        arrayFrames = np.asarray(listFrames, dtype=np.float32)                      # Convert all frames from list to array and change data type from float64 to float32
    
    #logging_text = ' Frame[0]: '+str(arrayFrames[0])    
    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    return arrayFrames                                                          # return a 3D- numpy array of frames

def SaveNpzFile_FloatFeaturesLabels(module_dir,db_type,session,StatusData,status_datas_schema,DirPath,FileName,features,labels):                       

    module = 'SaveNpzFile_FloatFeaturesLabels '
    npzFileOK = False
    npz_file = DirPath+FileName
    npz_file_name = npz_file+'.npz'
    #print('*** File ',npz_file+'.npz',' exists: ',os.path.isfile(npz_file+'.npz'))

    # check if the npz file for this data reference already exists
    if os.path.isfile(npz_file_name):
        #print ("File exist")
        logging_text = ' npz filename '+npz_file_name+' in directory '+DirPath+' already exists'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            npzFileOK=False
    else:
        #print ("File not exist") - new directory will be created
        if os.path.exists(DirPath) == False:
            try:
                os.mkdir(DirPath,0o777)
            except:
                logging_text = ' Creation of the directory '+DirPath+' failed'
                if db_type == "File":
                    write_log_api_file(module_dir,module,logging_text)  
                else:
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            else:
                logging_text = ' Creation of the directory '+DirPath+' was sucessfull'
                if db_type == "File":
                    write_log_api_file(module_dir,module,logging_text)  
                else:
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                npzFileOK=True

    features = features.astype(np.float32)
    if (labels.dtype == 'float64'):
        labels = labels.astype(np.float32)
    np.savez_compressed(npz_file, f=features, l=labels)                     
    
    return npzFileOK,npz_file     

def SaveNpzFile_IQDataLabels(module_dir,db_type,session,StatusData,status_datas_schema,DirPath,FileName,IQ_values,DimIQDataFrames,
                             labels,Frequency_Hz,Bw_Hz,SampleRate_Hz,SampleRateWidth_Hz,SNR_dB):                       

    module = 'SaveNpzFile_IQDataLabels '
    npzFileOK = False
    npz_file = DirPath+FileName

    # check if the npz file for this data reference already exists
    if os.path.isfile(npz_file+'.npz'):
        #print ("File exist")
        # create additional unique data reference from time
        #time.sleep(0.1) # Sleep for 0.1 seconds - to be sure that a new data ref is created
        #datetime_int = int(time.time()*1000)
        #data_ref_new = f"{datetime_int}"
        data_ref_new = str(uuid.uuid4())
        NewFileName = FileName +'_'+data_ref_new
        npz_file = DirPath+NewFileName
        logging_text = ' New npz filename '+NewFileName+' in directory '+DirPath+' created'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        npzFileOK=True
    else:
        logging_text = ' Outputfile does not exist and will be created'
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        if os.path.exists(DirPath) == False:
            try:
                os.mkdir(DirPath,0o777)
            except:
                logging_text = ' Creation of the directory '+DirPath+' failed'
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            else:
                logging_text = ' Creation of the directory '+DirPath+' was sucessfull'
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                npzFileOK=True

    # prepare I/Q Data Farems as numpy arrays
    IQDataFrames = prepareFeatureStream_2D_c(module_dir,db_type,session,StatusData,status_datas_schema,IQ_values,DimIQDataFrames)
    #frametype = '2D'
    #IQDataFrames = prepareFeatureStream(module_dir,db_type,session,StatusData,status_datas_schema,IQ_values,DimIQDataFrames,frametype)

    IQDataFrames = IQDataFrames.astype(np.float32)
    if (labels.dtype == 'float64'):
        labels = labels.astype(np.float32)

    # prepare meta data
    dim_metadata = 6
    meta_data = np.zeros(dim_metadata)
    meta_data = meta_data.astype(np.float32)
    meta_data[0] = Frequency_Hz                     # frequency in Hz
    meta_data[1] = SampleRateWidth_Hz               # internal bandwidth (for IQ data resampling) 
    meta_data[2] = 0                                # external bandwidth - unknown
    meta_data[3] = Bw_Hz                            # bandwidth - based on frequency plan
    meta_data[4] = SampleRate_Hz                    # sample rate
    meta_data[5] = SNR_dB                           # SNR

    np.savez_compressed(npz_file, f=IQDataFrames, l=labels, m=meta_data)                     
    
    return npzFileOK,npz_file                                            

def list_sort(my_list_1, my_list_2):
    zipped_list_pairs = zip(my_list_2, my_list_1)
    my_result = [x for _, x in sorted(zipped_list_pairs,reverse=True)]
    return my_result

def TensorServingGRPC_Client(x_test,module_dir,db_type,session,StatusData,status_datas_schema,
                             DlInitInfo,dl_init_infos_schema,ip_address_tf,port_tf,
                             ai_type_dl,DlInitRef,UserId,reject_thres,max_num_results):

    from basic_db_functions_cython import get_init_status
    #from basic_db_functions_cython import get_dl_init_data
    from basic_db_functions_cython import get_dl_init_info
    from predict_client.prod_client import ProdClient

    module = 'TensorServingGRPC_Client '

    # get current init StatusData data
    if db_type != "File":
        status_data = get_init_status(session, StatusData, status_datas_schema,UserId)    
        if len(status_data) > 0:
            status_current = status_data['status']
            status_info_current = status_data['status_info']
            knn_init_status_current = status_data['knn_init_status']
            knn_init_status_info_current = status_data['knn_init_status_info']
            dl_init_status_current = status_data['dl_init_status']
            dl_init_status_info_current = status_data['dl_init_status_info']
        else:
            dl_results = []
            dl_result = {
            'id': 0,
            'dl_result':'NONE',
            'dl_qval': 0,
            'dl_info':'DL init status not available',
            'dl_qvals': str(0),
            }
            dl_results.append(dl_result)
            return dl_results
        #print('dl_init_status_current:',dl_init_status_current)
        logging_text = ' dl_init_status_current: ' + dl_init_status_current
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    else:
        dl_init_status_current = 'OK'
        logging_text = ' dl_init_status_current: OK'
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    if dl_init_status_current == 'OK':

        #read  DL init info data ( for this dl_init_ref)
        # get classes meta data
        if db_type == "File":
            GetDlInitDataInfoOK,dl_init_data_filt = get_dl_init_info_file(module_dir,DlInitRef)
        else:
            GetDlInitDataInfoOK,dl_init_data_filt = get_dl_init_info(session,DlInitInfo,dl_init_infos_schema,DlInitRef)

        logging_text = 'ADDLOG dl_init_data_filt: '+str(dl_init_data_filt)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

        ts_model_classes = ''
        if GetDlInitDataInfoOK:
            ts_model_name = dl_init_data_filt[0]['dlinit_tsmodelname']
            ts_model_version = dl_init_data_filt[0]['dlinit_tsmodelversion']
            ts_model_classes = dl_init_data_filt[0]['dlinit_tsmodelclasses']
            logging_text = ' tensorflow serving model name, version and classes: '+ts_model_name+' '+ts_model_version+' '+ts_model_classes
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            if ts_model_classes == '':
                logging_text = ' classes meta data was not loaded sucessfully'
                if db_type == "File":
                    write_log_api_file(module_dir,module,logging_text)  
                else:
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
                dl_results = []
                dl_result = {
                'id': 0,
                'dl_result': 'NONE',
                'dl_qval': 0,
                'dl_info':' no predictions available ',
                'dl_qvals': str(0),
                }
                dl_results.append(dl_result)
                return dl_results
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            # generate Classes as list from ts_model_classes string
            Classes = ts_model_classes.split(";")
            logging_text = ' tensorflow serving model classes list: '+str(Classes)
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        else:            
            logging_text = ' tensorflow serving model name and version are not avaliable'
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            dl_results = []
            dl_result = {
            'id': 0,
            'dl_result': 'NONE',
            'dl_qval': 0,
            'dl_info':' no predictions available ',
            'dl_qvals': str(0),
            }
            dl_results.append(dl_result)
            return dl_results

        # check max_num_results
        if max_num_results < 1:
            max_num_results = 1
        if max_num_results > 3:
            max_num_results = 3

        # check reject_thres
        if reject_thres <= 0:
            logging_text = 'reject_thres is '+str(reject_thres)+' but must be > 0'
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            dl_results = []
            dl_result = {
                'id': 0,
                'dl_result': 'REJ',
                'dl_qval': -1000,
                'dl_info': logging_text,
                'dl_qvals': str(-1000),
                }
            dl_results.append(dl_result)
            return dl_results

        if reject_thres >= 1:
            logging_text = 'reject_thres is '+str(reject_thres)+' but must be > 1'
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            dl_results = []
            dl_result = {
                'id': 0,
                'dl_result': 'REJ',
                'dl_qval': -1000,
                'dl_info': logging_text,
                'dl_qvals': str(-1000),
                }
            dl_results.append(dl_result)
            return dl_results

        #host = 'localhost:8500'
        #host = '0.0.0.0:8500'
        host = ip_address_tf+':'+port_tf
        model_name = ts_model_name
        model_version = int(ts_model_version)
        #print('host:',host)
        #print('model_name:',model_name)
        #print('model_version:',model_version)
        client = ProdClient(host, model_name, model_version)
        t0 = time.time()
        #req_data = [{'data': np.asarray(x_test),'in_tensor_dtype': 'DT_FLOAT','in_tensor_name': 'input_1'}]
        req_data = [{'data': np.asarray(x_test),'in_tensor_dtype': 'DT_FLOAT','in_tensor_name': 'input_layer'}]
        # additional logging of input
        logging_text = 'ADDLOG gRPC tensor serving input request '+str(req_data)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        try:
            prediction = client.predict(req_data, request_timeout=100)
        except RuntimeError:
            pred_OK = False
            logging_text = ' exception on predictions '
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            logging_text = ' exception on predictions '
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            dl_results = []
            dl_result = {
            'id': 0,
            'dl_result': 'NONE',
            'dl_qval': 0,
            'dl_info':' exception on predictions ',
            'dl_qvals': str(0),
            }
            dl_results.append(dl_result)
            return dl_results
        else:
            pred_OK = True
            logging_text = ' no exception on predictions '
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        t1 = time.time() - t0
        logging_text = ' time for gRPC tensor serving predictions on '+str(len(x_test))+' test samples '+'{:.4f}'.format(t1) + ' s'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        #print(prediction)
        logging_text = 'ADDLOG gRPC tensor serving prediction result: '+str(prediction)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        if len(prediction) > 0:  # any predictions ?
            #pred = prediction["dense3_softmax"]
            pred = prediction["output_layer"]
            # additional logging of output
            logging_text = 'ADDLOG gRPC tensor serving pred result: '+str(pred)
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            pred_OK = True
        else:
            pred_OK = False
            logging_text = ' no predictions available '
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            dl_results = []
            dl_result = {
            'id': 0,
            'dl_result': 'NONE',
            'dl_qval': 0,
            'dl_info':' no predictions available ',
            'dl_qvals': str(0),
            }
            dl_results.append(dl_result)
            return dl_results
        #print(pred)
        if pred_OK == True:
            t0 = time.time()
            dl_results = []
            #print("Length of pred: ", len(pred))
            logging_text = ' number of predictions from TF serving: '+str(len(pred))
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

            for i in range (0,len(pred)):  
                if ai_type_dl == 'DL_Class':
                    #logging_text = 'ai_type_dl: '+str(ai_type_dl)
                    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                    #logging_text = ' pred[i] from TF serving: '+str(pred[i])
                    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                    if max_num_results > 1:  # more than one result was requested
                        cond_res = pred[i] > reject_thres
                        num_results_i = cond_res.sum()
                        res_string = ''
                        qval_string = ''
                        info_string = 'Success: '+str(num_results_i)+' multiple results avaliable'
                        max_qval = 0
                        res_string_list = []
                        res_qval_list = []
                        for j in range(0,len(pred[i])):
                            if cond_res[j] == True:
                                pred_min = np.abs(pred[i] - pred[i][j])   # identify index with prediction pred[i][j]
                                res_string_list.append(Classes[np.argmin(pred_min)])
                                res_qval_list.append(f'{100*pred[i][j]:1.0f}')
                                if int(100*pred[i][j]) > max_qval:
                                    max_qval = int(100*pred[i][j])
                        # sort res_string_list based on res_qval_list
                        res_string_list_s = list_sort(res_string_list, res_qval_list)
                        res_qval_list = sorted(res_qval_list,reverse=True)
                        # delete last num_del from sorted results if more than requested exist
                        if (len(res_string_list_s)) > max_num_results:
                            num_del = len(res_string_list_s) - max_num_results
                            del res_string_list_s[-num_del:]
                            del res_qval_list[-num_del:]
                        # create semicolon separated list
                        res_string =";".join(map(str,res_string_list_s))
                        qval_string =";".join(map(str,res_qval_list))
                        # remove last ; if necessery
                        if res_string.endswith(';'):
                            res_string = res_string[:-1]
                        if qval_string.endswith(';'):
                            qval_string = qval_string[:-1]
                        #logging_text = 'result string multiple decisions: '+res_string
                        #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                        #logging_text = 'qval string multiple decisions: '+qval_string
                        #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                        dl_result = {
                        'id': i,
                        'dl_result': res_string,
                        'dl_qval': max_qval,
                        'dl_info': info_string,
                        'dl_qvals': qval_string,
                        }
                        if num_results_i == 0:
                            dl_result = {
                            'id': i,
                            'dl_result': 'REJ',
                            'dl_qval': 0,
                            'dl_info':'Reject',
                            'dl_qvals': str(0),
                            }
                    else: # only one result was requested
                        if pred[i][np.argmax(pred[i])] > reject_thres:
                            dl_result = {
                            'id': i,
                            'dl_result': Classes[np.argmax(pred[i])],
                            'dl_qval': int(100*pred[i][np.argmax(pred[i])]),
                            'dl_info':'Success: Single result avaliable',
                            'dl_qvals': str(int(100*pred[i][np.argmax(pred[i])])),
                            }
                        else:
                            dl_result = {
                            'id': i,
                            'dl_result': 'REJ',
                            'dl_qval': 0,
                            'dl_info':'Reject',
                            'dl_qvals': str(0),
                            }

                    dl_results.append(dl_result)

                if ai_type_dl == 'DL_Reg':
                    #logging_text = str(module_dir)
                    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                    # check if AI module is AOA
                    if "_aoa_" in module_dir:
                        #logging_text = ' TF serving called from AOA service '
                        #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                        #convert unit vector to angle according to HS code
                        pred_rad = np.arctan2(pred[i, 0],pred[i, 1])
                        pred_rad = pred_rad + 2.0 * np.pi if pred_rad < 0.0 else pred_rad 
                        pred_deg = pred_rad / np.pi * 180.0
                        dl_result = {
                        'id': i,
                        'dl_result': str(f'{pred_deg:1.6f}'),
                        'dl_qval': int(100),
                        'dl_info':'Success',
                        'dl_qvals': str(100),
                        }
                    else: 
                        #logging_text = 'ai_type_dl: '+str(ai_type_dl)
                        #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                        #logging_text = ' pred[0] from TF serving: '+str(pred[0])
                        #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                        if max_num_results > 1:  # more than one result was requested
                            cond_res = pred[i] > reject_thres
                            num_results_i = cond_res.sum()
                            res_string = ''
                            qval_string = ''
                            info_string = 'Success: Multiple results avaliable'
                            max_qval = 0
                            res_string_list = []
                            res_qval_list = []
                            for j in range(0,len(pred[i])):
                                if cond_res[j] == True:
                                    pred_min = np.abs(pred[i] - pred[i][j])   # identify index with prediction pred[i][j]
                                    res_string_list.append( Classes[np.argmin(pred_min)])
                                    res_qval_list.append(f'{100*pred[i][j]:1.0f}')
                                    if int(100*pred[i][j]) > max_qval:
                                        max_qval = int(100*pred[i][j])
                            # sort res_string_list based on res_qval_list
                            res_string_list_s = list_sort(res_string_list, res_qval_list)
                            res_qval_list = sorted(res_qval_list,reverse=True)
                            # delete last num_del from sorted results if more than requested exist
                            if (len(res_string_list_s)) > max_num_results:
                                num_del = len(res_string_list_s) - max_num_results
                                del res_string_list_s[-num_del:]
                                del res_qval_list[-num_del:]
                            # create semicolon separated list
                            res_string =";".join(map(str,res_string_list_s))
                            qval_string =";".join(map(str,res_qval_list))
                            # remove last ; if necessery
                            if res_string.endswith(';'):
                                res_string = res_string[:-1]
                            if qval_string.endswith(';'):
                                qval_string = qval_string[:-1]
                            dl_result = {
                            'id': i,
                            'dl_result': res_string,
                            'dl_qval': max_qval,
                            'dl_info': info_string,
                            'dl_qvals': qval_string,
                            }
                            if num_results_i == 0:
                                dl_result = {
                                'id': i,
                                'dl_result': 'REJ',
                                'dl_qval': 0,
                                'dl_info':'Reject',
                                'dl_qvals': str(0),
                                }
                        else: # only one result was requested
                            if pred[i][np.argmax(pred[i])] > reject_thres:
                                dl_result = {
                                'id': i,
                                'dl_result': str(f'{pred[i][np.argmax(pred[i])]:1.6f}'),
                                'dl_qval': int(100*pred[i][np.argmax(pred[i])]),
                                'dl_info':'Success',
                                'dl_qvals': str(int(100*pred[i][np.argmax(pred[i])])),
                                }
                            else:
                                dl_result = {
                                'id': i,
                                'dl_result': 'REJ',
                                'dl_qval': 0,
                                'dl_info':'Reject',
                                'dl_qvals': str(0),
                                }

                    dl_results.append(dl_result)

            t1 = time.time() - t0
            logging_text = ' time for preparation of results '+'{:.4f}'.format(t1) + ' s'
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        else:
            logging_text = ' no predictions available '
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            dl_results = []
            dl_result = {
            'id': 0,
            'dl_result': 'NONE',
            'dl_qval': 0,
            'dl_info':' no predictions available ',
            'dl_qvals': str(0),
            }
            dl_results.append(dl_result)
            return dl_results
    else:
        logging_text = ' DL classifier not initialized '
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        dl_results = []
        dl_result = {
        'id': 0,
        'dl_result': 'NONE',
        'dl_qval': 0,
        'dl_info':'DL Classifier not initialized',
        'dl_qvals': str(0),
        }
        dl_results.append(dl_result)
        return dl_results

    return dl_results

def TFliteInference(x_test,module_dir,db_type,num_threads_gprc_server,session,StatusData,status_datas_schema,
                    DlInitInfo,dl_init_infos_schema,ip_address_tf,port_tf,
                    ai_type_dl,DlInitRef,UserId,reject_thres,max_num_results):

    from basic_db_functions_cython import get_init_status
    #from basic_db_functions_cython import get_dl_init_data
    from basic_db_functions_cython import get_dl_init_info

    import tflite_runtime.interpreter as tflite

    module = 'TFliteInference '

    # get current init StatusData data
    if db_type != "File":
        status_data = get_init_status(session, StatusData, status_datas_schema,UserId)    
        if len(status_data) > 0:
            status_current = status_data['status']
            status_info_current = status_data['status_info']
            knn_init_status_current = status_data['knn_init_status']
            knn_init_status_info_current = status_data['knn_init_status_info']
            dl_init_status_current = status_data['dl_init_status']
            dl_init_status_info_current = status_data['dl_init_status_info']
        else:
            dl_results = []
            dl_result = {
            'id': 0,
            'dl_result':'NONE',
            'dl_qval': 0,
            'dl_info':'DL init status not available',
            'dl_qvals': str(0),
            }
            dl_results.append(dl_result)
            return dl_results
        #print('dl_init_status_current:',dl_init_status_current)
        logging_text = ' dl_init_status_current: ' + dl_init_status_current
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    else:
        dl_init_status_current = 'OK'
        logging_text = ' dl_init_status_current: OK'
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    if dl_init_status_current == 'OK':

        #read  DL init info data ( for this dl_init_ref)
        # get classes meta data
        if db_type == "File":
            GetDlInitDataInfoOK,dl_init_data_filt = get_dl_init_info_file(module_dir,DlInitRef)
        else:
            GetDlInitDataInfoOK,dl_init_data_filt = get_dl_init_info(session,DlInitInfo,dl_init_infos_schema,DlInitRef)

        logging_text = 'ADDLOG dl_init_data_filt: '+str(dl_init_data_filt)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

        ts_model_classes = ''
        if GetDlInitDataInfoOK:
            ts_model_name = dl_init_data_filt[0]['dlinit_tsmodelname']
            #ts_model_version = dl_init_data_filt[0]['dlinit_tsmodelversion']
            ts_model_classes = dl_init_data_filt[0]['dlinit_tsmodelclasses']
            logging_text = ' tensorlite model name and classes: '+ts_model_name+' '+ts_model_classes
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            if ts_model_classes == '':
                logging_text = ' classes meta data was not loaded sucessfully'
                if db_type == "File":
                    write_log_api_file(module_dir,module,logging_text)  
                else:
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
                dl_results = []
                dl_result = {
                'id': 0,
                'dl_result': 'NONE',
                'dl_qval': 0,
                'dl_info':' no predictions available ',
                'dl_qvals': str(0),
                }
                dl_results.append(dl_result)
                return dl_results
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            # generate Classes as list from ts_model_classes string
            Classes = ts_model_classes.split(";")
            logging_text = ' tensorflow lite model classes list: '+str(Classes)
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        else:            
            logging_text = ' tensorflow lite model name and version are not avaliable'
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            dl_results = []
            dl_result = {
            'id': 0,
            'dl_result': 'NONE',
            'dl_qval': 0,
            'dl_info':' no predictions available ',
            'dl_qvals': str(0),
            }
            dl_results.append(dl_result)
            return dl_results

        # check max_num_results
        if max_num_results < 1:
            max_num_results = 1
        if max_num_results > 3:
            max_num_results = 3

        # check reject_thres
        if reject_thres <= 0:
            logging_text = 'reject_thres is '+str(reject_thres)+' but must be > 0'
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            dl_results = []
            dl_result = {
                'id': 0,
                'dl_result': 'REJ',
                'dl_qval': -1000,
                'dl_info': logging_text,
                'dl_qvals': str(-1000),
                }
            dl_results.append(dl_result)
            return dl_results

        if reject_thres >= 1:
            logging_text = 'reject_thres is '+str(reject_thres)+' but must be > 1'
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            dl_results = []
            dl_result = {
                'id': 0,
                'dl_result': 'REJ',
                'dl_qval': -1000,
                'dl_info': logging_text,
                'dl_qvals': str(-1000),
                }
            dl_results.append(dl_result)
            return dl_results

        # TF lite inference

        logging_text = ' tensorflow lite model name : '+ts_model_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

        t0 = time.time()

        # Load TFLite model and allocate tensors
        with open(ts_model_name, 'rb') as f:
            tflite_model = f.read()

        interpreter = tflite.Interpreter(model_content=tflite_model, num_threads=num_threads_gprc_server)
        #interpreter = tflite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # TF lite batch interence 
        #print('x_test.shape :',x_test.shape)
        #print('len(x_test) :',len(x_test))
        #print('len(x_test[0]) :',len(x_test[0]))
        #print('len(x_test[0][0]) :',len(x_test[0][0]))

        # resize input and output tensor shapes
        num_samples = len(x_test)
        num_dim_1 = len(x_test[0])
        num_dim_2 = len(x_test[0][0])
        num_classes = len(Classes)
        #print('num_samples,num_classes :',num_samples,num_classes)
        interpreter.resize_tensor_input(input_details[0]['index'], (num_samples,num_dim_1,num_dim_2))
        interpreter.resize_tensor_input(output_details[0]['index'], (num_samples, num_classes))
        interpreter.allocate_tensors()

        #print("== Input details ==")
        #print("name:", input_details[0]['name'])
        #print("shape:", input_details[0]['shape'])
        #print("type:", input_details[0]['dtype'])

        #print("\n== Output details ==")
        #print("name:", output_details[0]['name'])
        #print("shape:", output_details[0]['shape'])
        #print("type:", output_details[0]['dtype'])
  
        #start_time_eval = time.time()

        t1 = time.time() - t0
        logging_text = ' time for preparation of TFlite predictions '+'{:.4f}'.format(t1) + ' s'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

        interpreter.set_tensor(input_details[0]['index'], x_test)

        t0 = time.time()

        interpreter.invoke()

        """
        end_time_eval = time.time()
        total_time_eval = end_time_eval - start_time_eval
        mean_time_eval = total_time_eval / num_samples
        print(f"Mean time batch: {mean_time_eval}s = {mean_time_eval*1000}ms")
        """

        """
        print('prediction[0] :',prediction[0])
        print('prediction.shape :',prediction.shape)
        """

        t1 = time.time() - t0
        logging_text = ' time for TFlite predictions '+'{:.4f}'.format(t1) + ' s'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

        # access on batch results
        prediction = interpreter.get_tensor(output_details[0]['index'])[0:num_samples]

        if len(prediction) > 0:  # any predictions ?
            # additional logging of output
            logging_text = 'ADDLOG tensor lite prediction result: '+str(prediction)
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            pred_OK = True
        else:
            pred_OK = False
            logging_text = ' no predictions available '
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            dl_results = []
            dl_result = {
            'id': 0,
            'dl_result': 'NONE',
            'dl_qval': 0,
            'dl_info':' no predictions available ',
            'dl_qvals': str(0),
            }
            dl_results.append(dl_result)
            return dl_results
        #print(pred)
        if pred_OK == True:
            t0 = time.time()
            dl_results = []
            logging_text = ' number of predictions from TF serving: '+str(len(prediction))
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

            for i in range (0,len(prediction)):  
                if ai_type_dl == 'DL_Class_TFlite':
                    #logging_text = 'ai_type_dl: '+str(ai_type_dl)
                    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                    #logging_text = ' pred[i] from TF serving: '+str(pred[i])
                    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                    if max_num_results > 1:  # more than one result was requested
                        cond_res = prediction[i] > reject_thres
                        num_results_i = cond_res.sum()
                        res_string = ''
                        qval_string = ''
                        info_string = 'Success: '+str(num_results_i)+' multiple results avaliable'
                        max_qval = 0
                        res_string_list = []
                        res_qval_list = []
                        for j in range(0,len(prediction[i])):
                            if cond_res[j] == True:
                                pred_min = np.abs(prediction[i] - prediction[i][j])   # identify index with prediction pred[i][j]
                                res_string_list.append(Classes[np.argmin(pred_min)])
                                res_qval_list.append(f'{100*prediction[i][j]:1.0f}')
                                if int(100*prediction[i][j]) > max_qval:
                                    max_qval = int(100*prediction[i][j])
                        # sort res_string_list based on res_qval_list
                        res_string_list_s = list_sort(res_string_list, res_qval_list)
                        res_qval_list = sorted(res_qval_list,reverse=True)
                        # delete last num_del from sorted results if more than requested exist
                        if (len(res_string_list_s)) > max_num_results:
                            num_del = len(res_string_list_s) - max_num_results
                            del res_string_list_s[-num_del:]
                            del res_qval_list[-num_del:]
                        # create semicolon separated list
                        res_string =";".join(map(str,res_string_list_s))
                        qval_string =";".join(map(str,res_qval_list))
                        # remove last ; if necessery
                        if res_string.endswith(';'):
                            res_string = res_string[:-1]
                        if qval_string.endswith(';'):
                            qval_string = qval_string[:-1]
                        #logging_text = 'result string multiple decisions: '+res_string
                        #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                        #logging_text = 'qval string multiple decisions: '+qval_string
                        #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                        dl_result = {
                        'id': i,
                        'dl_result': res_string,
                        'dl_qval': max_qval,
                        'dl_info': info_string,
                        'dl_qvals': qval_string,
                        }
                        if num_results_i == 0:
                            dl_result = {
                            'id': i,
                            'dl_result': 'REJ',
                            'dl_qval': 0,
                            'dl_info':'Reject',
                            'dl_qvals': str(0),
                            }
                    else: # only one result was requested
                        if prediction[i][np.argmax(prediction[i])] > reject_thres:
                            dl_result = {
                            'id': i,
                            'dl_result': Classes[np.argmax(prediction[i])],
                            'dl_qval': int(100*prediction[i][np.argmax(prediction[i])]),
                            'dl_info':'Success: Single result avaliable',
                            'dl_qvals': str(int(100*prediction[i][np.argmax(prediction[i])])),
                            }
                        else:
                            dl_result = {
                            'id': i,
                            'dl_result': 'REJ',
                            'dl_qval': 0,
                            'dl_info':'Reject',
                            'dl_qvals': str(0),
                            }

                    dl_results.append(dl_result)

                if ai_type_dl == 'DL_Reg_TFlite':
                    #logging_text = str(module_dir)
                    #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                    # check if AI module is AOA
                    if "_aoa_" in module_dir:
                        #logging_text = ' TF serving called from AOA service '
                        #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
                        #convert unit vector to angle according to HS code
                        pred_rad = np.arctan2(prediction[i, 0],prediction[i, 1])
                        pred_rad = pred_rad + 2.0 * np.pi if pred_rad < 0.0 else pred_rad 
                        pred_deg = pred_rad / np.pi * 180.0
                        dl_result = {
                        'id': i,
                        'dl_result': str(f'{pred_deg:1.6f}'),
                        'dl_qval': int(100),
                        'dl_info':'Success',
                        'dl_qvals': str(100),
                        }
                    else: 
                        if max_num_results > 1:  # more than one result was requested
                            cond_res = prediction[i] > reject_thres
                            num_results_i = cond_res.sum()
                            res_string = ''
                            qval_string = ''
                            info_string = 'Success: Multiple results avaliable'
                            max_qval = 0
                            res_string_list = []
                            res_qval_list = []
                            for j in range(0,len(prediction[i])):
                                if cond_res[j] == True:
                                    pred_min = np.abs(prediction[i] - prediction[i][j])   # identify index with prediction pred[i][j]
                                    res_string_list.append( Classes[np.argmin(pred_min)])
                                    res_qval_list.append(f'{100*prediction[i][j]:1.0f}')
                                    if int(100*prediction[i][j]) > max_qval:
                                        max_qval = int(100*prediction[i][j])
                            # sort res_string_list based on res_qval_list
                            res_string_list_s = list_sort(res_string_list, res_qval_list)
                            res_qval_list = sorted(res_qval_list,reverse=True)
                            # delete last num_del from sorted results if more than requested exist
                            if (len(res_string_list_s)) > max_num_results:
                                num_del = len(res_string_list_s) - max_num_results
                                del res_string_list_s[-num_del:]
                                del res_qval_list[-num_del:]
                            # create semicolon separated list
                            res_string =";".join(map(str,res_string_list_s))
                            qval_string =";".join(map(str,res_qval_list))
                            # remove last ; if necessery
                            if res_string.endswith(';'):
                                res_string = res_string[:-1]
                            if qval_string.endswith(';'):
                                qval_string = qval_string[:-1]
                            dl_result = {
                            'id': i,
                            'dl_result': res_string,
                            'dl_qval': max_qval,
                            'dl_info': info_string,
                            'dl_qvals': qval_string,
                            }
                            if num_results_i == 0:
                                dl_result = {
                                'id': i,
                                'dl_result': 'REJ',
                                'dl_qval': 0,
                                'dl_info':'Reject',
                                'dl_qvals': str(0),
                                }
                        else: # only one result was requested
                            if prediction[i][np.argmax(prediction[i])] > reject_thres:
                                dl_result = {
                                'id': i,
                                'dl_result': str(f'{prediction[i][np.argmax(prediction[i])]:1.6f}'),
                                'dl_qval': int(100*prediction[i][np.argmax(prediction[i])]),
                                'dl_info':'Success',
                                'dl_qvals': str(int(100*prediction[i][np.argmax(prediction[i])])),
                                }
                            else:
                                dl_result = {
                                'id': i,
                                'dl_result': 'REJ',
                                'dl_qval': 0,
                                'dl_info':'Reject',
                                'dl_qvals': str(0),
                                }

                    dl_results.append(dl_result)

            t1 = time.time() - t0
            logging_text = ' time for preparation of results '+'{:.4f}'.format(t1) + ' s'
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        else:
            logging_text = ' no predictions available '
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            dl_results = []
            dl_result = {
            'id': 0,
            'dl_result': 'NONE',
            'dl_qval': 0,
            'dl_info':' no predictions available ',
            'dl_qvals': str(0),
            }
            dl_results.append(dl_result)
            return dl_results
    else:
        logging_text = ' DL classifier not initialized '
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        dl_results = []
        dl_result = {
        'id': 0,
        'dl_result': 'NONE',
        'dl_qval': 0,
        'dl_info':'DL Classifier not initialized',
        'dl_qvals': str(0),
        }
        dl_results.append(dl_result)
        return dl_results

    return dl_results

# normalize features columns to range [0,1] 
def normalize_columns(X,X_max,X_min):
    # normalize features columns in range [0,1]
    num_columns = X.shape[1]
    for i in range (0,num_columns):
        if (X_max[i]-X_min[i])>0:
            X[:,i]=((X[:,i]-X_min[i])/(X_max[i]-X_min[i]))*1.0
        else:
            X[:,i]=1.0
    return X

def limit_data(X,Thres):
    # set data below the threshold to the threshold value, for example -30 dB 
    X_cond = X < Thres
    X[X_cond] = Thres
    return X

def load_train_data_init_knn(module_dir,db_type,train_dir_knn,session,StatusData,status_datas_schema,DataRefInfo,data_ref_infos_schema,ai_type,n_neighbors,userid,Thres=-1000):

    from basic_db_functions_cython import get_ref_data_info

    ResOk = False
    module = 'load_train_data_init_knn '

    # load active reference train data info from database for knn init
    NPZ_file_name_list_train = []
    GetRefDataInfoOK,ref_data_active = get_ref_data_info(session,DataRefInfo,data_ref_infos_schema,'Active','ALL', userid)
    num_active_files = len(ref_data_active)
    logging_text = 'Number of active kNN train data files '+str(num_active_files) 
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    if GetRefDataInfoOK:
        for i in range (0,num_active_files):
            NPZ_file_name_list_train.append(ref_data_active[i]['dataref_filename'])
    else:
        logging_text = ' reference data info for knn init not loaded sucessfully from database'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        neigh_file_name = ''
        train_labels_file_name = ''
        train_min_file_name = ''
        train_max_file_name = ''
        init_ref = 0
        return ResOk,init_ref,neigh_file_name,train_labels_file_name,train_min_file_name,train_max_file_name

    t0 = time.time()
    data_labels_multiple_train = []
    data_features_multiple_train = []
    NPZ_file_num = 0
    for NPZ_file_name in NPZ_file_name_list_train:
        if ".npz" in NPZ_file_name:
            try:
                data_features_labels = np.load(NPZ_file_name,allow_pickle=True)
                features_has_nan = np.isnan(data_features_labels['f']).any()
                if features_has_nan:
                    NPZ_file_num = NPZ_file_num + 1
                    logging_text = 'kNN train data file '+NPZ_file_name+' not loaded because it contains NaN values'
                    if db_type == "File":
                        write_log_api_file(module_dir,module,logging_text)  
                    else:
                        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
                else:
                    data_features_multiple_train.append(data_features_labels['f'])
                    data_labels_multiple_train.append(data_features_labels['l'])
                    NPZ_file_num = NPZ_file_num + 1
                    logging_text = 'kNN train data file '+NPZ_file_name+' sucessfully loaded'
                    if db_type == "File":
                        write_log_api_file(module_dir,module,logging_text)  
                    else:
                        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
            except:
                logging_text = ' file '+NPZ_file_name+' in kNN train data directory do not exist'
                if db_type == "File":
                    write_log_api_file(module_dir,module,logging_text)  
                else:
                    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    if NPZ_file_num == 0:
        logging_text = ' files in kNN train data directory do not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        neigh_file_name = ''
        train_labels_file_name = ''
        train_min_file_name = ''
        train_max_file_name = ''
        init_ref = 0
        return ResOk,init_ref,neigh_file_name,train_labels_file_name,train_min_file_name,train_max_file_name

    #Initialize kNN Classifier/Regressor from multiple training files
    #neigh = KNeighborsRegressor(n_neighbors=n_neighbors,weights = 'distance')
    if ai_type == 'kNN_Reg':
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
    if ai_type == 'kNN_Class':
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    # create concateneted numpy array for train labels 
    y_train = np.array(data_labels_multiple_train[0])
    for i in range (1,len(data_labels_multiple_train)):
        y_train_i = np.array(data_labels_multiple_train[i])
        y_train = np.concatenate((y_train,y_train_i),axis=0)
    # create concateneted numpy array for train features 
    #print('first y_train -labels ',y_train[0:10])
    #print('last y_train -labels ',y_train[len(y_train)-10:len(y_train)])
    X_train = np.array(data_features_multiple_train[0])
    for i in range (1,len(data_features_multiple_train)):
        X_train_i = np.array(data_features_multiple_train[i])
        X_train = np.concatenate((X_train,X_train_i),axis=0)
    #limit data to power threshold , for example -30 db
    if Thres != -1000:  # -1000 is the default value initialized
        X_train = limit_data(X_train,Thres)
    # calculate minima and maxima from train data
    num_features = X_train.shape[1]
    X_train_min = np.empty(num_features)
    X_train_max = np.empty(num_features)
    for i in range (0,num_features):
        X_train_min[i] = X_train[:,i].min()
        X_train_max[i] = X_train[:,i].max()
    X_train = normalize_columns(X_train,X_train_max,X_train_min)
    # train the kNN classifier/regressor
    neigh.fit(X_train,y_train)
    t1 = time.time() - t0
    logging_text = 'Time elapsed for initializing kNN Classifier/Regressor: '+ ' {:.4f}'.format(t1)+ ' s'
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    #create file names for neigh and y_train
    #time.sleep(0.1) # Sleep for 0.1 seconds - to be sure that a new data ref is created
    #datetime_int = int(time.time()*1000)
    #init_ref = f"{datetime_int}"
    init_ref = str(uuid.uuid4())
    NPZ_file = 'kNN_Train'
    # set knn init file names
    neigh_file_name = train_dir_knn+'neigh_'+str(NPZ_file_num)+'_files_'+init_ref+'_'+NPZ_file+'.pickle'
    logging_text = 'Filename for neigh object: '+neigh_file_name
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    train_labels_file_name = train_dir_knn+'train_labels_'+str(NPZ_file_num)+'_files_'+init_ref+'_'+NPZ_file+'.pickle'
    logging_text = 'Filename for train labels object: '+ train_labels_file_name
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    train_min_file_name = train_dir_knn+'min_'+str(NPZ_file_num)+'_files_'+init_ref+'_'+NPZ_file+'.pickle'
    logging_text = 'Filename for train min object: '+ train_min_file_name
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    train_max_file_name = train_dir_knn+'max_'+str(NPZ_file_num)+'_files_'+init_ref+'_'+NPZ_file+'.pickle'
    logging_text = 'Filename for train max object: '+ train_max_file_name
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    # Save knn init objects as pickle files
    file_name_lock = neigh_file_name+'.lock'
    lock_file_name = FileLock(file_name_lock)
    with lock_file_name:
        with open(neigh_file_name, 'wb') as pickle_out:
            pickle.dump(neigh, pickle_out)
    try:
        os.remove(file_name_lock)
    except:
        pass
    file_name_lock = train_labels_file_name+'.lock'
    lock_file_name = FileLock(file_name_lock)
    with lock_file_name:
        with open(train_labels_file_name, 'wb') as pickle_out:
            pickle.dump(y_train, pickle_out)
    try:
        os.remove(file_name_lock)
    except:
        pass
    file_name_lock = train_min_file_name+'.lock'
    lock_file_name = FileLock(file_name_lock)
    with lock_file_name:
        with open(train_min_file_name, 'wb') as pickle_out:
            pickle.dump(X_train_min, pickle_out)
    try:
        os.remove(file_name_lock)
    except:
        pass
    file_name_lock = train_max_file_name+'.lock'
    lock_file_name = FileLock(file_name_lock)
    with lock_file_name:
        with open(train_max_file_name, 'wb') as pickle_out:
            pickle.dump(X_train_max, pickle_out)
    try:
        os.remove(file_name_lock)
    except:
        pass

    ResOk=True
    return ResOk,init_ref,neigh_file_name,train_labels_file_name,train_min_file_name,train_max_file_name

def load_test_data(module_dir,db_type,session,StatusData,status_datas_schema,test_data_file):
    module = 'load_test_data '
    x_test = []
    t0 = time.time()
    if ".npz" in test_data_file:
        # load testing data
        try:
            data_features_labels = np.load(test_data_file,allow_pickle=True)
        except:
            load_file = False
        else:
            x_test = data_features_labels['f']
            load_file = True
    else:
        load_file = False
    t1 = time.time() - t0
    logging_text = ' time for loading test data file: '+test_data_file+' {:.4f}'.format(t1) + ' s'
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    return x_test,load_file

def load_test_data_multiple(module_dir,db_type,session,StatusData,status_datas_schema,test_data_dir):
    LoadOK = False
    module = 'load_test_data_multiple '
    if os.path.exists(test_data_dir):
        logging_text = ' kNN test data directory: '+test_data_dir,' does exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    else:
        logging_text = ' kNN test data directory: '+test_data_dir,' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        X_test = np.zeros(1)
        return LoadOK,X_test
    #data_labels_multiple_test = []
    data_features_multiple_test = []
    NPZ_file_name_list_test = os.listdir(test_data_dir)
    NPZ_file_name_list_test.sort()
    t0 = time.time()
    for NPZ_file_name in NPZ_file_name_list_test:
        if ".npz" in NPZ_file_name:
            data_features_labels = np.load(test_data_dir+NPZ_file_name,allow_pickle=True)
            data_features_multiple_test.append(data_features_labels['f'])
            #data_labels_multiple_test.append(data_features_labels['l'])
            logging_text = 'kNN Test File '+str(NPZ_file_name)+' loaded'
            if db_type == "File":
                write_log_api_file(module_dir,module,logging_text)  
            else:
                write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    
    #Initialize kNN Classifier/Regressor from multiple test files
    #create concateneted numpy array for test labels 
    """
    y_test = np.array(data_labels_multiple_test[0])
    for i in range (1,len(data_labels_multiple_test)):
        y_test_i = np.array(data_labels_multiple_test[i])
        y_test = np.concatenate((y_test,y_test_i),axis=0)
    """
    # create concateneted numpy array for test features 
    X_test = np.array(data_features_multiple_test[0])
    for i in range (1,len(data_features_multiple_test)):
        X_test_i = np.array(data_features_multiple_test[i])
        X_test = np.concatenate((X_test,X_test_i),axis=0)
    t1 = time.time() - t0
    logging_text = 'Time elapsed for loading kNN test files: '+ ' {:.4f}'.format(t1)+ ' s'
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    LoadOK = True
    return LoadOK,X_test

def knn_check_reject(module_dir,db_type,session,StatusData,status_datas_schema,x_test,neigh,train_labels,ai_type,kNN_reject=3):

    module = 'knn_check_reject '
    num_reject = 0
    diff_knn_dist_test = []
    diff_labelval_test = []
    num_test_samples =  len(x_test)   
    rejects = np.zeros(num_test_samples)
    qvals = np.zeros(num_test_samples)   #best distance in percentage to reject threshold

    # read kNN init parameters parameters from JSON init file
    kNN,kNN_reject,reject_thres,max_num_results,reject_labelval_diff,pow_norm,feature_norm,softmax = \
    read_knn_init_param(module_dir,db_type,session,StatusData,status_datas_schema)
    
    t0 = time.time()
    dist_neigh,dist_neigh_ind = neigh.kneighbors(x_test,kNN_reject)
    #print(x_test[0])
    #print(dist_neigh[0])
    #print(dist_neigh_ind[0])

    if ai_type == 'kNN_Reg':
        for i in range (0,num_test_samples):
            #min/max labelled distance in kNN set
            kNN_MinLabelVal = np.min(train_labels[dist_neigh_ind[i]])
            kNN_MaxLabelVal = np.max(train_labels[dist_neigh_ind[i]])
            maxdiff_labelval_test = (kNN_MaxLabelVal-kNN_MinLabelVal)
            diff_labelval_test.append(maxdiff_labelval_test)
            knn_min_dist_test = dist_neigh[i][0]
            diff_knn_dist_test.append(knn_min_dist_test)
            qvals[i] = 100-((knn_min_dist_test)/reject_thres)*100
            if (knn_min_dist_test>reject_thres)or(maxdiff_labelval_test>reject_labelval_diff):
                num_reject = num_reject+1    
                rejects[i] = 1

    if ai_type == 'kNN_Class':
        num_kNN_LabelNotSame = 0
        for i in range (0,num_test_samples):
            #min/max labelled distance in kNN set
            #logging_text = 'Reject check labels: '+str(train_labels[dist_neigh_ind[i]])
            #write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
            if len(set(train_labels[dist_neigh_ind[i]])) == 1: # labels within train_labels[dist_neigh_ind[i]] are the same
                kNN_LabelNotSame = False
            else:
                kNN_LabelNotSame = True
                num_kNN_LabelNotSame = num_kNN_LabelNotSame+1
            knn_min_dist_test = dist_neigh[i][0]
            diff_knn_dist_test.append(knn_min_dist_test)
            qvals[i] = 100-((knn_min_dist_test)/reject_thres)*100
            if (knn_min_dist_test>reject_thres)or(kNN_LabelNotSame): # count also reject if the labels within train_labels[dist_neigh_ind[i]] are not the same
                num_reject = num_reject+1    
                rejects[i] = 1

    t1 = time.time() - t0
    logging_text = ' time elapsed for kNN reject predictions: '+' {:.4f}'.format(t1) + ' s for '+str(num_test_samples)+' test samples'
    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Results including rejects on test data: '
    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Maximum distance on test samples: '+' {:.4f}'.format(max(diff_knn_dist_test))
    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Minimum distance on test samples: '+' {:.4f}'.format(min(diff_knn_dist_test))
    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Avarage distance on test samples: '+' {:.4f}'.format((sum(diff_knn_dist_test)/len(diff_knn_dist_test)))
    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Distance threshold rejects:'+' {:.4f}'.format(reject_thres)
    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    logging_text = 'Reject rate:'+' {:.4f}'.format(num_reject/num_test_samples)
    write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    if ai_type == 'kNN_Reg':
        logging_text = 'Maximum kNN label value difference on test samples: '+' {:.4f}'.format(max(diff_labelval_test))
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        logging_text = 'Minimum kNN label value difference on test samples: '+' {:.4f}'.format(min(diff_labelval_test))
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        logging_text = 'Average kNN label value difference on test samples: '+' {:.4f}'.format((sum(diff_labelval_test)/len(diff_labelval_test)))
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        logging_text = 'Label value difference threshold rejects::'+' {:.4f}'.format(reject_labelval_diff)
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    if ai_type == 'kNN_Class': 
        logging_text = 'Labels not the same rate:'+' {:.4f}'.format(num_kNN_LabelNotSame/num_test_samples)
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)

    return rejects,qvals

def read_knn_init_param(module_dir,db_type,session,StatusData,status_datas_schema):

    module = 'read_knn_init_param'

    init_file_name = module_dir+'/init_knn_param.json'
    if os.path.exists(init_file_name):
        # read init file
        file_name_lock = init_file_name+'.lock'
        lock_file_name = FileLock(file_name_lock)
        with lock_file_name:
            with open(init_file_name, 'r') as myfile:
                data=myfile.read()
        try:
            os.remove(file_name_lock)
        except:
            pass
        # parse file
        obj = json.loads(data)
        try:
            kNN = int(obj['kNN'])  #initial defatult 1
        except:
            kNN = 1
        # reject parameters
        try:
            kNN_reject = int(obj['kNN_reject'])  #initial defatult 3
        except:
            kNN_reject = 3        
        try:
            reject_thres = float(obj['reject_thres']) #initial defatult 0.1
        except:
            reject_thres = 1000 
        try:
            max_num_results = int(obj['max_num_results']) #initial defatult 3
        except:
            max_num_results = 1000 
        try:
            reject_labelval_diff = int(obj['reject_labelval_diff'])  #initial defatult 20
        except:
            reject_labelval_diff = 1000 
        try:
            pow_norm = str(obj['pow_norm'])  #initial defatult "YES"
        except:
            pow_norm = "" 
        try:
            feature_norm = str(obj['feature_norm'])  #initial defatult "YES"
        except:
            feature_norm = "" 
        try:
            softmax = str(obj['softmax'])  #initial defatult "YES"
        except:
            softmax = "" 
        logging_text = 'kNN init parameters loaded from init file:' + init_file_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        logging_text = 'kNN Init Values: ',kNN,kNN_reject,reject_thres,max_num_results,reject_labelval_diff,pow_norm,feature_norm,softmax
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    else:
        logging_text = 'kNN init file ' + init_file_name + ' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        kNN = 0
        kNN_reject = 0
        reject_thres = 1000.0
        max_num_results = 1000
        reject_labelval_diff = 1000
        pow_norm = "" 
        feature_norm  = ""
        softmax = "" 

    return kNN,kNN_reject,reject_thres,max_num_results,reject_labelval_diff,pow_norm,feature_norm,softmax

def read_dl_init_param(module_dir,db_type,session,StatusData,status_datas_schema):
    module = 'read_dl_init_param '
    init_file_name = module_dir+'/init_dl_param.json'
    if os.path.exists(init_file_name):
        # read init file
        file_name_lock = init_file_name+'.lock'
        lock_file_name = FileLock(file_name_lock)
        with lock_file_name:
            with open(init_file_name, 'r') as myfile:
                data=myfile.read()
        try:
            os.remove(file_name_lock)
        except:
            pass
        # parse file
        obj = json.loads(data) 
        try:
            reject_thres = float(obj['reject_thres'])  
        except:
            reject_thres = 0.0
        try:
            max_num_results = int(obj['max_num_results'])  
        except:
            max_num_results = -1   
        try:
            oc_reject_thres = float(obj['oc_reject_thres'])  
        except:
            oc_reject_thres = 0.0
        try:
            oc_num_single_results = int(obj['oc_num_single_results']) 
        except:
            oc_num_single_results = -1         

        logging_text = 'DL init parameters loaded from init file:' + init_file_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        logging_text = 'DL init parameters reject_thres,max_num_results,oc_reject_thres,oc_num_single_results :' + str(reject_thres)+ ',' + str(max_num_results)+ str(oc_reject_thres)+ ',' + str(oc_num_single_results)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
    else:
        logging_text = 'DL init file ' + init_file_name + ' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        reject_thres = -1
        max_num_results = -1
        oc_reject_thres = 0.0
        oc_num_single_results = -1   
    return reject_thres,max_num_results,oc_reject_thres,oc_num_single_results

def write_dl_init_param(module_dir,db_type,session,StatusData,status_datas_schema,
                        reject_thres,max_num_results,oc_reject_thres,oc_num_single_results):

    module = 'write_dl_init_param '
    init_file_name = module_dir+'/init_dl_param.json'
    Error = False

    # storing dl init data in dictionary
    dl_init_data = {"description": "dl init parameters",
                     "reject_thres": reject_thres,
                     "max_num_results": max_num_results,
                     "oc_reject_thres": oc_reject_thres,
                     "oc_num_single_results": oc_num_single_results
                     }

    if os.path.exists(module_dir):
        # write init file
        try: 
            file_name_lock = init_file_name+'.lock'
            lock_file_name = FileLock(file_name_lock)
            with lock_file_name:
                with open(init_file_name, 'w') as myfile:
                    json.dump(dl_init_data, myfile)
            try:
                os.remove(file_name_lock)
            except:
                pass
        except:
            Error = True

        logging_text = 'DL init parameters written to init file:' + init_file_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        logging_text = 'DL Init Values: ' + str(reject_thres) + ','+str(max_num_results)
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    else:
        logging_text = 'DL init file directory' + module_dir + ' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        Error = True

    return Error   

def write_knn_init_param(module_dir,db_type,session,StatusData,status_datas_schema,
                         kNN,kNN_reject,reject_thres,max_num_results,
                         reject_labelval_diff,softmax,pow_norm,feature_norm):

    module = 'write_knn_init_param '
    init_file_name = module_dir+'/init_knn_param.json'
    Error = False

    # storing knn init data in dictionary
    knn_init_data = {"description": "knn init parameters", "kNN": kNN,"kNN_reject": kNN_reject,
                     "reject_thres": reject_thres,"max_num_results": max_num_results,
                     "reject_labelval_diff": reject_labelval_diff,"pow_norm": pow_norm,
                     "feature_norm": feature_norm,"softmax": softmax}

    if os.path.exists(module_dir):
        # write init file
        try: 
            file_name_lock = init_file_name+'.lock'
            lock_file_name = FileLock(file_name_lock)
            with lock_file_name:
                with open(init_file_name, 'w') as myfile:
                    json.dump(knn_init_data, myfile)
            try:
                os.remove(file_name_lock)
            except:
                pass
        except:
            Error = True

        logging_text = 'kNN init parameters written to init file:' + init_file_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        logging_text = 'kNN Init Values: ',kNN,kNN_reject,reject_thres,reject_labelval_diff,softmax,pow_norm,feature_norm
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    else:
        logging_text = 'kNN init file directory' + module_dir + ' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        Error = True

    return Error   

def write_init_param(module_dir,db_type,session,StatusData,status_datas_schema,
                     ai_type,ai_type_dl,ai_type_knn,db_type_n,num_threads_gprc_server):

    module = 'write_init_param '
    init_file_name = module_dir+'/init_param.json'
    Error = False

    # storing init data in dictionary
    init_data = {"description": "written AI service init parameters", "ai_type": ai_type, 
                 "ai_type_dl": ai_type_dl,"ai_type_knn": ai_type_knn,"db_type": db_type_n,
                 "num_threads_gprc_server":num_threads_gprc_server}

    if os.path.exists(module_dir):
        # write init file
        try: 
            file_name_lock = init_file_name+'.lock'
            lock_file_name = FileLock(file_name_lock)
            with lock_file_name:
                with open(init_file_name, 'w') as myfile:
                    json.dump(init_data, myfile)
            try:
                os.remove(file_name_lock)
            except:
                pass
        except:
            Error = True
        logging_text = 'AI init parameters written to init file:' + init_file_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        logging_text = 'AI Init Values: ',ai_type,ai_type_dl,ai_type_knn,db_type_n
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    else:
        logging_text = 'AI init file directory' + module_dir + ' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        Error = True

    return Error   

def write_one_class_results_param(module_dir,db_type,session,StatusData,status_datas_schema,
                                  oc_num_single_results,oc_reject_thres):

    module = 'write_one_class_results_param '
    init_file_name = module_dir+'/one_class_result_param.json'
    Error = False

    # storing init data in dictionary
    one_class_results_data = {"description": "written one class result parameters", "oc_num_single_results": oc_num_single_results, 
                     "oc_reject_thres": oc_reject_thres}

    if os.path.exists(module_dir):
        # write init file
        try: 
            file_name_lock = init_file_name+'.lock'
            lock_file_name = FileLock(file_name_lock)
            with lock_file_name:
                with open(init_file_name, 'w') as myfile:
                    json.dump(one_class_results_data, myfile)
            try:
                os.remove(file_name_lock)
            except:
                pass
        except:
            Error = True

        logging_text = 'one class result parameters written to init file:' + init_file_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        logging_text = 'one class result: ',oc_num_single_results,oc_reject_thres
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    else:
        logging_text = 'one class result file directory' + module_dir + ' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        Error = True

    return Error  

def write_pre_process_param(module_dir,db_type,session,StatusData,status_datas_schema,
                            frames_dim_new,frames_interval_fact,num_frames_ave,
                            step_frames_ave,num_integration_intervalls,pow_sub_range_width,
                            pow_sub_range_step,cut_low_pow_values):

    module = 'write_pre_process_param '
    init_file_name = module_dir+'/pre_process_param.json'
    Error = False

    # storing init data in dictionary
    pre_process_param_data = {"description": "written preprocessing parameters", "frames_dim_new": frames_dim_new,"frames_interval_fact": frames_interval_fact,
                              "num_frames_ave": num_frames_ave,"step_frames_ave": step_frames_ave,"num_integration_intervalls": num_integration_intervalls,
                              "pow_sub_range_width": pow_sub_range_width,"pow_sub_range_step": pow_sub_range_step,"cut_low_pow_values": cut_low_pow_values}

    if os.path.exists(module_dir):
        # write init file
        try: 
            file_name_lock = init_file_name+'.lock'
            lock_file_name = FileLock(file_name_lock)
            with lock_file_name:
                with open(init_file_name, 'w') as myfile:
                    json.dump(pre_process_param_data, myfile)
            try:
                os.remove(file_name_lock)
            except:
                pass
        except:
            Error = True

        logging_text = 'preprocessing parameters written to init file:' + init_file_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        logging_text = 'preprocessing parameters: ',frames_dim_new,frames_interval_fact,num_frames_ave,step_frames_ave,step_frames_ave
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    else:
        logging_text = 'preprocessing parameters file directory' + module_dir + ' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        Error = True

    return Error 

def reset_default_init_param(module_dir,db_type,session,StatusData,status_datas_schema,mode):

    module = 'reset_default_init_param '

    init_default_file_name = ""
    init_file_name = ""
    if mode == 'init_knn_param':
        init_default_file_name = module_dir+'/init_knn_param_default.json'
        init_file_name = module_dir+'/init_knn_param.json'
    if mode == 'init_dl_param':
        init_default_file_name = module_dir+'/init_dl_param_default.json'
        init_file_name = module_dir+'/init_dl_param.json'
    # these parameters are now allowed to be reset
    #if mode == 'init_param':
    #    init_default_file_name = module_dir+'/init_param_default.json'
    #    init_file_name = module_dir+'/init_param.json'
    if mode == 'one_class_result_param':
        init_default_file_name = module_dir+'/one_class_result_param_default.json'
        init_file_name = module_dir+'/one_class_result_param.json'
    if mode == 'pre_process_param':
        init_default_file_name = module_dir+'/pre_process_param_default.json'
        init_file_name = module_dir+'/pre_process_param.json'

    if init_default_file_name == "":
        Error = True
        return Error

    logging_text = 'default parameter init file:' + init_default_file_name
    if db_type == "File":
        write_log_api_file(module_dir,module,logging_text)  
    else:
        write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    Error = False

    if os.path.exists(module_dir):
        # write init file
        try: 
            shutil.copyfile(init_default_file_name,init_file_name)
            Error = False
        except:
            Error = True

        logging_text = 'default init parameters written to init file:' + init_file_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 

    else:
        logging_text = 'init file directory' + module_dir + ' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text) 
        Error = True

    return Error  

def read_pre_process_param(module_dir,db_type,session,StatusData,status_datas_schema):
    module = 'read_pre_process_param '
    init_file_name = module_dir+'/pre_process_param.json'
    if os.path.exists(init_file_name):
        # read init file
        file_name_lock = init_file_name+'.lock'
        lock_file_name = FileLock(file_name_lock)
        with lock_file_name:
            with open(init_file_name, 'r') as myfile:
                data=myfile.read()
        try:
            os.remove(file_name_lock)
        except:
            pass
        # parse file
        obj = json.loads(data)
        try:
            frames_dim_new = int(obj['frames_dim_new'])  
        except:
            frames_dim_new = -1000
        try:
            frames_interval_fact = float(obj['frames_interval_fact'])  
        except:
            frames_interval_fact = -1000.0   
        try:
            num_frames_ave = int(obj['num_frames_ave'])  
        except:
            num_frames_ave = -1000
        try:
            step_frames_ave = int(obj['step_frames_ave']) 
        except:
            step_frames_ave = -1000       
        try:
            num_integration_intervalls = int(obj['num_integration_intervalls']) 
        except:
            num_integration_intervalls = -1000 
        try:
            pow_sub_range_width = int(obj['pow_sub_range_width']) 
        except:
            pow_sub_range_width = 0
        try:
            pow_sub_range_step = int(obj['pow_sub_range_step']) 
        except:
            pow_sub_range_step = 0
        try:
            cut_low_pow_values = int(obj['cut_low_pow_values']) 
        except:
            cut_low_pow_values = 0

        logging_text = ' frame preprocessing parameters loaded from file:' + init_file_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        logging_text = ' frame preprocessing parameters: ',frames_dim_new,frames_interval_fact,num_frames_ave,step_frames_ave,num_integration_intervalls,pow_sub_range_width,pow_sub_range_step,cut_low_pow_values
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    else:
        logging_text = ' frame preprocessing parameters ' + init_file_name + ' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        frames_dim_new = -1000
        frames_interval_fact = -1000.0
        num_frames_ave = -1000
        step_frames_ave = -1000
        num_integration_intervalls = -1000
    return frames_dim_new,frames_interval_fact,num_frames_ave,step_frames_ave,num_integration_intervalls,pow_sub_range_width,pow_sub_range_step,cut_low_pow_values

def read_one_class_result_param(module_dir,db_type,session,StatusData,status_datas_schema):
    module = 'read_one_class_result_param '
    init_file_name = module_dir+'/one_class_result_param.json'
    if os.path.exists(init_file_name):
        # read init file
        file_name_lock = init_file_name+'.lock'
        lock_file_name = FileLock(file_name_lock)
        with lock_file_name:
            with open(init_file_name, 'r') as myfile:
                data=myfile.read()
        try:
            os.remove(file_name_lock)
        except:
            pass
        # parse file
        obj = json.loads(data)
        try:
            oc_num_single_results = int(obj['oc_num_single_results'])  
        except:
            oc_num_single_results = -1000
        try:
            oc_reject_thres = float(obj['oc_reject_thres'])  
        except:
            oc_reject_thres = -1000.0   
        logging_text = ' one class result parameters loaded from file:' + init_file_name
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        logging_text = ' one class result parameters: ',oc_num_single_results,oc_reject_thres
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
    else:
        logging_text = ' one class result parameters ' + init_file_name + ' does not exist'
        if db_type == "File":
            write_log_api_file(module_dir,module,logging_text)  
        else:
            write_log_api(module_dir,session,StatusData,status_datas_schema,module,logging_text)
        num_single_results = -1000
        reject_thres = -1000.0
    return oc_num_single_results,oc_reject_thres

# calculate mean value data on data_frames(num_frames_all,frames_dim):  
# calculate mean values of data frames over num_frames_ave frames in step_frames_ave steps
def MeanValDataFrames(data_frames,num_frames_ave,step_frames_ave):
 
    num_frames_all = len(data_frames)
    frames_dim = len(data_frames[0])

    # count first num_mean_frames for mean_cms_data_pow np array allocation 
    num_mean_val_frames=0
    for i in range (0,num_frames_all-num_frames_ave,step_frames_ave):
        num_mean_val_frames=num_mean_val_frames+1

    # now calculated the mean values 
    mean_val_data_frames = np.zeros((num_mean_val_frames,frames_dim))
    num_mean_val_frames=0
    for i in range (0,len(data_frames)-num_frames_ave,step_frames_ave):
        mean_val_data_frames[num_mean_val_frames] = np.mean(data_frames[i:i+num_frames_ave],axis=0)
        num_mean_val_frames=num_mean_val_frames+1

    return mean_val_data_frames

# Resample data frames to dimension frames_dim_new starting from min_index to max_index (in data frames)
def ReSampleDataFrames(data_frames,frames_dim_new,min_ind=0,max_ind=0): 
 
    num_frames_all = len(data_frames)
    frames_dim = len(data_frames[0])
    if max_ind == 0:
        max_ind = frames_dim
    
    decimate_factor = int((max_ind-min_ind)/frames_dim_new)

    # dimension of extracted and resampled power data - n_dim
    resampled_data_frames = np.zeros((num_frames_all,frames_dim_new))
    # resample maximun power data to frames_dim_new values
    if decimate_factor < 4:
        for i in range (0,num_frames_all):
            resampled_data_frames[i] = signal.resample(data_frames[i][min_ind:max_ind],frames_dim_new)
    else:
        for i in range (0,num_frames_all):
            #decimated_data_frame = signal.decimate(data_frames[i][min_ind:max_ind],decimate_factor)
            #resampled_data_frames[i] = signal.resample(decimated_data_frame,frames_dim_new)
            resampled_data_frames[i] = signal.resample(data_frames[i][min_ind:max_ind:decimate_factor],frames_dim_new)

    return resampled_data_frames

# normalize data frames to mean value = 0 and divide by 2.5 standard deviation 
def NormlizeDataFrames(data_frames): 
 
    num_frames_all = len(data_frames)
    frames_dim = len(data_frames[0])

    data_frames_n = np.zeros((num_frames_all,frames_dim))

    #  normalize data frames to mean value = 0 and divide by 2.5 standard deviation 
    for i in range (0,num_frames_all):
        mean_val = np.mean(data_frames[i][:])
        std_val = np.std(data_frames[i][:])
        data_frames_n[i][:] = data_frames[i][:]-mean_val
        data_frames_n[i][:] = data_frames_n[i][:]/(2.5*std_val)

    return data_frames_n

# normalize data frames to mean value = 0 and divide by 2.5 standard deviation 
def NormlizeDataFrames2D(data_frames): 
 
    num_frames_all = len(data_frames)
    frames_dim = len(data_frames[0])

    data_frames_n = np.zeros((num_frames_all,frames_dim,2))

    #  normalize data frames to mean value = 0 and divide by 2.5 standard deviation 
    mean_val_0 = np.mean(data_frames[:,:, 0])
    std_val_0 = np.std(data_frames[:,:, 0])
    mean_val_1 = np.mean(data_frames[:,:, 1])
    std_val_1 = np.std(data_frames[:,:, 1])
    data_frames_n[:,:, 0] = data_frames[:,:, 0]-mean_val_0
    data_frames_n[:,:, 0] = data_frames_n[:,:, 0]/(2.5*std_val_0)
    data_frames_n[:,:, 1] = data_frames[:,:, 1]-mean_val_1
    data_frames_n[:,:, 1] = data_frames_n[:,:, 1]/(2.5*std_val_1)

    return data_frames_n

# calculate power in watts from dbm
def DBmToPowWatt(data_frames_dbm): 
 
    num_frames_all = len(data_frames_dbm)
    frames_dim = len(data_frames_dbm[0])

    data_frames_watt = np.zeros((num_frames_all,frames_dim))

    # calculate power in watt from dbm: pow_watt = 10**((pow_dbm-30)/10)
    for i in range (0,num_frames_all):
        data_frames_watt[i][:] = 10**((data_frames_dbm[i][:]-30.0)/10.0)

    return data_frames_watt

# calculate dbm from power in watts
def PowWattTodBm(data_frames_watt): 
 
    num_frames_all = len(data_frames_watt)
    frames_dim = len(data_frames_watt[0])

    data_frames_dbm = np.zeros((num_frames_all,frames_dim))

    # calculate power in dbm from watts: pow_dbm = 10*log(pow_watt) +30
    for i in range (0,num_frames_all):
        data_frames_dbm[i][:] = 10*np.log(data_frames_watt[i][:]) + 30.0

    return data_frames_dbm

# Integrate data frames within num_integration_intervalls, modes: 'integral' or 'average'
def IntegrateDataFrames(data_frames,num_integration_intervalls,mode='average'): 
 
    #from scipy.integrate import simps

    num_frames_all = len(data_frames)
    frames_dim = len(data_frames[0])

    # calculate the frame dimension for the integration on the frame intervalls
    frames_dim_integration = int(frames_dim/num_integration_intervalls)
    #print('frames_dim_integration',frames_dim_integration)
    xp_red = np.linspace(0., 1.0, frames_dim_integration)
    data_frames_integrated = np.zeros((num_frames_all,frames_dim_integration))
    data_frames_interval = np.zeros((num_frames_all,frames_dim_integration))

    # calculate integrals (or averages) for all frame intervalls
    for i in range (0,num_frames_all):
        for j in range (0,frames_dim_integration):
            data_frames_interval = data_frames[i][j*num_integration_intervalls:(j+1)*num_integration_intervalls]
            #print('frame interval',data_frames_interval)
            if mode == 'integral':
                #data_frames_integrated[i][j] = simps(data_frames_interval,x=xp_red)
                data_frames_integrated[i][j] = np.average(data_frames_interval)
            else:
                data_frames_integrated[i][j] = np.average(data_frames_interval)

    return data_frames_integrated

def read_init_param(module_dir):
    init_file_name = module_dir+'/init_param.json'
    if os.path.exists(init_file_name):
        # read init file
        file_name_lock = init_file_name+'.lock'
        lock_file_name = FileLock(file_name_lock)
        with lock_file_name:
            with open(init_file_name, 'r') as myfile:
                data=myfile.read()
        try:
            os.remove(file_name_lock)
        except:
            pass
        # parse file
        obj = json.loads(data)
        try:
            ai_type = str(obj['ai_type'])  #current values - DL,kNN
        except:
            ai_type = ''
        try:
            ai_type_dl = str(obj['ai_type_dl'])  #current values - DL_Class, DL_Reg, DL_Class_TFlite, DL_Reg_TFlite,
        except:
            ai_type_dl = ''        
        try:
            ai_type_knn = str(obj['ai_type_knn'])  #current values - kNN_Class, kNN_Reg
        except:
            ai_type_knn = ''        
        try:
            db_type = str(obj['db_type'])  #current values - PostgreSQL, SQlite, File
        except:
            db_type = ''
        try:
            num_threads_gprc_server = int(obj['num_threads_gprc_server'])  
        except:
            num_threads_gprc_server = 1
    else:
        ai_type = ''
        ai_type_dl = ''
        ai_type_knn = ''  
        db_type = ''
        num_threads_gprc_server = 1
    return ai_type,ai_type_dl,ai_type_knn,db_type,num_threads_gprc_server


def set_ref_data_info_file(module_dir,dataref,dataref_info,dataref_filename,dataref_status,
                           dataref_userid,dataref_unitname,dataref_chname,dataref_freq,
                           dataref_bw,dataref_sw,dataref_eqdetailspath,dataref_eqdetails):
    
    module = 'set_ref_data_info_file '

    # set data path 
    #data_file_path = module_dir+"/data/"
    if is_docker():
        data_file_path = "/filedata/"  #using docker volumes to support docker swarm
    else:
        data_file_path = module_dir+"/data/"

    data_ref_file_name = 'data_ref_info_'+dataref+'.json'

    # data file name
    data_ref_file =  data_file_path+data_ref_file_name

    logging_text = 'dataref file:'+data_ref_file
    write_log_api_file(module_dir,module,logging_text)

    now = datetime.now()
    dataref_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
    # storing all global data in dictionary
    ref_data_info = {"dataref": dataref,                        # unique reference of data as npz file
                     "dataref_info": dataref_info,              # information on data stored as npz file
                     "dataref_datetime": dataref_datetime,      # datetime when data info was stored 
                     "dataref_filename": dataref_filename,      # filename where data have been stored in npz file               
                     "dataref_status": dataref_status,          # status of data Values: (Active, Notactive), any other value will be interpreted as Notactive
                     "dataref_userid": dataref_userid,          # id of the user who created this data 
                     "dataref_unitname": dataref_unitname,      # additional data:  Unit name         
                     "dataref_chname": dataref_chname,          # additional data:  Channel name          
                     "dataref_freq": dataref_freq,              # additional data:  Frequency in Hz     
                     "dataref_bw": dataref_bw,                  # additional data:  Bandwidth in Hz
                     "dataref_sw": dataref_sw,                  # additional data:  Stepwidth in Hz
                     "dataref_eqdetailspath": dataref_eqdetailspath,  # additional data:  File path (directory+filemane+extension) to equipment details file 
                     "dataref_eqdetails": dataref_eqdetails
                     }

    logging_text = 'dataref info:'+str(ref_data_info)
    write_log_api_file(module_dir,module,logging_text)

    SetRefDataInfoOK = True
    if os.path.exists(data_file_path):
        # write data ref file
        try: 
            file_name_lock = data_ref_file+'.lock'
            lock_file_file = FileLock(file_name_lock)
            with lock_file_file:
                with open(data_ref_file, 'w') as myfile:
                    json.dump(ref_data_info, myfile)
            try:
                os.remove(file_name_lock)
            except:
                pass
            logging_text = 'ref data info file '+data_ref_file+' created'
            write_log_api_file(module_dir,module,logging_text)
        except:
            SetRefDataInfoOK = False
            logging_text = 'ref data info file '+data_ref_file+' could not be created'
            write_log_api_file(module_dir,module,logging_text)

    return SetRefDataInfoOK

def get_ref_data_info_file(module_dir, status_in, dataref_in, userid_in):  

    module = 'get_ref_data_info_file '

    # set data path 
    #data_file_path = module_dir+"/data/"
    if is_docker():
        data_file_path = "/filedata/"  #using docker volumes to support docker swarm
    else:
        data_file_path = module_dir+"/data/"

    # get all file in data directory 
    file_name_list = os.listdir(data_file_path)

    logging_text = 'file_name_list '+str(file_name_list)
    write_log_api_file(module_dir,module,logging_text)

    ref_data_info_list = []
    dataref = ''
    dataref_info = ''
    dataref_datetime = ''  
    dataref_filename = ''
    dataref_status = ''
    dataref_userid = ''
    dataref_unitname = ''
    dataref_chname = ''  
    dataref_freq = 0
    dataref_bw = 0
    dataref_sw = 0
    dataref_eqdetailspath = ''
    dataref_eqdetails = ''
    GetRefDataInfoOK = True
    ref_data_info = {"dataref": dataref,                        # unique reference of data as npz file
                    "dataref_info": dataref_info,              # information on data stored as npz file
                    "dataref_datetime": dataref_datetime,      # datetime when data info was stored 
                    "dataref_filename": dataref_filename,      # filename where data have been stored in npz file               
                    "dataref_status": dataref_status,          # status of data Values: (Active, Notactive), any other value will be interpreted as Notactive
                    "dataref_userid": dataref_userid,          # id of the user who created this data 
                    "dataref_unitname": dataref_unitname,      # additional data:  Unit name         
                    "dataref_chname": dataref_chname,          # additional data:  Channel name          
                    "dataref_freq": dataref_freq,              # additional data:  Frequency in Hz     
                    "dataref_bw": dataref_bw,                  # additional data:  Bandwidth in Hz
                    "dataref_sw": dataref_sw,                  # additional data:  Stepwidth in Hz
                    "dataref_eqdetailspath": dataref_eqdetailspath,  # additional data:  File path (directory+filemane+extension) to equipment details file 
                    "dataref_eqdetails": dataref_eqdetails
                    }
    for file_name in file_name_list:
        if ("data_ref_info" in file_name)and(".json" in file_name)and(".lock" not in file_name):
            #logging_text = 'file_name '+str(file_name)
            #write_log_api_file(module_dir,module,logging_text)
            if os.path.exists(data_file_path+file_name):
                # read init file
                file_name_lock = data_file_path+file_name+'.lock'
                lock_file_name = FileLock(file_name_lock)
                with lock_file_name:
                    with open(data_file_path+file_name, 'r') as myfile:
                        data=myfile.read()
                try:
                    os.remove(file_name_lock)
                except:
                    pass
                # parse file
                obj = json.loads(data)
                try:
                    dataref = str(obj['dataref']) 
                except:
                    dataref = ''
                try:
                    dataref_info = str(obj['dataref_info']) 
                except:
                    dataref_info = ''        
                try:
                    dataref_datetime = str(obj['dataref_datetime']) 
                except:
                    dataref_datetime = ''        
                try:
                    dataref_filename = str(obj['dataref_filename']) 
                except:
                    dataref_filename = ''
                try:
                    dataref_status = str(obj['dataref_status']) 
                except:
                    dataref_status = ''
                try:
                    dataref_userid = str(obj['dataref_userid']) 
                except:
                    dataref_userid = ''
                try:
                    dataref_unitname = str(obj['dataref_unitname']) 
                except:
                    dataref_unitname = ''
                try:
                    dataref_chname = str(obj['dataref_chname']) 
                except:
                    dataref_chname = ''
                try:
                    dataref_freq = int(obj['dataref_freq']) 
                except:
                    dataref_freq = 0
                try:
                    dataref_bw = int(obj['dataref_bw']) 
                except:
                    dataref_bw = 0
                try:
                    dataref_sw = int(obj['dataref_sw']) 
                except:
                    dataref_sw = 0
                try:
                    dataref_eqdetailspath = str(obj['dataref_eqdetailspath']) 
                except:
                    dataref_eqdetailspath = ''
                try:
                    dataref_eqdetails = str(obj['dataref_eqdetails']) 
                except:
                    dataref_eqdetails = ''

                ref_data_info = {"dataref": dataref,                        # unique reference of data as npz file
                                "dataref_info": dataref_info,              # information on data stored as npz file
                                "dataref_datetime": dataref_datetime,      # datetime when data info was stored 
                                "dataref_filename": dataref_filename,      # filename where data have been stored in npz file               
                                "dataref_status": dataref_status,          # status of data Values: (Active, Notactive), any other value will be interpreted as Notactive
                                "dataref_userid": dataref_userid,          # id of the user who created this data 
                                "dataref_unitname": dataref_unitname,      # additional data:  Unit name         
                                "dataref_chname": dataref_chname,          # additional data:  Channel name          
                                "dataref_freq": dataref_freq,              # additional data:  Frequency in Hz     
                                "dataref_bw": dataref_bw,                  # additional data:  Bandwidth in Hz
                                "dataref_sw": dataref_sw,                  # additional data:  Stepwidth in Hz
                                "dataref_eqdetailspath": dataref_eqdetailspath,  # additional data:  File path (directory+filemane+extension) to equipment details file 
                                "dataref_eqdetails": dataref_eqdetails
                                }

                #logging_text = 'ref_data_info '+str(ref_data_info)
                #write_log_api_file(module_dir,module,logging_text)

                append_record = 0
                if (status_in == "ALL")and(dataref_in == "ALL")and(userid_in == "ALL"):
                    append_record = 1
                elif (status_in == "ALL")and(dataref_in == "ALL"):
                    if dataref_userid == userid_in:
                        append_record = 1
                elif (status_in == "ALL")and(userid_in == "ALL"):
                    if dataref == dataref_in:
                        append_record = 1
                elif (dataref_in == "ALL")and(userid_in == "ALL"):
                    if status_in == dataref_status:
                        append_record = 1
                elif (status_in == "ALL"):
                    if (userid_in == dataref_userid)and(dataref_in == dataref):
                        append_record = 1
                elif (dataref_in == "ALL"):
                    if (userid_in == dataref_userid)and(status_in == dataref_status):
                        append_record = 1
                elif (userid_in == "ALL"):
                    if (dataref_in == dataref)and(status_in == dataref_status):
                        append_record = 1

                if append_record == 1:
                    ref_data_info_list.append(ref_data_info)
            else:
                GetRefDataInfoOK = False
    
    logging_text = 'ref_data_info_list '+str(ref_data_info_list)
    write_log_api_file(module_dir,module,logging_text)

    return GetRefDataInfoOK,ref_data_info_list

def activate_ref_data_file(module_dir, dataref_list, status_in):  

    module = 'activate_ref_data_file '

    # set data path 
    if is_docker():
        data_file_path = "/filedata/"  #using docker volumes to support docker swarm
    else:
        data_file_path = module_dir+"/data/"

    # get all file in data directory 
    file_name_list = os.listdir(data_file_path)

    logging_text = 'file_name_list '+str(file_name_list)
    write_log_api_file(module_dir,module,logging_text)

    ref_data_info_list = []
    ActDataRefOK = True
    num_files_updated = 0
    for i in range (0,len(dataref_list)):
        for file_name in file_name_list:
            if ("data_ref_info" in file_name)and(".json" in file_name)and(".lock" not in file_name) \
                and(dataref_list[i] in file_name):   # loop over all files defined by dataref_list
                logging_text = 'file_name '+str(file_name)
                write_log_api_file(module_dir,module,logging_text)
                if os.path.exists(data_file_path+file_name):
                    # read init file
                    file_name_lock = data_file_path+file_name+'.lock'
                    lock_file_name = FileLock(file_name_lock)
                    with lock_file_name:
                        with open(data_file_path+file_name, 'r') as myfile:
                            data=myfile.read()
                    try:
                        os.remove(file_name_lock)
                    except:
                        pass
                    # parse file
                    obj = json.loads(data)
                    try:
                        dataref = str(obj['dataref']) 
                    except:
                        dataref = ''
                    try:
                        dataref_info = str(obj['dataref_info']) 
                    except:
                        dataref_info = ''        
                    try:
                        dataref_datetime = str(obj['dataref_datetime']) 
                    except:
                        dataref_datetime = ''        
                    try:
                        dataref_filename = str(obj['dataref_filename']) 
                    except:
                        dataref_filename = ''
                    try:
                        dataref_status = str(obj['dataref_status']) 
                    except:
                        dataref_status = ''
                    try:
                        dataref_userid = str(obj['dataref_userid']) 
                    except:
                        dataref_userid = ''
                    try:
                        dataref_unitname = str(obj['dataref_unitname']) 
                    except:
                        dataref_unitname = ''
                    try:
                        dataref_chname = str(obj['dataref_chname']) 
                    except:
                        dataref_chname = ''
                    try:
                        dataref_freq = int(obj['dataref_freq']) 
                    except:
                        dataref_freq = 0
                    try:
                        dataref_bw = int(obj['dataref_bw']) 
                    except:
                        dataref_bw = 0
                    try:
                        dataref_sw = int(obj['dataref_sw']) 
                    except:
                        dataref_sw = 0
                    try:
                        dataref_eqdetailspath = str(obj['dataref_eqdetailspath']) 
                    except:
                        dataref_eqdetailspath = ''
                    try:
                        dataref_eqdetails = str(obj['dataref_eqdetails']) 
                    except:
                        dataref_eqdetails = ''
                else:
                    dataref = ''
                    dataref_info = ''
                    dataref_datetime = ''  
                    dataref_filename = ''
                    dataref_status = ''
                    dataref_userid = ''
                    dataref_unitname = ''
                    dataref_chname = ''  
                    dataref_freq = 0
                    dataref_bw = 0
                    dataref_sw = 0
                    dataref_eqdetailspath = ''
                    dataref_eqdetails = ''
                    ActDataRefOK = False

                ref_data_info = {"dataref": dataref,                       # unique reference of data as npz file
                                "dataref_info": dataref_info,              # information on data stored as npz file
                                "dataref_datetime": dataref_datetime,      # datetime when data info was stored 
                                "dataref_filename": dataref_filename,      # filename where data have been stored in npz file               
                                "dataref_status": status_in,               # status of data Values: (Active, Notactive), any other value will be interpreted as Notactive
                                "dataref_userid": dataref_userid,          # id of the user who created this data 
                                "dataref_unitname": dataref_unitname,      # additional data:  Unit name         
                                "dataref_chname": dataref_chname,          # additional data:  Channel name          
                                "dataref_freq": dataref_freq,              # additional data:  Frequency in Hz     
                                "dataref_bw": dataref_bw,                  # additional data:  Bandwidth in Hz
                                "dataref_sw": dataref_sw,                  # additional data:  Stepwidth in Hz
                                "dataref_eqdetailspath": dataref_eqdetailspath,  # additional data:  File path (directory+filemane+extension) to equipment details file 
                                "dataref_eqdetails": dataref_eqdetails
                                }

                #logging_text = 'ref_data_info '+str(ref_data_info)
                #write_log_api_file(module_dir,module,logging_text)

                ActDataRefOK = True
                try: 
                    file_name_lock = data_file_path+file_name+'.lock'
                    lock_file_name = FileLock(file_name_lock)
                    with lock_file_name:
                        with open(data_file_path+file_name, 'w') as myfile:
                            json.dump(ref_data_info, myfile)
                    try:
                        os.remove(file_name_lock)
                    except:
                        pass
                    num_files_updated = num_files_updated +1
                    logging_text = 'ref data info file '+file_name+' updated'
                    write_log_api_file(module_dir,module,logging_text)
                except:
                    ActDataRefOK = False
                    logging_text = 'ref data info file '+file_name+' could not be created'
                    write_log_api_file(module_dir,module,logging_text)

    return ActDataRefOK,num_files_updated

def delete_ref_data_file(module_dir, dataref_list):  

    module = 'delete_ref_data_file '
    
    # set data path 
    #data_file_path = module_dir+"/data/"
    if is_docker():
        data_file_path = "/filedata/"  #using docker volumes to support docker swarm
    else:
        data_file_path = module_dir+"/data/"

    # get all file in data directory 
    file_name_list = os.listdir(data_file_path)

    logging_text = 'file_name_list '+str(file_name_list)
    write_log_api_file(module_dir,module,logging_text)

    ai_module = get_module_from_dir(module_dir)
    logging_text = 'ai_module:'+ai_module
    write_log_api_file(module_dir,module,logging_text) 
    train_dir_knn = set_train_dir(ai_module,'knn')
    logging_text = 'train_dir_knn:'+train_dir_knn
    write_log_api_file(module_dir,module,logging_text)
    train_dir_dl = set_train_dir(ai_module,'dl')
    logging_text = 'train_dir_dl:'+train_dir_dl
    write_log_api_file(module_dir,module,logging_text)

    DelDataRefOK = True
    num_files_deleted = 0
    for i in range (0,len(dataref_list)):
        for file_name in file_name_list:
            if ("data_ref_info" in file_name)and(".json" in file_name)and(".lock" not in file_name) \
                and(dataref_list[i] in file_name):   # loop over all files defined by dataref_list
                #logging_text = 'file_name '+str(file_name)
                #write_log_api_file(module_dir,module,logging_text)
                if os.path.exists(data_file_path+file_name):
                    file_name_lock = data_file_path+file_name+'.lock'
                    lock_file_name = FileLock(file_name_lock)
                    with lock_file_name:
                        try: 
                            os.remove(data_file_path+file_name)
                            num_files_deleted = num_files_deleted + 1
                            logging_text = 'file '+str(data_file_path+file_name)+ ' deleted'
                            write_log_api_file(module_dir,module,logging_text)
                        except:
                            pass
                            #DelDataRefOK = False
                    try:
                        os.remove(file_name_lock)
                    except:
                        pass

    #remove the sub diretrories with its content for knn
    for i in range (0,len(dataref_list)):
        data_ref = dataref_list[i]
        train_sub_dir = train_dir_knn+data_ref+'/'
        try:
            shutil.rmtree(train_sub_dir)
            logging_text = 'train_sub_dir '+train_sub_dir+' removed'
            write_log_api_file(module_dir,module,logging_text)
        except:
            pass
            #DelKnnInitRefOK = False

    #remove the sub diretrories with its content for dl
    for i in range (0,len(dataref_list)):
        data_ref = dataref_list[i]
        train_sub_dir = train_dir_dl+data_ref+'/'
        try:
            shutil.rmtree(train_sub_dir)
            logging_text = 'train_sub_dir '+train_sub_dir+' removed'
            write_log_api_file(module_dir,module,logging_text)
        except:
            pass
            #DelKnnInitRefOK = False

    return DelDataRefOK,num_files_deleted

def set_knn_init_file(module_dir,knninit_ref,knninit_info,knninit_info_details,
                      neigh_file_name,labels_file_name,min_file_name,max_file_name,
                      softmax_file_name,dataref_list,knninit_userid,knn_param_id,sv_param_id):
    
    module = 'set_knn_init_file '

    # set data path 
    #knn_init_file_path = module_dir+"/data/"
    if is_docker():
        knn_init_file_path = "/filedata/"  #using docker volumes to support docker swarm
    else:
        knn_init_file_path = module_dir+"/data/"
    knn_init_file_name = 'knn_init_info_'+knninit_ref+'.json'

    # knn init file name
    knn_init_file =  knn_init_file_path+knn_init_file_name

    logging_text = 'knn init file:'+knn_init_file
    write_log_api_file(module_dir,module,logging_text)

    now = datetime.now()
    knninit_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
    # storing all global data in dictionary
    knn_init_info = {"knninit_ref": knninit_ref,                # unique reference of initialized knn classifier
                     "knninit_info": knninit_info,              # info for initialized knn classifier
                     "knninit_info_details": knninit_info_details,  # details for initialized knn classifier
                     "knninit_datetime": knninit_datetime,
                     "knninit_userid":knninit_userid,
                     "neigh_file_name": neigh_file_name,        # file name of initialized knn classifier
                     "labels_file_name": labels_file_name,      # file name of labels used for knn training
                     "min_file_name": min_file_name,            # file name of minima for normalization
                     "max_file_name": max_file_name,            # file name of maxima for normalization
                     "softmax_file_name": softmax_file_name,    # file name of for softmax data
                     "data_ref_list": str(dataref_list),        # list of data refs used for this knn classifier
                     "knn_param_id": knn_param_id,              # knn_par_id of local knn parameters
                     "sv_param_id": sv_param_id                 # sv_par_id of local service specific parameters (for example savd_param_id)
                     }

    logging_text = 'knninit_ref:'+str(knninit_ref)
    write_log_api_file(module_dir,module,logging_text)

    SetInitInfoOK = True
    if os.path.exists(knn_init_file_path):
        try: 
            file_name_lock = knn_init_file+'.lock'
            lock_file_name = FileLock(file_name_lock)
            with lock_file_name:
                with open(knn_init_file, 'w') as myfile:
                    json.dump(knn_init_info, myfile)
            try:
                os.remove(file_name_lock)
            except:
                pass
            logging_text = 'knn init info file '+knninit_ref+' created'
            write_log_api_file(module_dir,module,logging_text)
        except:
            SetInitInfoOK = False
            logging_text = 'knn init info file '+knninit_ref+' could not be created'
            write_log_api_file(module_dir,module,logging_text)

    return SetInitInfoOK

def get_knn_init_info_file(module_dir, knn_init_ref_list_str, userid_in):  

    module = 'get_knn_init_info_file '

    # set data path 
    #knn_init_file_path = module_dir+"/data/"
    if is_docker():
        knn_init_file_path = "/filedata/"  #using docker volumes to support docker swarm
    else:
        knn_init_file_path = module_dir+"/data/"

    knn_init_ref_list_str = knn_init_ref_list_str.replace("[","")
    knn_init_ref_list_str = knn_init_ref_list_str.replace("]","")
    knn_init_ref_list_str = knn_init_ref_list_str.replace("'","")
    knn_init_ref_list_str = knn_init_ref_list_str.replace('"','')
    knn_init_ref_list_str = knn_init_ref_list_str.replace(" ","")
    knninit_ref_in_list = knn_init_ref_list_str.split(",")

    logging_text = 'knninit_ref_in_list '+str(knninit_ref_in_list)
    write_log_api_file(module_dir,module,logging_text)

    logging_text = 'userid_in '+str(userid_in)
    write_log_api_file(module_dir,module,logging_text)

    # get all file in data directory 
    file_name_list = os.listdir(knn_init_file_path)

    logging_text = 'file_name_list '+str(file_name_list)
    write_log_api_file(module_dir,module,logging_text)

    # set knninit_ref_in to all for emtpy input list
    knninit_ref_in = ""
    if (len(knninit_ref_in_list) == 0):
        knninit_ref_in = "ALL"
    # set knninit_ref_in to all for input list = ''
    if (len(knninit_ref_in_list) == 1):
        if (knninit_ref_in_list[0] == ''):
            knninit_ref_in = "ALL"

    knn_init_data_info_list = []
    knninit_ref = ''
    knninit_info = ''
    knninit_info_details = ''  
    knninit_datetime = ''
    knninit_userid = ''
    neigh_file_name = ''
    labels_file_name = ''
    min_file_name = ''
    max_file_name = ''
    softmax_file_name = ''  
    data_ref_list = '' 
    knn_param_id = ''
    sv_param_id = ''
    knn_init_data_info = {"knninit_ref": knninit_ref,                        
        "knninit_info": knninit_info,              
        "knninit_info_details": knninit_info_details,      
        "knninit_datetime": knninit_datetime,
        "knninit_userid":knninit_userid,
        "neigh_file_name": neigh_file_name,      
        "labels_file_name": labels_file_name,         
        "min_file_name": min_file_name,         
        "max_file_name": max_file_name,     
        "softmax_file_name": softmax_file_name,
        "data_ref_list": data_ref_list,
        "knn_param_id": knn_param_id,            
        "sv_param_id": sv_param_id     
        }
    GetKnnInitDataInfoOK = False
    for file_name in file_name_list:
        if ("knn_init_info" in file_name)and("json" in file_name)and(".lock" not in file_name):
            #logging_text = 'file_name '+str(file_name)
            #write_log_api_file(module_dir,module,logging_text)
            if os.path.exists(knn_init_file_path+file_name):
                # read init file
                file_name_lock = knn_init_file_path+file_name+'.lock'
                lock_file_name = FileLock(file_name_lock)
                with lock_file_name:
                    with open(knn_init_file_path+file_name, 'r') as myfile:
                        data=myfile.read()
                try:
                    os.remove(file_name_lock)
                except:
                    pass
                # parse file
                obj = json.loads(data)
                try:
                    knninit_ref = str(obj['knninit_ref']) 
                except:
                    knninit_ref = ''
                try:
                    knninit_info = str(obj['knninit_info']) 
                except:
                    knninit_info = ''        
                try:
                    knninit_info_details = str(obj['knninit_info_details']) 
                except:
                    knninit_info_details = ''  
                try:
                    knninit_datetime = str(obj['knninit_datetime']) 
                except:
                    knninit_datetime = ''     
                try:
                    knninit_userid = str(obj['knninit_userid']) 
                except:
                    knninit_userid = ''   
                try:
                    neigh_file_name = str(obj['neigh_file_name']) 
                except:
                    neigh_file_name = ''
                try:
                    labels_file_name = str(obj['labels_file_name']) 
                except:
                    labels_file_name = ''
                try:
                    min_file_name = str(obj['min_file_name']) 
                except:
                    min_file_name = ''
                try:
                    max_file_name = str(obj['max_file_name']) 
                except:
                    max_file_name = ''
                try:
                    softmax_file_name = str(obj['softmax_file_name']) 
                except:
                    softmax_file_name = ''
                try:
                    data_ref_list = str(obj['data_ref_list']) 
                except:
                    data_ref_list = ''
                try:
                    knn_param_id = str(obj['knn_param_id']) 
                except:
                    knn_param_id = ''
                try:
                    sv_param_id = str(obj['sv_param_id']) 
                except:
                    sv_param_id = ''

                if knninit_ref != '':
                    knn_init_data_info = {"knninit_ref": knninit_ref,                        
                                    "knninit_info": knninit_info,              
                                    "knninit_info_details": knninit_info_details,   
                                    "knninit_datetime": knninit_datetime,
                                    "knninit_userid":knninit_userid,   
                                    "neigh_file_name": neigh_file_name,      
                                    "labels_file_name": labels_file_name,         
                                    "min_file_name": min_file_name,         
                                    "max_file_name": max_file_name,     
                                    "softmax_file_name": softmax_file_name,
                                    "data_ref_list": data_ref_list,       
                                    "knn_param_id": knn_param_id,  
                                    "sv_param_id": sv_param_id
                                    }

                    logging_text = 'knn_init_data_info '+str(knn_init_data_info)
                    write_log_api_file(module_dir,module,logging_text)

                    #print('knninit_ref_in :',knninit_ref_in)
                    #print('userid_in :',userid_in)
                    #print('knninit_ref_in_list :',knninit_ref_in_list)
                    #print('knninit_ref :',knninit_ref)

                    append_record = 0
                    if (knninit_ref_in == "ALL")and(userid_in == "ALL"):
                        append_record = 1
                    if (knninit_ref_in == "ALL"):
                        if userid_in == knninit_userid:
                            append_record = 1
                    if (userid_in == "ALL"):
                        if knninit_ref in knninit_ref_in_list:
                            append_record = 1
                    if (userid_in == knninit_userid):
                        if knninit_ref in knninit_ref_in_list:
                            append_record = 1

                    if append_record == 1:
                        knn_init_data_info_list.append(knn_init_data_info)
                        GetKnnInitDataInfoOK = True

    logging_text = 'knn_init_data_info_list '+str(knn_init_data_info_list)
    write_log_api_file(module_dir,module,logging_text)

    return GetKnnInitDataInfoOK,knn_init_data_info_list

def delete_knn_info_data_file(module_dir,knninit_ref_list,train_dir_knn):  

    module = 'delete_knn_info_data_file '
    # set data path 
    #knn_init_file_path = module_dir+"/data/"
    if is_docker():
        knn_init_file_path = "/filedata/"  #using docker volumes to support docker swarm
    else:
        knn_init_file_path = module_dir+"/data/"

    # get all file in data directory 
    file_name_list = os.listdir(knn_init_file_path)

    logging_text = 'file_name_list '+str(file_name_list)
    write_log_api_file(module_dir,module,logging_text)

    DelKnnInitRefOK = True
    num_files_deleted = 0
    for i in range (0,len(knninit_ref_list)):
        for file_name in file_name_list:
            if ("knn_init_info" in file_name)and(".json" in file_name)and(".lock" not in file_name) \
                and(knninit_ref_list[i] in file_name):  
                logging_text = 'deleted knn_init_info file_name: '+str(file_name)
                write_log_api_file(module_dir,module,logging_text)
                if os.path.exists(knn_init_file_path+file_name):
                    file_name_lock = knn_init_file_path+file_name+'.lock'
                    lock_file_name = FileLock(file_name_lock)
                    with lock_file_name:
                        try: 
                            os.remove(knn_init_file_path+file_name)
                            num_files_deleted = num_files_deleted + 1
                        except:
                            pass
                            #DelDataRefOK = False
                    try:
                        os.remove(file_name_lock)
                    except:
                        pass

    # get list of files in train_data_dir
    file_name_list_train = os.listdir(train_dir_knn)
    train_num_files_deleted = 0
    for i in range (0,len(knninit_ref_list)):
        for file_name in file_name_list_train:
            if knninit_ref_list[i] in file_name:   
                #logging_text = 'file_name '+str(file_name)
                #write_log_api_file(module_dir,module,logging_text)
                if os.path.exists(train_dir_knn+file_name):
                    try: 
                        os.remove(train_dir_knn+file_name)
                        train_num_files_deleted = train_num_files_deleted + 1
                    except:
                        pass
                        #DelKnnInitRefOK = False

    return DelKnnInitRefOK,num_files_deleted

def delete_dl_info_data_file(module_dir,dlinit_ref_list):  

    module = 'delete_dl_info_data_file '

    # set data path 
    #dl_init_file_path = module_dir+"/data/"
    if is_docker():
        dl_init_file_path = "/filedata/"  #using docker volumes to support docker swarm
    else:
        dl_init_file_path = module_dir+"/data/"

    # get all file in data directory 
    file_name_list = os.listdir(dl_init_file_path)

    logging_text = 'file_name_list '+str(file_name_list)
    write_log_api_file(module_dir,module,logging_text)

    DelDlInitRefOK = True
    num_files_deleted = 0
    for i in range (0,len(dlinit_ref_list)):
        for file_name in file_name_list:
            if ("dl_init_info" in file_name)and(".json" in file_name)and(".lock" not in file_name) \
                and(dlinit_ref_list[i] in file_name):  
                logging_text = 'deleted dl_init_info file_name: '+str(file_name)
                write_log_api_file(module_dir,module,logging_text)
                if os.path.exists(dl_init_file_path+file_name):
                    file_name_lock = dl_init_file_path+file_name+'.lock'
                    lock_file_name = FileLock(file_name_lock)
                    with lock_file_name:
                        try: 
                            os.remove(dl_init_file_path+file_name)
                            num_files_deleted = num_files_deleted + 1
                        except:
                            pass
                            #DelDataRefOK = False
                    try:
                        os.remove(file_name_lock)
                    except:
                        pass

    return DelDlInitRefOK,num_files_deleted

def set_dl_init_info_file(module_dir,dl_init_ref,DL_InfoStr,FreqBwModelName,FreqBwModelVersion,IQModelClasses,UserId,dl_param_id):
    
    module = 'set_dl_init_info_file '

    # set data path 
    #data_file_path = module_dir+"/data/"
    if is_docker():
        data_file_path = "/filedata/"  #using docker volumes to support docker swarm
    else:
        data_file_path = module_dir+"/data/"

    dl_init_info_file_name = 'dl_init_info_'+dl_init_ref+'.json'

    # data file name
    dl_init_info_file =  data_file_path+dl_init_info_file_name

    logging_text = 'dl_init_info file:'+dl_init_info_file
    write_log_api_file(module_dir,module,logging_text)

    now = datetime.now()
    dlinit_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
    # storing all global data in dictionary
    dl_init_info = {"dlinit_ref": dl_init_ref,                       # unique reference of dl init data created by dl_init
                    "dlinit_info": DL_InfoStr,                       # information on this dl initialisation
                    "dlinit_datetime": dlinit_datetime,              # datetime when data info was stored 
                    "dlinit_tsmodelname": FreqBwModelName,           # name of tensorflow serving model           
                    "dlinit_tsmodelversion": FreqBwModelVersion,     # version of tensorflow serving model
                    "dlinit_tsmodelclasses": IQModelClasses,  # classes of tensorflow serving model
                    "dlinit_dl_param_id": dl_param_id         # this is the ID of the local dl parameters which are used for dl recognition (reject thres, num results, ...), If DlParamId = '' dl init parameters will be used , If DlParamId = '0' global parameters will be used
                     }

    logging_text = 'dl_init_info:'+str(dl_init_info)
    write_log_api_file(module_dir,module,logging_text)

    SetDlInitInfoOK = True
    if os.path.exists(data_file_path):
        # write data ref file
        try: 
            file_name_lock = dl_init_info_file+'.lock'
            lock_file_name = FileLock(file_name_lock)
            with lock_file_name:
                with open(dl_init_info_file, 'w') as myfile:
                    json.dump(dl_init_info, myfile)
            try:
                os.remove(file_name_lock)
            except:
                pass
            logging_text = 'dl init info file '+dl_init_info_file+' created'
            write_log_api_file(module_dir,module,logging_text)
        except:
            SetDlInitInfoOK = False
            logging_text = 'dl init info file '+dl_init_info_file+' could not be created'
            write_log_api_file(module_dir,module,logging_text)

    return SetDlInitInfoOK

def get_dl_init_info_file(module_dir,dl_init_ref):
    
    module = 'get_dl_init_info_file '

    # set data path 
    dl_init_file_path = module_dir+"/data/"
    if is_docker():
        dl_init_file_path = "/filedata/"  #using docker volumes to support docker swarm
    else:
        dl_init_file_path = module_dir+"/data/"
    dl_init_info_file_name = 'dl_init_info_'+dl_init_ref+'.json'

    dl_init_info_list = []
    dlinit_ref = ''
    dlinit_info = ''
    dlinit_datetime = ''  
    dlinit_tsmodelname = ''
    dlinit_tsmodelversion = ''
    dlinit_tsmodelclasses = ''
    dl_param_id = ''

    dl_init_info = {"dlinit_ref": dlinit_ref,                           # unique reference of dl init data created by dl_init
                    "dlinit_info": dlinit_info,                         # information on this dl initialisation
                    "dlinit_datetime": dlinit_datetime,                 # datetime when data info was stored 
                    "dlinit_tsmodelname": dlinit_tsmodelname,           # name of tensorflow serving model           
                    "dlinit_tsmodelversion": dlinit_tsmodelversion,     # version of tensorflow serving model
                    "dlinit_tsmodelclasses": dlinit_tsmodelclasses,     # classes of tensorflow serving model
                    "dlinit_dl_param_id": dl_param_id         # this is the ID of the local dl parameters which are used for dl recognition (reject thres, num results, ...), If DlParamId = '' dl init parameters will be used , If DlParamId = '0' global parameters will be used
                     }

    GetDlInitDataInfoOK = False
    dl_init_data_info_list = []
    if os.path.exists(dl_init_file_path+dl_init_info_file_name):
        # read init file
        file_name_lock = dl_init_file_path+dl_init_info_file_name+'.lock'
        lock_file_name = FileLock(file_name_lock)
        with lock_file_name:
            with open(dl_init_file_path+dl_init_info_file_name, 'r') as myfile:
                data=myfile.read()
        try:
            os.remove(file_name_lock)
        except:
            pass
        # parse file
        obj = json.loads(data)
        try:
            dlinit_ref = str(obj['dlinit_ref']) 
        except:
            dlinit_ref = ''
        try:
            dlinit_info = str(obj['dlinit_info']) 
        except:
            dlinit_info = ''        
        try:
            dlinit_datetime = str(obj['dlinit_datetime']) 
        except:
            dlinit_datetime = ''  
        try:
            dlinit_tsmodelname = str(obj['dlinit_tsmodelname']) 
        except:
            dlinit_tsmodelname = ''     
        try:
            dlinit_tsmodelversion = str(obj['dlinit_tsmodelversion']) 
        except:
            dlinit_tsmodelversion = ''   
        try:
            dlinit_tsmodelclasses = str(obj['dlinit_tsmodelclasses']) 
        except:
            dlinit_tsmodelclasses = ''
        try:
            dl_param_id = str(obj['dlinit_dl_param_id']) 
        except:
            dl_param_id = ''

        if dlinit_ref != '':
            dl_init_info = {"dlinit_ref": dlinit_ref,                           # unique reference of dl init data created by dl_init
                            "dlinit_info": dlinit_info,                         # information on this dl initialisation
                            "dlinit_datetime": dlinit_datetime,                 # datetime when data info was stored 
                            "dlinit_tsmodelname": dlinit_tsmodelname,           # name of tensorflow serving model           
                            "dlinit_tsmodelversion": dlinit_tsmodelversion,     # version of tensorflow serving model
                            "dlinit_tsmodelclasses": dlinit_tsmodelclasses,      # classes of tensorflow serving model
                            "dlinit_dl_param_id": dl_param_id  # this is the ID of the local dl parameters which are used for dl recognition (reject thres, num results, ...), If DlParamId = '' dl init parameters will be used , If DlParamId = '0' global parameters will be used
                            }

        dl_init_data_info_list.append(dl_init_info)
        GetDlInitDataInfoOK = True

    logging_text = 'dl_init_data_info_list '+str(dl_init_data_info_list)
    write_log_api_file(module_dir,module,logging_text)

    return GetDlInitDataInfoOK,dl_init_data_info_list