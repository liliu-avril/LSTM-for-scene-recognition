import numpy
import scipy.io as spio
import scipy.io as sio

nb_file = 88  #476, 138
nb_file_train = int(0.8*nb_file)-1
nb_file_test = nb_file - nb_file_train

arr_index_ = numpy.arange(nb_file)
arr_index_even = arr_index_ [::2]
numpy.random.shuffle(arr_index_even)

arr_index_even[0:int(nb_file_train/2)]+numpy.ones(int(nb_file_train/2))
train_index = numpy.concatenate((arr_index_even[0:int(nb_file_train/2)],arr_index_even[0:int(nb_file_train/2)]+1), axis=0)
test_index = numpy.concatenate((arr_index_even[int(nb_file_train/2):],arr_index_even[int(nb_file_train/2):]+1), axis=0)
arr_index = numpy.concatenate((train_index,test_index), axis=0)
arr_index = numpy.array(arr_index)
print(arr_index)
sio.savemat('/media/liliu/Sony_LiLIU/linux_liliu_beifen/Lip_recognition/random_tab_88.mat',{'arr_index':arr_index})
