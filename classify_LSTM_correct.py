# LSTM for phone recognition with input a-b lip
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
#from keras.models import Graph
from keras.layers import Dense
from keras.layers import LSTM
#from keras.layers import TimeDistributedDense
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import h5py
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import scipy.io as spio

#for nb_Cell in [10,15]:
workdir = '/media/liliu/Sony_LiLIU/linux_liliu_beifen/Lip_recognition/Hand_position_experiment_88/'

nb_Cell = 50
#print('nb_Cell = %d' % nb_Cell)
nb_epoch=  10
batchsize = 1
#nb_sbj = 10
#nb_w = 50
#nb_file = nb_sbj*nb_w

nb_file = 88
nb_file_train = int(0.7*nb_file)
nb_file_test = int(0.3*nb_file)
drop = 0.5
dim_input = 2
normalization_coeff = 1

# fix random seed for reproducibility
#numpy.random.seed(7)

#maxVal_IU,maxVal_MP,maxVal_SP,maxVal_GT,maxVal_FX = 6,5,5,5,8
maxVal_Phone = 5 #6
def to_categorical(seq_data,maxVal):
	category_data = numpy.zeros((len(seq_data),maxVal))
	for i in range(len(seq_data)):
		if seq_data[i] >= maxVal:
			category_data[i,maxVal-1] = 1.0
		else: 
			category_data[i,int(seq_data[i])-1]=1.0
	return category_data


def find_max(n):
	max_len = -1
	for i in range(n):
		filename = workdir +'data_combine_code_1D/data_'+str(i+1);
		data_in = pandas.read_csv(filename, usecols=[0], engine='python',header=None, sep=r"\s+")
		len_data = len(data_in)
		if max_len < len_data:
			max_len = len_data
	return max_len

max_len = find_max(nb_file)


def padding_zero(data_):
	len_data_ = len(data_)
	if len_data_ < max_len:
		add_zeros = numpy.zeros(max_len-len_data_,dtype=numpy.int)
		data_padding = numpy.concatenate((data_,add_zeros),axis = 0)
	else:
		data_padding = data_
	return data_padding


def padding_zero_one(data_):
	len_data_ = len(data_)
	if len_data_ < max_len:
		add_ones = numpy.ones(max_len-len_data_,dtype=numpy.int)
		data_padding = numpy.concatenate((data_,add_ones),axis = 0)
	else:
		data_padding = data_
	return data_padding

# def TimeDistributedDense(x, w, b=None, dropout=None,
#                            input_dim=None, output_dim=None, timesteps=None):
#     '''Apply y.w + b for every temporal slice y of x.
#     '''
#     if not input_dim:
#         # won't work with TensorFlow
#         input_dim = K.shape(x)[2]
#     if not timesteps:
#         # won't work with TensorFlow
#         timesteps = K.shape(x)[1]
#     if not output_dim:
#         # won't work with TensorFlow
#         output_dim = K.shape(w)[1]

#     if dropout:
#         # apply the same dropout pattern at every timestep
#         ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
#         dropout_matrix = K.dropout(ones, dropout)
#         expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
#         x *= expanded_dropout_matrix

#     # collapse time dimension and batch dimension together
#     x = K.reshape(x, (-1, input_dim))

#     x = K.dot(x, w)
#     if b:
#         x = x + b
#     # reshape to 3D tensor
#     x = K.reshape(x, (-1, timesteps, output_dim))
#     return x


def load_file(i):
#i = 0
	filename = workdir + 'data_combine_code_1D/data_'+str(i+1);
	data = pandas.read_csv(filename, engine='python',header=None, sep=r"\s+")
	data = data.values
	
	A = []
	for i in range(dim_input):
		A.append(padding_zero(data[:, i])/normalization_coeff)
			
	Phone = padding_zero_one(data[:,dim_input])
	len_data = max_len
	Phone = to_categorical(Phone,maxVal_Phone)
	
	A2 = []
	for a in A:
		A2.append(numpy.reshape(a,(1,len_data,1)))
	
	Phone = numpy.reshape(Phone, (1,len_data, maxVal_Phone))
	return A2, Phone


A2, Phone = load_file(0)
for i in range(nb_file-1):	
	Ak, Phone_k = load_file(i+1)
	for i in range(dim_input):
		A2[i] = numpy.concatenate((A2[i],Ak[i]), axis=0)	
	
	Phone = numpy.concatenate((Phone,Phone_k), axis=0)

def find_len(n):
	files_len = []
	for i in range(n):
		filename = workdir + 'data_combine_code_1D/data_'+str(i+1);
		data_in = pandas.read_csv(filename, usecols=[0], engine='python',header=None, sep=r"\s+")
		files_len.append(len(data_in))
	return files_len

files_len = find_len(nb_file)	

def classification_evaluation(Predict,Target):
	classification_score = 0
	for i in range(len(Predict)):
		if int(Predict[i]) == int(Target[i]):
			classification_score = classification_score+1
	classification_score = classification_score/float(len(Predict))
	return classification_score

a_trainScoreClassify1=[]
a_testScoreClassify1=[]

# here is to obtain the random number of the training set
mat = spio.loadmat('/media/liliu/Sony_LiLIU/linux_liliu_beifen/Lip_recognition/random_tab_88.mat', squeeze_me=True)
arr_index = mat['arr_index']
#arr_index = arr_index-1
print(arr_index)

train_A = []
for i in range(dim_input):
	train_A.append(A2[i][arr_index[0:nb_file_train]])
		
train_Phone = Phone[arr_index[0:nb_file_train]]

files_len = numpy.array(files_len)
files_len = files_len[arr_index]

trainX = numpy.concatenate(train_A, axis=2)
input_shape = dim_input

########## build the LSTM model
inputs = Input(shape=(None,input_shape),name = 'inputs')
lstm_out = LSTM(nb_Cell,return_sequences=True,)(inputs)
drop_out = Dropout(drop)(lstm_out)
output1 = Dense(maxVal_Phone,activation="softmax",name='output1')(drop_out)
#output1 = TimeDistributedDense(maxVal_Phone,activation="softmax",name='output1')(drop_out)
#output2 = TimeDistributedDense(maxVal_GT,activation="softmax",name='output2')(drop_out)
#output3 = TimeDistributedDense(maxVal_FX,activation="softmax",name='output3')(drop_out)
#model = Model(input=inputs, output = [output1,output2,output3])
model = Model(input = inputs, output = output1)
#model.compile(optimizer='rmsprop', loss = {'output1':'categorical_crossentropy','output2':'categorical_crossentropy','output3':'categorical_crossentropy'},loss_weights={'output1': 1., 'output2': 1., 'output3': 1.})
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy')	
model.fit(trainX,train_Phone, nb_epoch=nb_epoch,batch_size=1, verbose=2)
	
savefile = workdir + '/model_save/classify_LSTM_1out_nbC_'+str(nb_Cell)+'_nbE_'+str(nb_epoch)+'_nbTf_'+str(nb_file_train)+'.h5';
model.save(savefile)
#model = load_model(savefile)

trainPredict1= model.predict(trainX)	
index_trainPredict1 = numpy.argmax(trainPredict1, axis=2)
index_trainTarget1 = numpy.argmax(train_Phone, axis=2)

trainScoreClassify1 = 0

for i in range(len(index_trainPredict1)):	
	trainScoreClassify1 = trainScoreClassify1+classification_evaluation(index_trainPredict1[i,:], index_trainTarget1[i,:])

trainScoreClassify1 = trainScoreClassify1/float(len(index_trainPredict1));


#for index_filetest in arr_index[nb_file_train:]:
#for index_filetest in range(nb_file_test)+numpy.ones(nb_file_test)*nb_file_train:
for index_filetest in range(nb_file):
	test_A = []

	for i in range(dim_input):
		#len_file_i = files_len[arr_index[index_filetest]]
		len_file_i = files_len[index_filetest]	
		#print(len_file_i)
		test_A.append(A2[i][arr_index[index_filetest],0:len_file_i])
	
		test_Phone = Phone[arr_index[index_filetest],0:len_file_i]
	
	
		test_A[i] = numpy.reshape(test_A[i], (1,len_file_i, 1))
		
	test_Phone = numpy.reshape(test_Phone, (1,len_file_i, maxVal_Phone))
	testX = numpy.concatenate(test_A,axis=2)
	
	testPredict1 = model.predict(testX)
	index_testPredict1 = numpy.argmax(testPredict1, axis=2)
	index_testTarget1 = numpy.argmax(test_Phone, axis=2)
	
	testScoreClassify1 = 0
	for i in range(len(index_testPredict1)):	
		testScoreClassify1 = testScoreClassify1+classification_evaluation(index_testPredict1[i,:], index_testTarget1[i,:])
	
	testScoreClassify1 = testScoreClassify1/float(len(index_testPredict1));
	a_testScoreClassify1.append(testScoreClassify1)
	
a_testScoreClassify1 = numpy.array(a_testScoreClassify1)
print(a_testScoreClassify1)
print(len(a_testScoreClassify1))
avg_testScoreClassify1 = numpy.average(a_testScoreClassify1[nb_file_train:])


print('Train Score: %.2f percent' % (trainScoreClassify1*100))	
print('Test Score: %.2f percent' % (avg_testScoreClassify1*100))	
		
with open("classify_LSTM_lip.log", "a") as f:
	f.write("nbCell: %d --nbEpoch: %d --nbTrainf: %d --TrScCl1:%f --TeScCl1:%f --Dropout:%f\n" %(nb_Cell,nb_epoch,nb_file_train, trainScoreClassify1, avg_testScoreClassify1,drop))

plt.figure()
plt.plot(a_testScoreClassify1,'ro-')
plt.grid()

