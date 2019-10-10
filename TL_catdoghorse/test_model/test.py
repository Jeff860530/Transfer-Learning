import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image###
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator###
from keras.models import Model,load_model
from keras.optimizers import Adam
from keras import layers
from keras import models

# In[1]:
model = load_model('for_591jpg.h5')
model.summary()
# In[2]:
for i,layer in enumerate(model.layers):
  if i == 87:
      print(layer.input,'\n',layer.output,'\n',layer.input_shape,'\n',layer.output_shape)
  #print(i,layer.name)
# In[0]:
model.summary()
# In[0]:
model1.summary()
# In[3]:
for i in range(4):
  model.layers.pop()
# In[0]:
for i,layer in enumerate(model1.layers):
  print(i,layer.name)
# In[4]:
#x= models.Sequential()
x=GlobalAveragePooling2D()(model.layers[-5].output)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(512,activation='relu')(x) #dense layer 2
x=Dense(256,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation
# In[5]:
model1=Model(inputs=model.input,outputs=preds)
# In[6]:#check which layer is trainable 
for i,layer in enumerate(model1.layers):
    if layer.trainable==True:
        print(i,layer.name,'Ture')
    else:
        print(i,layer.name,'False')
# In[6]:    
for layer in model1.layers:
    layer.trainable=False
# In[6]:  
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model1.layers[:87]:
    layer.trainable=False
for layer in model1.layers[87:]:
    layer.trainable=True


# In[3]:
print('Number of trainable weights after  freezing the model1:', len(model1.trainable_weights))
# In[3]:
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) 
#included in our dependencies
train_generator=train_datagen.flow_from_directory('./test_train197jpg/',
 # this is where you specify the path to the main data folder
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)
# In[4]:
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
model1.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=10)
