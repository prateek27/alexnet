import tensorflow as tf
if __name__=='__main__':
    tf.enable_eager_execution()

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D,Dropout,Activation,BatchNormalization,MaxPool2D,Flatten,Dense

class Alexnet(tf.keras.Model):
    def __init__(self,classes):
        super(Alexnet,self).__init__()
        
        self.input_names = "None"
        #1st Conv Layer
        self.conv1 = Sequential()
        self.conv1.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),padding='valid'))
        self.conv1.add(Activation('relu'))
        self.conv1.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
        self.conv1.add(BatchNormalization())
        
        #2nd Conv Layer
        self.conv2 = Sequential()
        self.conv2.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding='same'))
        self.conv2.add(Activation('relu'))
        self.conv2.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
        self.conv2.add(BatchNormalization())
            
        #3rd Conv Layer    
        self.conv3 = Sequential()
        self.conv3.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same'))
        self.conv3.add(Activation('relu'))
        self.conv3.add(BatchNormalization())
        
        #4th Conv Layer
        self.conv4 = Sequential()
        self.conv4.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same'))
        self.conv4.add(Activation('relu'))
        self.conv4.add(BatchNormalization())
            
        #5th Conv Layer    
        self.conv5 = Sequential()
        self.conv5.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same'))
        self.conv5.add(Activation('relu'))
        self.conv5.add(BatchNormalization())
        self.conv5.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        
        #6th Layer  & 7th Layer
        self.fc = Sequential()
        self.fc.add(Flatten())
        self.fc.add(Dense(4096))
        self.fc.add(Activation('relu'))
        self.fc.add(Dropout(0.4))
        
        #8th Layer
        self.fc.add(Dense(4096))
        self.fc.add(Activation('relu'))
        self.fc.add(Dropout(0.4))
                           
        #9th Output Layer
        self.classification_layer = Sequential()
        self.classification_layer.add(Dense(classes))
    
    def call(self,x):
        #print(x.shape)
        conv1 = self.conv1(x)
        #print(conv1.shape)
        conv2 = self.conv2(conv1)
        #print(conv2.shape)
        conv3 = self.conv3(conv2)
        #print(conv3.shape)
        conv4 = self.conv4(conv3)
        #print(conv4.shape)
        conv5 = self.conv5(conv4)
        #print(conv5.shape)
        fc = self.fc(conv5)
        #print(fc.shape)
        out = self.classification_layer(fc)
        #print(out.shape)
        return out
        