# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:24:29 2021

@author: prajw
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:19:33 2021

@author: prajw
"""

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.models import Model

inp = K.layers.Input([256, 256, 3], name = 'Input')


    
c1 = K.layers.Conv2D(16, (2,2), activation = 'relu', name = 'b1_1_conv1')(inp)
drop1 = K.layers.Dropout(0.25,  name = 'b1_1_drop1')(c1)
mp1 = K.layers.MaxPool2D((2,2),  name = 'b1_1_pool1')(drop1)
ic1 = K.layers.Conv2DTranspose((16), (2,2),  name = 'b1_1_inv_conv1')(drop1)

output1 = K.layers.Conv2D(3, (1, 1), activation='sigmoid', name = 'b1_output') (ic1)
    
    
c2_1 = K.layers.Conv2D(32, (2,2), activation = 'relu',  name = 'b2_1_conv1')(ic1)
drop2_1 = K.layers.Dropout(0.25,  name = 'b2_1_drop1')(c2_1)
mp2 = K.layers.MaxPool2D((2,2),  name = 'b2_1_pool1')(drop2_1)
ic2_1 = K.layers.Conv2DTranspose((32), (2,2),  name = 'b2_1_inv_conv1')(drop2_1)
drop2_2 = K.layers.Dropout(0.25,  name = 'b2_1_drop2')(ic2_1)
c2_2 = K.layers.Conv2D(16, (2,2), activation = 'relu',  name = 'b2_1_conv2')(drop2_2)
ic2_2 = K.layers.Conv2DTranspose((16), (2,2),  name = 'b2_1_inv_conv2')(c2_2)

output2 = K.layers.Conv2D(3, (1, 1), activation='sigmoid',  name = 'b2_ouptut1') (ic2_2)


    
#output = K.layers.concatenate([u9, c1], axis=3)
    
model = Model(inputs=[inp], outputs=[output1, output2])

model.summary()


