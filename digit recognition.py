import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train , y_train) , (x_test , y_test)=keras.datasets.mnist.load_data()
#use the '/Applications/Python\ 3.11/Install\ Certificates.command' 
#command to bypass sslcertverification error


#scaling the data

x_train=x_train/255
x_test=x_test/255


# flatten the image array (6000,28,28)->(6000,28*28)


x_train_flat=x_train.reshape(len(x_train),28*28)
x_test_flat=x_test.reshape(len(x_test),28*28)

#building neural network


model=keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train_flat,y_train,epochs=5)

#model has been trained

#time to evaluate
model.evaluate(x_test_flat,y_test)

#predicting values from test data
y_predicted = model.predict(x_test_flat)
print(np.argmax(y_predicted[17]))

y_predicted_labels=[np.argmax for i in y_predicted]
cm=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
