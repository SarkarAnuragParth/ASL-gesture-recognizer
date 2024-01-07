import tensorflow as tf
import os
from data import data_builder
def train(args):
    train_data,val_data=data_builder(args)
    model=tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
    include_top=True,
    classes=36)
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy,metrics=[tf.keras.metrics.Accuracy,tf.keras.metrics.CategoricalAccuracy])
    model.fit(train_data,epochs=args.epochs,callbacks=[callback],verbose='auto',validation_data=val_data)
    model.save(os.path.join(args.path_to_model,'ASl_Classifier'))   
    
def test(args):
    train_data=data_builder(args)
    model=tf.keras.models.load_model(args.path_to_trained_model)
    model.evaluate(train_data)
    