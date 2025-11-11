from tensorflow import keras
from keras import layers

def build_mobilenet_v3_small(input_shape=(224,224,3), num_classes=37,
                             train_backbone=False, dropout=0.3, l2_reg=1e-5):
    base = keras.applications.MobileNetV3Small(
        include_top=False, input_shape=input_shape, weights="imagenet"
    )
    base.trainable = train_backbone

    inputs = keras.Input(shape=input_shape)
    x = keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    return keras.Model(inputs, outputs)
