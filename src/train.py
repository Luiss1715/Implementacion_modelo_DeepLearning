import os, argparse, json
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks

AUTOTUNE = tf.data.AUTOTUNE

def load_ds(root, img_size=(224,224), batch_size=16, seed=42):
    ds_train = keras.utils.image_dataset_from_directory(
        os.path.join(root, "train"),
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        seed=seed
    )
    class_names = ds_train.class_names  

    ds_val = keras.utils.image_dataset_from_directory(
        os.path.join(root, "val"),
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        seed=seed,
        shuffle=False
    )

    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.2),
    ])

    ds_train = ds_train.map(lambda x,y: (aug(x, training=True), y),
                            num_parallel_calls=AUTOTUNE)

    return ds_train.prefetch(AUTOTUNE), ds_val.prefetch(AUTOTUNE), class_names

def build_model(num_classes,
                input_shape=(224,224,3),
                dropout=0.3,
                l2_reg=1e-5,
                train_backbone=False,
                backbone="mobilenet_v3_small"):
    if backbone == "mobilenet_v3_small":
        Base = keras.applications.MobileNetV3Small
        preprocess = keras.applications.mobilenet_v3.preprocess_input
    elif backbone == "efficientnet_b0":
        Base = keras.applications.EfficientNetB0
        preprocess = keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Backbone no soportado: {backbone}")

    base = Base(include_top=False, input_shape=input_shape, weights="imagenet")
    base.trainable = train_backbone

    inputs = keras.Input(shape=input_shape)
    x = preprocess(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    model = keras.Model(inputs, outputs)
    return model


def get_backbone(model):
    for lyr in model.layers:
        if isinstance(lyr, keras.Model) and len(lyr.layers) > 10:
            return lyr
    raise ValueError("No se encontró el backbone dentro del modelo.")

def main(args):
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)

    ds_train, ds_val, class_names = load_ds(args.data_dir, batch_size=args.batch_size)
    num_classes = len(class_names)

    #Baseline 
    model = build_model(num_classes=num_classes, train_backbone=False,
                        dropout=args.dropout, l2_reg=args.l2)
    model.compile(optimizer=keras.optimizers.Adam(args.lr_base),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    os.makedirs("runs", exist_ok=True)
    cbs = [
        callbacks.ModelCheckpoint("runs/best.keras", monitor="val_accuracy", save_best_only=True),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        callbacks.CSVLogger("runs/train_log.csv", append=False)
    ]

    history = model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, callbacks=cbs)


    with open("runs/class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # Fine-tuning 
    if args.finetune:
        
        model = keras.models.load_model("runs/best.keras")

        backbone = get_backbone(model)
        n = len(backbone.layers)
        cut = int(n * (1 - args.unfreeze_ratio))
        for i, layer in enumerate(backbone.layers):
            layer.trainable = (i >= cut)

        model.compile(optimizer=keras.optimizers.Adam(args.lr_finetune),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        cbs_ft = [
            callbacks.ModelCheckpoint("runs/best_finetune.keras", monitor="val_accuracy", save_best_only=True),
            callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
            callbacks.CSVLogger("runs/train_log_finetune.csv", append=False)
        ]

        model.fit(ds_train, validation_data=ds_val, epochs=args.epochs_finetune, callbacks=cbs_ft)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed/pets")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--lr_base", type=float, default=1e-3)

    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--epochs_finetune", type=int, default=8)
    ap.add_argument("--unfreeze_ratio", type=float, default=0.30)  # último 30%
    ap.add_argument("--lr_finetune", type=float, default=1e-5)

    args = ap.parse_args()
    main(args)
