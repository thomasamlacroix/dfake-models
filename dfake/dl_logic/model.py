from keras.models import Model
from keras import Sequential, Input, layers, optimizers, callbacks

from keras.utils import image_dataset_from_directory

#Pretrained model for transfer learning
from keras.applications.efficientnet import EfficientNetB3, preprocess_input


def data_augmentation():
    data_augmentation = Sequential()

    data_augmentation.add(layers.RandomFlip("horizontal"))
    # data_augmentation.add(layers.RandomFlip("vertical"))
    data_augmentation.add(layers.RandomZoom(0.1))
    data_augmentation.add(layers.RandomTranslation(0.2, 0.2))
    data_augmentation.add(layers.RandomRotation(0.1))

    return data_augmentation


def initialize_model(input_shape, base_model):
    """
    Initialize the Neural Network with random weights
    """

    #Freezing weights of pretrained model
    base_model.trainable = False

    #Input
    inputs = Input(shape=input_shape)

    # x = data_augmentation(inputs)
    x = preprocess_input(inputs) #Preprocessing layer specifically designed for the pretrained model
    x = base_model(x) #Transfer learning model


    x = layers.Flatten()(x)

    #Dense layers
    x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    # x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    print("✅ Model initialized")

    return model


def compile_model(model, learning_rate=0.0005):
    """
    Compile the Neural Network
    """
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy', 'recall', 'precision'])

    print("✅ Model compiled")

    return model


def train_model(
        model,
        train_ds,
        patience=3,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print("\nTraining model...")

    LRreducer = callbacks.ReduceLROnPlateau(monitor="val_loss",
                                            factor=0.1,
                                            patience=patience,
                                            verbose=1,
                                            min_lr=0)

    EarlyStopper = callbacks.EarlyStopping(monitor='val_loss',
                                        patience=patience,
                                        verbose=0,
                                        restore_best_weights=True)


    history = model.fit(train_ds,
                    epochs=10,
                    validation_data=validation_data,
                    callbacks=[LRreducer, EarlyStopper],
                    verbose=1)

    print(f"✅ Model trained")

    return model, history


def evaluate_model(model,
                   test_ds,
                   batch_size=128
                   ):
    """
    Evaluate trained model performance on the dataset
    """

    print("\nEvaluating model...")

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    evaluation = model.evaluate(
        test_ds,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
        )

    accuracy = evaluation["accuracy"]
    recall = evaluation["recall"]
    precision = evaluation["precision"]

    print("✅ Model evaluated")
    print(f"Accuracy: {round(accuracy, 2)}")
    print(f"Recall: {round(recall, 2)}")
    print(f"Precision: {round(precision, 2)}")

    return evaluation
