Documentation of Parameters and Results of Traffic assignment

model = tf.keras.models.Sequential([
# Convolutional layer. Learn 32 filters using a 3x3 kernel     
tf.keras.layers.Conv2D(
    32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
),
# Max-pooling layer, using 2x2 pool size     
tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
# Flatten units     
tf.keras.layers.Flatten(),
# Add a hidden layer with dropout     
tf.keras.layers.Dense(128, activation="relu"),
tf.keras.layers.Dropout(0.5),
# Add an output layer with output units for all categories     
tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

/step - loss: 3.4972 - accuracy: 0.0532
333/333 - 3s - loss: 3.4999 - accuracy: 0.0573

    # Create a convolutional neural network 
    model = tf.keras.models.Sequential([
    # Convolutional layer. Learn 32 filters using a 3x3 kernel     
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    # Max-pooling layer, using 2x2 pool size     
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # Flatten units     
    tf.keras.layers.Flatten(),
    # Add a hidden layer with dropout     
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),    
    # Add an output layer with output units for all categories     
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

/step - loss: 3.5036 - accuracy: 0.0544
333/333 - 2s - loss: 3.4980 - accuracy: 0.0551

Same as above but remove MaxPooling2D

training took about 30s ETAs

/step - loss: 3.4982 - accuracy: 0.0576
333/333 - 4s - loss: 3.4983 - accuracy: 0.0543

Same as above but change pool_size to (3, 3)

about 7s ETAs
/step - loss: 3.5039 - accuracy: 0.0547
333/333 - 2s - loss: 3.4914 - accuracy: 0.0579

kept pool_size at (3, 3) but add another dense layer 128

about 9s ETAs

/step - loss: 0.2523 - accuracy: 0.9341
333/333 - 2s - loss: 0.3383 - accuracy: 0.9248

added another dense layer (total of 3)

about 9s ETAs

/step - loss: 0.2897 - accuracy: 0.9195
333/333 - 2s - loss: 0.3430 - accuracy: 0.9190

change back to 2 dense layers but change to pool_size (2, 2)

about 12s ETAs

/step - loss: 3.5038 - accuracy: 0.0527
333/333 - 2s - loss: 3.4999 - accuracy: 0.0508

keep at 2 dense layers but change pool_size (4, 4)

about 9s ETAs

step - loss: 0.2475 - accuracy: 0.9299
333/333 - 2s - loss: 0.2778 - accuracy: 0.9334

keep at 2 dense layers but change pool_size (5, 5)
about 7s ETAs

step - loss: 0.9918 - accuracy: 0.6699
333/333 - 2s - loss: 0.6227 - accuracy: 0.7891

keep at 2 dense layers, revert to pool_size (4, 4) but change to 64 filters
tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),

17s 33ms/step - loss: 0.5979 - accuracy: 0.8103
333/333 - 3s - loss: 0.5142 - accuracy: 0.8429


change to 16 filter
    
500/500 [==============================] - 7s 15ms/step - loss: 0.3864 - accuracy: 0.8841
333/333 - 2s - loss: 0.3087 - accuracy: 0.9120

double the Conv2D and MaxPooling2D

    model = tf.keras.models.Sequential([
    # Convolutional layer. Learn 32 filters using a 3x3 kernel     
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    # Max-pooling layer, using 2x2 pool size     
    tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", 
    ),    
    tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),    
    # Flatten units     
    tf.keras.layers.Flatten(),
    # Add a hidden layer with dropout     
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    # Add an output layer with output units for all categories     
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

500/500 [==============================] - 10s 20ms/step - loss: 0.9735 - accuracy: 0.6771
333/333 - 2s - loss: 0.8266 - accuracy: 0.7279

change dropout to 0.3 and Conv2D at 16 layers
# Create a convolutional neural network 
    model = tf.keras.models.Sequential([
    # Convolutional layer. Learn 32 filters using a 3x3 kernel     
    tf.keras.layers.Conv2D(
        16, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    # Max-pooling layer, using 2x2 pool size     
    tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
    # Flatten units     
    tf.keras.layers.Flatten(),
    # Add a hidden layer with dropout     
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    # Add an output layer with output units for all categories     
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    500/500 [==============================] - 7s 14ms/step - loss: 0.2486 - accuracy: 0.9240
333/333 - 2s - loss: 0.2963 - accuracy: 0.9339

changed above to sgd for optimizer

500/500 [==============================] - 7s 14ms/step - loss: 0.4930 - accuracy: 0.8472
333/333 - 2s - loss: 0.4788 - accuracy: 0.8565


change hidden layer nodes from 128 to NUM_CATEGORIES

500/500 [==============================] - 6s 13ms/step - loss: 0.9096 - accuracy: 0.7090
333/333 - 2s - loss: 0.6692 - accuracy: 0.7971

add 2 dense layer of 64 nodes 

    # Create a convolutional neural network 
    model = tf.keras.models.Sequential([
    # Convolutional layer. Learn 32 filters using a 3x3 kernel     
    tf.keras.layers.Conv2D(
        16, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
    # Max-pooling layer, using 2x2 pool size     
    tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
    # Flatten units     
    tf.keras.layers.Flatten(),
    # Add a hidden layer with dropout     
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    # Add an output layer with output units for all categories     
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

500/500 [==============================] - 7s 13ms/step - loss: 0.2429 - accuracy: 0.9276
333/333 - 2s - loss: 0.3743 - accuracy: 0.9069

changed above 64 to 128 for dense layer
500/500 [==============================] - 7s 13ms/step - loss: 0.2393 - accuracy: 0.9300
333/333 - 2s - loss: 0.2766 - accuracy: 0.9263

change dropout to 0.1

500/500 [==============================] - 7s 14ms/step - loss: 0.2711 - accuracy: 0.9266
333/333 - 2s - loss: 0.4223 - accuracy: 0.8971


change Conv2D to (4, 4) from (3, 3)
    tf.keras.layers.Conv2D(
        16, (4, 4), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ),
500/500 [==============================] - 7s 14ms/step - loss: 0.3524 - accuracy: 0.8979
333/333 - 1s - loss: 0.3075 - accuracy: 0.9234


change pool_size and Conv2D to (2, 2)
500/500 [==============================] - 9s 17ms/step - loss: 3.5057 - accuracy: 0.0525
333/333 - 2s - loss: 3.4974 - accuracy: 0.0572

change pool_size and Conv2D to (3, 3)

500/500 [==============================] - 7s 14ms/step - loss: 1.4789 - accuracy: 0.5391
333/333 - 2s - loss: 1.1342 - accuracy: 0.6469

tried "hinge" and "mse" loss, accuracy in the 0.0x
