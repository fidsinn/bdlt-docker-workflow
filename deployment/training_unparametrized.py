import tensorflow as tf

CONFIG = {
    "model": "distilbert-base-uncased",
    "seq_length": 512,
    "num_classes": 20,
    "batch_size": 64,
}

train_ds = tf.data.experimental.load("20ng_bert_data/train_ds").cache()
val_ds = tf.data.experimental.load("20ng_bert_data/val_ds").cache()
test_ds = tf.data.experimental.load("20ng_bert_data/test_ds").cache()

batched_train_ds = (
    train_ds
    .shuffle(2 * CONFIG["batch_size"])
    .repeat()
    .batch(CONFIG["batch_size"])
)
batched_val_ds = (
    val_ds
    .batch(CONFIG["batch_size"])
)
batched_test_ds = (
    test_ds
    .batch(CONFIG["batch_size"])
)

input_embeddings = tf.keras.layers.Input(shape=(768,), name="input_embeddings", dtype="float32")

x = tf.keras.layers.Dropout(.1)(input_embeddings)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(.2)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(.2)(x)
output = tf.keras.layers.Dense(CONFIG["num_classes"], activation='softmax')(x)

model = tf.keras.Model(input_embeddings, output)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
    optimizer=tf.keras.optimizers.Adam(), 
    metrics=['accuracy']
)

callbacks=[]

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=3,          # Stop after 3 epochs of no improvement
    monitor='val_loss',  # Look at validation_loss
    min_delta=0.01,      # After 0 change
    mode='min',          # Stop when quantity has stopped decreasing
    verbose=1
)

callbacks.append(early_stopping)

history = model.fit(
    batched_train_ds, 
    validation_data = batched_val_ds,
    epochs = 20, 
    steps_per_epoch = 15076 // CONFIG["batch_size"],
    callbacks = callbacks
)
