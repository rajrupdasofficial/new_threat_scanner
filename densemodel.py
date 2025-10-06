import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
import logging
import collections
from pyspark.sql import SparkSession
from pyspark import StorageLevel
from pyspark.sql.functions import coalesce, lit, concat_ws, col, broadcast

# ---------------------------------------------------------------------
# Environment & global acceleration
# ---------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduce TF log chatter

# Mixed precision for NVIDIA Tensor Cores
mixed_precision.set_global_policy('mixed_float16')  # recommended policy on recent TF/Keras
# Prefer TF32 on Ampere+ (e.g., RTX 3050)
try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except Exception:
    pass

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# GPU config
# ---------------------------------------------------------------------
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logger.info(f"GPU configured: {gpus}")
        except RuntimeError as e:
            logger.error(f"GPU config error: {e}")
    else:
        logger.warning("No GPU detected. Falling back to CPU.")
    return gpus
# ---------------------------------------------------------------------
# Spark data loading + stratified sampling (200k)
# ---------------------------------------------------------------------
def load_and_merge_datasets_stratified():
    t0 = time.time()
    spark = SparkSession.builder \
        .appName("VulnDetection") \
        .config("spark.driver.memory", "12g") \
        .config("spark.executor.memory", "12g") \
        .config("spark.driver.maxResultSize", "8g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "6g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.shuffle.partitions", "64") \
        .config("spark.default.parallelism", "64") \
        .getOrCreate()

    cve_df = spark.read.csv('dataset/cve.csv', header=True, inferSchema=True)
    if '_c0' in cve_df.columns:
        cve_df = cve_df.withColumnRenamed('_c0', 'cve_id')
        logger.info("Renamed '_c0' to 'cve_id' in cve.csv")

    cve_df = cve_df.select(col('cve_id'), col('cwe_name'), col('summary'))

    products_df = spark.read.csv('dataset/products.csv', header=True, inferSchema=True) \
        .select(col('cve_id'), col('vulnerable_product'))

    vendor_product_df = spark.read.csv('dataset/vendor_product.csv', header=True, inferSchema=True) \
        .select(col('vendor'), col('product'))

    vendors_df = spark.read.csv('dataset/vendors.csv', header=True, inferSchema=True) \
        .select(col('vendor'))

    # Joins: broadcast only small dimension table(s)
    merged_df = cve_df.join(products_df, on='cve_id', how='left')
    merged_df = merged_df.join(
        vendor_product_df,
        merged_df['vulnerable_product'] == vendor_product_df['product'],
        how='left'
    )
    merged_df = merged_df.join(broadcast(vendors_df), on='vendor', how='left')

    # Compose text and filter
    merged_df = merged_df.withColumn(
        'text',
        concat_ws(
            ' ',
            coalesce(merged_df['summary'], lit('')),
            coalesce(merged_df['vulnerable_product'], lit('')),
            coalesce(merged_df['vendor'], lit(''))
        )
    ).filter((col('text') != '') & col('cwe_name').isNotNull())

    # Persist reused DF to avoid recomputation across actions
    merged_df = merged_df.persist(StorageLevel.MEMORY_AND_DISK)

    # Compute label fractions
    label_counts = merged_df.groupBy('cwe_name').count().collect()
    label_fractions = {}
    for row in label_counts:
        current_count = row['count']
        if current_count < 1000:
            min_fraction = min(3.0 / current_count, 0.01) if current_count > 0 else 0
        else:
            min_fraction = 0.002
        label_fractions[row['cwe_name']] = min_fraction

    sampled_df = merged_df.stat.sampleBy('cwe_name', fractions=label_fractions, seed=42).limit(200000)

    total_count = sampled_df.count()
    logger.info(f"Dataset size after stratified sampling: {total_count}")

    # Materialize to driver for tokenization
    rows = sampled_df.select('text', 'cwe_name').collect()
    text_list = [row['text'] for row in rows]
    cwe_name_list = [row['cwe_name'] for row in rows]

    spark.stop()
    logger.info(f"Final dataset size: {len(text_list)} samples")
    logger.info(f"Spark pipeline time: {(time.time() - t0):.1f}s")
    return text_list, cwe_name_list

# ---------------------------------------------------------------------
# Tokenization & encoding
# ---------------------------------------------------------------------
def preprocess_data(text_list, cwe_name_list, max_words=10000, max_len=300):
    t0 = time.time()
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(text_list)
    sequences = tokenizer.texts_to_sequences(text_list)

    # Filter out empty sequences
    valid_indices = [i for i, seq in enumerate(sequences) if len(seq) > 0]
    filtered_sequences = [sequences[i] for i in valid_indices]
    filtered_cwe = [cwe_name_list[i] for i in valid_indices]
    logger.info(f"Filtered out {len(sequences) - len(filtered_sequences)} empty sequences.")

    X = pad_sequences(
        filtered_sequences, maxlen=max_len, padding='post', truncating='post', dtype='int32'
    )
    embedding_input_dim = max_words + 1

    unique_labels_set = sorted(list(set(filtered_cwe)))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels_set)}
    y_int = np.array([label_to_int[label] for label in filtered_cwe], dtype=np.int32)

    logger.info(f"Preprocessed {len(filtered_sequences)} samples in {(time.time() - t0):.1f}s.")
    return X, y_int, tokenizer, embedding_input_dim

# ---------------------------------------------------------------------
# Label filtering & one-hot
# ---------------------------------------------------------------------
def filter_and_reencode(X, y_int, min_samples=2):
    t0 = time.time()
    label_counts = collections.Counter(y_int)
    valid_labels = {label for label, count in label_counts.items() if count >= min_samples}
    mask = np.array([label in valid_labels for label in y_int])

    X_filtered = X[mask]
    y_int_filtered = y_int[mask]

    unique_final_labels = sorted(list(set(y_int_filtered)))
    final_label_map = {old_label: new_label for new_label, old_label in enumerate(unique_final_labels)}

    y_int_remapped = np.array([final_label_map[label] for label in y_int_filtered], dtype=np.int32)
    num_final_classes = len(unique_final_labels)
    y_one_hot = tf.keras.utils.to_categorical(y_int_remapped, num_classes=num_final_classes)

    logger.info(f"Filtered to {len(X_filtered)} samples and {num_final_classes} classes in {(time.time() - t0):.1f}s.")
    return X_filtered, y_one_hot, y_int_remapped, num_final_classes

# ---------------------------------------------------------------------
# Model with proper mask into MHA; dense stack unchanged
# ---------------------------------------------------------------------
def build_deep_model_fixed(input_dim, input_length, num_classes):
    inputs = tf.keras.Input(shape=(input_length,), dtype='int32')
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=input_dim, output_dim=256, mask_zero=True, name='embed'
    )
    embedding = embedding_layer(inputs)

    # Obtain boolean padding mask from the Embedding
    mask = embedding_layer.compute_mask(inputs)  # shape [B, T], dtype bool

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(embedding)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Expand mask to [B, T, T] for self-attention
    # The attention mask should prevent padded positions from being attended to
    def create_attention_mask(m):
        # m has shape [B, T]
        # We need shape [B, T, T] where mask[b, i, j] = m[b, j]
        # This means query at position i can attend to key at position j only if m[b, j] is True
        m = tf.cast(m, tf.bool)  # [B, T]
        m = tf.expand_dims(m, axis=1)  # [B, 1, T]
        # Broadcast to [B, T, T]
        seq_len = tf.shape(m)[-1]
        m = tf.tile(m, [1, seq_len, 1])  # [B, T, T]
        return m

    attn_mask = tf.keras.layers.Lambda(create_attention_mask, name='create_attn_mask')(mask)

    # MultiHeadAttention with explicit attention_mask
    mha_layer = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128, name='mha')
    attn_out = mha_layer(query=x, value=x, key=x, attention_mask=attn_mask)

    # Masked average pooling to avoid padded positions
    def masked_avg(tensors):
        feats, m = tensors
        m = tf.cast(m, feats.dtype)               # [B, T]
        m = tf.expand_dims(m, -1)                 # [B, T, 1]
        sum_feats = tf.reduce_sum(feats * m, axis=1)   # [B, C]
        denom = tf.reduce_sum(m, axis=1) + 1e-6        # [B, 1]
        return sum_feats / denom

    x = tf.keras.layers.Lambda(masked_avg, name='masked_avg')([attn_out, mask])

    # Dense stack unchanged; output in float32 for numerical stability under mixed precision
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='vuln_mha_lstm')

    # Compile without jit_compile since it's not supported
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    configure_gpu()

    # Timers for visibility
    t_total = time.time()

    logger.info("Starting data loading and preprocessing...")
    text_list, cwe_name_list = load_and_merge_datasets_stratified()

    X, y_int, tokenizer, embedding_input_dim = preprocess_data(text_list, cwe_name_list)

    X_final, y_final_one_hot, y_final_int, num_final_classes = filter_and_reencode(X, y_int)

    # Train/val split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, y_int_train, y_int_test = train_test_split(
        X_final, y_final_one_hot, y_final_int, test_size=0.2, random_state=42, stratify=y_final_int
    )

    # Class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_int_train), y=y_int_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Build model
    logger.info(f"Building model with {num_final_classes} classes...")
    model = build_deep_model_fixed(input_dim=embedding_input_dim, input_length=300, num_classes=num_final_classes)
    model.summary(print_fn=lambda s: logger.info(s))

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)

    # Optional: TQDM progress bar (TensorFlow Addons or keras-tqdm)
    tqdm_callback = None
    try:
        from tensorflow_addons.callbacks import TQDMProgressBar
        tqdm_callback = TQDMProgressBar(show_epoch_progress=True, show_overall_progress=True)
        logger.info("Using TensorFlow Addons TQDMProgressBar")
    except Exception:
        try:
            from keras_tqdm import TQDMCallback as _TQDMCallback
            tqdm_callback = _TQDMCallback()
            logger.info("Using keras-tqdm TQDMCallback")
        except Exception:
            logger.info("TQDM progress bar not available; default Keras progress will be used")

    callbacks = [early_stopping, lr_scheduler] + ([tqdm_callback] if tqdm_callback is not None else [])

    # tf.data pipelines with cache+prefetch to keep GPU busy
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(min(len(X_train), 10000), reshuffle_each_iteration=True)
        .batch(64, drop_remainder=True)
        .cache()
        .prefetch(AUTOTUNE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(64, drop_remainder=False)
        .cache()
        .prefetch(AUTOTUNE)
    )

    logger.info("Starting training...")
    t_fit = time.time()
    history = model.fit(
        train_dataset,
        epochs=40,
        validation_data=test_dataset,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=0 if tqdm_callback is not None else 1  # avoid double bars
    )
    logger.info(f"Training time: {(time.time() - t_fit):.1f}s")

    # Evaluate
    test_loss, test_acc = model.evaluate(test_dataset, verbose=0 if tqdm_callback is not None else 1)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model and tokenizer (Keras v3 .keras is recommended)
    model.save('deep_model.keras')
    tokenizer_config = tokenizer.to_json()
    with open('tokenizer.json', 'w') as f:
        f.write(tokenizer_config)

    # Optionally save label map for inference
    meta = {
        "num_classes": int(model.output_shape[-1]),
        "max_len": 300
    }
    with open('meta.json', 'w') as f:
        json.dump(meta, f)

    logger.info(f"Training completed successfully in {(time.time() - t_total):.1f}s")
