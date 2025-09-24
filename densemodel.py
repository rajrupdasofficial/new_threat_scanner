import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
import collections
from pyspark.sql import SparkSession
from pyspark.sql.functions import coalesce, lit, concat_ws, col, broadcast

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Explicit GPU Configuration
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

# Step 1: Load and Stratified Sample Datasets using PySpark (increased to 200k samples)
def load_and_merge_datasets_stratified():
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
        .getOrCreate()

    cve_df = spark.read.csv('dataset/cve.csv', header=True, inferSchema=True)
    if '_c0' in cve_df.columns:
        cve_df = cve_df.withColumnRenamed('_c0', 'cve_id')
        logger.info("Renamed '_c0' to 'cve_id' in cve.csv")

    cve_df = cve_df.select(
        col('cve_id'), col('cwe_name'), col('summary')
    )

    products_df = spark.read.csv('dataset/products.csv', header=True, inferSchema=True).select(
        col('cve_id'), col('vulnerable_product')
    )

    vendor_product_df = spark.read.csv('dataset/vendor_product.csv', header=True, inferSchema=True).select(
        col('vendor'), col('product')
    )

    vendors_df = spark.read.csv('dataset/vendors.csv', header=True, inferSchema=True).select(col('vendor'))

    merged_df = cve_df.join(broadcast(products_df), on='cve_id', how='left')
    merged_df = merged_df.join(broadcast(vendor_product_df),
                               merged_df['vulnerable_product'] == vendor_product_df['product'], how='left')
    merged_df = merged_df.join(broadcast(vendors_df), on='vendor', how='left')

    merged_df = merged_df.withColumn(
        'text',
        concat_ws(' ',
                  coalesce(merged_df['summary'], lit('')),
                  coalesce(merged_df['vulnerable_product'], lit('')),
                  coalesce(merged_df['vendor'], lit(''))
        )
    )
    merged_df = merged_df.filter((merged_df['text'] != '') & merged_df['cwe_name'].isNotNull())

    label_counts = merged_df.groupBy('cwe_name').count().collect()
    label_fractions = {}

    for row in label_counts:
        current_count = row['count']
        if current_count < 1000:
            min_fraction = min(3.0 / current_count, 0.01) if current_count > 0 else 0
        else:
            min_fraction = 0.002
        label_fractions[row['cwe_name']] = min_fraction

    sampled_df = merged_df.stat.sampleBy('cwe_name', fractions=label_fractions, seed=42)
    sampled_df = sampled_df.limit(200000)  # Increased sample size to 200k

    total_count = sampled_df.count()
    logger.info(f"Dataset size after stratified sampling: {total_count}")

    rows = sampled_df.select('text', 'cwe_name').collect()
    text_list = [row['text'] for row in rows]
    cwe_name_list = [row['cwe_name'] for row in rows]

    spark.stop()
    logger.info(f"Final dataset size: {len(text_list)} samples")
    return text_list, cwe_name_list

# Step 2: Preprocess Data using TensorFlow (with filtering for empty sequences)
def preprocess_data(text_list, cwe_name_list, max_words=10000, max_len=300):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(text_list)
    sequences = tokenizer.texts_to_sequences(text_list)

    # Filter out empty sequences
    valid_indices = [i for i, seq in enumerate(sequences) if len(seq) > 0]
    filtered_sequences = [sequences[i] for i in valid_indices]
    filtered_cwe = [cwe_name_list[i] for i in valid_indices]
    logger.info(f"Filtered out {len(sequences) - len(filtered_sequences)} empty sequences.")

    X = pad_sequences(filtered_sequences, maxlen=max_len, padding='post', truncating='post')

    embedding_input_dim = max_words + 1

    unique_labels_set = sorted(list(set(filtered_cwe)))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels_set)}
    y_int = np.array([label_to_int[label] for label in filtered_cwe], dtype=np.int32)

    logger.info(f"Preprocessed {len(filtered_sequences)} samples.")
    return X, y_int, tokenizer, embedding_input_dim

# Step 3: Filter classes and re-encode labels
def filter_and_reencode(X, y_int, min_samples=2):
    label_counts = collections.Counter(y_int)
    valid_labels = {label for label, count in label_counts.items() if count >= min_samples}
    mask = np.array([label in valid_labels for label in y_int])

    X_filtered = X[mask]
    y_int_filtered = y_int[mask]

    unique_final_labels = sorted(list(set(y_int_filtered)))
    final_label_map = {old_label: new_label for new_label, old_label in enumerate(unique_final_labels)}

    y_int_remapped = np.array([final_label_map[label] for label in y_int_filtered])
    num_final_classes = len(unique_final_labels)
    y_one_hot = tf.keras.utils.to_categorical(y_int_remapped, num_classes=num_final_classes)

    logger.info(f"Filtered to {len(X_filtered)} samples and {num_final_classes} classes.")
    return X_filtered, y_one_hot, y_int_remapped, num_final_classes

# Step 4: Build Deep Model with MultiHeadAttention for better mask handling
def build_deep_model(input_dim, input_length, num_classes):
    inputs = tf.keras.Input(shape=(input_length,))
    embedding_layer = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=256, mask_zero=True)
    embedding = embedding_layer(inputs)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(embedding)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Use MultiHeadAttention instead of Attention for robust self-attention and mask propagation
    mha_layer = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)
    attn_out = mha_layer(query=x, value=x, key=x)

    x = tf.keras.layers.GlobalAveragePooling1D()(attn_out)

    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main Training Logic
if __name__ == "__main__":
    configure_gpu()

    logger.info("Starting data loading and preprocessing...")
    text_list, cwe_name_list = load_and_merge_datasets_stratified()
    X, y_int, tokenizer, embedding_input_dim = preprocess_data(text_list, cwe_name_list)

    X_final, y_final_one_hot, y_final_int, num_final_classes = filter_and_reencode(X, y_int)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, y_int_train, y_int_test = train_test_split(
        X_final, y_final_one_hot, y_final_int, test_size=0.2, random_state=42, stratify=y_final_int)

    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_int_train), y=y_int_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    logger.info(f"Building model with {num_final_classes} classes...")
    model = build_deep_model(input_dim=embedding_input_dim, input_length=300, num_classes=num_final_classes)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)

    # Use tf.data.Dataset for consistent batches (batch size 64 for larger dataset)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64, drop_remainder=True)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64, drop_remainder=True)

    logger.info("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=40,
        validation_data=test_dataset,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weight_dict,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    model.save('deep_model.keras')
    tokenizer_config = tokenizer.to_json()
    with open('tokenizer.json', 'w') as f:
        f.write(tokenizer_config)

    logger.info("Training completed successfully")
