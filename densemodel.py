import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import coalesce, lit, concat_ws, col

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

# Step 1: Load and Merge Datasets using PySpark with aggressive sampling
def load_and_merge_datasets():
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

    # Load CVE CSV
    cve_df = spark.read.csv('dataset/cve.csv', header=True, inferSchema=True)

    # Handle column renaming
    if '_c0' in cve_df.columns:
        cve_df = cve_df.withColumnRenamed('_c0', 'cve_id')
        logger.info("Renamed '_c0' to 'cve_id' in cve.csv")

    # Select required columns
    cve_df = cve_df.select(
        col('cve_id'), col('mod_date'), col('pub_date'), col('cvss'), col('cwe_code'),
        col('cwe_name'), col('summary'), col('access_authentication'), col('access_complexity'),
        col('access_vector'), col('impact_availability'), col('impact_confidentiality'), col('impact_integrity')
    )

    products_df = spark.read.csv('dataset/products.csv', header=True, inferSchema=True).select(
        col('cve_id'), col('vulnerable_product')
    )

    vendor_product_df = spark.read.csv('dataset/vendor_product.csv', header=True, inferSchema=True).select(
        col('vendor'), col('product')
    )

    vendors_df = spark.read.csv('dataset/vendors.csv', header=True, inferSchema=True).select(col('vendor'))

    # Merge using joins with broadcast hint for smaller tables
    from pyspark.sql.functions import broadcast

    merged_df = cve_df.join(broadcast(products_df), on='cve_id', how='left')
    merged_df = merged_df.join(broadcast(vendor_product_df),
                               merged_df['vulnerable_product'] == vendor_product_df['product'], how='left')
    merged_df = merged_df.join(broadcast(vendors_df), on='vendor', how='left')

    # Combine text features
    merged_df = merged_df.withColumn(
        'text',
        concat_ws(' ',
                  coalesce(merged_df['summary'], lit('')),
                  coalesce(merged_df['vulnerable_product'], lit('')),
                  coalesce(merged_df['vendor'], lit(''))
        )
    )

    # Filter out invalid rows
    merged_df = merged_df.filter((merged_df['text'] != '') & merged_df['cwe_name'].isNotNull())

    # AGGRESSIVE sampling to 0.1% (0.001) to get manageable dataset size
    merged_df = merged_df.sample(fraction=0.001, seed=42)

    # Limit to maximum 50,000 rows for training
    merged_df = merged_df.limit(50000)

    total_count = merged_df.count()
    logger.info(f"Dataset size after aggressive sampling: {total_count} rows")

    # Save to parquet first, then read back (more memory efficient)
    temp_path = "temp_data.parquet"
    merged_df.write.mode("overwrite").parquet(temp_path)

    # Read back and collect
    sampled_df = spark.read.parquet(temp_path).select('text', 'cwe_name')

    # Collect in smaller chunks using takeOrdered
    text_list = []
    cwe_name_list = []

    # Use take() instead of collect() to avoid memory issues
    chunk_size = 5000
    offset = 0

    while True:
        # Get a chunk of data
        chunk = sampled_df.limit(chunk_size).offset(offset).collect()

        if not chunk:  # No more data
            break

        for row in chunk:
            text_list.append(row['text'])
            cwe_name_list.append(row['cwe_name'])

        offset += chunk_size
        logger.info(f"Collected {len(text_list)} samples so far...")

        if len(text_list) >= 50000:  # Safety limit
            break

    # Cleanup temp file
    import shutil
    try:
        shutil.rmtree(temp_path)
    except:
        pass

    spark.stop()
    logger.info(f"Final dataset size: {len(text_list)} samples")
    return text_list, cwe_name_list

# Step 2: Preprocess Data using only TensorFlow - FIXED VERSION
def preprocess_data(text_list, cwe_name_list, max_words=10000, max_len=300):
    # Tokenize text using TensorFlow Tokenizer with OOV token
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(text_list)
    sequences = tokenizer.texts_to_sequences(text_list)

    # CRITICAL FIX: Calculate proper embedding input dimension
    # When using num_words with OOV token, embedding needs max_words + 1 dimension
    embedding_input_dim = max_words + 1

    # Defensive clipping: ensure no index exceeds max_words-1
    oov_token_idx = tokenizer.word_index.get("<OOV>", 1)

    # Clip any indices that are >= max_words to OOV token index
    clipped_sequences = []
    for seq in sequences:
        clipped_seq = []
        for idx in seq:
            if idx < max_words:
                clipped_seq.append(idx)
            else:
                clipped_seq.append(oov_token_idx)
        clipped_sequences.append(clipped_seq)

    # Pad sequences
    X = pad_sequences(clipped_sequences, maxlen=max_len, padding='post', truncating='post')

    # Manual label encoding using TensorFlow
    unique_labels_set = list(set(cwe_name_list))
    unique_labels_set.sort()  # Ensure consistent ordering

    label_to_int = {label: idx for idx, label in enumerate(unique_labels_set)}
    num_classes = len(unique_labels_set)

    # Map labels to integers
    y = np.array([label_to_int[label] for label in cwe_name_list], dtype=np.int32)

    # One-hot encoding for categorical loss
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    logger.info(f"Preprocessed {len(text_list)} samples with {num_classes} classes")
    logger.info(f"Embedding input dim: {embedding_input_dim}")

    return X, y_one_hot, tokenizer, label_to_int, num_classes, embedding_input_dim

# Step 3: Manual Train/Test Split using NumPy
def manual_train_test_split(X, y, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)  # For reproducibility
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split_idx = int(num_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

# Step 4: Build Deep Model with EXPANDED layers (added 2048-dim layers as requested)
def build_deep_model(input_dim, input_length, num_classes):
    model = tf.keras.Sequential([
        # Use mask_zero=True to handle padding tokens properly
        tf.keras.layers.Embedding(input_dim=input_dim, output_dim=256,
                                input_length=input_length, mask_zero=True),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.3),

        # EXPANDED LAYERS - Added 2048-dimensional layers as requested
        tf.keras.layers.Dense(2048, activation='relu'),  # NEW LAYER
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LayerNormalization(),

        tf.keras.layers.Dense(2048, activation='relu'),  # NEW LAYER
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main Training Logic
if __name__ == "__main__":
    configure_gpu()

    # Load data using Spark
    logger.info("Starting data loading...")
    text_list, cwe_name_list = load_and_merge_datasets()

    if len(text_list) == 0:
        logger.error("No data loaded! Check your CSV files.")
        exit(1)

    # Preprocess using TensorFlow - NOW RETURNS embedding_input_dim
    logger.info("Starting preprocessing...")
    X, y, tokenizer, label_to_int, num_classes, embedding_input_dim = preprocess_data(text_list, cwe_name_list)

    # Manual split
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = manual_train_test_split(X, y)

    # Build model - NOW USES CALCULATED embedding_input_dim
    logger.info(f"Building model with {num_classes} classes...")
    model = build_deep_model(input_dim=embedding_input_dim, input_length=300, num_classes=num_classes)
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Train
    logger.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model using TensorFlow with proper extension
    model.save('deep_model.keras')
    logger.info("Model saved successfully")

    # Save tokenizer configuration
    tokenizer_config = tokenizer.to_json()
    with open('tokenizer.json', 'w') as f:
        f.write(tokenizer_config)

    # Save label mapping
    with open('label_to_int.txt', 'w') as f:
        for label, idx in label_to_int.items():
            f.write(f"{label}:{idx}\n")

    logger.info("Training completed successfully")
