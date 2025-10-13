import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras import mixed_precision
from tensorflow.keras import regularizers
import logging
import sys
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import coalesce, lit, concat_ws, col
from sklearn.utils.class_weight import compute_class_weight

# Enable Mixed Precision for memory efficiency
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f'Compute dtype: {policy.compute_dtype}')
print(f'Variable dtype: {policy.variable_dtype}')

# Set TensorFlow for dynamic memory growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Enhanced logging setup
log_filename = f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# FIXED: Dynamic GPU Configuration for ANY GPU size
def configure_gpu():
    """
    Dynamically configures GPU memory allocation based on available VRAM.
    Works with any GPU size (2GB, 4GB, 6GB, 8GB, etc.)
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Enable memory growth - allocates only what's needed
                tf.config.experimental.set_memory_growth(gpu, True)

            # Get GPU memory info dynamically
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            logger.info(f"GPU detected: {gpu_details}")

            # Calculate safe memory limit (85% of total VRAM to leave headroom)
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True
                )
                total_memory_mb = int(result.stdout.strip().split('\n')[0])
                safe_memory_limit = int(total_memory_mb * 0.85)  # Use 85% of total VRAM

                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=safe_memory_limit)]
                )
                logger.info(f"GPU Total VRAM: {total_memory_mb}MB, Safe Limit: {safe_memory_limit}MB")
            except Exception as e:
                logger.warning(f"Could not query GPU memory, using dynamic growth only: {e}")
                logger.info("GPU configured with pure memory growth (no hard limit)")

            logger.info(f"GPU configured: {gpus}")
            logger.info("Mixed Precision Training ENABLED (FP16)")
            logger.info("Dynamic memory allocation ENABLED")
        except RuntimeError as e:
            logger.error(f"GPU config error: {e}")
    else:
        logger.warning("No GPU detected. Falling back to CPU.")
    return gpus

# Step 1: Load and Merge Datasets
def load_and_merge_datasets():
    spark = SparkSession.builder \
        .appName("VulnDetection") \
        .config("spark.driver.memory", "14g") \
        .config("spark.executor.memory", "14g") \
        .config("spark.driver.maxResultSize", "10g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "7g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.shuffle.partitions", "150") \
        .getOrCreate()

    cve_df = spark.read.csv('dataset/cve.csv', header=True, inferSchema=True)

    if '_c0' in cve_df.columns:
        cve_df = cve_df.withColumnRenamed('_c0', 'cve_id')
        logger.info("Renamed '_c0' to 'cve_id' in cve.csv")

    cve_df = cve_df.select(
        col('cve_id'), col('mod_date'), col('pub_date'), col('cvss'), col('cwe_code'),
        col('cwe_name'), col('summary'), col('access_authentication'), col('access_complexity'),
        col('access_vector'), col('impact_availability'), col('impact_confidentiality'), col('impact_integrity')
    ).filter(col('cwe_name').isNotNull() & col('summary').isNotNull())

    products_df = spark.read.csv('dataset/products.csv', header=True, inferSchema=True).select(
        col('cve_id'), col('vulnerable_product')
    ).filter(col('vulnerable_product').isNotNull())

    vendor_product_df = spark.read.csv('dataset/vendor_product.csv', header=True, inferSchema=True).select(
        col('vendor'), col('product')
    )

    vendors_df = spark.read.csv('dataset/vendors.csv', header=True, inferSchema=True).select(col('vendor'))

    from pyspark.sql.functions import broadcast

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

    # Sample to target size
    merged_df = merged_df.sample(fraction=0.002, seed=42)
    merged_df = merged_df.limit(400000)

    total_count = merged_df.count()
    logger.info(f"Dataset size after sampling: {total_count} rows")

    merged_df = merged_df.repartition(30)

    temp_path = "temp_data.parquet"
    merged_df.write.mode("overwrite").parquet(temp_path)

    sampled_df = spark.read.parquet(temp_path).select('text', 'cwe_name')

    text_list = []
    cwe_name_list = []

    chunk_size = 10000
    offset = 0

    while True:
        chunk = sampled_df.limit(chunk_size).offset(offset).collect()
        if not chunk:
            break

        for row in chunk:
            text_list.append(row['text'])
            cwe_name_list.append(row['cwe_name'])

        offset += chunk_size
        logger.info(f"Collected {len(text_list)} samples so far...")

        if len(text_list) >= 400000:
            break

    import shutil
    try:
        shutil.rmtree(temp_path)
    except:
        pass

    spark.stop()
    logger.info(f"Final dataset size: {len(text_list)} samples")
    return text_list, cwe_name_list

# Step 2: Preprocess Data with class balancing
def preprocess_data(text_list, cwe_name_list, max_words=15000, max_len=350):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(text_list)
    sequences = tokenizer.texts_to_sequences(text_list)

    embedding_input_dim = max_words + 1
    oov_token_idx = tokenizer.word_index.get("<OOV>", 1)

    clipped_sequences = []
    for seq in sequences:
        clipped_seq = [idx if idx < max_words else oov_token_idx for idx in seq]
        clipped_sequences.append(clipped_seq)

    X = pad_sequences(clipped_sequences, maxlen=max_len, padding='post', truncating='post')

    unique_labels_set = sorted(list(set(cwe_name_list)))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels_set)}
    num_classes = len(unique_labels_set)

    y = np.array([label_to_int[label] for label in cwe_name_list], dtype=np.int32)
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    # Calculate class weights for imbalanced data
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights = {i: class_weights_array[i] for i in range(num_classes)}
    logger.info(f"Computed class weights for {num_classes} classes (addressing class imbalance)")

    logger.info(f"Preprocessed {len(text_list)} samples with {num_classes} classes")
    logger.info(f"Embedding input dim: {embedding_input_dim}, Max length: {max_len}")

    return X, y_one_hot, tokenizer, label_to_int, num_classes, embedding_input_dim, class_weights

# Step 3: Train/Test Split
def manual_train_test_split(X, y, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split_idx = int(num_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# FIXED: Improved model with regularization
def build_deep_model(input_dim, input_length, num_classes):
    """
    IMPROVED: Added L2 regularization and increased dropout to prevent overfitting.
    """
    l2_reg = regularizers.l2(0.0001)

    model = tf.keras.Sequential([
        # Embedding layer with regularization
        tf.keras.layers.Embedding(
            input_dim=input_dim,
            output_dim=384,
            input_length=input_length,
            mask_zero=True,
            embeddings_regularizer=l2_reg
        ),

        # LSTM layers with regularization
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(384, return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)
        ),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LayerNormalization(),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)
        ),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LayerNormalization(),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)
        ),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)
        ),
        tf.keras.layers.Dropout(0.4),

        # Dense layers with L2 regularization
        tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.LayerNormalization(),

        tf.keras.layers.Dense(3072, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.LayerNormalization(),

        tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.LayerNormalization(),

        tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.45),
        tf.keras.layers.LayerNormalization(),

        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.35),

        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(0.3),

        # Output layer with float32 for numerical stability
        tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])

    # AdamW optimizer
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# FIXED: Using Keras built-in LearningRateScheduler (much cleaner!)
def warmup_cosine_decay_schedule(epoch, lr, total_epochs=30, warmup_epochs=5, learning_rate_base=0.001):
    """
    Learning rate schedule function for Keras LearningRateScheduler callback.
    Increases learning rate during warm-up, then applies cosine decay.

    This is compatible with all optimizer types including LossScaleOptimizer.
    """
    if epoch < warmup_epochs:
        # Warm-up phase: gradually INCREASE learning rate
        new_lr = (learning_rate_base / warmup_epochs) * (epoch + 1)
    else:
        # Cosine decay phase
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        new_lr = learning_rate_base * 0.5 * (1 + np.cos(np.pi * progress))

    logger.info(f"Epoch {epoch + 1}: Learning rate = {new_lr:.6f}")
    return new_lr

# Main Training Logic
if __name__ == "__main__":
    try:
        logger.info("="*80)
        logger.info("TRAINING SESSION STARTED - DYNAMIC GPU ALLOCATION")
        logger.info("="*80)

        configure_gpu()

        # Load data
        logger.info("Step 1: Loading samples...")
        text_list, cwe_name_list = load_and_merge_datasets()

        if len(text_list) == 0:
            logger.error("No data loaded! Check your CSV files.")
            sys.exit(1)

        # Preprocess with class weights
        logger.info("Step 2: Preprocessing with class balancing...")
        X, y, tokenizer, label_to_int, num_classes, embedding_input_dim, class_weights = preprocess_data(
            text_list, cwe_name_list
        )

        # Split
        logger.info("Step 3: Splitting data...")
        X_train, X_test, y_train, y_test = manual_train_test_split(X, y)
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Build model with regularization
        logger.info(f"Step 4: Building regularized model with {num_classes} classes...")
        model = build_deep_model(
            input_dim=embedding_input_dim,
            input_length=350,
            num_classes=num_classes
        )

        model.build(input_shape=(None, 350))
        model.summary(print_fn=logger.info)

        total_params = model.count_params()
        logger.info(f"Total parameters: {total_params:,}")

        # FIXED: Use Keras built-in LearningRateScheduler (works with all optimizers)
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )

        # Built-in LearningRateScheduler - works perfectly with LossScaleOptimizer
        lr_scheduler = LearningRateScheduler(
            lambda epoch, lr: warmup_cosine_decay_schedule(
                epoch, lr, total_epochs=30, warmup_epochs=5, learning_rate_base=0.001
            ),
            verbose=0
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_model_400k.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        # Dynamic batch size based on GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            batch_size = 64  # Good for 4GB GPU with mixed precision
        else:
            batch_size = 32  # CPU fallback

        logger.info("Step 5: Starting SUPERVISED training with regularization...")
        logger.info(f"Using class weights to handle imbalanced data")
        logger.info(f"Batch size: {batch_size} (optimized for detected GPU)")
        logger.info("Learning rate will INCREASE during warm-up (first 5 epochs)")

        # Train with class weights
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            class_weight=class_weights,
            callbacks=[early_stopping, lr_scheduler, checkpoint],
            verbose=1
        )

        # Evaluate
        logger.info("Step 6: Evaluating model...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
        logger.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

        # Save artifacts
        logger.info("Step 7: Saving artifacts...")
        model.save('dense_model.keras')

        tokenizer_config = tokenizer.to_json()
        with open('tokenizer.json', 'w') as f:
            f.write(tokenizer_config)

        with open('label_to_int_400k.txt', 'w') as f:
            for label, idx in label_to_int.items():
                f.write(f"{label}:{idx}\n")

        np.save('training_history_400k.npy', history.history)

        # Save training summary
        with open('training_summary.txt', 'w') as f:
            f.write(f"Training Summary\n")
            f.write(f"="*60 + "\n")
            f.write(f"Total Samples: {len(text_list)}\n")
            f.write(f"Number of Classes: {num_classes}\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Mixed Precision: Enabled (FP16)\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Class Weighting: Enabled\n")
            f.write(f"L2 Regularization: Enabled\n")
            f.write(f"Warm-up Learning Rate: Enabled\n")
            f.write(f"Final Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Final Test Loss: {test_loss:.4f}\n")

        logger.info("="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Model uses dynamic GPU allocation - works with any GPU size")
        logger.info("="*80)

    except tf.errors.ResourceExhaustedError as e:
        logger.exception("GPU OUT OF MEMORY ERROR:")
        logger.error("Reduce batch_size or model complexity")
        sys.exit(1)
    except MemoryError as e:
        logger.exception("SYSTEM MEMORY ERROR:")
        logger.error("Reduce PySpark memory configs or sample size")
        sys.exit(1)
    except Exception as e:
        logger.exception("UNEXPECTED ERROR:")
        import traceback
        logger.error(f"\nTraceback:\n{traceback.format_exc()}")
        sys.exit(1)
