import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


def check_gpu():
    # Check if TensorFlow can detect GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs detected:")
        for gpu in gpus:
            print(f" - {gpu.name} ({gpu.device_type})")
        
        # Print GPU memory information
        for i, gpu in enumerate(gpus):
            try:
                # Use the correct format for device name
                details = tf.config.experimental.get_memory_info(f"GPU:{i}")
                print(f"Memory Info for {gpu.name}:")
                print(f" - Total memory: {details['current'] / 1024 ** 2:.2f} MB")
                print(f" - Peak memory: {details['peak'] / 1024 ** 2:.2f} MB")
            except Exception as e:
                print(f"Failed to get memory info for {gpu.name}: {e}")
        
        # Validate TensorFlow is running on GPU
        try:
            with tf.device('/GPU:0'):
                print("Running a small test computation on GPU...")
                test_tensor = tf.constant([1.0, 2.0, 3.0]) ** 2
                print("Test computation result:", test_tensor.numpy())
        except Exception as e:
            print(f"Error running computation on GPU: {e}")
    else:
        print("No GPUs detected by TensorFlow. Please check your GPU setup.")

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    check_gpu()
