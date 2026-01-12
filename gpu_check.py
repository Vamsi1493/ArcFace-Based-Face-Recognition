import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs available:", len(gpus))
for gpu in gpus:
    print(gpu)
    



