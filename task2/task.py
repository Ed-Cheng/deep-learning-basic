import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model
from PIL import Image
import numpy as np

class DenseNet3(Model):
    """ 
    A Keras Model subclass
    Contains the DenseNet3 Model I built according to the cw requirements.

    Func:
        __init__: defines all layers that could be predifined
        dense_block: main structure, contains 4 conv layers
        call: build and return the whole model to the user

    """
    def __init__(self, k, theta):
        """
        Defines all layers that could be predifined

        Args:
            k: scalar, the growth rate
            theta: scalar, reduces the percentage number of feature-maps at transition layers. 

        """
        super(DenseNet3, self).__init__()
        # define all layers except dense_block

        self.k = k
        self.theta = theta
        self.compress = 6*k * theta # was initially tf.keras.backend.int_shape(d)[-1], we found that equals 6*k by experiment

        # The convolution & pooling layer
        self.conv_init = layers.Conv2D(2*self.k, 7, strides=2, padding='same')
        self.pool_init = layers.MaxPool2D(3, strides=2, padding='same')

        # The layers that dense_block and transition will use
        self.batchNorm = layers.BatchNormalization()
        self.Relu = layers.ReLU()
        self.concat = layers.Concatenate()

        self.trans = layers.Conv2D(self.compress, 1, padding='same')
        self.avePool = layers.AvgPool2D(pool_size=2, strides=2, padding='same')

        # finishing layers
        self.globAve = layers.GlobalAvgPool2D()
        self.dense = layers.Dense(10, activation='softmax')


    # 4 convolutional layers in total 
    def dense_block(self, x):
        """
        Creates the dense block architecture, contains 4 conv layers

        Args:
            x: tenser, with n layers of dense block

        Return:
            x: tensor, with n+1 layers of dense block

        """
        for _ in range(4):
            # bottleneck
            y = layers.BatchNormalization()(x)
            y = self.Relu(y)
            y = layers.Conv2D(4*self.k, 1, padding='same')(y)

            # One 3X3 conv layer
            z = layers.BatchNormalization()(y)
            z = self.Relu(z)
            z = layers.Conv2D(1*self.k, 3, padding='same')(z)

            x = self.concat([x, z])
        return x

    # construct the whole model
    def call(self, input_tensor):
        """
        Use this function to create the final DenseNet3 model, contains 3 dense_block

        Args:
            input_tensor: tenser, inlcudes the shape of our traning data

        Return:
            self.dense(x): tensor, final build of the DenseNet3 model

        """
        # Start with one convolution and pooling layer
        x = self.conv_init(input_tensor)
        x = self.pool_init(x)

        # require 3 dense_block
        for _ in range(3):
            # One layer of dense_block
            d = self.dense_block(x)

            # One layer of transition
            d = self.trans(d)
            d = self.avePool(d)

        x = self.globAve(d)

        return self.dense(x)


# The paper used 16*16 as a cutout size for CIFAR-10, so our max s will be 16
def cutout(train_images):
    """
    Add one random size black box in random location to all the image files.
    Done by applying 0,1 masks (0 for the black box)

    Args:
        train_images: n by image_size vector, contains images

    Return:
        cutout_img: n by image_size vector, train_images with random size black box in random location
    """
    Img_size = train_images.shape
    cutout_img = np.copy(train_images)
    mask = np.ones(Img_size)

    # Randomly generate s and the location, boundaries are considered (this part is vectorized instead of one by one)
    s = np.random.randint(16, size=Img_size[0])
    # The (32-s) here assures that the cutout of s stays in the image
    row_cut = np.random.randint(32-s, size=Img_size[0])
    column_cut = np.random.randint(32-s, size=Img_size[0])

    for i in range(Img_size[0]):
      mask[i, row_cut[i]:row_cut[i] + s[i], column_cut[i]:column_cut[i] + s[i], :] = 0

    cutout_img = cutout_img * mask

    return cutout_img

def compile_train(k, theta, train_images, train_labels, test_images, test_labels):
    """
    train the DenseNet3 model based on the given parameters and images

    Args:
        k: scalar, the growth rate (the paper got best result on k=40)
        theta: scalar, reduces the percentage number of feature-maps at transition layers (the paper used theta=0.5)
        train_images: n by image_size vector, contains training_images
        train_labels: n by 1 vector, the label of the train_images
        test_images: n by image_size vector, contains testing_images
        test_labels n by 1 vector, the label of the testing_images
    Return:
        model: keras.engine model, the trained DenseNet3 model
        history: keras.callbacks, stores the information of the training process

    """
    input_shape = (32, 32, 3)
    inputs = layers.Input(input_shape)

    output = DenseNet3(k, theta).call(inputs)
    model = Model(inputs, output)
    model.summary()

    ## compile with loss and optimiser
    opt = tf.keras.optimizers.Adam(learning_rate=0.001) # 0.001 is the default adam learning rate (works pretty good)
    model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    ## train (tried batch_size={12,24,32,64} and 64 has the best trade-off for time-accuracy)
    history = model.fit(train_images, 
                        train_labels, 
                        batch_size=64,
                        epochs=10,
                        validation_data=(test_images, test_labels))
    print('Training done.')

    return model, history



if __name__ == '__main__':
    """
    1. Load train, test data for validation training
    2. Add cutout to train data
    3. Train the DenseNet3 model and save it
    4. Load the saved model and test data to visualize the result (36 png saved with captions printed out)
    """
    ## cifar-10 training dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # normalize to [0,1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0


    # Show 16 examples of cutouts
    cutout_images = cutout(train_images)
    store_img = tf.concat([cutout_images[i,...] for i in range(16)],1).numpy()
    im = Image.fromarray((store_img * 255).astype(np.uint8))
    im.save("cutout.png")

    # Using the best combination of k and theta given in the paper 
    model, history = compile_train(40, 0.5, cutout_images, train_labels, test_images, test_labels)
    
    # save trained model
    model.save('saved_model_tf')
    print('Model saved.')


    ## load the trained model and convert test_images back for final plotting
    loaded_model = models.load_model('saved_model_tf')

    ## cifar-10 testing dataset
    (_, _), (test_images, test_labels) = datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    ## load the trained model
    new_model = models.load_model('saved_model_tf')

    ## inference
    num_images = 36
    outputs = new_model.predict(test_images[:num_images,...]/255.0)
    predicted = tf.argmax(outputs, 1)
    acc = 0
    for i in range(num_images):
      print('Ground-truth vs Predicted: ', class_names[test_labels[i,0]],'\t', class_names[predicted[i]])
      if class_names[test_labels[i,0]] == class_names[predicted[i]]:
        acc += 1
    print('accuray: ', acc/num_images)

    # example images
    im = Image.fromarray(tf.concat([test_images[i,...] for i in range(num_images)],1).numpy())
    im.save("result.png")
    print('result.png saved.')