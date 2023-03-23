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

def compile_train(k, theta, summary, train_images, train_labels, test_images, test_labels):
    """
    train the DenseNet3 model based on the given parameters and images

    Args:
        k: scalar, the growth rate (the paper got best result on k=40)
        theta: scalar, reduces the percentage number of feature-maps at transition layers (the paper used theta=0.5)
        summary: boolin, if True print the model summary
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
    if summary:
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

def split_kfold(kfold, image, label):
    """
    split the data into k fold 

    Args:
        kfold: scalar, number of folds
        image: n by image_size vector, contains images
        label: n by 1 vector, the label of the images
    Return:
        kfold_data: kfold by floor(n/kfold) by image_size vector, contains splited images
        kfold_label: kfold by floor(n/kfold) vector, contains label of kfold_data
    """
    nData = label.shape[0]
    # There will be max (kfold-1) data loss. Ignorable compare to whole dataset. 
    nDataPerFold = int(np.floor(nData/kfold))

    # Create array to store the splited data
    kfold_data = np.zeros(np.append((kfold, nDataPerFold), image[0].shape))
    kfold_label =np.zeros(np.append((kfold, nDataPerFold), label[0].shape))
    for i in range(kfold):
        start = nDataPerFold*i
        end = nDataPerFold*(i+1)
        kfold_data[i] = image[start:end]
        kfold_label[i] = label[start:end]

    return kfold_data, kfold_label

def perform_kfold_CV(kfold, split_set, split_lab, test_set, test_lab, data_name):
    """
    perform k-fold cross-validation and check the performance with test set

    Args:
        kfold: scalar, number of folds
        split_set: 0.8*n by image_size vector, contains 80% total images
        split_lab: 0.8*n by 1 vector, the label of the split_set
        test_set: 0.2*n by image_size vector, contains 20% total images
        test_lab: 0.2*n by 1 vector, the label of the test_set
        data_name: string, printed out during training to clarify which cv session this belongs
    Return:
        ave_val_acc: scalar, average validation accuracy over the k-fold cv
        ave_val_loss: scalar, average validation loss over the k-fold cv
        ave_test_acc: scalar, average test accuracy over the k-fold cv
    """

    # empty list to store the results
    val_acc_per_fold = []
    val_loss_per_fold = []
    test_acc_per_fold = []

    for cv in range(kfold):
        # Split the images into train and val 
        test_images = split_set[cv]
        test_labels = split_lab[cv]

        train_idx = [i for i in range(kfold)]
        train_idx.remove(cv)
        train_images = split_set[train_idx[0]]
        train_labels = split_lab[train_idx[0]]
        for to_train in range(kfold-2):
            train_images = np.vstack((train_images, split_set[train_idx[to_train+1]]))
            train_labels = np.vstack((train_labels, split_lab[train_idx[to_train+1]]))

        # summary of the train/val set split
        print(f"\nTotal dataset {[i for i in range(kfold)]} \n" +
              f"Training with set {train_idx}, train_images shape: {train_images.shape} \n" +
              f"Validating with set [{cv}], test_images shape: {test_images.shape}\n")


        model, history = compile_train(40, 0.5, False, train_images, train_labels, test_images, test_labels)

        num_test = test_lab.shape[0]
        outputs = model.predict(test_set)
        predicted = tf.argmax(outputs, 1)
        test_acc = np.sum(np.where(test_lab[:,0] == predicted[:], 1, 0)) / num_test

        test_acc_per_fold.append(test_acc)
        val_acc_per_fold.append(history.history['val_accuracy'][-1])
        val_loss_per_fold.append(history.history['val_loss'][-1])

        print(f"-------------------- Fold {cv+1} finished ({data_name}) --------------------")

    ave_test_acc = np.round(np.mean(test_acc_per_fold),3)
    ave_val_acc = np.round(np.mean(val_acc_per_fold),3)
    ave_val_loss = np.round(np.mean(val_loss_per_fold),3)

    print(f"\nAverage val_accuracy: {ave_val_acc} \n" +
    f"Average val_loss: {ave_val_loss} \n" +
    f"Average test_accuracy: {ave_test_acc} \n" +
    f"-------------------- Cross-Validation finished ({data_name}) --------------------\n")

    return ave_val_acc, ave_val_loss, ave_test_acc

if __name__ == '__main__':
    """
    0. Print our my choice of modification
    1. Load train, test data. Merge and shuffle them.
    2. Split the dataset into develop-test (80-20) and 3-fold (evenly divided by 3)
    3. Performing cross-validation on cutout and non-cutout data
    4. Train 2 models with full development set on cutout and non-cutout data
    5. Summarize the performance
    """
    print("\nI choose 'with and without the Cutout data augmentation' \n")

    ## cifar-10 training dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # normalize to [0,1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Merge and shuffle train and test set (we will split into 80-20 development set and test set later)
    all_labels = np.concatenate((train_labels, test_labels), axis=0)
    all_images = np.concatenate((train_images, test_images), axis=0)
    all_images_cut = np.concatenate((cutout(train_images), cutout(test_images)), axis=0)

    shuffle_idx = np.random.permutation(all_labels.shape[0])
    all_images = all_images[shuffle_idx]
    all_labels = all_labels[shuffle_idx]

    # Split develop-test (80-20), split K-fold, get cutout data
    kfold_k = 3
    total_develop_set = int(np.floor(0.8*all_labels.shape[0]))

    test_set, test_lab = all_images[total_develop_set:], all_labels[total_develop_set:]
    develop_set, develop_label = all_images[:total_develop_set], all_labels[:total_develop_set]
    develop_set_cut, develop_label_cut = cutout(develop_set), develop_label

    split_set, split_lab = split_kfold(kfold_k, develop_set, develop_label)
    split_set_cut, split_lab_cut = split_kfold(kfold_k, develop_set_cut, develop_label_cut)

    # performing cross-validation
    NoCut_val_acc, NoCut_val_loss, NoCut_test_acc = perform_kfold_CV(kfold_k, split_set, split_lab, test_set, test_lab, "No cutout")
    Cut_val_acc, Cut_val_loss, Cut_test_acc = perform_kfold_CV(kfold_k, split_set_cut, split_lab_cut, test_set, test_lab, "With cutout")

    # Train with full development set
    print("\nStart training with full development set without cutout\n")
    model_nocut, history_nocut = compile_train(40, 0.5, False, develop_set, develop_label, test_set, test_lab)
    model_nocut.save('saved_model_nocut_tf')
    print('model_nocut saved.')
    
    print("\nStart training with full development set with cutout\n")
    model_cut, history_cut = compile_train(40, 0.5, False, develop_set_cut, develop_label_cut, test_set, test_lab)
    model_cut.save('saved_model_cut_tf')
    print('model_cut saved.')

    # The validation set in the entire development set is the holdout test set
    nocut_acc = np.round(history_nocut.history['val_accuracy'][-1],3)
    nocut_loss = np.round(history_nocut.history['val_loss'][-1],3)

    cut_acc = np.round(history_cut.history['val_accuracy'][-1],3)
    cut_loss = np.round(history_cut.history['val_loss'][-1],3)

    print("-------------------- Final Summary -------------------- \n" +
          f"Cross-Validation:\n" + 
          f"With cutout, test set avg acc  = {Cut_test_acc}, avg val_acc = {Cut_val_acc}, avg val_loss = {Cut_val_loss}\n" +
          f"No   cutout, test set avg acc = {NoCut_test_acc}, avg val_acc = {NoCut_val_acc}, avg val_loss = {NoCut_val_loss}\n" +
          f"Full development set:\n" + 
          f"With cutout, test set acc = {cut_acc}, test set loss = {cut_loss} \n" +
          f"No   cutout, test set acc = {nocut_acc}, test set loss = {nocut_loss} \n" +
          "------------------------------------------------------- \n")