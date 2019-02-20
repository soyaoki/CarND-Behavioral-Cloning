import os
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import sklearn
import keras
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Lambda,  Dropout, Cropping2D, ELU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from random import shuffle
import glob
from keras.utils.vis_utils import plot_model

PATH = './data' 
PATH = './train_data' # track1
PATH = './train_data_2' # track2

# Data visualizitation
# First of all, we need to look data distribution
def visualizing_data(samples, name):
    angle = []
    for line in samples:
        angle.append(float(line[3]))
    
    plt.figure()
    plt.hist(angle, bins=20,range=(-1,1))
    plt.xlabel('Steering angle')
    plt.ylabel('Frequency')
    plt.savefig(name+".png")

# Data Augumentation
# There are too many data of steering angle = 0. So we need augumentation of the angle > 0.1 or < -0.1
def flip_img(images, angles, images_left, images_right, angles_left, angles_right, batch_size=32):
    ind = np.where(np.abs(angles) > 0.1)[0]
    images_flipped = np.empty([len(ind), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    images_l = np.empty([len(ind), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    images_flipped_l = np.empty([len(ind), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    images_r = np.empty([len(ind), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    images_flipped_r = np.empty([len(ind), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    
    angles_flipped = []
    angles_l = []
    angles_flipped_l = []
    angles_r = []
    angles_flipped_r = []
    
    for i in range(0, len(ind)):
        images_flipped[i] = np.fliplr(images[ind[i]])
        images_l[i] = images_left[ind[i]]
        images_flipped_l[i] = np.fliplr(images_right[ind[i]])
        images_r[i] = images_right[ind[i]]
        images_flipped_r[i] = np.fliplr(images_right[ind[i]])
        
        angles_flipped.append(-angles[ind[i]])
        angles_l.append(angles_left[ind[i]])
        angles_flipped_l.append(-angles_left[ind[i]])
        angles_r.append(angles_right[ind[i]])
        angles_flipped_r.append(-angles_right[ind[i]])
    
    angles.extend(angles_flipped)
    angles.extend(angles_l)
    angles.extend(angles_flipped_l)
    angles.extend(angles_r)
    angles.extend(angles_flipped_r)
    
    return np.vstack((images, images_flipped, images_l, images_flipped_l, images_r, images_flipped_r)), np.array(angles)

# Data generator
def generator(samples):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size] # batch sample

            images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
            images_left = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
            images_right = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
            angles = []
            angles_left = []
            angles_right = []
            
            for i in range(0, len(batch_samples)):
                batch_sample = batch_samples[i]
                name = PATH+'/IMG/'+batch_sample[0].split('/')[-1]
                name_l = PATH+'/IMG/'+batch_sample[1].split('/')[-1]
                name_r = PATH+'/IMG/'+batch_sample[2].split('/')[-1]
                images[i] = mpimg.imread(name)
                angles.append(float(batch_sample[3]))

                images_left[i] = mpimg.imread(name_l)
                images_right[i] = mpimg.imread(name_r)
                angles_left.append(angles[i] + 0.2)
                angles_right.append(angles[i] - 0.2)
            
            # trim image to only see section with road
            s = len(angles)
            images = images[:s,:,:,:]
            images_left = images_left[:s,:,:,:]
            images_right = images_right[:s,:,:,:]
            X_train, y_train = flip_img(images, angles, images_left, images_right, angles_left, angles_right)
            yield sklearn.utils.shuffle(X_train, y_train) # each batch, return X and y for training

    
if __name__ == '__main__':
    samples = []
    with open(PATH+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print("Train data : "+str(len(train_samples)))
    print("Valid data : "+str(len(validation_samples)))

    # Visualize data distribution
    visualizing_data(train_samples,"train")
    visualizing_data(validation_samples,"validation")

    # Size of image from simulator
    IMAGE_HEIGHT = 160
    IMAGE_WIDTH = 320
    IMAGE_CHANNELS = 3
    batch_size = 32

    # compile and train the model using the generator function
    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

    # show images
    lis = glob.glob(PATH+"/IMG/*")
    print(lis[1])
    
    c = mpimg.imread(lis[0])
    l = mpimg.imread(lis[1])
    r = mpimg.imread(lis[2])
    cf = np.fliplr(c)
    lf = np.fliplr(l)
    rf = np.fliplr(r)
    plt.figure()
    plt.subplot(231)
    plt.imshow(c)
    plt.title('Center')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(cf)
    plt.title('Center (Flipped)')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(l)
    plt.title('Left')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(lf)
    plt.title('Left (Flipped)')
    plt.axis('off')    
    plt.subplot(233)
    plt.imshow(r)
    plt.title('Right')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rf)
    plt.title('Right (Flipped)')
    plt.axis('off')
    plt.savefig("images_aug.png", facecolor="azure", bbox_inches='tight', pad_inches=0)
    
    test_image = mpimg.imread(lis[1])
    print(test_image.shape)
    test_image_crp = test_image[70:-25, :, :]
    plt.figure()
    plt.subplot(131)
    plt.imshow(test_image)
    plt.title('Image from simulator')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(test_image_crp)
    plt.title('Croped imgae')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(np.fliplr(test_image_crp))
    plt.title('Flipped image')
    plt.axis('off')
    plt.savefig("images.png", facecolor="azure", bbox_inches='tight', pad_inches=0)

    # ------------------ NVIDIA Model ------------------
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x /255.0 -0.5, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))
    # Croping
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    # 1x1 Convolution
    model.add(Conv2D(48,(1,1),activation="elu"))
    # NVIDIA model
    model.add(Conv2D(24,(5,5),strides=(2,2),activation="elu"))
    model.add(Conv2D(36,(5,5),strides=(2,2),activation="elu"))
    model.add(Conv2D(48,(5,5),strides=(2,2),activation="elu"))
    model.add(Conv2D(64,(3,3),activation="elu"))
    model.add(Conv2D(64,(3,3),activation="elu"))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.summary()
    
    plot_model(model, to_file="model.png", show_shapes=True)
    
    if os.path.isfile("model.h5"):
        model.load_weights("model.h5", by_name=False)
        print("parames had been loaded.")

    model.compile(loss='mse', optimizer='adam')
#    history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples) - len(train_samples)%batch_size, validation_data = validation_generator, 
#                                         nb_val_samples = len(validation_samples) - len(validation_samples)%batch_size, nb_epoch=5, verbose=1)
    history_object = model.fit_generator(train_generator, samples_per_epoch = 128, validation_data = validation_generator, 
                                         nb_val_samples = 128, nb_epoch=1, verbose=1)

    model.save('model.h5')

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("training_results.png")

    """
    If the above code throw exceptions, try 
    model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
    validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
    """

    # visualize feature of 1st layer
    layer_name = 'conv2d_1'
    hidden_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    vis_img = np.empty([1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    vis_img[0,:,:,:] = test_image
    print(vis_img.shape)
    hidden_output = hidden_layer_model.predict(vis_img)
    print(hidden_output.shape)
    plt.figure()
    plt.subplot(311)
    plt.imshow(hidden_output[0,:,:,0])
    plt.title('Hidden layer output 1')
    plt.axis('off')
    plt.subplot(312)
    plt.imshow(hidden_output[0,:,:,1])
    plt.title('Hidden layer output 2')
    plt.axis('off')
    plt.subplot(313)
    plt.imshow(hidden_output[0,:,:,2])
    plt.title('Hidden layer output 3')
    plt.axis('off')
    plt.savefig("hidden_layer_output.png", facecolor="azure", bbox_inches='tight', pad_inches=0)
    