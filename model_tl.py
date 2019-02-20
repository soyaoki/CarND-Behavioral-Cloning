from keras.applications.inception_v3 import InceptionV3, preprocess_input
import cv2
import numpy as np
import sklearn
import keras
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Lambda,  Dropout, Cropping2D, Input, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from random import shuffle
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from keras import backend as K
from keras.utils.vis_utils import plot_model

PATH = './data' 
#PATH = './train_data' # track1
#PATH = './train_data_2' # track2

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

# InceptionV3 need (299,299,3)
def img_resize(image):
    return cv2.resize(image, dsize=(299, 299))

# Data generator
def generator(samples, batch_size=32):
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

#####################################################
###################### Pipeline #####################
#####################################################
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

    IMAGE_HEIGHT = 160
    IMAGE_WIDTH = 320
    IMAGE_CHANNELS =3
    batch_size=32
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    lis = glob.glob(PATH+"/IMG/*")
    print(lis[1])
    test_image = cv2.imread(lis[1])
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    print(test_image.shape)
    test_image_crp = test_image[60:-25, :, :]
    test_image_rsz = img_resize(test_image_crp)
    plt.figure()
    plt.subplot(131)
    plt.imshow(test_image)
    plt.title('Image from simulator')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(test_image_rsz)
    plt.title('Cropped & resized image')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(np.fliplr(test_image_rsz))
    plt.title('Flipped image')
    plt.axis('off')
    plt.savefig("images_tl.png", facecolor="azure", bbox_inches='tight', pad_inches=0)


    # ------------- Model architecture -------------
    # Inpur image form simulator directly
    input_image_tensor = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)) # from simulator
    ppl = Lambda(lambda x: (x /255.0 -0.5)*2)(input_image_tensor) # pre-process of InceptionV3
    crpl = Cropping2D(cropping=((60,25),(0,0)))(ppl) # crop
    rszl = Lambda(lambda image: K.tf.image.resize_images(image, (299, 299)))(crpl) # resize

    img_width, img_height = 299, 299
    input_tensor = Input(shape=(img_width, img_height, 3)) # RGB
    model_v3 = InceptionV3(weights='imagenet', include_top=False, input_tensor=rszl)
#    plot_model(model_v3, to_file="inceptionV3.png", show_shapes=True)
    for i, layer in enumerate(model_v3.layers):
       print(i, layer.name)

    print(model_v3.layers[43].name)
    print(model_v3.layers[43].output_shape)
    print(model_v3.layers[43])
    print(model_v3.input)
    print(model_v3.output)

    # 40th layer is 1st mixed layer
#    l1 = Dropout(0.5)(model_v3.layers[43].output)
    # 20th layer is start of convolution
    l1 = Dropout(0.5)(model_v3.layers[20].output)
    l2 = Flatten()(l1)
    l3 = Dense(100, activation='elu')(l2)
    l4 =Dense(50, activation='elu')(l3)
    l5 = Dense(10, activation='elu')(l4)
    l6 =Dense(1)(l5)

    model_tl = Model(inputs=model_v3.input, output=[l6])
    model_tl.summary()

    for i, layer in enumerate(model_tl.layers):
       print(i, layer.name)

    # Params of InceptionV3 are setted not to trainable.
#    for layer in model_tl.layers[:43]:
#        layer.trainable = False
    for layer in model_tl.layers[:20]:
        layer.trainable = False

    model_tl.summary()
    plot_model(model_tl, to_file="model_tl.png", show_shapes=True)

    # As we want to train model agan and again, load last model's params.
    if os.path.isfile("model_tl.h5"):
        model_tl.load_weights("model_tl.h5", by_name=True)
        print("parames had been loaded.")
    # ------------- Model architecture -------------

    model_tl.compile(loss='mse', optimizer='adam')
    history_object = model_tl.fit_generator(train_generator, samples_per_epoch = 128, validation_data = validation_generator, nb_val_samples = 128, nb_epoch=5, verbose=1)
    model_tl.save('model_tl.h5')

    # print the keys contained in the history object
    print(history_object.history.keys())

    # plot the training and validation loss for each epoch
    plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("training_results_tl.png")

    # visualize feature of 1st layer
    layer_name = 'conv2d_1'
    hidden_layer_model = Model(inputs=model_tl.input, outputs=model_tl.get_layer(layer_name).output)
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
    plt.savefig("hidden_layer_output_tl.png", facecolor="azure", bbox_inches='tight', pad_inches=0)
