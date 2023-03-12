from keras.layers import Input
from keras.optimizers import gradient_descent_v2
from models.slowfast import SlowFast_body, bottleneck
import cv2
import numpy as np
import os

def resnet50(inputs, **kwargs):
    model = SlowFast_body(inputs, [3, 4, 6, 3], bottleneck, **kwargs)
    return model

def frames_from_video(video_dir, nb_frames = 25, img_size = 224):

    # Opens the Video file
    cap = cv2.VideoCapture(video_dir)
    i=0
    frames = []
    while(cap.isOpened() and i<nb_frames):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()
    return np.array(frames) / 255.0

def predictions(video_dir, model, nb_frames = 25, img_size = 224):

    X = frames_from_video(video_dir, nb_frames, img_size)
    X = np.reshape(X, (1, nb_frames, img_size, img_size, 3))
    
    predictions = model.predict(X)
    preds = predictions.argmax(axis = 1)

    classes = []
    with open(os.path.join('output', 'classes.txt'), 'r') as fp:
        for line in fp:
            classes.append(line.split()[1])

    for i in range(len(preds)):
        print('Prediction - {} -- {}'.format(preds[i], classes[preds[i]]))

# Load the model with pre-configured parameters
x = Input(shape = (25, 224, 224, 3))
model = resnet50(x, num_classes=14)

model.compile(loss='categorical_crossentropy',
              optimizer=gradient_descent_v2.SGD(learning_rate=0.01, momentum=0.9), 
              metrics=['accuracy'])

model.summary()

# We load the weights direclty
model.load_weights('output/slowfast_finalmodel.hdf5')

model.save('output/slowfast_finalmodel_new.hdf5')

predictions(video_dir = 'test/Arrest048_x264_21.mp4', model = model, nb_frames = 25, img_size = 224)