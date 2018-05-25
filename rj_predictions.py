from keras.models import load_model
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle

def resize_to_fit(image, width, height):

    (h, w) = image.shape[:2]

    # Resize the image according to which parameter is bigger
    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)

    # Simple functions to add required padding
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # Bordering the image with the padding
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    return image


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "tester"


# Let's load up our model
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained convolutional neural network
model = load_model(MODEL_FILENAME)

# Let's get some random images from our test data set. 
# Remember this is a very simple captcha classifier, and does not account for more advanced captchas
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)

for image_file in captcha_image_files:

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letter_image_regions = []
    contours = contours[0] if imutils.is_cv2() else contours[1]

    for cnt in contours:
	#print cnt
        (x,y,w,h) = cv2.boundingRect(cnt)
        
        # If width is significantly more than the height, there might be two characters overlapping,
        # hence we need to split it in between
        if w/h>1.25:
            new_w=w/2
            letter_image_regions.append((x,y,new_w,h))
            letter_image_regions.append((x+new_w,y,new_w,h))
        else:
            letter_image_regions.append((x,y,w,h))

    letter_image_regions.sort(key=lambda x:x[0])

    predictions = []

    # loop over the lektters
    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box

        letter_image = image[y - 1:y + h + 1, x - 1:x + w + 1]

        letter_image = resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        prediction = model.predict(letter_image)

        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)


    # Print the captcha's text
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))
