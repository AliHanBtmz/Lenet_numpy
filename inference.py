from model.lnet5 import LNet5
import numpy as np
import os
import cv2
from natsort import natsorted

model = LNet5()
model.load_weights("lenet5_weights_full.npz")

test_path = "my_test_images/"
for file in natsorted(os.listdir(test_path)):
    img = cv2.imread(test_path + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 1, 28, 28)
    img = img.astype(np.float32)
    img = img / 255.0
    pred = model.predict(img)
    print("Orginal images: ", file[0], "Predicted: ", pred)

    orginal_img = cv2.imread(test_path + file, cv2.IMREAD_COLOR)
    orginal_img = cv2.resize(orginal_img, (28, 28))
    cv2.imshow("results", orginal_img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
