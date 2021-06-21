from keras.models import load_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras import models

model = load_model('model.h5')

def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def visualize(current_activation):
    cols = 5
    rows = 5
    f, ax = plt.subplots(5, cols, figsize=(30, 5))

    for i in range(rows):
        for j in range(cols):
            index = i*cols + j
            if index < current_activation.shape[-1]:
                ax[i][j].imshow(current_activation[0, :, :, i*cols + j], cmap='viridis')
            ax[i][j].axis('off')

    file_name = "visualization.png"
    plt.savefig(file_name, dpi=300)


def main():

    model = load_model('model.h5')
    center_image = process_image(mpimg.imread('./center_lane.jpg'))
    img = np.expand_dims(center_image, axis=0)
    layer_outputs = [layer.output for layer in model.layers[:15]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img, batch_size=1)
    layer_2 = activations[2]
    visualize(layer_2)


if __name__ == '__main__':
    main()