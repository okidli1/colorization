Running the model should be easy. I have provided weights for the model we used in the report, and there is code to skip training if weights are already saved. In the bottom cell, you should be able to enter the path of any black and white image in order to colourize it. I have provided some of the tests we used in case you would like to recreate them.

# Image Colorization Model

### How to Reproduce Main Figures
This repository contains the model weights `colorization_model_weights.pth` so you do not need to train the model to reproduce any of the results or figures shown in the report.

Run the main notebook `colorize_model.ipynb`. It will:
1. Load the CIFAR-10 dataset
2. Convert image to grayscale
3. Use the pretrained model to predict RGB colors
4. Display the output as Original vs Grayscale vs Colorized images

### Colorize Your Own Grayscale Image
Download any grayscale image and edit the image path in the notebook to your own image:

`img_path = "path/to/your/image.jpg"`

Run the notebook and it will display the colorized output and your grayscale image side-by-side.
