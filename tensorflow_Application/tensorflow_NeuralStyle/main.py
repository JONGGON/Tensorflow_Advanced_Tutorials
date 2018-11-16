import os
import model

# download URL : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
if not os.path.exists("weights"):
    os.makedirs("weights")
model_file_path = 'weights/imagenet-vgg-verydeep-19.mat'

# content_a  / style_b = 1/1000
content_image = "content/person.jpg"
style_image = "style/seated-nude.jpg"
initial_noise_image = "content_image"  # or style image or noise -> Assigning an initial value to the content image is faster than assigning noise.

# image_size height , width -> is expected to be at least 224.
# optimizer_selection -> Adam, RMSP, SGD
model.neuralstyle(model_file_path=model_file_path, epoch=500, show_period=100, optimizer_selection="Adam", \
                  learning_rate=0.1,
                  image_size=(380, 683), \
                  content_image=content_image, style_image=style_image, content_a=1, style_b=1000, \
                  initial_noise_image=initial_noise_image)
