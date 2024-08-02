from captcha.image import ImageCaptcha
import numpy as np
import random
import string
import os

def random_captcha_text(char_set=string.ascii_uppercase + string.digits, captcha_size=4):
    return ''.join(random.choices(char_set, k=captcha_size))

def generate_captcha_image(output_dir='data/raw', num_images=10000):
    image = ImageCaptcha(width=160, height=60)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(num_images):
        captcha_text = random_captcha_text()
        captcha = image.generate_image(captcha_text)
        captcha.save(os.path.join(output_dir, f'{captcha_text}_{i}.png'))

if __name__ == "__main__":
    generate_captcha_image()
