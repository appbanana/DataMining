import numpy as np
import PIL as pil
import skimage


def create_captcha(text, shear, size=(100, 24)):
    im = pil.Image.new('L', size, 'black')
    draw = pil.ImageDraw.Draw(im)
    font = pil.ImageFont.truetype(r'../data/Coval-Book.otf', 22)
    draw.text((0, 0), text, fill=1, font=font)
    image = np.array(im)
    affine_tf = skimage.transform.AffineTransform(shear=shear)
    image = skimage.transform.wrap(image, affine_tf)
    return image / image.max()


if __name__ == '__main__':
    image = create_captcha("GENE", shear=0.5)
