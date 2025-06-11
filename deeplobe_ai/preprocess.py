from PIL import Image, ExifTags, ImageOps
import numpy as np


# auto-orient the image based on exif metadata
def auto_orient(img):
    # Check if the image has EXIF data
    if hasattr(img, "getexif") and img.getexif() is not None:
        # Retrieve the orientation from the EXIF data
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = dict(img.getexif().items())
        orientation = exif.get(orientation)
        if orientation is not None:
            if orientation == 1:
                # Normal orientation
                pass
            elif orientation == 2:
                # Flip the image horizontally
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                # Rotate the image 180 degrees
                img = img.transpose(Image.ROTATE_180)
            elif orientation == 4:
                # Flip the image vertically
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            elif orientation == 5:
                # Rotate the image 270 degrees and flip it horizontally
                img = img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                # Rotate the image 270 degrees
                img = img.transpose(Image.ROTATE_270)
            elif orientation == 7:
                # Rotate the image 90 degrees and flip it horizontally
                img = img.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                # Rotate the image 90 degrees
                img = img.transpose(Image.ROTATE_90)
    else:
        # The image does not have any EXIF data
        pass
    return img


# resize options
def fill_center_crop(image, output_size):
    """
    Resize and center crop an image to the desired output size.
    """
    # Get the original image size
    original_size = np.array(image.size)
    # print("original size of image:", original_size)

    # Calculate the aspect ratios of the original and output sizes
    original_aspect_ratio = original_size[0] / original_size[1]
    output_aspect_ratio = output_size[0] / output_size[1]
    # print("original aspect ratio:", original_aspect_ratio)

    # Determine which dimension needs to be cropped
    if original_aspect_ratio > output_aspect_ratio:
        # The image is wider than the output, so crop the sides
        new_width = int(original_size[1] * output_aspect_ratio)
        left = int((original_size[0] - new_width) / 2)
        right = int((original_size[0] + new_width) / 2)
        image = image.crop((left, 0, right, original_size[1]))
    else:
        # The image is taller than the output, so crop the top and bottom
        new_height = int(original_size[0] / output_aspect_ratio)
        top = int((original_size[1] - new_height) / 2)
        bottom = int((original_size[1] + new_height) / 2)
        image = image.crop((0, top, original_size[0], bottom))

    # Resize the cropped image to the desired output size
    image = image.resize(output_size)

    return image


def fit_within(image, output_size):
    """
    Resize an image to fit within the desired output size while maintaining the aspect ratio.
    """
    original_size = np.array(image.size)

    # Calculate the aspect ratios of the original and output sizes
    original_aspect_ratio = original_size[0] / original_size[1]
    output_aspect_ratio = output_size[0] / output_size[1]

    # Determine which dimension needs to be scaled
    if original_aspect_ratio > output_aspect_ratio:
        # The image is wider than the output, so scale the width
        new_width = output_size[0]
        new_height = int(new_width / original_aspect_ratio)
    else:
        # The image is taller than the output, so scale the height
        new_height = output_size[1]
        new_width = int(new_height * original_aspect_ratio)
    # Resize the image to fit within the desired output size
    image = image.resize((new_width, new_height), resample=Image.BILINEAR)
    return image


def fit_reflect_edges(image, output_size):
    original_size = np.array(image.size)

    original_aspect_ratio = original_size[0] / original_size[1]
    output_aspect_ratio = output_size[0] / output_size[1]

    # Determine which dimension needs to be scaled
    if original_aspect_ratio > output_aspect_ratio:
        # The image is wider than the output, so scale the width
        new_width = output_size[0]
        new_height = int(new_width / original_aspect_ratio)
    else:
        # The image is taller than the output, so scale the height
        new_height = output_size[1]
        new_width = int(new_height * original_aspect_ratio)

    # Resize the image to fit within the desired output size
    resized_image = image.resize((new_width, new_height), resample=Image.BILINEAR)

    # Calculate the reflection padding size
    reflect_width = (output_size[0] - new_width) // 2
    reflect_height = (output_size[1] - new_height) // 2

    # print("reflected width, height:", reflect_width, reflect_height)

    output_image = Image.new(mode="RGB", size=output_size, color=(0, 0, 0))

    # Paste the resized image onto the output image with reflection padding
    output_image.paste(resized_image, (reflect_width, reflect_height))

    if original_aspect_ratio > output_aspect_ratio:
        # Reflect the top and bottom padding areas
        top_area = output_image.crop(
            (0, reflect_height, output_size[0], 2 * reflect_height)
        )
        top_reflected = ImageOps.flip(top_area)
        output_image.paste(top_reflected, (0, reflect_width))

        bottom_area = output_image.crop(
            (
                0,
                output_size[1] - 2 * reflect_height,
                output_size[0],
                output_size[1] - reflect_height,
            )
        )
        bottom_reflected = ImageOps.flip(bottom_area)
        output_image.paste(
            bottom_reflected, (reflect_width, output_size[1] - (reflect_height))
        )  # (0, )

    else:
        # Reflect the left and right padding areas
        left_area = output_image.crop(
            (
                reflect_width,
                reflect_height,
                2 * reflect_width,
                output_size[1] - reflect_height,
            )
        )
        left_reflected = ImageOps.mirror(left_area)
        output_image.paste(left_reflected, (0, reflect_height))

        right_area = output_image.crop(
            (
                output_size[0] - 2 * reflect_width,
                reflect_height,
                output_size[0] - reflect_width,
                output_size[1] - reflect_height,
            )
        )
        right_reflected = ImageOps.mirror(right_area)
        output_image.paste(
            right_reflected, (output_size[0] - reflect_width, reflect_height)
        )

    return output_image


def resize_image(image, output_size, method):
    # Apply the desired resizing method
    if method == "stretch":
        resized_image = image.resize(output_size)
    elif method == "filled_center":
        resized_image = fill_center_crop(image, output_size)
    elif method == "fit_reflect_edges":
        resized_image = fit_reflect_edges(image, output_size)
    elif method == "fit_black_edges":
        resized_image = fit_within(image, output_size)
        black = (0, 0, 0)  # Black color
        resized_image = ImageOps.pad(
            resized_image, (output_size[0], output_size[1]), color=black
        )
    elif method == "fit_white_edges":
        resized_image = fit_within(image, output_size)
        white = (255, 255, 255)  # white color
        resized_image = ImageOps.pad(
            resized_image, (output_size[0], output_size[1]), color=white
        )
    else:
        raise ValueError(f"Invalid resizing method '{method}'")

    return resized_image
