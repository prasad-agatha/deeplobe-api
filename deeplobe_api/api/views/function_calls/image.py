import io
import os
import cv2
import uuid
import wget
import shutil
import requests
import subprocess

from io import BytesIO

from PIL import Image

from urllib.request import urlretrieve

from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile

from deeplobe_api.api.serializers import MediaAssetSerializer


def resize_image(url=None, width=None, height=None):
    if url is not None:
        response = requests.get(url, stream=True)
        buffer = io.BytesIO(response.content)
        image = Image.open(buffer)

        # resize in new format
        format = (width, height)

        # new image
        resized_image = image.resize(format, Image.ANTIALIAS)

        # Get the Django file object
        new_image = BytesIO()
        # converted RGBA RGB
        if resized_image.mode in ["RGBA", "P"]:
            rgb_image = resized_image.convert("RGB")
            rgb_image.save(new_image, format="JPEG")
        else:
            resized_image.save(new_image, format="JPEG")
        resized_image_ = ContentFile(new_image.getvalue())
        file_name = os.path.basename(url)
        django_file_object = InMemoryUploadedFile(
            resized_image_,
            None,
            f"{file_name}",
            "image/jpeg",
            resized_image_.tell,
            None,
        )
        return django_file_object
    raise Exception("This object doesn't have valid url")


def convert_file_django_object(url):
    file_name = "file"
    urlretrieve(url, file_name)
    # Get the Django file object
    with open(file_name, "rb") as img:
        image = ContentFile(img.read())
        django_file_object = InMemoryUploadedFile(
            image,
            None,
            f"{file_name}.jpg",
            "image/jpeg",
            image.tell,
            None,
        )
        if os.path.exists(file_name):
            os.remove(file_name)
        return django_file_object


def save_frame_in_s3(base_dir):
    import boto3, botocore
    from decouple import config

    access_key = config("AWS_ACCESS_KEY_ID")
    secret_key = config("AWS_SECRET_ACCESS_KEY")
    region = config("AWS_REGION")
    bucket = config("AWS_S3_BUCKET_NAME")
    config = botocore.config.Config(
        read_timeout=900, connect_timeout=900, retries={"max_attempts": 0}
    )
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    s3 = session.resource("s3", config=config)
    # s3.Bucket(bucket).put_object(Key=name, Body=body, ACL="public-read")
    aws_urls = []
    frames_path = f"{base_dir}/frames/"
    for filename in os.listdir(frames_path):
        filepath = f"{base_dir}/frames/{filename}"
        key = f"frames/{os.path.basename(filename)}"
        s3.Bucket(bucket).put_object(
            Key=key, Body=open(filepath, "rb"), ACL="public-read"
        )
        location = session.client("s3").get_bucket_location(Bucket=bucket)[
            "LocationConstraint"
        ]
        # uploaded_url = f"https://s3-{location}.amazonaws.com/{bucket}/{key}"
        uploaded_url = f"https://{bucket}.s3.amazonaws.com/{key}"
        aws_urls.append(uploaded_url)
    # remove frames directory
    shutil.rmtree(frames_path)
    return aws_urls


def frames_count(video_path=None, seconds=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return {"Number_of_image_frames": frame_count, "fps": fps}


def raw_resize_image(img=None, input_width=None, input_height=None, buffer=None):
    image = Image.open(img)

    # original image width, height
    original_width, original_height = image.size
    ratio = original_width / original_height
    width = input_width
    height = width / ratio
    if original_width < width and original_height < input_height:
        height = original_height
        width = original_width
    elif height > input_height:
        height = input_height
        width = height * ratio

    # resize in new format
    format = (int(width), int(height))

    # new image
    resized_image = image.resize(format, Image.ANTIALIAS)

    # Get the Django file object
    new_image = BytesIO()
    # converted RGBA RGB
    if resized_image.mode in ["RGBA", "P"]:
        rgb_image = resized_image.convert("RGB")
        rgb_image.save(new_image, format="JPEG")
    else:
        resized_image.save(new_image, format="JPEG")
    resized_image_ = ContentFile(new_image.getvalue())
    if buffer:
        file_name = f"{uuid.uuid4().hex}.jpg"
    else:
        file_name = os.path.basename(img)
    django_file_object = InMemoryUploadedFile(
        resized_image_,
        None,
        f"{file_name}",
        "image/jpeg",
        resized_image_.tell,
        None,
    )
    return django_file_object


def save_frame_in_s3_with_resize(base_dir, height, width, uuid):
    aws_urls = []
    frames_path = f"{base_dir}/frames/"
    for filename in os.listdir(frames_path):
        try:
            # new image
            # user format will be (width, height)
            resized_image = raw_resize_image(
                img=f"{frames_path}/{filename}",
                input_width=width,
                input_height=height,
            )
        except Exception as e:
            print(str(e))

        dict_data = {"name": filename, "asset": resized_image}
        serializer = MediaAssetSerializer(data=dict_data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            aws_urls.append(serializer.data)
    # remove frames directory
    shutil.rmtree(frames_path)
    return aws_urls


# def get_frames(video_path=None, images_count=None):
#     base_dir = os.path.dirname(
#         os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     )
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)

#     # Create a directory to save the images
#     if not os.path.exists(f"{base_dir}/frames/"):
#         os.makedirs(f"{base_dir}/frames/")

#     # Loop through the frames and save them as images
#     # Get the frame rate and number of frames in the video
#     frame_rate = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # if seconds is None:
#     #     # Calculate the length of the video in seconds
#     #     length_in_seconds = frame_count / frame_rate
#     # else:
#     #     length_in_seconds = seconds

#     # Calculate the number of image frames
#     image_frame_count = int(images_count * frame_rate)
#     # image_frame_count = images_count

#     frame_count = 1
#     while frame_count <= image_frame_count:
#         # Read a frame from the video
#         ret, frame = cap.read()

#         # If there are no more frames, break out of the loop
#         if not ret:
#             break

#         # Save the frame as an image
#         frame_path = os.path.join(f"{base_dir}/frames/", f"frame{frame_count}.jpg")

#         cv2.imwrite(frame_path, frame)

#         # Increment the frame count
#         frame_count += 1

#     # Release the video capture object
#     cap.release()

#     # upload image into s3
#     try:
#         # url = save_frame_in_s3(f"frame{frame_count}.jpg", open(frame_path, "rb"))
#         aws_urls = save_frame_in_s3(base_dir)
#         return aws_urls
#     except Exception as e:
#         print(e)


def extract_frames(video_path, gfps, height, width, uuid):
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    output_dir = f"{base_dir}/frames/"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    # Calculate interval between frames
    interval = round(total_frames // (duration * float(gfps)))
    if interval == 0:
        interval = 1

    # Loop through video frames
    count = 0
    while cap.isOpened():
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as image
        if count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{count:05d}.jpg")
            cv2.imwrite(frame_path, frame)

        count += 1

    # Release video capture object
    cap.release()

    # upload image into s3
    try:
        # url = save_frame_in_s3(f"frame{frame_count}.jpg", open(frame_path, "rb"))
        aws_urls = save_frame_in_s3_with_resize(base_dir, height, width, uuid)
        return aws_urls
    except Exception as e:
        print(e)


def upload_video_frames():
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    try:
        # url = save_frame_in_s3(f"frame{frame_count}.jpg", open(frame_path, "rb"))
        aws_urls = save_frame_in_s3(base_dir)
        return aws_urls
    except Exception as e:
        print(e)


def save_video_in_s3(path, video_filename):
    import boto3, botocore
    from decouple import config

    access_key = config("AWS_ACCESS_KEY_ID")
    secret_key = config("AWS_SECRET_ACCESS_KEY")
    region = config("AWS_REGION")
    bucket = config("AWS_S3_BUCKET_NAME")
    config = botocore.config.Config(
        read_timeout=900, connect_timeout=900, retries={"max_attempts": 0}
    )
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    s3 = session.resource("s3", config=config)
    key = os.path.basename(f"{path}/{video_filename}")
    s3.Bucket(bucket).put_object(
        Key=key, Body=open(f"{path}/{video_filename}", "rb"), ACL="public-read"
    )
    location = session.client("s3").get_bucket_location(Bucket=bucket)[
        "LocationConstraint"
    ]
    # uploaded_url = f"https://s3-{location}.amazonaws.com/{bucket}/{key}"
    uploaded_url = f"https://{bucket}.s3.amazonaws.com/{key}"
    # shutil.rmtree(path)
    return uploaded_url


def frames_to_video(urls):
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    path = f"{base_dir}/frame_images"
    urls = urls
    # Create a directory to save the images
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)

    for url in urls:
        wget.download(url)
    video_filename = "output.mp4"
    cmd = f"ffmpeg -framerate 24 -i frames_images/frame%04d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p frames_images/{video_filename}"
    subprocess.run(cmd, shell=True)
    url = save_video_in_s3(path, video_filename)
    for file in os.listdir(path):
        os.remove(file)
    return url


def delete_object_in_s3(key):
    import boto3, botocore
    from decouple import config

    access_key = config("AWS_ACCESS_KEY_ID")
    secret_key = config("AWS_SECRET_ACCESS_KEY")
    region = config("AWS_REGION")
    bucket = config("AWS_S3_BUCKET_NAME")
    config = botocore.config.Config(
        read_timeout=900, connect_timeout=900, retries={"max_attempts": 0}
    )
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    s3 = session.resource("s3", config=config)
    bucket = s3.Bucket(bucket)
    obj = bucket.Object(key)

    # Delete the object
    response = obj.delete()

    print(response)  # Print the response from AWS

    # video_key = os.path.basename(video_path)
    # s3.delete_object(Bucket=bucket, Key=video_key)


def save_django_file_in_s3(file):
    import boto3, botocore
    from decouple import config

    access_key = config("AWS_ACCESS_KEY_ID")
    secret_key = config("AWS_SECRET_ACCESS_KEY")
    region = config("AWS_REGION")
    bucket = config("AWS_S3_BUCKET_NAME")
    config = botocore.config.Config(
        read_timeout=900, connect_timeout=900, retries={"max_attempts": 0}
    )
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    s3 = session.resource("s3", config=config)
    key = file.name
    s3.Bucket(bucket).put_object(Key=key, Body=file.read(), ACL="public-read")
    location = session.client("s3").get_bucket_location(Bucket=bucket)[
        "LocationConstraint"
    ]
    # uploaded_url = f"https://s3-{location}.amazonaws.com/{bucket}/{key}"
    uploaded_url = f"https://{bucket}.s3.amazonaws.com/{key}"
    # Delete the object in s3 bucket
    return uploaded_url, key


def video_images_prediction(request):
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    frames_path = f"{base_dir}/frames"
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    input_video_file = request.data.get("video")
    url, key = save_django_file_in_s3(input_video_file)
    # video to frames
    subprocess.run(
        f"ffmpeg -i {url} frames/frame%04d.jpg",
        shell=True,
    )

    # for file in os.listdir(frames_path):
    #     os.remove(file)
    # # frames to video
    # video_filename = "output_video.mp4"
    # subprocess.run(
    #     f"ffmpeg -framerate 24 -i frames/frame%04d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p frames/{video_filename}",
    #     shell=True,
    # )
    # url = save_video_in_s3(frames_path, video_filename)
    # return Response({"video_url": url})
