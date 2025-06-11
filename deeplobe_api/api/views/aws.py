import boto3
import os
from decouple import config


def save_frame_in_s3(name, body, content_type=""):
    access_key = config("AWS_ACCESS_KEY_ID")
    secret_key = config("AWS_SECRET_ACCESS_KEY")
    region = config("AWS_REGION")
    bucket = config("AWS_S3_BUCKET_NAME")

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    s3 = session.resource("s3")
    s3.Bucket(bucket).put_object(
        Key=name, Body=body, ACL="public-read", ContentType=content_type
    )
    uploaded_url = f"https://{bucket}.s3.amazonaws.com/{name}"
    return uploaded_url

def upload_folder_to_s3(folder_path, bucket):
    uploaded_urls = []  # To store the uploaded S3 URLs
    s3 = boto3.client("s3")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_key = os.path.relpath(local_file_path, folder_path)
            s3_key = s3_key.replace("\\", "/")  # Convert backslashes to forward slashes for S3 path
            with open(local_file_path, "rb") as f:
                content_type = ""  # Add the appropriate content type if needed
                s3.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=f,
                    ACL="public-read",
                    ContentType=content_type,
                )
            # Get the uploaded S3 URL for each file in the "vectorDB" folder
            uploaded_url = f"https://{bucket}.s3.amazonaws.com/{s3_key}"
            uploaded_urls.append(uploaded_url)
            print(f"Uploaded {local_file_path} to {uploaded_url}")
    return uploaded_urls

