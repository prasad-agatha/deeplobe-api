import torchvision
from PIL import Image

import requests
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")

# https://pytorch.org/vision/0.8/models.html
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def get_prediction(urls):
    """
    A function to take list of urls of images. This function takes urls, get each image from url, send such image to model
    get predictions from model, use threshold and get boxes > threshold. resize the bboxes to original size.

    input:
          List of Image urls
    output:
           A dictionary of URLs, CLASSes/LABELS, BOXES.
    """

    pred_classes = []
    pred_boxess = []
    img_urls = []

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.eval()

    for image_url in urls:
        try:
            img = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
            t_img = transform(img)
            pred = model([t_img])

            scores = pred[0]["scores"]
            labels = list(pred[0]["labels"].detach().numpy())
            boxes = list(pred[0]["boxes"].detach().numpy())

            score_indices = [idx for idx in range(len(scores)) if scores[idx] > 0.75]
            filtered_labels = [labels[idx] for idx in score_indices]

            filtered_boxes = [boxes[idx] for idx in score_indices]
            y_scale, x_scale = img.size[0] / 224, img.size[1] / 224
            bbox = [
                [(i[0] * y_scale, i[1] * x_scale), (i[2] * y_scale, i[3] * x_scale)]
                for i in filtered_boxes
            ]

            pred_classes.append(
                [COCO_INSTANCE_CATEGORY_NAMES[i] for i in filtered_labels]
            )
            pred_boxess.append(bbox)
            img_urls.append(image_url)

        except:
            pred_classes.append("empty")
            pred_boxess.append("empty")
            img_urls.append(image_url)
    final_dict = {"Images": img_urls, "Classes": pred_classes, "Boxes": pred_boxess}
    return final_dict


## TEst
# https://deeplobe.s3.amazonaws.com/media_assets/13_hIcekEv_rFiCHZy.jpg
# https://deeplobe.s3.amazonaws.com/media_assets/4_XZ7BXRk_Pj7L04c.jpg
