import json  # TODO: better to use "import ujson as json" for the best performance
import logging
import os
import sys
import uuid
import xml.etree.ElementTree as ET

from label_studio_converter.imports.label_config import generate_label_config
from label_studio_converter.utils import ExpandFullPath

logger = logging.getLogger('root')

def new_task(out_type, root_url, file_name):
    return {
        "data": {"image": os.path.join(root_url, file_name)},
        # 'annotations' or 'predictions'
        out_type: [
            {
                "result": [],
                "ground_truth": False,
            }
        ],
    }

def create_bbox(annotation, from_name, image_height, image_width, to_name):
    label = annotation["label"]
    x, y, width, height = annotation['bbox']
    x, y, width, height = float(x), float(y), float(width), float(height)
    item = {
        "id": uuid.uuid4().hex[0:10],
        "type": "rectanglelabels",
        "value": {
            "x": x / image_width * 100.0,
            "y": y / image_height * 100.0,
            "width": width / image_width * 100.0,
            "height": height / image_height * 100.0,
            "rotation": 0,
            "rectanglelabels": [label],
        },
        "to_name": to_name,
        "from_name": from_name,
        "image_rotation": 0,
        "original_width": image_width,
        "original_height": image_height,
    }
    return item

def read_pascal_voc(file_path:str, image_id:int=0):
    """Reads in a Pascal VOC xml label file for conversion to label-studio json

    Args:
        file_path (str): Path to VOC file to be loaded in 
        image_id (int): Index of the file to be used when writing to Label Studio JSON
    Returns:
        annotations: List of dictionary objects containing a bbox, label, filename, width(px), height(px), and the image id
    """
    tree = ET
    try:
        tree = ET.parse(file_path)
    except Exception as e:
        logger.fatal(f"Unable to read {file_path} . File is not xml or does not exist")
        logger.exception(e)
        raise
    
    root = tree.getroot()
    image_size = root.find("size")
    image_width = int(image_size.find("width").text)
    image_height = int(image_size.find("height").text)
    image_filename = root.find("filename").text
    annotations = []
    categories = []
    logger.info(f'Found {len(root.findall("object"))} labels in {file_path}')
    for label in root.findall("object"):
        annotation_dict = {}
        bbox = label.find("bndbox")
        bbox_x = float(bbox.find("xmin").text)
        bbox_y = float(bbox.find("ymin").text)
        bbox_width = float(bbox.find("xmax").text) - bbox_x 
        bbox_height = float(bbox.find("ymax").text) - bbox_y 
        annotation_dict["bbox"] = (bbox_x, bbox_y, bbox_width, bbox_height)
        annotation_dict["label"] = label.find("name").text
        annotation_dict["image_filename"] = image_filename 
        annotation_dict["image_height"] = image_height
        annotation_dict["image_width"] = image_width
        annotation_dict["image_id"] = image_id 
        if annotation_dict["label"] not in categories:
            categories.append(annotation_dict["label"])
        logger.info(f'\t{annotation_dict}')
        annotations.append(annotation_dict)
    return annotations, categories

def convert_voc_to_ls(
    input_dir,
    out_file,
    to_name='image',
    from_name='label',
    out_type="annotations",
    image_root_url='/data/local-files/?d=',
):
    """Convert VOC labeling to Label Studio JSON

    :param input_dir: Directory with Pascal VOC annotations 
    :param out_file: output file with Label Studio JSON tasks
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param out_type: annotation type - "annotations" or "predictions"
    :param image_root_url: root URL path where images will be hosted, e.g.: http://example.com/images
    """

    tasks = {}  # image_id => task
    """
        # logger.info('Reading COCO notes and categories from %s', input_file)

        # with open(input_file, encoding='utf8') as f:
        #     coco = json.load(f)

        # # build categories => labels dict
        # new_categories = {}
        # # list to dict conversion: [...] => {category_id: category_item}
        # categories = {int(category['id']): category for category in coco['categories']}
        # ids = sorted(categories.keys())  # sort labels by their origin ids

        # for i in ids:
        #     name = categories[i]['name']
        #     new_categories[i] = name

        # # mapping: id => category name
        # categories = new_categories

        # # mapping: image id => image
        # images = {item['id']: item for item in coco['images']}

        # logger.info(
        #     f'Found {len(categories)} categories, {len(images)} images and {len(coco["annotations"])} annotations'
        # )
    """
    annotations = []
    categories = []
    logger.info(f'Importing Pascal VOC annotations from {input_dir}')
    input_paths = [os.path.join(input_dir, path) for path in os.listdir(input_dir)]
    logger.info(f'Found {len(input_paths)} input files') 
    for i, input_file in enumerate(input_paths):
        logger.info(f'Reading in file {i} of {len(input_paths)}: {input_file}')
        annotation_list, categories[len(categories):] = read_pascal_voc(input_file, i)     
        tasks[i] = new_task(out_type, image_root_url, annotation_list[0]["image_filename"])
        for ann in annotation_list:
            annotations.append(ann)
    categories = dict(enumerate(categories))
    logger.info(f'Found {len(input_paths)} input files with {len(annotations)} annotations')

    # flags for labeling config composing
    bbox = False
    bbox_once = False
    rectangles_from_name = from_name + '_rectangles'
    tags = {}

    for i, annotation in enumerate(annotations):
        bbox |= 'bbox' in annotation
        
        if bbox and not bbox_once:
            tags.update({rectangles_from_name: 'RectangleLabels'})
            bbox_once = True

        # read image sizes
        image_id = annotation['image_id']
        image_file_name = annotation['image_filename']
        image_width = annotation['image_width']
        image_height = annotation['image_height']

        task = tasks[image_id]

        if 'bbox' in annotation:
            item = create_bbox(
                annotation,
                rectangles_from_name,
                image_height,
                image_width,
                to_name,
            )
            task[out_type][0]['result'].append(item)

        tasks[image_id] = task

    # generate and save labeling config
    label_config_file = out_file.replace('.json', '') + '.label_config.xml'
    generate_label_config(categories, tags, to_name, from_name, label_config_file)

    if len(tasks) > 0:
        tasks = [tasks[key] for key in sorted(tasks.keys())]
        logger.info('Saving Label Studio JSON to %s', out_file)
        with open(out_file, 'w') as out:
            json.dump(tasks, out)

        print(
            '\n'
            f'  1. Create a new project in Label Studio\n'
            f'  2. Use Labeling Config from "{label_config_file}"\n'
            f'  3. Setup serving for images [e.g. you can use Local Storage (or others):\n'
            f'     https://labelstud.io/guide/storage.html#Local-storage]\n'
            f'  4. Import "{out_file}" to the project\n'
        )
    else:
        logger.error('No labels converted')


def add_parser(subparsers):
    voc = subparsers.add_parser('voc')

    voc.add_argument(
        '-i',
        '--input',
        dest='input',
        required=True,
        help='directory with voc where images, labels, notes.json are located',
        action=ExpandFullPath,
    )
    voc.add_argument(
        '-o',
        '--output',
        dest='output',
        help='output file with Label Studio JSON tasks',
        default='output.json',
        action=ExpandFullPath,
    )
    voc.add_argument(
        '--to-name',
        dest='to_name',
        help='object name from Label Studio labeling config',
        default='image',
    )
    voc.add_argument(
        '--from-name',
        dest='from_name',
        help='control tag name from Label Studio labeling config',
        default='label',
    )
    voc.add_argument(
        '--out-type',
        dest='out_type',
        help='annotation type - "annotations" or "predictions"',
        default='annotations',
    )
    voc.add_argument(
        '--image-root-url',
        dest='image_root_url',
        help='root URL path where images will be hosted, e.g.: http://example.com/images',
        default='/data/local-files/?d=',
    )
    # voc.add_argument(
    #     '--point-width',
    #     dest='point_width',
    #     help='key point width (size)',
    #     default=1.0,
    #     type=float,
    # )
