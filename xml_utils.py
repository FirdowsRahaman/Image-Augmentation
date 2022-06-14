import os
import numpy as np
import xml.etree.ElementTree as ET


# Define XML annotation format for creating new XML files
xml_body_1="""<annotation>
        <folder>{FOLDER}</folder>
        <filename>{FILENAME}</filename>
        <path>{PATH}</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>{WIDTH}</width>
            <height>{HEIGHT}</height>
            <depth>3</depth>
        </size>
"""
xml_object=""" <object>
                <name>{CLASS}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>{XMIN}</xmin>
                    <ymin>{YMIN}</ymin>
                    <xmax>{XMAX}</xmax>
                    <ymax>{YMAX}</ymax>
                </bndbox>
        </object>
"""
xml_body_2="""</annotation>        
"""


# Create XML files
def create_xml(folder, image_path, xml_path, img_size, cls, bboxes):
    # Get image size and filename
    img_height, img_width = img_size[1], img_size[0]
    filename = os.path.split(image_path)[-1]

    # Create XML file and write data
    with open(xml_path,'w') as f:
        f.write(xml_body_1.format(**{'FOLDER':folder, 'FILENAME':filename, 'PATH':image_path,
                                     'WIDTH':img_width, 'HEIGHT':img_height}))

        for c, bbox in zip(cls, bboxes):
            f.write(xml_object.format(**{'CLASS':c, 'XMIN':bbox[0], 'YMIN':bbox[1],
                                         'XMAX':bbox[2], 'YMAX':bbox[3]}))

        f.write(xml_body_2)


def read_annotation_data(xml_file):
    file = open(xml_file,'r')
    tree = ET.parse(file)
    root = tree.getroot()
        
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    img_size = [img_width, img_height]
        
    bboxes = []
    classes = []

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
            
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymin = int(xmlbox.find('ymin').text)
        ymax = int(xmlbox.find('ymax').text)

        bboxes.append([xmin, ymin, xmax, ymax])
        classes.append(cls)

    return bboxes, classes, img_size


def copy_xml_file(num_images):
        tmp_xmls = []
        
        for count, f in enumerate(num_images):
            f_name, f_ext = os.path.splitext(f)
            f_name = f_name + ".xml"
            tmp_xmls.append(f_name)
            
        return tmp_xmls


if __name__ == "__main__":
    a, b, c = read_annotation_data("../try/maxresdefault.xml")
    print("box -----> ", a)
    print("/n")
    # print("cls--------------> ", b)
    # d = np.array(a)
    # print(d[:, 0], d[:, 2])
    # print(d[:, 0] - d[:, 2])
    # print(c[:1] - d[:, 0])
    # for i in a:
        # print(i[ 0])
