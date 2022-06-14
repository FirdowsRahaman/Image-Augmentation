import os 
import cv2
import glob
import shutil
import random

import xml_utils
import aug_utils
import numpy as np
import xml.etree.ElementTree as ET


class ImageAugmentation:
    def __init__(self, img_dir, out_dir):
        '''
        :param img_dir: Path to the images.
        :param out_dir: Output directory to save augmentation images.
        '''
        self.input_dir = img_dir
        self.output_dir = out_dir

        self.image_list = glob.glob(f'{self.input_dir}/*.jpg')
        self.xml_list = glob.glob(f'{self.input_dir}/*.xml')

        self.num_imgs = len(self.image_list)
        self.num_xmls = len(self.xml_list)
        assert self.num_imgs == self.num_xmls

        print(f"Total number of images: {self.num_imgs}")
        print(f"Total number of xmls: {self.num_xmls}")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    

    def rotate(self, size=None, angle=(-30, 40), scale=1.0):
        '''
        Rotate the image
        :param size: How many images to be augmented to this category. Should be less than total num of images.
        :param angle: Rotation angle in degrees (int or tuple (angle range)). Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        if size is None:
            size = self.num_imgs
        random_images = random.sample(self.image_list, size)
        random_xmls = xml_utils.copy_xml_file(random_images)
        
        temp_angle = angle
        for img_path, xml_path in zip(random_images, random_xmls):
            Extension = str(img_path.split("\\")[-1])
            image = cv2.imread(img_path)

            if type(angle) == tuple:
                angle = random.randint(angle[0], angle[1])
        
            # Grab the dimensions of the image and then determine the centre
            (h, w) = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            
            # Grab the rotation matrix (applying the negative of the
            # angle to rotate clockwise), then grab the sine and cosine
            # (i.e., the rotation components of the matrix)
            M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # Compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # Adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY

            # Perform the actual rotation and return the image
            image = cv2.warpAffine(image, M, (nW, nH))

            scale_factor_x = image.shape[1] / w
            scale_factor_y = image.shape[0] / h
            image = cv2.resize(image, (w,h))
            
            bboxes, cls, img_size = xml_utils.read_annotation_data(xml_path)
            bboxes = np.array(bboxes)
            corners = aug_utils.get_corners(bboxes)
            corners = np.hstack((corners, bboxes[:,4:])).astype("float32")
            corners[:,:8] = aug_utils.rotate_box(corners[:,:8], angle, cX, cY, h, w)
            new_bbox = aug_utils.get_enclosing_box(corners).astype('float32')
            new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
            bboxes_rot  = new_bbox.astype('int32')
            bboxes_rot = aug_utils.clip_box(bboxes_rot, [0, 0 ,w, h], 0.25)
            bboxes_rot = bboxes_rot.tolist()
            
            # Save image and XML files to the directory
            image_path = f"{self.output_dir}/_rotate_{str(angle)}_{Extension}"
            f_name, f_ext = os.path.splitext(image_path)
            xml_path = f_name + ".xml"
            cv2.imwrite(image_path, image)
            xml_utils.create_xml(self.output_dir, image_path, xml_path, [w, h], cls, bboxes_rot)
            angle = temp_angle


    def flip(self, size=None, vflip=False, hflip=False):
        '''
        Flip the image
        :param size: Num of images to be augmented to this category. Should be less than total num of images.
        :param vflip: Whether to flip the image vertically.
        :param hflip: Whether to flip the image horizontally.
        '''
        if size is None:
            size = self.num_imgs
        random_images = random.sample(self.image_list, size)
        random_xmls = xml_utils.copy_xml_file(random_images)
        
        for img_path, xml_path in zip(random_images, random_xmls):
            Extension = str(img_path.split("\\")[-1])
            image = cv2.imread(img_path)
            
            if hflip or vflip:
                if hflip and vflip:
                    c = -1
                else:
                    c = 0 if vflip else 1
                    
            # flip image horizontally
            image = cv2.flip(image, flipCode=c)
            bbox, cls, img_size = xml_utils.read_annotation_data(xml_path)

            img_width = img_size[0]
            img_height = img_size[1]

            # horizontal flipping
            if c == 1:
                bbox_aug = abs(bbox - np.array([img_width, 0, img_width, 0]))
            # vertical flipping
            elif c == 0:
                bbox_aug = abs(bbox - np.array([0, img_height, 0, img_height]))
            # horizontal and vertical flipping
            elif c == -1:
                bbox_aug = abs(bbox - np.array([img_width, img_height, img_width, img_height]))
            
            # Save image and XML files to the directory
            image_path = f"{self.output_dir}/_flip_{str(c)}_{Extension}"
            f_name, f_ext = os.path.splitext(image_path)
            xml_path = f_name + ".xml"
            cv2.imwrite(image_path, image)
            xml_utils.create_xml(self.output_dir, image_path, xml_path, [img_width, img_height], cls, bbox_aug.astype('int32'))


    def shear_x(self, size=None, shear_factor=(0.1, 0.9)):
        '''
        Shears an image in horizontal direction.
        :param size: Num of images to be augmented to this category. Should be less than total num of images.
        :param shear_factor: Shearing images in horizontal dimension (int or tuple (shear_factor range)).
        '''
    
        if size is None:
            size = self.num_imgs
        random_images = random.sample(self.image_list, size)
        random_xmls = xml_utils.copy_xml_file(random_images)
        
        temp_shear_factor = shear_factor
        for img_path, xml_path in zip(random_images, random_xmls):
            Extension = str(img_path.split("\\")[-1])
            image = cv2.imread(img_path)
                
            img_width, img_height = image.shape[1], image.shape[0]
            bboxes, cls, img_size = xml_utils.read_annotation_data(xml_path)
            bboxes = np.array(bboxes)

            if isinstance(shear_factor, tuple):
                shear_factor = abs(random.uniform(shear_factor[0], shear_factor[1]))
                    
            # flip image horizontally
            if shear_factor < 0:
                image = cv2.flip(image, flipCode=1)
                bboxes = (bboxes - np.array([img_width, 0, img_width, 0]))
            
            M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
            nW =  image.shape[1] + abs(shear_factor*image.shape[0])
            
            bboxes[:,[0,2]] += ((bboxes[:,[1,3]]) * abs(shear_factor) ).astype(int) 
            image = cv2.warpAffine(image, M, (int(nW), image.shape[0]))
            
            # flip image horizontally
            if shear_factor < 0:
                image = cv2.flip(image, flipCode=1)
                bboxes = (bboxes - np.array([img_width, 0, img_width, 0]))
                
            image = cv2.resize(image, (img_width, img_height))
            scale_factor_x = nW / img_width
            bboxes = bboxes.astype("float32")
            bboxes[:,:4] /= [scale_factor_x, 1, scale_factor_x, 1] 
            
            # Save image and XML files to the directory
            image_path = f"{self.output_dir}/shear_x_{Extension}"
            f_name, f_ext = os.path.splitext(image_path)
            xml_path = f_name + ".xml"
            cv2.imwrite(image_path, image)
            xml_utils.create_xml(self.output_dir, image_path, xml_path, [img_width, img_height], cls, bboxes.astype('int32'))
            shear_factor = temp_shear_factor


    def shear_y(self, size=None, shear_factor=(0.1, 0.9)):
        '''
        Shears an image in vertical direction.
        :param size: Num of images to be augmented to this category. Should be less than total num of images.
        :param shear_factor: Shearing images in horizontal dimension (int or tuple (shear_factor range)).
        '''
        if size is None:
            size = self.num_imgs
        random_images = random.sample(self.image_list, size)
        random_xmls = xml_utils.copy_xml_file(random_images)
        
        temp_shear_factor = shear_factor
        for img_path, xml_path in zip(random_images, random_xmls):
            Extension = str(img_path.split("\\")[-1])
            image = cv2.imread(img_path)
            
            img_width, img_height = image.shape[1], image.shape[0]
            bboxes, cls, img_size = xml_utils.read_annotation_data(xml_path)
            bboxes = np.array(bboxes)

            if isinstance(shear_factor, tuple):
                shear_factor = abs(random.uniform(shear_factor[0], shear_factor[1]))
                    
            # flip image vertically
            if shear_factor < 0:
                image = cv2.flip(image, flipCode=0)
                bboxes = abs(bboxes - np.array([0, img_height, 0, img_height]))
            
            M = np.array([[1, 0, 0],[abs(shear_factor), 1, 0]])
            nH = image.shape[0] + abs(shear_factor*image.shape[1])
            
            bboxes[:,[1,3]] += ((bboxes[:,[0,2]]) * abs(shear_factor) ).astype(int) 
            image = cv2.warpAffine(image, M, (image.shape[1], int(nH)))
            
            # flip image vertically
            if shear_factor < 0:
                image = cv2.flip(image, flipCode=0)
                bboxes = abs(bboxes - np.array([0, img_height, 0, img_height]))
                
            image = cv2.resize(image, (img_width, img_height))
            scale_factor_y = nH / img_height
            bboxes = bboxes.astype("float32")
            bboxes[:,:4] /= [1, scale_factor_y, 1, scale_factor_y] 
            
            # Save image and XML files to the directory
            image_path = f"{self.output_dir}/shear_y_{Extension}"
            f_name, f_ext = os.path.splitext(image_path)
            xml_path = f_name + ".xml"
            cv2.imwrite(image_path, image)
            xml_utils.create_xml(self.output_dir, image_path, xml_path, [img_width, img_height], cls, bboxes.astype('int32'))
            shear_factor = temp_shear_factor


    def contrast(self, size=None, contrast=(10, 50)):
        if size is None:
            size = self.num_imgs
        random_images = random.sample(self.image_list, size)
        random_xmls = xml_utils.copy_xml_file(random_images)

        temp_contrast = contrast
        for img_path, xml_path in zip(random_images, random_xmls):
            Extension = str(img_path.split("\\")[-1])
            image = cv2.imread(img_path)

            if isinstance(contrast, tuple):
                contrast = abs(random.uniform(contrast[0], contrast[1]))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image[:,:,2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in image[:,:,2]]
            image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            # Save image and XML files to the directory
            image_path = f"{self.output_dir}/contrast_{str(contrast)}_{Extension}"
            f_name, f_ext = os.path.splitext(image_path)
            xml_aug = f_name + ".xml"
            cv2.imwrite(image_path, image)
            shutil.copy2(xml_path, xml_aug)  
            contrast = temp_contrast
    

    def hue(self, size=None, value=(10, 30)):
        if size is None:
            size = self.num_imgs
        random_images = random.sample(self.image_list, size)
        random_xmls = xml_utils.copy_xml_file(random_images)

        temp_value = value
        for img_path, xml_path in zip(random_images, random_xmls):
            Extension = str(img_path.split("\\")[-1])
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            if isinstance(value, tuple):
                value = abs(random.uniform(value[0], value[1]))

            v = image[:, :, 2]
            v = np.where(v <= 255 + value, v - value, 255)
            image[:, :, 2] = v
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            # Save image and XML files to the directory
            image_path = f"{self.output_dir}/hue_{str(value)}_{Extension}"
            f_name, f_ext = os.path.splitext(image_path)
            xml_aug = f_name + ".xml"
            
            cv2.imwrite(image_path, image)
            shutil.copy2(xml_path, xml_aug) 
            value = temp_value
            

    def saturation(self, size=None, saturation=30):
        if size is None:
            size = self.num_imgs
        random_images = random.sample(self.image_list, size)
        random_xmls = xml_utils.copy_xml_file(random_images)
        for img_path, xml_path in zip(random_images, random_xmls):
            Extension = str(img_path.split("\\")[-1])
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            v = image[:, :, 2]
            v = np.where(v <= 255 - saturation, v + saturation, 255)
            image[:, :, 2] = v

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            # Save image and XML files to the directory
            image_path = self.output_dir + "/_saturation-" + str(saturation) + Extension
            f_name, f_ext = os.path.splitext(image_path)
            xml_aug = f_name + ".xml"
            cv2.imwrite(image_path, image)
            shutil.copy2(xml_path, xml_aug) 


    def gausian_blur(self, size=None, blur=0.20):
        if size is None:
            size = self.num_imgs
        random_images = random.sample(self.image_list, size)
        random_xmls = xml_utils.copy_xml_file(random_images)
        for img_path, xml_path in zip(random_images, random_xmls):
            Extension = str(img_path.split("\\")[-1])
            image = cv2.imread(img_path)
            image = cv2.GaussianBlur(image,(5,5),blur)

            # Save image and XML files to the directory
            image_path = self.output_dir + "/_gausian_blur-" + str(blur) + Extension
            f_name, f_ext = os.path.splitext(image_path)
            xml_aug = f_name + ".xml"
            cv2.imwrite(image_path, image)
            shutil.copy2(xml_path, xml_aug) 
            # os.rename(xml_path, xml_aug)
        

    def averageing_blur(self, size=None, shift=3):
        if size is None:
            size = self.num_imgs
        random_images = random.sample(self.image_list, size)
        random_xmls = xml_utils.copy_xml_file(random_images)
        for img_path, xml_path in zip(random_images, random_xmls):
            Extension = str(img_path.split("\\")[-1])
            image = cv2.imread(img_path)
            image=cv2.blur(image, (shift, shift))

            # Save image and XML files to the directory
            image_path = self.output_dir + "/_averageing_blur-" + str(shift) + Extension
            f_name, f_ext = os.path.splitext(image_path)
            xml_aug = f_name + ".xml"
            cv2.imwrite(image_path, image)
            shutil.copy2(xml_path, xml_aug) 
            # os.rename(xml_path, xml_aug)


    def median_blur(self, size=None, shift=3):
        if size is None:
            size = self.num_imgs
        random_images = random.sample(self.image_list, size)
        random_xmls = xml_utils.copy_xml_file(random_images)
        for img_path, xml_path in zip(random_images, random_xmls):
            Extension = str(img_path.split("\\")[-1])
            image = cv2.imread(img_path)
            image=cv2.medianBlur(image, shift)

            # Save image and XML files to the directory
            image_path = self.output_dir + "/_median_blur-" + str(shift) + Extension
            f_name, f_ext = os.path.splitext(image_path)
            xml_aug = f_name + ".xml"
            cv2.imwrite(image_path, image)
            shutil.copy2(xml_path, xml_aug) 
            # os.rename(xml_path, xml_aug)
        

    def sharpen(self, size=None):
        if size is None:
            size = self.num_imgs
        random_images = random.sample(self.image_list, size)
        random_xmls = xml_utils.copy_xml_file(random_images)
        for img_path, xml_path in zip(random_images, random_xmls):
            Extension = str(img_path.split("\\")[-1])
            image = cv2.imread(img_path)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            image = cv2.filter2D(image, -1, kernel)

            # Save image and XML files to the directory
            image_path = self.output_dir + "/_sharpen-" + Extension
            f_name, f_ext = os.path.splitext(image_path)
            xml_aug = f_name + ".xml"
            cv2.imwrite(image_path, image)
            shutil.copy2(xml_path, xml_aug) 
            # os.rename(xml_path, xml_aug)
    

    def addeptive_gaussian_noise(self, size=None):
        if size is None:
            size = self.num_imgs
        random_images = random.sample(self.image_list, size)
        random_xmls = xml_utils.copy_xml_file(random_images)
        for img_path, xml_path in zip(random_images, random_xmls):
            Extension = str(img_path.split("\\")[-1])
            image = cv2.imread(img_path)
            h,s,v=cv2.split(image)
            s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            image=cv2.merge([h,s,v])

            # Save image and XML files to the directory
            image_path = self.output_dir + "/_addeptive_gaussian_noise-" + Extension
            f_name, f_ext = os.path.splitext(image_path)
            xml_aug = f_name + ".xml"
            cv2.imwrite(image_path, image)
            shutil.copy2(xml_path, xml_aug)


    def random_crop(self, size=None, value=(50, 70)):
        '''
        Crops an image.
        :param size: Num of images to be augmented to this category. Should be less than total num of images.
        :param shear_factor: Shearing images in horizontal dimension (int or tuple (shear_factor range)).
        '''
    
        if size is None:
            size = self.num_imgs
        random_images = random.sample(self.image_list, size)
        random_xmls = xml_utils.copy_xml_file(random_images)
        
        temp_value = value
        for img_path, xml_path in zip(random_images, random_xmls):
            Extension = str(img_path.split("\\")[-1])
            image = cv2.imread(img_path)
                
            img_width, img_height = image.shape[1], image.shape[0]
            bboxes, classes, img_size = xml_utils.read_annotation_data(xml_path)

            if isinstance(value, tuple):
                value = random.randint(value[0], value[1])
            
            max_limit_width = int(img_width * (value / 100))
            max_limit_height = int(img_height * (value / 100))

            random_xmin = random.randint(0, img_width - max_limit_width)
            random_ymin = random.randint(0, img_height - max_limit_height)

            nWmin = int(random_xmin)
            nWmax = int(random_xmin + max_limit_width) 
            nHmin = int(random_ymin) 
            nHmax = int(random_ymin + max_limit_height) 
            window = [nWmin, nHmin, nWmax, nHmax]
            print(window)
            
            cropped_img = image[nHmin:nHmax, nWmin:nWmax]
            resized_img = cv2.resize(cropped_img, (img_width, img_height))
            
            filtered_boxes, filtered_classes = aug_utils.filter_bboxes(bboxes, classes, window)
            if filtered_boxes != []:
                cropped_bboxes, cropped_classes = aug_utils.crop_bboxes(filtered_boxes, filtered_classes, window)
                bboxes = aug_utils.scale_bboxes(cropped_bboxes, window).astype("float32")
                
                scale_factor_x = img_width / resized_img.shape[1]
                scale_factor_y = img_height / resized_img.shape[0]
                
                # bboxes[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
                bboxes = bboxes.astype('int32').tolist()
                
                # Save image and XML files to the directory
                image_path = f"{self.output_dir}/random_crop_{value}_{Extension}"
                f_name, f_ext = os.path.splitext(image_path)
                xml_path = f_name + ".xml"
                cv2.imwrite(image_path, cropped_img)
                xml_utils.create_xml(self.output_dir, image_path, xml_path, [img_width, img_height], cropped_classes, bboxes)
                value = temp_value

    

if __name__ == "__main__":
    aug = ImageAugmentation("./images", "./try_aug")
    
    # aug.shear_y(size=100)
    # aug.shear_x(size=100)
    aug.random_crop(size=100)

    # aug.rotate(size=100)
    #aug.flip(hflip=True, size=100)

    # aug.contrast(size=300)
    # aug.hue(size=300)
    # aug.saturation(size=300)

    # aug.gausian_blur(size=300)
    # aug.averageing_blur(size=300)
    # aug.median_blur(size=400)

    # aug.sharpen(size=500)
    #aug.addeptive_gaussian_noise(size=600)
    
        