import numpy as np
import cv2
import copy


def bbox_area(bbox):
        return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])


def clip_box(bbox, clip_box, alpha):
        """Clip the bounding boxes to the borders of an image
        
        Parameters
        ----------
        
        bbox: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`
        
        clip_box: numpy.ndarray
            An array of shape (4,) specifying the diagonal co-ordinates of the image
            The coordinates are represented in the format `x1 y1 x2 y2`
            
        alpha: float
            If the fraction of a bounding box left in the image after being clipped is 
            less than `alpha` the bounding box is dropped. 
        
        Returns
        -------
        
        numpy.ndarray
            Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes left are being clipped and the bounding boxes are represented in the
            format `x1 y1 x2 y2` 
        
        """
        ar_ = (bbox_area(bbox))
        x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
        y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
        x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
        y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
        
        bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))
        
        delta_area = ((ar_ - bbox_area(bbox))/ar_)
        
        mask = (delta_area < (1 - alpha)).astype(int)
        
        bbox = bbox[mask == 1,:]
        return bbox


def get_corners(bboxes):
    
        """Get corners of bounding boxes
        
        Parameters
        ----------
        
        bboxes: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`
        
        returns
        -------
        
        numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
            
        """
        width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
        height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
        
        x1 = bboxes[:,0].reshape(-1,1)
        y1 = bboxes[:,1].reshape(-1,1)
        
        x2 = x1 + width
        y2 = y1 
        
        x3 = x1
        y3 = y1 + height
        
        x4 = bboxes[:,2].reshape(-1,1)
        y4 = bboxes[:,3].reshape(-1,1)
        
        corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
        return corners
    

def rotate_box(corners, angle,  cx, cy, h, w):
    
        """Rotate the bounding box.
        
        
        Parameters
        ----------
        
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        
        angle : float
            angle by which the image is to be rotated
            
        cx : int
            x coordinate of the center of image (about which the box will be rotated)
            
        cy : int
            y coordinate of the center of image (about which the box will be rotated)
            
        h : int 
            height of the image
            
        w : int 
            width of the image
        
        Returns
        -------
        
        numpy.ndarray
            Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        """

        corners = corners.reshape(-1,2)
        corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
        
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        calculated = np.dot(M,corners.T).T
        
        calculated = calculated.reshape(-1,8)
        return calculated
    

def get_enclosing_box(corners):
        """Get an enclosing box for ratated corners of a bounding box
        
        Parameters
        ----------
        
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
        
        Returns 
        -------
        
        numpy.ndarray
            Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`
            
        """
        x_ = corners[:,[0,2,4,6]]
        y_ = corners[:,[1,3,5,7]]
        
        xmin = np.min(x_,1).reshape(-1,1)
        ymin = np.min(y_,1).reshape(-1,1)
        xmax = np.max(x_,1).reshape(-1,1)
        ymax = np.max(y_,1).reshape(-1,1)
        
        final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
        return final


def filter_bboxes(boxes, classes, window):
    temp_boxes = copy.deepcopy(boxes)
    temp_classes = copy.deepcopy(classes)

    nWmin = window[0]
    nHmin = window[1]
    nWmax = window[2]
    nHmax = window[3]
    
    for bbox, cls in zip(boxes, classes):
        if bbox[0] and bbox[2] < nWmin:
            temp_boxes.remove(bbox)
            temp_classes.remove(cls)

        elif bbox[0] & bbox[2] > nWmax:
            temp_boxes.remove(bbox)
            temp_classes.remove(cls)
            
        elif bbox[1] and bbox[3] < nHmin:
            temp_boxes.remove(bbox)
            temp_classes.remove(cls)
            
        elif bbox[1] & bbox[3] > nHmax:
            temp_boxes.remove(bbox)
            temp_classes.remove(cls)
            
    return temp_boxes, temp_classes


def crop_bboxes(filtered_boxes, filtered_classes, window):
    x_min, y_min, x_max, y_max = np.split(np.array(filtered_boxes),  4, axis=1)

    win_x_min = window[0]
    win_y_min = window[1]
    win_x_max = window[2]
    win_y_max = window[3]

    x_min_clipped = np.maximum(np.minimum(x_min, win_x_max), win_x_min)
    x_max_clipped = np.maximum(np.minimum(x_max, win_x_max), win_x_min)
    y_min_clipped = np.maximum(np.minimum(y_min, win_y_max), win_y_min)
    y_max_clipped = np.maximum(np.minimum(y_max, win_y_max), win_y_min)

    crop_bboxes = np.concatenate([x_min_clipped, y_min_clipped, x_max_clipped, y_max_clipped], axis=1)
    
    return crop_bboxes, filtered_classes


def scale_bboxes(bboxes, window):
    temp_array = np.array([window[0], window[1], window[0], window[1]])
    boxlist = bboxes - temp_array
    return boxlist


if __name__ == "__main__":
    a  = get_enclosing_box(None)
    