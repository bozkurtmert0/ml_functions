import os
import glob
import os
import glob
import random
from PIL import Image
from ultralytics import YOLO

#----------------------------------------------------------------------------------------------

def image_path_to_list(folder_path):
  """
  Appends the file paths of all images in the given directory

  :param folder_path: File path of images
  :return: List of file paths of image files
  """ 

  image_files = glob.glob(os.path.join(folder_path, "*.jpg"))  
  image_files = glob.glob(os.path.join(folder_path,  "*.[jpg|jpeg|png|gif|bmp]")) 
  image_list = image_files

  return image_path_list

#----------------------------------------------------------------------------------------------

def yolo_pred_image(image_path_list,model):
  """
  
  :param image_path_list: List of image path you want to predict
  :param model: Yolo model for predict
  :return: List of predicted images
  """
  
  pred_imgs_list = []
  for i in range(0,len(image_path_list)):
    result = model(image_path_list[i])
    for r in result:
      im_array = r.plot(line_width = 1,
                        font_size = 0.5,
                        kpt_radius = 2

      )  
      # plot a BGR numpy array of predictions
      im = Image.fromarray(im_array[..., ::-1])
      pred_imgs_list.append(im)

  return pred_imgs_list

#------------------------------------------------------------------------------------------


def path2imgList(image_paths):
  """
  :return: List of images
  """
  
  images = [Image.open(image_path) for image_path in image_paths]

  return images

#---------------------------------------------------------------------------------------------
  

def img_merge(img_list,types = "horizontal",save_img = False ):
  # Birleştirmek istediğiniz fotoğrafların dosya yollarını belirtin
  #image_paths = [image_list[1]]

# Birleştirme türünü seçin: "horizontal" (yanyana) veya "vertical" (alt alta)
  merge_type = types

# Fotoğrafları açın ve boyutlarını alın
  images = img_list
  image_widths, image_heights = zip(*(image.size for image in images))

  # Birleştirilecek resimlerin toplam genişliği ve yüksekliği hesaplayın
  if merge_type == types:
      total_width = sum(image_widths)
      max_height = max(image_heights)
  else:
      max_width = max(image_widths)
      total_height = sum(image_heights)

# Birleştirme için yeni bir boş resim oluşturun
  if merge_type == types:
      merged_image = Image.new("RGB", (total_width, max_height))
  else:
      merged_image = Image.new("RGB", (max_width, total_height))

# Fotoğrafları yeni resim üzerine kopyalayın
  x_offset = 0
  y_offset = 0
  for image in images:
      if merge_type == "horizontal":
          merged_image.paste(image, (x_offset, 0))
          x_offset += image.width
      else:
          merged_image.paste(image, (0, y_offset))
          y_offset += image.height
  if save_img:
    merged_image.save("path/to/merged_image.jpg")

  return merged_image



def get_images_with_annotations(dataset_folder):
    """
    bgr_image = cv2.cvtColor(labeled_images[0], cv2.COLOR_RGB2BGR)
    pil_image = Image.fromarray(bgr_image)

    """
  
    image_files = glob.glob(os.path.join(dataset_folder, "images", "*.jpg"))
    label_files = glob.glob(os.path.join(dataset_folder, "labels", "*.txt"))

    annotated_images = []
    class_dict = {
        0: "anarsialineatella",
        1: "anarsia_lineatella",
        2: "mosquito",
        3: "other",
    }
    color_palette = {
        "anarsialineatella": (0, 0, 255),  # Mavi renk
        "anarsia_lineatella": (150, 255, 0),  # Yeşil renk
        "mosquito" :(255, 0, 0),
        "other":(100, 100, 100),
        # Diğer etiketler
    }

    for image_path in image_files:
        image = cv2.imread(image_path)

        label_path = os.path.join(dataset_folder, "labels", os.path.basename(image_path)[:-4] + ".txt")

        with open(label_path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())

                img_height, img_width, _ = image.shape
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height

                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)


                class_name = class_dict.get(int(class_id), "unknown")
                color = color_palette.get(class_name, (0, 0, 0))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        annotated_images.append(image)

    return annotated_images
