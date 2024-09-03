import time
import cv2
from tensorflow.keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from models.unets import Unet2D
from models.deeplab import Deeplabv3, relu6, BilinearUpsampling, DepthwiseConv2D
from models.FCN import FCN_Vgg16_16s
from utils.learning.metrics import dice_coef, precision, recall
from utils.BilinearUpSampling import BilinearUpSampling2D
from utils.io.data import load_data, save_results, save_rgb_results, save_history, load_test_images, DataGen


# settings
input_dim_x = 224
input_dim_y = 224
color_space = 'rgb'
path = './data/wound_dataset/'
weight_file_name = '2024-06-05 12_35_06.557170.hdf5'
pred_save_path = '2019-12-19 01%3A53%3A15.480800/'
data_gen = DataGen(path, split_ratio=0.0, x=input_dim_x, y=input_dim_y, color_space=color_space)
x_test, test_label_filenames_list = load_test_images(path)


# Load model
model = Deeplabv3(input_shape=(input_dim_x, input_dim_y, 3), classes=1)
model = load_model('./training_history/' + weight_file_name,
                   custom_objects={'recall': recall,
                                   'precision': precision,
                                   'dice_coef': dice_coef,
                                   'relu6': relu6,
                                   'DepthwiseConv2D': DepthwiseConv2D,
                                   'BilinearUpsampling': BilinearUpsampling})

start_time = time.time()

for image_batch, label_batch in data_gen.generate_data(batch_size=len(x_test), test=True):
    prediction = model.predict(image_batch, verbose=1)
    save_results(prediction, 'rgb', path + 'test/predictions/' + pred_save_path, test_label_filenames_list)
    
    # Calculate and print metrics
    y_true = label_batch
    y_pred = prediction > 0.5  # Assuming binary classification, threshold at 0.5
    precision_score = precision(y_true, y_pred).numpy()
    recall_score = recall(y_true, y_pred).numpy()
    dice_score = dice_coef(y_true, y_pred).numpy()
    print(f'Precision: {precision_score}')
    print(f'Recall: {recall_score}')
    print(f'Dice Coefficient: {dice_score}')
    break

end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time:.2f} seconds")
