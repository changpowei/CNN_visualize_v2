# File: config.py
# Author: Qian Ge <geqian1001@gmail.com>


"Training config"

# directory of pre-trained VGG19 parameters
vgg_dir = '../lib/nets/pretrained/vgg19.npy'

# directory of training image data
data_dir = '../CAM_data/training_data/256_ObjectCategories/'

# directory of validation image data
#所有驗證圖片的路徑，用於訓練實驗證訓練準確率和loss
val_data_dir = '../CAM_data/training_data/256_ObjectCategories/'

# directory of the image use for inference class activation map during training (put only one image)
#僅一張驗證圖片，紀錄訓練過程中CAM的變化
infer_data_dir = '../CAM_data/training_inference_image/'

# directory of saving inference result (saved every 50 training steps)
infer_dir = '../CAM_data/training_inference_result/'

# directory of saving summaries (saved every 10 training steps)
#存放summary data的日誌檔案，可用tensorboard查看訓練時的數據
summary_dir = '../CAM_data/summaries/'

# directory of saving trained model (saved every 100 training steps)
checkpoint_dir = '../CAM_data/retrained_model/'



"Testing config"

# directory of trained model parameters
#存放model的路徑，路徑下包含如 checkpoint, model-24800.data, model-24800.index, model-24800.meta的資料
#應該會和訓練時，checkpoint_dir的路徑相同
model_dir = '../CAM_data/retrained_model/Caltech-256_GPU_v5/'

# directory of testing data
test_data_dir = '../data'

# directory of saving testing images
result_dir = '../CAM_data/CAM_test_result/'
