import os
import onnxruntime as rt
import numpy as np
from PIL import Image
providers = ['CPUExecutionProvider']

def load_model():
    model='ResNet50_imagenet_max'  
    model_path = os.path.join(os.path.dirname(__file__), f'{model}.onnx')
    print('Acquiring model "{}"'.format(model))

    m = rt.InferenceSession(model_path, providers=providers)
    m.pooling = None
    print('\rSuccessfully acquired model\t\t\t\t\t')
    return m

def _extract_batch(fp_list, source_path, model):
    # Initialize a list to store the preprocessed images
    img_data_list = []
    os.chdir(source_path)
    for fp in fp_list:
        img = Image.open(fp).resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        features = model.run(None, {'input': img_array})[0]
        img_data_list.append(features)

    return np.concatenate(img_data_list, axis=0)    