from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import torchvision
import numpy as np
import os
import pandas as pd

# Initialize MTCNN for face detection
mtcnn = MTCNN()
# Load pre-trained Inception ResNet model
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

#Returns an array of the coordinates of the edges of the box containing the face in the image
def get_face_box(image):
    box_coords, _ = mtcnn.detect(image)
    if box_coords is not None:
        return box_coords

#Returns a tensor of the features of a face found on an image
def create_face_features_tensor(image):
    face_detected = mtcnn(image)
    if face_detected is not None:
        feature_tensor = resnet(face_detected.unsqueeze(0)).detach()
        return feature_tensor

#Returns the distance between 2 feature tensors using Euclidean Distance
def tensors_distance(tensor_1,tensor_2):
    distance = (tensor_1 - tensor_2).norm().item()
    return distance

def crop_faces(image, boxes):
    cropped_faces = []
    for box in boxes:
        cropped_face = image.crop(box.tolist())
        cropped_faces.append(cropped_face)
    return cropped_faces

# Creates tensor database containing the information of all faces
def create_database():
    results = []
    for filename in os.listdir('data'):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # Load the image
            image_path = os.path.join('data', filename)
            image = Image.open(image_path)
            
            # Get feature tensor
            feature_tensor = create_face_features_tensor(image)
            # To a list to save if in a json fila
            tensor_list = feature_tensor.tolist()

            # Append the result to the list
            results.append({'Image': filename, 'Tensor_Info': tensor_list})

    image_names = [entry['Image'] for entry in results]
    tensors = [entry['Tensor_Info'] for entry in results]

    # Convert the list of tensors to a numpy array
    tensors_array = np.array(tensors)

    # Save the numpy array to a .npz file
    np.savez('features_base.npz', tensors=tensors_array, image_names=image_names)

def search_in_database(image):
    results = []
    ref_tensor = create_face_features_tensor(image)
    data = np.load('features_base.npz')

    tensors_array = data['tensors']
    image_names = data['image_names']

    # Convert the numpy array back to a list of tensors
    tensors_list = [torch.tensor(tensor) for tensor in tensors_array]


    for image_name, tensor in zip(image_names, tensors_list):

        distance = tensors_distance(ref_tensor, tensor)

        if distance < 1.0:  # You can adjust the threshold for verification
            result = "Same person"
        else:
            result = "Different persons"
        
        results.append({'Image': image_name.replace('data/',''), 'Similarity': distance, 'Result': result})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Similarity', ascending=True)

    return results_df
