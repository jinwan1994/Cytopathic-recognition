# Cytopathic-recognition
Code of image classification with pytorch 
# dataset loder
## load code

    print("Load dataset......")
    image_datasets = {x: customData(img_path='data/',
                                    txt_path=('data/TxtFile/' + x + '.txt'),
                                    data_transforms=data_transforms,
                                    dataset=x) for x in ['train', 'val']}
## Train
### Begin to train

    ```bash
    # Training
    python resnet_train.py
    ```


