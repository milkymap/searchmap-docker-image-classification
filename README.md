# searchmap-docker-image-classification
This tools allows to build an image classification deep learning model 

# requirepents 
* create the directory workspace
```bash
    mkdir workspace
    mkdir workspace/dataset
    mkdir workspace/features
    mkdir workspace/checkpoints 

    mv path2source_images/* ./workspace/dataset
    mv labels_config.csv ./workspace 
```

* create the directory models 
```bash
    mkdir models 
```

# docker env variables 
* WORKSPACE
    * this is a main workspace
    * it must contain dataset, features and checkpoints subdir
    * add the csv file that describe the task : image_name => labels   
    * DATASET
        * path where images are located 
        * all images should be there
    * FEATURES
        * path where extracted features are saved
        * features will be saved in pickle format 
        * the extracted features filename is embeddings.pkl 
    * CHECKPOINTS
        * path where snapshot will be saved 
    * labels_config.csv 

* MODELS
    * path where extractors (vectorizer) are saved
    * resnet18, resnet34, resnet50, ... vgg16, alexnet ...! 

# build and run server-mode for cpu 
* build image 
```bash
docker build -t searchmap:classification -f Dockerfile.cpu .
``` 
* run container 
```bash
docker run --rm --name classification --tty --interactive -v /home/ibrahima/Datasets/Cancer/:/home/solver/workspace -v /home/ibrahima/Models/features_extractor/:/home/solver/models -p 8500:8000 searchmap:classification --backbone resnet18 --img_extension '*.jpg' --task_labels_config label_config.csv --embeddings_filename embeddings.pkl processing --nb_workers 2 training --nb_epochs 32 --batch_size 8 --optimizer_name Adam serving --server_port 8000 --hostname '0.0.0.0' --prefix '/backend'
```

# rebuild the features 
to rebuild the features, use the option --force in processing stage
to rebuild the model, use --force in training stage 
