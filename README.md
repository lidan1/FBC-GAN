# FBC-GAN

# dataset
    -cub2011
        -images folder
        -segmentation folder
        -images.txt
        -bounding_boxes.txt  
# files
    main_fbg.py
    trainer_fbg.py
    foreground.py
    background.py

# trian
<!-- python main_bg.py --cfg ./cfg/train_bg.yml --gpu 0 -->
python main_fbg.py --cfg ./cfg/train_bg.yml --gpu 0

# To Do:
    training and tuning hyper params
    testing and evaluation script