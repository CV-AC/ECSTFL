## EC-STFL: Expression-Clustered Spatiotemporal Feature Learning


This is an official implementation of EC-STFL, which is proposed in the paper "DFEW: A Large-Scale Database for Recognizing Dynamic Facial Expressions in the Wild"[[1]](https://dl.acm.org/doi/10.1145/3394171.3413620). 

EC-STFL is a loss function to improve model performance for Facial Expression Recognition (FER) task. It is proposed for video-based Facial Expression Recognition (FER) task, but it also works for image-based FER method. We provide code for inference testing and finetuning training. Theese code can be conducted on both GPU and CPU devices. We also release the pretrained checkpoint (model weights) for you.




### Updates
Sep, 6th, 2024: We released the code of first version. 


### Requirements
* torch
* torchvision
* sklearn 
* pandas 


### Inferencing (Testing) the pretrained models on images/videos 
* Preparing the path file. 
  * If you are testing images, putting the path file named 'Imgs_test.csv' in "./annotation_infer/".
     ```markdown
     # The following shows the details of path file (imgs; .csv file)
     filePath,expression
     xxx1.jpg,1
     xxx2.jpg,0
     ...
     ```
  * If you are testing videos, putting the path file named 'Videos_test.txt' in "./annotation_infer/".
    ```markdown
    # The following shows the details of path file (videos; .txt file). 
    # The first column is folder name of each video clip; the second column is images number of one certain video clip; the third column is the corresponding label. 
    video1_folder 96 3
    video2_folder 48 1
    ...
    ```
   * Downloading the pretrained checkpoint (model weights); Putting them into the path of "./checkpoint_load/".
   * Conducting the program
     ```shell
     # PREPARING THE PATH FILE.
     # This path could be passed if you have prepared the path file by yourselves. 
     # The following step is to prepare the path file by revising the main path of data in pre-defined annotation file. 
     cd $main_path\ECSTFL\annotation_tr
     python script.py --data_root main_datapath
     #python script.py --data_root D:\Dataset\AffectNet\Manually_Annotated # AffectNet
     cd ..
     # If you process image data
     python infer.py --type img --model_name resnet18_mscele1m --device gpu
     # If you process video data
     # gamma is the hyper-parameter to balance cross-entropy loss and EC-STFL loss. 
     python infer.py --type video --model_name r3d_dfew --device gpu    
     ```
### Finetuning the pretrained models on your own dataset
   * Preparing the path file
     * If you are finetuning model in image data, putting the path file named "{xx}_train.csv" and "{xx}_test.csv" to "./annotation/" folder. The data format is same as the requirement of testing.
         ```markdown
         # The following shows the details of path file (imgs; .csv file)
         filePath,expression
         xxx1.jpg,1
         xxx2.jpg,0
         ...
         ```
     * If you are finetuning model in video data, putting the path file named "{xx}_train.txt" and "{xx}_test.txt" to "./annotation/" folder. The data format is same as the requirement of testing.
        ```markdown
        # The following shows the details of path file (videos; .txt file). 
        # The first column is folder name of each video clip; the second column is images number of one certain video clip; the third column is the corresponding label. 
        video1_folder 96 3
        video2_folder 48 1
        ...
        ```
     * Note that the label mapping is different among datasets. It is crucial to keep the label mappings of pre-training dataset and finetuning datasets to be same. We list some mapping from class number to facial expression name.
        ```markdown
        For DFEW, 7 emotions:
        0 - Happy
        1 - Sad 
        2 - Neutral
        3 - Angry
        4 - Surprise 
        5 - Disgust
        6 - Fear
        ```
   * Downloading the pretrained checkpoint (model weights); Putting them into the path of "./checkpoint_load/".
   * Conducting the program
     ```shell
     # PREPARING THE PATH FILE.
     # This path could be passed if you have prepared the path file by yourselves. 
     # The following step is to prepare the path file by revising the main path of data in pre-defined annotation file. 
     cd $main_path\ECSTFL\annotation  
     python script.py --data_root main_datapath
     cd ..
     # If you process image data
     # xx_train is the path file of training data; xx_val is the path file of testing data.
     python finetune.py --type img --model_name resnet18_ECSTFL_rafdb --device gpu --gamma 10 --list_file_train ./annotataion/xx_train.csv --list_file_val ./annotation/xx_val.csv 
     # If you process video data
     # gamma is the hyper-parameter to balance cross-entropy loss and EC-STFL loss. If gamma=0, EpiScale loss degrades to Cross Entropy loss. 
     # xx_train is the path file of training data; xx_val is the path file of testing data. 
     python finetune.py --type video --model_name r3d_ECSTFL_dfew --device gpu --gamma 10 --list_file_train ./annotation/xx_train.txt --list_file_val ./annotation/xx_val.txt    
     ```

### Checkpoint
| Loss function     | Modality | Backbone    | Dataset  | accuracy | Checkpoint                                                                                                                                                                                                     |
|:------------------|:---------|-------------|----------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cross-Entropy     | Video    | r3d_18      | DFEW fd1 | 62.02    | [[baidu drive (code:50ff)]](https://pan.baidu.com/s/1EdphDWzsDKCOB47DpRaqUw ) [[drop box]](https://www.dropbox.com/scl/fi/r0mh851p1x8q94goe7be8/r3d_dfew.pth?rlkey=3vg0n2c1obx1l02rjea1kmans&st=s70rp8tm&dl=0) |
| EC-STFL(gamma=10) | Video    | r3d_18      | DFEW fd1 | 64.08    | [[baidu drive (code:50ff)]](https://pan.baidu.com/s/1EdphDWzsDKCOB47DpRaqUw ) [[drop box]](https://www.dropbox.com/scl/fi/pxlr57ag4ntjhe2x5uk49/r3d_ECSTFL_dfew.pth?rlkey=mydlbngt0kgb8xwl4atri63ak&st=2j6c4ccz&dl=0)                                                                                        |
| Cross-Entropy     | Img      | resnet18    | RAF-DB   | xx.xx    | [[baidu drive (code:50ff)]](https://pan.baidu.com/s/1EdphDWzsDKCOB47DpRaqUw) [[drop box]](https://www.google.com/)                                                                                         |
| EC-STFL           | Img      | resnet18    | RAF-DB   | xx.xx    | [[baidu drive (code:50ff)]](https://pan.baidu.com/s/1EdphDWzsDKCOB47DpRaqUw) [[drop box]](https://www.google.com/)                                                                                         |



### Citation
if you use this code, please cite:

```mardown
@inproceedings{jiang2020dfew,
  title={DFEW: A large-scale database for recognizing dynamic facial expressions in the wild},
  author={Jiang, Xingxun and Zong, Yuan and Zheng, Wenming and Tang, Chuangao and Xia, Wanchuang and Lu, Cheng and Liu, Jiateng},
  booktitle={Proceedings of the 28th ACM international conference on multimedia},
  pages={2881--2889},
  year={2020}
}
```

### License
Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0 International Licence (CC BY-NC-ND) license.







