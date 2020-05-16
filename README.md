# DD2424---Project-Covid-19
A forked implementation of ResNet architecture from calmisential https://github.com/calmisential/TensorFlow2.0_ResNet

# TensorFlow2.0_ResNet
A ResNet(**ResNet18, ResNet34, ResNet50, ResNet101, ResNet152**) implementation using TensorFlow-2.0

See https://github.com/calmisential/Basic_CNNs_TensorFlow2.0 for more CNNs.

## Train
1. Requirements:
+ Python >= 3.6
+ Tensorflow == 2.0.0
2. To train the ResNet on your own dataset, you can put the dataset under the folder **original dataset**, and the directory should look like this:
```
|——original dataset
   |——class_name_0
   |——class_name_1
   |——class_name_2
   |——class_name_3
```
3. Run the script **split_dataset.py** to split the raw dataset into train set, valid set and test set.
4. Change the corresponding parameters in **config.py**.
5. Run **train.py** to start training.
## Evaluate
Run **evaluate.py** to evaluate the model's performance on the test dataset.

## The repo contains following archs implemented with tensorflow2.0:
+ [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152](https://github.com/calmisential/TensorFlow2.0_ResNet)
+ [InceptionV3](https://github.com/calmisential/TensorFlow2.0_InceptionV3)


## References
1. The original paper: https://arxiv.org/abs/1512.03385
2. The TensorFlow official tutorials: https://tensorflow.google.cn/beta/tutorials/quickstart/advanced

################### Instructions: Apply CLAHE and augment images ##################################

techniques:
	zoom
	flip
	contrast
	brightness
	gaussian_blur
	gaussian_noise	
	elastic deformation

1. Download the .rar datasets from Drive and store them in a folder which is called 'datasets'
	- so this folder is supposed to have 3 subfolders given by the different datasets on Drive
	- go into these folders and create in each a new folder 'dataset_raw' where you move all the 		  subfolders with the classification classes
		COVID_19
		NORMAL
		PNEUNOMIA
	- this looks then like e.g.: .../datasets/DataSet_NoArrows/dataset_raw/COVID-19
							          	      /NORMAL
							     		      /PNEUNOMIA		
	- now the basic structure is given

2. Next, execute ApplyClaheAndAugmentation.py, before you do that you have to:
	- change dirDataset to: dirDataset = yourpath/datasets/DataSet_NoArrows/ (the DataSet you want 	 										  to use for training)
	- now it should work for you, you dont have to do adaptions to any of the other folder-strings!

3. After execution, check whether the augmented data was created:
	- there should be three folders now in the .../datasets/DataSet_NoArrows/ (e.g.) folder:
		1. dataset_raw
		2. dataset_clahe
		3. dataset_clahe_augmented
	- check if augmentation was applied as expected:
		zoom (random)
		flip (vertically)
		contrast (random)
		brightness (random)

4. The next augmentation techhniques are applied using the CLODSA library: open the file augmentImages.py

	- the settings for the augmentation techniques
		gaussian_blur
		gaussian_noise	
		elastic deformation
		(none)
	  are stored in config_augmentImages.json, I have tested the parameters and they should deliver 	  appropiate augmentation
	- open config_augmentImages.json and change the input and output folders to:
		"input_path":"yourpath/datasets/DataSet_NoArrows/dataset_clahe/"
		"output_path":"youtpath/datasets/DataSet_NoArrows/dataset_clahe_augmentedCLODSA/"
	- the new folder with the augmented images is created automatically, if it exists beforehand it 	  causes and error!

5. Now we have 2 folders with CLAHE and different augmentation techniques applied!
	merge them to a single folder if you want
