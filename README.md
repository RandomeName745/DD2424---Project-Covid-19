# DD2424---Project-Covid-19

Instructions: Apply CLAHE and augment images:

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
