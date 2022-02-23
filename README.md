simpsons_images
==============================

Purpose
====================
Build a series of CNN models to identify the main Simpsons character in an image file.
The following models will be built:
1. Model with no hyperparameter tuning
2. Model with no hyperparameter tuning, but utilizing a callback to adjust the LR
3. Model with hyperparameter tuning
4. Model using transfer learning, leveraging a prebuilt CNN model from TensorFlow Hub
5. Model using transfer learning and fine tuning the final layers

Project Organization
----

	├── LICENSE
	├── Makefile           			<- Makefile with commands like `make data` or `make train`
	├── README.md         	 		<- This file
	├── data
	│   └── images
	|    		├── simspons_dataset	<- Contains set of subfolders with image files for training			
	│				└── simpsons_testset	<- Contains set of images for testing
	|
	├── docs               			<- Documentation
	│
	├── models             			<- Trained and serialized models, model predictions, or model summaries
	│
	├── notebooks          			<- Jupyter notebooks. Naming convention is a number (for ordering),
	│                         	 	 the creator's initials, and a short `-` delimited description, e.g.
	│                         		 `1.0-jqp-initial-data-exploration`.
	│
	├── references         			<- Data dictionaries, manuals, and all other explanatory materials.
	│
	├── reports            			<- Generated analysis as HTML, PDF, LaTeX, etc.
	│   └── figures        			<- Generated graphics and figures to be used in reporting
	│
	├── requirements.txt   			<- The requirements file for reproducing the analysis environment, e.g.
	│                         	 	 generated with `pip freeze > requirements.txt`
	│
	├── setup.py  
	│
	├── src                			<- Source code for use in this project.
	│   ├── __init__.py    			<- Makes src a Python module
	│   │
	│   ├── d00_utils						<- Functions used across the project
	│   │
	│   ├── d01_data       			<- Scripts to download or generate data
	│   │
	│   ├── d02_intermediate  	<- Scripts to turn raw data into intermediate
	│   │
	│   ├── d03_processing    	<- Scripts to turn intermediate into modeling output
	│   │												
	│   ├── d04_modeling				<- Scripts to train models and use models for predictions                 
	│   │
	│   ├── d05_evaluations			<- Scripts to analyze model performance                
	│   │
	│   ├── d06_reporting  			<- Scripts to produce reports/tables               
	│   │
	│   └── d07_visualizations	<-Scripts to create plots/graphics                 
	│
	└── tox.ini            			<- tox file with settings for running tox; see tox.readthedocs.io


----
