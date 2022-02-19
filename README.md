simpsons_images
==============================

Identify Simpsons character in image file

Project Organization
----

	├── LICENSE
	├── Makefile           			<- Makefile with commands like `make data` or `make train`
	├── README.md         	 		<- The top-level README for developers using this project.
	├── data
	│   ├── 01_raw         			<- The original, immutable data dump.
	│   ├── 02_interim     			<- Intermediate data that has been transformed.
	│   └── 03_processed   			<- The final, canonical data sets for modeling.
	│
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

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
