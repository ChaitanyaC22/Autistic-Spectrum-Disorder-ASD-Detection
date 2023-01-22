## Project Objective: To Detect Autistic Spectrum Disorder (ASD) cases 

### Autism Screening Adult Data Set 

---
![](Autism_Img.jpg)
---

This *advanced* level data set has Autistic Spectrum Disorder (ASD) Screening Test Data for 704 adults. There are 21 attributes in the dataset that include test takers' 
demographics such as *Age*, *Gender*, *Ethnicity* etc. The screening test included 10 questions (A1 to A10) that test takers answered. In each of these 10 questions, 
the test takers were given a statement with which they had to agree or disagree. For example, the statement at question A1 is "I often notice small sounds when other do not".
The test takers' responses to A1 and A10 are coded as binary values (0,1). Once the test taker has answered all 10 questions, his/her status on ASD is determined which 
is recorded under **Class/ASD** variable. You can encouraged to explore the [ASD Test Mobile App](http://asdtests.com/#)           


This data set is recommended for learning and practicing your skills in **exploratory data analysis**, **data visualization**, and **classification modelling techniques**. 
Feel free to explore the data set with multiple **supervised** and **unsupervised** learning techniques. The Following data dictionary gives more details on this data set:

---

### Data Dictionary 

| Column   Position 	| Atrribute Name  	| Definition                                                                                                                                          	| Data Type    	| Example                            	| % Null Ratios 	|
|-------------------	|-----------------	|-----------------------------------------------------------------------------------------------------------------------------------------------------	|--------------	|------------------------------------	|---------------	|
| 1                 	| A1_Score        	| Question 1 Answer: Binary (0, 1)                                                                                                                    	| Quantitative 	| 0, 1                               	| 0             	|
| 2                 	| A2_Score        	| Question 2 Answer: Binary (0, 1)                                                                                                                    	| Quantitative 	| 0, 2                               	| 0             	|
| 3                 	| A3_Score        	| Question 3 Answer: Binary (0, 1)                                                                                                                    	| Quantitative 	| 0, 3                               	| 0             	|
| 4                 	| A4_Score        	| Question 4 Answer: Binary (0, 1)                                                                                                                    	| Quantitative 	| 0, 4                               	| 0             	|
| 5                 	| A5_Score        	| Question 5 Answer: Binary (0, 1)                                                                                                                    	| Quantitative 	| 0, 5                               	| 0             	|
| 6                 	| A6_Score        	| Question 6 Answer: Binary (0, 1)                                                                                                                    	| Quantitative 	| 0, 6                               	| 0             	|
| 7                 	| A7_Score        	| Question 7 Answer: Binary (0, 1)                                                                                                                    	| Quantitative 	| 0, 7                               	| 0             	|
| 8                 	| A8_Score        	| Question 8 Answer: Binary (0, 1)                                                                                                                    	| Quantitative 	| 0, 8                               	| 0             	|
| 9                 	| A9_Score        	| Question 9 Answer: Binary (0, 1)                                                                                                                    	| Quantitative 	| 0, 9                               	| 0             	|
| 10                	| A10_Score       	| Question 10 Answer: Binary (0,   1)                                                                                                                 	| Quantitative 	| 0, 10                              	| 0             	|
| 11                	| Age             	| Age in years                                                                                                                                        	| Quantitative 	| 24, 32, 40                         	| 1             	|
| 12                	| Gender          	| Gender (m: Male, f: Female)                                                                                                                         	| Qualitative  	| "m", "f"                           	| 0             	|
| 13                	| Ethnicity       	| List of common ethnicities   (White-European, Latino, Others, Black, Asian, Middle Eastern, Pasifika,   South Asian, Hispanic, Turkish)             	| Qualitative  	| "Middle-Eastern", "Asian", "Black" 	| 13            	|
| 14                	| Jundice         	| Whether the case was born with   Jundice (Yes, No)                                                                                                  	| Qualitative  	| "yes", "no"                        	| 0             	|
| 15                	| Austim          	| Whether any immediate family   member has a PDD (Yes, No)                                                                                           	| Qualitative  	| "yes", "no"                        	| 0             	|
| 16                	| Country_of_res  	| Country of residence (List of   countries)                                                                                                          	| Qualitative  	| "Austria", "Ireland", "Jordan"     	| 0             	|
| 17                	| Used_app_before 	| Whether the user has used the   screening app before (Yes, No)                                                                                      	| Qualitative  	| "yes", "no"                        	| 0             	|
| 18                	| Result          	| Screening score: The final score   obtained based on the scoring algorithm of the screening method used. This   was computed in an automated manner 	| Quantitative 	| 5, 8, 10                           	| 0             	|
| 19                	| Age_desc        	| Age description                                                                                                                                     	| Qualitative  	| "18 and more"                      	| 0             	|
| 20                	| Relation        	| Who is completing the test   (Self, Parent, Health care professional, Relative, etc)                                                                	| Qualitative  	| "Parent", "Self", "Relative"       	| 13            	|
| 21                	| Class/ASD       	| yes, no                                                                                                                                             	| Qualitative  	| "yes", "no"                        	| 0             	|

---
### Project files related information:

**1. `Autistic-Spectrum-Disorder-ASD-Detection-Project.ipynb`:** Jupyter notebook showcases a thorough examination of data through exploratory analysis and data visualization techniques, as well as the selection and training of models pertinent to the classification task of identifying individuals with ASD. The notebook also includes evaluations of the chosen models' performance.<br><br>
**2. `Autistic-Spectrum-Disorder-ASD-Detection-Project.html`:** Web-page displaying Autistic-Spectrum-Disorder-ASD-Detection-Project.ipynb<br><br>
**3. `helper_functions.py` script:** comprises of all the essential auxiliary functions or methods that are required to perform various operations such as data visualization through the creation of graphs, and other miscellaneous tasks that are necessary for the completion of the project.<br><br>
**4. `model_pipelines.py` script:** encompasses a collection of model pipelines that are used to construct models for the binary classification task of determining the status of an individual with regards to Autistic Spectrum Disorder (ASD). These pipelines provide a streamlined and efficient method for building, training and evaluating the models for this classification task.<br><br>
**5. `Saved Model` Folder:** contains best estimator/model's pickle file (format: .sav)<br><br>
**6. `results.csv`:** model evaluation results of all the classifiers taken into account to perform the set of experiments<br><br>


### Acknowledgement

This data set has been sourced from the Machine Learning Repository of University of California, Irvine [Autism Screening Adult Data Set (UC Irvine)](https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult). 
The UCI page mentions the following as the original source of the data set:

+ Fadi Fayez Thabtah, Department of Digital Technology, Manukau Institute of Technology, Auckland, New Zealand 
<br><br><br>

Thank you for taking the time to visit this repository!

