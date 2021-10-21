
![image info](./pictures/ncl.jpg) 


# 
<h1 align="center"> CSC8639 - MSc Data Science  Project and Dissertation </h1>
<h1 align="center"> Machine Learning for Medicine:
Predicting Fibre Classification for Patient and Control Datasets using clustering Algorithms
</h1>
<h1 align="center"> Author: Frestie Ngongo <br> Supervisors: Stephen McGough, Amy Vincent, Conor Lawless, Atif Khan </h1>

## 1. Abstract
---
The diagnosis of patients with mitochondrial disease can be prone to human error with studies highlighting that 55% of patients with mitochondrial disease were initially misdiagnosed. This resulted from an insufficiency in understanding and clinical tests for identifying key factors in patients with mitochondrial diseases, to help with diagnosis. Therefore, conducting this research in order to find out if protein intensities could be used as suitable factors was essential, because the heterogenous expression of oxidative phosphorylation proteins is known to be a characteristic finding in mitochondrial disease. For this reason, the proteins ‘NDUFB8 ‘and ‘NDUFA13’, both part of the oxidative phosphorylation process, were analysed using machine learning unsupervised models. Clustering algorithms – K-Means and Gaussian Mixture Model – were used to identify whether these proteins could be used to separate out patient fibres into two groups, those with the reactive–chain deficiency, and those without. This produced plots that showed the Gaussian Mixture Model was better fitted to the data due to producing two clusters, unlike the K-Means model. Additionally, the CI variant of the mitochondrial disease was found to have 100% of patient fibres as reactive-chain deficient, whilst only a small percentage, ~4%, of the patient fibres of those with the deletion variant of mitochondrial disease were predicted as reactive-chain deficient in the Gaussian Mixture Model. This showed that the model may have identified the pattern for each patient and used this to predict their clinical outcome.
## 2. Introduction
---
Mitochondrial dysfunction is identified as a particularly heterogenous pathological change [1]. Mitochondrial diseases have been difficult to diagnose due to the different ways it presents in various individuals, including the many different organs [2]. In addition, there is no single lab or diagnostic test that can be carried out to confirm whether an individual has the disease [2]. Currently, gene analysis is the most reliable method used to confirm mitochondrial disease states in individuals, alongside taking a family history, blood and urine tests, and physical examinations [3]. However, studies have found that 55% of patients with mitochondrial disease were initially misdiagnosed on first admission and of these, 32% were misdiagnosed twice [4]. These figures are alarming and highlight the impact of human error. Introduction of an alternative method of diagnosis could therefore help to solve this issue, i.e., the use of Machine Learning (ML), such as the use of the decision tree classifier and naive Bayesian classifier to analyse medical records. All that needs doing would be to input the patient records with known correct diagnosis into a computer program to run a learning algorithm. This would then produce an output and the accuracy of the algorithm can be checked [5].


The dataset used in this study was obtained from The Welcome Centre for Mitochondrial Disease of which were 3 controls and 9 patients. Imaging Mass Cytometry (IMC) was used to analyse the proteins in these samples. IMC works to analyse up to 40 protein markers simultaneously, using metal-labelled antibodies with laser ablation, followed by detection using mass cytometry by time-of-flight [6]. This was used to produce images that were then used to quantify the protein mean intensities. The proteins looked at were NDUFA13, NDUFB8, VDAC1, COX4+4L2, OSCP, MTCO1, SDHA, UqCRC2. The VDAC1 protein is known to be a mitochondrial mass marker and was used as a standard for creating a ratio with the other proteins [7]. Information regarding the 8 different proteins within the sample, the myofibers locations, the area of the myofibers, cell circularity and perimeter were obtained, and results presented in a table. The focus was how the protein mean intensities affected the categorisation of the fibres. This is because the proteins were more realistic factors that could contribute to the diagnosis of whether an individual had the mitochondrial disease (patient) or they did not (control).


The use of Machine Learning Algorithms may help to analyse these proteins and determine whether they are useful markers for diagnosis of mitochondrial diseases. For example, a study looked at developing three ML predictive models for cancer diagnosis and managed to achieve a maximum accuracy of 96% using the support vector machines algorithm [7]. This was used to separate the data into two groups - those with cancer and those without. Applying this knowledge, the use of ML may be used to categorise the fibres of the individuals in this study into those with a reactive chain (RC) deficiency and those without. RC deficiency is a characteristic of those with the mitochondrial disease and the proteins NDUFB8 and NDUFA13 are known as NADH:ubiquinone oxidoreductase subunit B8 and A13 respectively. They function in the transfer of electrons from NADH to the respiratory chain, therefore an abnormality in the numbers of these proteins present may cause a break in the oxidative phosphorylation process, hence reducing the amount of energy produced for the proper functioning of the cell. For this reason, the NDUFB8 and NDUFA13 proteins were focussed on in this project.


Machine Learning as a branch of Artificial Intelligence and Computer Science uses algorithms to imitate the way humans learn, while gradually improving its accuracy [8]. Being able to predict whether an individual has the mitochondria disease or not and going further to identify what variation of the disease they may have, is important, but being able to do so accurately is even more essential. Therefore, running algorithms, including clustering algorithms, on the dataset, and analysing their accuracy and precision could potentially provide a means to make the diagnosis of mitochondrial diseases easier and accurate, due to pattern recognition, and less human errors.

The aim of this project is to design a model to accurately classify fibres into one of two groups: healthy or RC deficient. Then going further to classify those with mitochondrial disease into one of six groups, those with the mitochondrial disease and those without. This will involve carrying out two clustering algorithms to identify whether either of one of them is able to produce two clusters: one for each classification.

## 3. Background and Relevant Work
---
There are four types of machine learning: supervised learning (SL); unsupervised learning (UL), semi-supervised learning (SSL), reinforcement learning (RL). For SL models, input and output data are fed into the algorithm, whilst for UL models, they independently identify patterns in the input data and use this to predict an output. In SSL algorithms, they can use a mix of classified and unclassified data to build problem-solving models. For RL models, they make use of a rewards system, in that the algorithms get rewarded for desired actions and punished for undesired actions. The clustering algorithms used in this project make use of unsupervised learning. This is ideal as the model can predict what category to place the fibres into based on patterns identified, hence aiding to see how accurately the model can be used in diagnosis of mitochondrial diseases [9].


A study conducted by Warren. C et al., looking at the decoding of mitochondrial heterogeneity in single muscle fibres using imaging mass cytometry (IMC), formed the foundation of this project. [7] IMC makes use of the antibody-conjugated isotopes of rare earth metals with laser ablation, and detection using mass cytometry by time-of-flight [6]. It analyses up to 40 protein markers simultaneously to create images of high definition from a single tissue section [6]. This produced the dataset used for the study by Warren. C et al., and the same dataset used in this project. The result of the study by Warren. C et al., was that they were able to demonstrate the accurate quantification of protein levels using IMC. From this they accurately measured the deficiency of oxidative phosphorylation for common mitochondrial DNA variants and witnessed a compensatory upregulation in the number of unaffected oxidative phosphorylation components [7]. This led to the construction of this project, to identify whether either of the NDUFB8 and NDUFA13 proteins contributed to the deficiency in oxidative phosphorylation witnessed, because the heterogenous expression of oxidative phosphorylation proteins, and resulting respiratory deficiency, are characteristics found in fibres with a mitochondrial dysfunction.


A recent trial had made use of machine learning algorithms (namely the SL models) to predict those that fall into the at-risk category for COVID-19 in a timely manner, hence reducing death rates [10]. In this study, they identified 20 features they deemed as significant for predicting the survival chance of an individual and ran this against SL models (logistic regression, random forest, and extreme gradient boosting) [10]. The outcome of the study was that the random forest model outperformed the others. The study conducted by Sumayh S. Aljameel et al., was similar to the one carried out in this research, in that specific features were used to predict one thing or the other. 


Alternatively, another study by Hany Alashwal et al., used multiple clustering algorithms for partitioning patients of Alzheimer’s disease based on their similarity [11]. They used K-Means clustering to identify whether it could classify individuals into the correct bio-profile. They observed that for those with Alzheimer’s disease, more than 69% of them and about half of those with mild cognitive impairment were always assigned to the pathological bio-bioprofile. This led to the belief that the K-means algorithm could predict datasets with clinical features into specific labels. This prompted the use of clustering algorithms in this research project. Also, the idea that the data passed through the model is unlabelled is beneficial as it means that the model is required to identify its own pattern and predict an outcome. Plus, the article by Hany Alashwal et al., also highlighted that unsupervised learning algorithms have been proven to be powerful for discovering patterns. [11]

## 4. Methods
---
The work carried out in this project involved using the notebook from anaconda using python software to calculate the log of the mean intensities of the proteins and using that value for the rest of the project. The first part of the project included analysing the data using graphs to view the log intensity of the proteins. This was followed by constructing a K-Means clustering algorithm, then a Gaussian Mixture Model (GMM) clustering algorithm.



### 4.1 Exploratory Data Analysis

For the analysis, the raw data, which was in long format, was reshaped to a Y format. This splits the data into key columns in a data frame. From this, the key columns, containing the logs of the proteins, the patient type, cell ID, patient ID, and subject group. Then the log mean intensity of the VDAC1 protein with the log mean intensities of all the other proteins were plotted on a scatter plot in python. The proteins belonging to the patients were coloured in blue and those of the control were coloured in orange.

### 4.2 K-Means

Following on from this, the K-Means clustering algorithm was constructed. This involved setting the number of clusters to 2, as we knew that there were two labels: those with the mitochondrial disease and those without the disease. The data applied to this model included all the 6 proteins. The K-Means was left to use this data to predict a diagnosis of whether the fibre belonged to a patient or a control. The results of this algorithm were presented using two separate scatter plots similar to the raw data comparing the proteins: logNDUFA13 and logNDUFB8 with the logVDAC1 protein. The scatterplot was also coloured in red for the control and blue for the patient to show which fibres the model had predicted to fall into which category.

### 4.3 GMM
A similar method used for the K-Means was applied to the GMM. So, the GMM was used to predict 2 clusters using data consisting of only two proteins at a time; logNDUFB8 with logVDAC1; then logNDUFA13 with logVDAC1. The result of this algorithm was used to create two scatterplots of the logNDUFB8 and logNDUFA13 with the logVDAC1. The plots were coloured in blue for the fibres predicted to be non-RC deficient and in green for the fibres predicted to be RC deficient.

### 4.4 Analysis of Clusters

The results of the K-Means and GMM clustering algorithms were used to create tables consisting of the proportions of fibres predicted to be RC deficient and those predicted to be non-RC deficient. The tables presented these proportions for everyone (both controls and patients), plus the proportions predicted for each disease type.

## 5. Results
---
This project aimed to use clustering algorithms to identify any patterns in the data and to see whether the models could be used to split the data into 2 clusters: those with mitochondrial disease and those without. 

### 5. 1 2D Mito Plot
![image info](./pictures/figure1.png) 
>Figure 1. Plots of 5 different proteins against the LOG_VDAC1 protein.
The plots of 5 different proteins against a sixth protein (logVDAC1), that are found in the individuals’ fibres. The plots identify that the control fibres, in orange, seem to be less spread out than the patients’ fibres, in blue.


The fibres from the IMC were analysed and it was found in the exploratory data analysis (Figure 1) that the data for the patients were very spread out for each protein. However, for the control, the data appeared to be less spread out. For this reason, it was apparent that the patients and control had some differences in the protein compositions. This led to the use of different clustering algorithms to identify whether the proteins were key in labelling an individual as a patient or control.


### 5.2 NDUFA13 

The K-Means and GMM algorithm were ran and produced plots of the outcome based on the NDUFB8 protein.

![image info](./pictures/figure2.png) 
>Figure 2. Plots of the amounts of logNDUFA13 vs the amounts of logVDAC1 proteins in the raw data, K-Means Model and Gaussian Mixture Model (GMM).
The K-Means algorithm showed no distinct separation in the clusters in comparison to the GMM model, which showed two separate groups. The control group appeared to be in the lower left corner of the graph of the KMeans in red, whilst the results for patients showed them in the higher end of the graph in blue. For GMM, there was a clear separation with an observation of 2 distinct clusters forming the ‘V’ shape: control in blue, patients in green.
	

It was observed that the GMM model produced a better prediction than the K-Means graph to classify the fibres, as it produced two clusters. The next step was to see if the same was true for the NDUFB8 protein.

### 5.3 NDUFB8

The K-Means and GMM algorithm were ran and produced plots of the outcome based on the NDUFB8 protein.

## 5.4 K-Mean results
---

## 5.5 GMM Table of Results part 1:  LogVDAC1 & LogNDUFA13
---
## 5.6 GMM Table of Results part 2: LogVDAC1 & LogNDUFB8
---

## 6. Conclusion
---

## 7. Contribution to the States of the Art
---

## 8. Scope and Limits of the Work and Future Work
---

[Githublink](https://github.com/FNgongo/MScDissertationLeighSyndrome/blob/main/EDAandBasicML-FNVesion-Copy1.ipynb)

## References
---
1.	Alston, Charlotte L et al. “The genetics and pathology of mitochondrial disease.” The Journal of pathology vol. 241,2 (2017): 236-250. doi:10.1002/path.4809
2.	Koenig, Mary Kay. “Presentation and diagnosis of mitochondrial disorders in children.” Pediatric neurology vol. 38,5 (2008): 305-13. doi:10.1016/j.pediatrneurol.2007.12.001
3.	Cleveland Clinic. 2021. Mitochondrial Diseases: Causes, Symptoms, Diagnosis & Treatment. [online] Available at: <https://my.clevelandclinic.org/health/diseases/15612-mitochondrial-diseases>
4.	Columbia University's Mailman School of Public Health. "Mitochondrial disease patients face difficult road to diagnosis: On average, patients see more than eight physicians, undergo multiple tests, and receive misdiagnoses before finally being diagnosed with a mitochondrial disease." ScienceDaily. ScienceDaily, 26 March 2018. <www.sciencedaily.com/releases/2018/03/180326161004.htm>.
5.	Kononenko I. Machine learning for medical diagnosis: history, state of the art and perspective. Artificial Intelligence in Medicine. 2001;23(1):89-109. 
6.	Veenstra J, Dimitrion P, Yao Y, Zhou L, Ozog D, Mi Q. Research Techniques Made Simple: Use of Imaging Mass Cytometry for Dermatological Research and Clinical Applications. Journal of Investigative Dermatology. 2021;141(4):705-712.e1. 
7.	Warren, C., McDonald, D., Capaldi, R. et al. Decoding mitochondrial heterogeneity in single muscle fibres by imaging mass cytometry. Sci Rep 10, 15336 (2020). https://doi.org/10.1038/s41598-020-70885-3
8.	Education, I., 2021. What is Machine Learning?. [online] Ibm.com. Available at: <https://www.ibm.com/cloud/learn/machine-learning>
9.	Patel, Yash. Machine Learning in HealthCare. ResearchGate. 2021
10.	Sumayh S. Aljameel, Irfan Ullah Khan, Nida Aslam, Malak Aljabri, Eman S. Alsulmi, "Machine Learning-Based Model to Predict the Disease Severity and Outcome in COVID-19 Patients", Scientific Programming, vol. 2021, Article ID 5587188, 10 pages, 2021. https://doi.org/10.1155/2021/5587188
11.	Alashwal H, El Halaby M, Crouse J, Abdalla A, Moustafa A. The Application of Unsupervised Clustering Methods to Alzheimer’s Disease. Frontiers in Computational Neuroscience. 2019;13.

## 
---

## 
---

## 
---
<h1 align="center"> CSC8639 :  Data Science MSc Project and Dissertation </h1>
## Machine Learning for Medicine : 


This is some text .

* Here is the list of item
* Here is another one.
<!-- Headings -->

# Heading 1
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6


<!-- Italics -->
*This text* is itaslic

\*This text\* is itaslic

_This text_ is italic

<!-- Strong -->
__This text__is strong

**This text** is strong

<!-- Strikethrough -->

~~This text~~ is a strikethrough

<!-- Horizontal rule -->
---

---

<!-- Blockquote -->
>This is a quote

<!-- Links -->

[Goldendoer](http://www.goldendoer.co.uk)

[Tables Generator](https://www.tablesgenerator.com/markdown_tables
"Table Generator")

<!-- UL -->
* Item 1
* Item 2
* Item 3
    * Nested Item 1
    * Nested Item 2

<!-- OL -->
1. Item 1
1. Item 2
12. Item 3

<!-- Inline code block -->

<p> This is a paragraph </p>

<!-- Images -->
![Markdown Logo](https://markdown-here.com/img/icon256.png)



<!-- Github Markdown -->
    
<!-- Code Blocks -->

```bash
 npm install 
 npm start
```

```javascript
   function add(num1, num2){
       return num1 + num2;
   }
```

```python
def add(num1, num2):
  return num1 +num2
```

<!-- Table -->

| Name       |  Email             |
|------------|--------------------|
| Frestie    |  fretie@iCloud.com |
| John       |  john@iCloud.com   |


<!-- Task List -->

* [x] Task 1
* [x] Task 2
* [ ] Task 3




