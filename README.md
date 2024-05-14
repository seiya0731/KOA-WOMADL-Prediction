=======================================================
The "Raw Data" contains all source data and programs used in this study.


1. The original clinical and image data can be found in the OAI official website: https://nda.nih.gov/oai. 
   
2. The image data we selected from the OAI and used in this study were saved in our Nutstore Net Disk：https://www.jianguoyun.com/p/DbuN3pIQufC4DBiL9cIFIAA, and contains the following three sub-folders:
    2.1 "dataOA" contains the original images without any procedures of image processing, which was selected from the OAI database. The selected 269 patients information is shown in the "data.csv" 
    2.2 "niiDataOA" contains the images after the procedrure of image preprocessing, using the code shown in "preprocessing_MR.py"
    2.3 "ROImaskOA" contains the ROI images segmented by the radiologists.

3. "Raw Data\data.csv" contained all 269 patients' clinical characteristics and extracted radiomics features. The procedure of feature extraction is implemented using the code shown in "batch-features.py".

4. Raw Data\results4plot.R" is the main statistical analysis and machine learning program. The uLR+mRMR+LASSO feature reduction and modeling method is implemented using this program. The LASSO, Calibration curve, and Nomogram  related figures and tables are generated using the codes in this program. Meanwhile, it generated the results files of "score_train.csv" and  "score_test.csv".

5. "score_train.csv" and  "score_test.csv" are used to generate the ROC, DCA, ROC delong test related figures and tables, using the program "general_fig_300ppi.py".
Note“.R"  and ".py" files are the programs written using R and python language, respectively.
 
