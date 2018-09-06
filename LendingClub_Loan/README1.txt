1. There are two cvs files in this package:
   Dataset 1: lending_club_data.csv (from coursera)
   Dataset 2: page-blocks.csv (download from http://sci2s.ugr.es/keel/category.php?cat=clas&order=ins#sub2)
   
2. My code for each dataset using different learning algorithms are named in the format: Data#_algorithmName.py.
   For Dataset1, the lending_club_data.csv, the python scripts Data1_DecisionTree.py, 
   Data1_Neural_Network.py, Data1_Boosting.py, Data1_svm.py and Data1_kNN.py can be run individually. 
   For Dataset2, the page-blocks.csv, the python scripts Data2_DecisionTree.py, 
   Data2_Neural_Network.py, Data2_Boosting.py, Data2_svm.py and Data2_kNN.py can be run individually. 
   
3. Python 2.7 is the environment that I run my python scripts. Python packages bumpy, pandas, sklearn and matplotlib are imported in my code. If all those four packages are already installed, simply run each python script in the directory containing the datasets.

4. The output of each script will have values and data frames printed to the terminal and figures will be output as .png files in the same directory.

5. The syan62-analysis.pdf contains brief description about what optimization I have done and the explanation of the reason why I perform those optimizations and why I got different results, as well as the figures. It is easier to read each section of the syan62-analysis.pdf and to run the python script corresponding to that section at the same time.