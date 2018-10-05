
Enhanced Feedback Enabled Cascaded Classification Models
---------------------------------------------------------
+--------- Readme constructed with Notepad++ -----------+

NOTE: For better visualization please open with Notepad++


I. Prerequisites
----------------
The program is based on the OpenCV libraries and the Knitro Nonlinear Optimization Solver.
In order to properly run the program, please download and install the necessary dependencies.

OpenCV - http://github.com/opencv/opencv
         Please note that the DPM-Module of the opencv_contrib repository is required.
		 For more information please visit http://github.com/opencv/opencv_contrib
Knitro - http://www.artelys.com/en/optimization-tools/knitro/downloads

Our files as well as our trained models can be found online in:
https://drive.google.com/drive/folders/1QAuqgdiRryafMRpXHTXRlwJv9TDfdJMM?usp=sharing

A. Datasets:
------------
The datasets are available online at:

Scene Categorization - http://people.csail.mit.edu/torralba/code/spatialenvelope/
Object Recognition   - http://host.robots.ox.ac.uk/pascal/VOC/voc2006/index.html
		
II. Using the EFE-CCM
---------------------
The program is run by executing the main program file "main.cpp". We made available several
variables to manage the program's flow. These variables are located in "globalVarsnDefs.h".
We have provided intuitive names for all our variables in order to facilitate its usage.
Additionally, due to an extensive documentation, we consider the program very straight-forward
to use. 

A. Constructing GIST Descriptor and Ground Truth Labels for the Scene Classification Dataset

Torralba et. al. provided a MATLAB implementation for their GIST Descriptor. Furthermore, they
provide their ground-truth labels in LabelMe format. We implemented a small MATLAB script 
"GISTLabelsnDescriptors.m" to generate the GIST-Descriptors and Ground-truth labels of their 
work into a .txt file, which is easily importable in C++. The generated files use an ';' as value
separator. 

The script generates the files SC_GistLabels.txt and SC_GistDescriptors.txt. The former file 
contains the name of the image and its label, while the later contains the corresponding Gist 
Descriptor for the image in that position. 

IMPORTANT: Note that the code of Torralba et. al. is required to execute "GISTLabelsnDescriptors.m"

B. Constructing the Ground-truth labels for Object Recognition

For this data set, we want to entangle the PASCAL ground-truth labeling format into a more readable
structure as the one defined in II.A. Here, we constructed the "fromPascalStructureToSimplified.m" 
script, which generates a .txt file with the following structure:

filename;x_left;y_top;x_right;y_bot;class_label

In our experiments, class_label can take the values:
1 - car
2 - person
3 - horse
4 - cow

III. File list
--------------

A. Scripts For Input-Data Generation
------------------------------------

GISTLabelsnDescriptors.m				See II.A

fromPascalStructureToSimplified.m		See II.B


B. Scripts For Output-Data Postprocesing
-----------------------------------------

GraphsExplainedVariance.ipynb			Script for graphical description of the explained variance as
										a function of the selected principal components.
										
GraphsClassificationAccuracy.ipynb		Script for graphical description of the evolution over iterations
										of the classification accuracy.
										
GraphVarImportance.ipynb				Script for graphical interpretation of the obtained variable
										importance in the second layer classifiers.

										
C. Required files for the main program execution
------------------------------------------------	

1. Datasets

SC Dataset 			/spatial_envelope_256x256_static_8outdoorcategories
OR Train Dataset 	/voc2006_trainval/VOCdevkit/VOC2006/PNGImages
OR Test Dataset		/voc2006_test/VOCdevkit/VOC2006/PNGImages
			
2. Descriptors and labels for the SC Task

	SC_GistLabels.txt		----+-> 	See II.A
	SC_GistDescriptors.txt	----+		Note that this files are obtainable with the .m provided files.										
	OR_GistLabels.txt		----+
	OR_GistDescriptors.txt	----+
	OR_GistLabels_test.txt	----+
	OR_GistDescriptors_test	----+		

3. DPM Models from Felzenswalb
	car 	----+->	 DPM_Models/"name".xml || "name" = object class name. 
	person	----+
	horse	----+
	cow		----+
	

D. Files Generated During Training
----------------------------------

1. General
----------

SC_structure.json						||	contains a "_1", "_2" structure which contains all features, labels 
											and correspondence for all the labels in the SC training set								
OR_in_SCDataset_FeatureMap.json 		||  contains the FeatureMaps for all considered OR classes. 


TEST_SC_structure.json					||	contains a "_1", "_2" structure which contains all features, labels 
											and correspondence for all the labels in the SC test set								   
TEST_OR_in_SCDataset_FeatureMap.json	||  contains the FeatureMaps for all considered OR classes. 


OR_structure.json						||	contains a "_1", "_2" structure which contains all features, labels 
											and correspondence for all the labels in the OR training set								   
SC_in_ORDataset_FeatureMap.json			||  contains the FeatureMap for SC recognition.


TEST_OR_structure.json					||	contains a "_1", "_2" structure which contains all features, labels 
											and correspondence for all the labels in the OR test set								   
TEST_SC_in_ORDataset_FeatureMap.json	||  contains the FeatureMap for SC recognition.
											
											
2. FE-CCM
----------
NOTE: The CCM model corresponds to the files listed below with i = 1.

firstLyrSC_svm_it"i".json 			|| "i" is the iteration id
firstLyrOR_"c"svm_it"i".json		|| "c" is the class id and "i" is the iteration id
secLyrSC_RF_it"i".json				|| "i" is the iteration id
secLyrOR_"c"RF_it"i".json			|| "c" is the class id and "i" is the iteration id

3. PCA-FE-CCM 
--------------

firstLyrSC_svm_it"i".json 			|| "i" is the iteration id
firstLyrOR_"c"svm_it"i".json		|| "c" is the class id and "i" is the iteration id
secLyrSC_RF_it_PCA_FE_"i".json		|| "i" is the iteration id
secLyrOR_PCA_FE_"c"RF_it"i".json	|| "c" is the class id and "i" is the iteration id

PCA_FE_SC_it"i".json				|| PCA transformation for SC data set
PCA_FE_OR"c"_it"i".json				|| PCA transformation for OR data set

4. LDA-FE-CCM 
--------------

firstLyrSC_svm_it"i".json 			|| "i" is the iteration id
firstLyrOR_"c"svm_it"i".json		|| "c" is the class id and "i" is the iteration id
secLyrSC_RF_it_LDA_FE_"i".json		|| "i" is the iteration id
secLyrOR_LDA_FE_"c"RF_it"i".json	|| "c" is the class id and "i" is the iteration id

_pca_LDA_FE_SC_it"i".json 			|| PCA transformation component for SC data set
_lda_LDA_FE_SC_it"i".json			|| LDA transformation component for SC data set

_pca_LDA_FE_OR"c"_it"i".json		|| PCA transformation component for SC data set
_lda_LDA_FE_OR"c"_it"i".json		|| LDA transformation component for SC data set

IMPORTANT: The obtained classification results are shown in the console. We have summarized their
		   results in several .txt files for all instances.		
		
		
E. Main Files:
--------------

globalVarsnDefs.h 			Definition of global variables and definitions used throughout the program.
	|						Definition of several program control flow variables.			
	|						Definition of several filenames (input and output)
	|						
	+-> Task1
	+-> FeatureMap
	+-> HOGDescriptor
	+-> res_HeatMapSize		Standardized Detection Map Size
	
	
generalPurpose.h			General purpose functions - header
generalPurpose.cpp			General purpose functions - implementation
	|
	+-> void split(const std::string &s, char delim, std::vector<std::string> &elems);
	+-> int factorial(int n, int n_min);
	+-> int factorial(int n);
	+-> int multinom_coeff(int n, int k);
	+-> int getIntApprox(double x);
	+-> cv::Mat sigmoid(const cv::Mat& input);
	+-> void calc_rbf(double gamma, int vec_count, int var_count, cv::Mat& sv, cv::Mat& other, 
			cv::Mat& results);

			
fileStorage.h				Functions for file storage and information consistency - header
fileStorage.cpp				Functions for file storage and information consistency - implementation
	|
	namespace fs;
	|
	+-> enum MODEL;
	+-> void _inner_saveTask_1(cv::FileStorage& fs, Task_1& task_1);
	+-> void _inner_loadTask_1(cv::FileStorage& fs, Task_1& task_1);
	+-> void _inner_saveFeatureMap(cv::FileStorage& fs, FeatureMap& task_2);
	+-> void _inner_loadFeatureMap(cv::FileStorage& fs, FeatureMap& task_2);
	+-> void _inner_saveFeatureMap(cv::FileStorage& fs, std::map<int, FeatureMap>& task_2);
	+-> void _inner_loadFeatureMap(cv::FileStorage& fs, std::map<int, FeatureMap>& task_2);
	+-> void saveLoadSet(std::string filename, MODEL saveLoad, Task_1& task_1);
	+-> void saveLoadSet(std::string filename, MODEL saveLoad, FeatureMap& task_2);
	+-> void saveLoadSet(std::string filename, MODEL saveLoad, std::map<int, FeatureMap>& task_2);
	+-> void saveLoadSet(std::string filename, MODEL saveLoad, Task_1& task_1, FeatureMap& task_2);
	+-> void saveLoadSet(std::string filename, MODEL saveLoad, Task_1& task_1, 
	|		std::map<int, FeatureMap>& task_2);
	+-> void saveLoadDPMDetections(std::string detections_filename, MODEL saveLoad,
	|		std::map<int, std::map<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>>>& dpm_detections);
	+-> void readGroundTruthFile(std::string filename,
	|		std::map<int, std::map<std::string, std::vector<cv::Rect>>> &records_map);
	+-> void readGISTfromFile(std::string filename_labels, std::string filename_descriptors, Task_1& task_1,
	|	FeatureMap& task_2);
	+-> std::string generateName(std::string first_part, int index, std::string second_part, int iter);
	+-> std::string generateName(std::string first_part, int iter);
	
	
ml.h 						Machine Learning Functions - header
ml.cpp 						Machine Learning Functions - implementation
	|
	namespace ml;
	|
	+-> enum TRAIN_SUBSET;
	+-> void divideTrainTestSet(FeatureMap& train, FeatureMap&test, const FeatureMap& src, 
	|		TRAIN_SUBSET train_subset, int folds_or_samples);
	+-> void divideTrainTestSet(Task_1& task_1_train, FeatureMap& task_2_train, Task_1& task_1_test, 
	|		FeatureMap& task_2_test, const Task_1& task_1, const FeatureMap& task_2, 
	|		TRAIN_SUBSET train_subset, int folds_or_samples);
	+-> void divideTrainTestSet(Task_1& task_1_train, std::map<int, FeatureMap>& task_2_train,
	|		Task_1& task_1_test, std::map<int, FeatureMap>& task_2_test, const Task_1& task_1, 
	|		const std::map<int, FeatureMap>& task_2, TRAIN_SUBSET train_subset, int folds_or_samples);
	+-> void svmTrain(cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& trainMat, const cv::Mat& trainLabels, 
	|		cv::ml::SVM::KernelTypes kernel_type);
	+-> void svmTrainWeigthed(cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& trainMat, const cv::Mat& trainLabels,
	|		cv::ml::SVM::KernelTypes kernel_type);
	+-> void RFTrain(cv::Ptr<cv::ml::RTrees>& rf, cv::Mat& trainMat, cv::Mat& trainLabels);
	+-> void LCTrain(cv::Ptr<cv::ml::LogisticRegression>& lc, cv::Mat& trainMat, cv::Mat& trainLabels);
	+-> template<typename T>
	|	void inference(cv::Ptr<T>& classifier, cv::Mat& testResponse, const cv::Mat& testMat);
	+-> void svmInference_RAW(cv::Ptr<cv::ml::SVM>& svm, cv::Mat& testResponse, cv::Mat& prob_response,
	|		const cv::Mat& testMat, const int trueLbl);
	+-> void RFInference_RAW(cv::Ptr<cv::ml::RTrees>& rf, cv::Mat& prob_response, const cv::Mat& testMat);
	+-> void evaluate(const cv::Mat& testResponse, double& count, const cv::Mat& testLabels, int max_lbl);
	+-> void predict_LogOdds(cv::Mat&oddsPrediction, cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& testFeatures, int N_class);
	+-> void logOdds_OR(cv::Mat& firstLyrOR_oddsPredictions, std::map<int, FeatureMap>& secLyr_OR_2, 
			cv::Ptr<cv::ml::SVM>& firstLyr_SC_svm, const FeatureMap& firstLyr_SC_OR_2, int firstLyr_OR_featSize_cols,
			const Task_1& OR_1);

			
structConsruct.h			Data-structure construction functions - header
structConsruct.cpp			Data-structure construction functions - implementation
	|
	+-> const double jaccardScore(const cv::Rect roi1, const cv::Rect roi2);
	+-> cv::Mat paddedROI(const cv::Mat& input, const cv::Rect& rect);
	+-> void getFeatureMapfromDPMDetections(const std::map<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>>& cl_i_dpmDetections,
	|		const std::map<std::string, std::vector<cv::Rect>>& class_i_gt, Task_1& task_1, 
	|		FeatureMap& task_2,	int class_id, double jaccardThresh = 0.5);
	+-> void independizeLayers(FeatureMap& secLyr, const FeatureMap & firstLyr, int cols_extraFeat);
	+-> void independizeLayers(std::map<int, FeatureMap>& secLyr, const std::map<int, FeatureMap>& firstLayer,
	|		int cols_extraFeat);
	+-> void gen_flatCoords(std::map<std::string, std::vector<int>>& flatCoords, const std::string filename,
	|		const std::map<std::string, int> ROI_strVect, int _offset);
	+-> void heatmaps_SC(FeatureMap& secLyr_SC_2, std::map<int, cv::Mat>& OR_SC_probs, std::map<int, cv::Mat>& OR_SC_inference,
	|		std::map<int, cv::Ptr<cv::ml::SVM>>& firstLyr_OR_svm, const std::map<int, FeatureMap>& firstLyr_OR_SC_2,
	|		const Task_1& SC_1, const std::map<std::string, std::map<std::string, std::vector<int>>>& SC_flatCoords);
	+-> void heatmaps_OR(std::map<int, FeatureMap>& secLyr_OR_2, std::map<int, cv::Mat>& OR_inference,
			std::map<int, cv::Ptr<cv::ml::SVM>>& firstLyr_OR_svm, const std::map<int, FeatureMap>& firstLyr_OR_2,
			const Task_1& OR_1, const std::map<std::string, std::map<std::string, std::vector<int>>>& OR_flatCoords,
			int secLyr_OR_featSize_cols, int firstLyr_OR_featSize_cols);
			

featReduction.h 			Feature reduction fuctions - header
featReduction.cpp			Feature reduction fuctions - implementation
	|
	namespace fr;
	|
	+-> enum BASE;
*	+-> std::pair<fr::BASE, float> featRed_SC_params; 				Definition of the found parameters for the SC dataset
	|																during feature reduction
*	+-> std::vector<std::pair<fr::BASE, float>> featRed_OR_params;	Definition of the found parameters for the OR dataset
	|																during feature reduction
	+->	void reduceDataset(cv::PCA& pca, cv::Mat& train, cv::Mat& test, BASE base, float ratio_feats);
	+-> void trainPCA(cv::PCA& pca, cv::Mat& train, std::string pca_filename);
	+-> void loadPCA(cv::PCA& pca, std::string pca_filename);
	+-> void trainLDA_SC(cv::LDA& lda, std::string lda_filename, Task_1& SC_1, cv::Mat& _lda_feats,
	|		FeatureMap& secLyr_SC_2);
	+-> void trainLDA_OR(std::map<int,cv::LDA>& lda_map, std::string lda_filename_1stpart, Task_1& OR_1, 
	|		std::map<int, FeatureMap>& OR_2, std::map<int, cv::Mat>& OR_probs,
	|		std::map<std::string, std::map<std::string, std::vector<int>>>& OR_flatCoords, int iter);
	+-> void loadLDA_OR(std::map<int, cv::LDA>& lda_map, std::string lda_filename_1stpart, int iter);
	+-> void init_datasets_LDA_FE(FeatureMap& secLyr_SC_2, FeatureMap& test_secLyr_SC_2,
	|		cv::Mat& _pca_feats, cv::Mat& _lda_feats, cv::Mat& _pca_feats_test, cv::Mat& _lda_feats_test);
	+-> void init_datasets_LDA_FE(std::map<int, FeatureMap>& secLyr_OR_2, std::map<int, FeatureMap>& test_secLyr_OR_2,
	|		std::map<int, cv::Mat>& _pca_feats,	std::map<int, cv::Mat>& _lda_feats,
	|		std::map<int, cv::Mat>& _pca_feats_test, std::map<int, cv::Mat>& _lda_feats_test);
	+-> void applyLDA(cv::LDA& lda, cv::Mat& feats);
	+-> void applyLDA(std::map<int, cv::LDA>& lda_map, std::map<int, cv::Mat>& feats_map);
	+-> void comb_outputs_LDA_FE(FeatureMap& secLyr_SC_2, FeatureMap& test_secLyr_SC_2,
	|		cv::Mat& _pca_feats, cv::Mat& _lda_feats, cv::Mat& _pca_feats_test, cv::Mat& _lda_feats_test);
	+-> void comb_outputs_LDA_FE(std::map<int, FeatureMap>& secLyr_OR_2, 
			std::map<int, FeatureMap>& test_secLyr_OR_2, std::map<int, cv::Mat>& _pca_feats, 
			std::map<int, cv::Mat>& _lda_feats,	std::map<int, cv::Mat>& _pca_feats_test, 
			std::map<int, cv::Mat>& _lda_feats_test);
			
			
optProb.h					Formulation of the MINLP problem - header
optProb.cpp					Formulation of the MINLP problem - implementation
	|
	namespace opt;
	|
*	+-> float min_prob;		Defines the minimum value for the an aparent probability of 0. Used to avoid log(0)
*	+-> float max_prob;		Defines the minimum value for the an aparent probability of 1. Used to avoid log(1)
	+-> class ProblemMINLP_SC : public knitro::KTRProblem;
	|	|
	|	+-> typedef std::map<std::string, std::vector<int>> flatCoordStruct_i;
	|	|
	|	private:
	|	|
	|	+-> void setObjectiveProperties():
	|	+-> void setVariableProperties();
	|	+-> void setConstraintProperties():
	|	|
	|	public:
	|	|
	|	members:
	|	|	|
	|	|	+-> const cv::Mat* probabilities_SC;
	|	|	+-> const cv::Mat* probabilities_OR;
	|	|	+-> flatCoordStruct_i* ptr_SC_flatCoords_i;
	|	|	+-> cv::Mat* feat_vector;
	|	|	+-> std::map<std::string, int>* ptr_labels;
	|	|	+-> cv::Ptr<cv::ml::RTrees>* ptr_classifier;
	|	|	+-> int nVars;
	|	|	+-> int secLyrTrueLabel;
	|	|
	|	+-> ProblemMINLP_SC(int _nVars, int nConst, cv::Mat& _probabilities_SC, cv::Mat& _probabilities_OR, cv::Mat& _feat_Vector,
	|	|		cv::Ptr<cv::ml::RTrees>& _ptr_classifier, flatCoordStruct_i &_SC_flatCoords_i, std::map<std::string, int>& _labels,
	|	|		int trueLabel);
	|	|
	|	+-> double evaluateFC(const double* x, double* c, double* objGrad, double* jac) override;
	|	
	+-> class ProblemMINLP_OR : public knitro::KTRProblem;
		|
		+-> typedef std::map<std::string, std::vector<int>> flatCoordStruct_i;
		|
		private:
		|
		+-> void setObjectiveProperties():
		+-> void setVariableProperties();
		+-> void setConstraintProperties():
		|
		public:
		|
		members:
		|	|
		|	+-> const cv::Mat* probabilities_SC;
		|	+-> const cv::Mat* probabilities_OR;
		|	+-> flatCoordStruct_i* ptr_OR_flatCoords_i;
		|	+-> std::vector<cv::Mat>* feat_vectors;
		|	+-> std::map<std::string, int>* ptr_labels;
		|	+-> std::map<int, cv::Ptr<cv::ml::RTrees>>* ptr_classifiers;
		|	+-> int nVars;
		|	+-> std::vector<int>* trueLabels;
		|	+-> std::vector<std::pair<int, int>>* feat_ids;
		|	+-> std::map<int, FeatureMap>* ptr_features_i;
		|
		+-> ProblemMINLP_OR(int _nVars, int nConst, cv::Mat& _probabilities_SC, cv::Mat& _probabilities_OR, std::vector<cv::Mat>& _feat_vectors,
		|	std::map<int, cv::Ptr<cv::ml::RTrees>>& _ptr_classifiers, flatCoordStruct_i &_OR_flatCoords_i, std::map<std::string, int>& _labels,
		|	std::vector<int>& _trueLabels, std::vector<std::pair<int, int>>& _feat_ids, std::map<int, FeatureMap>& _features_i):
		|	
		+-> double evaluateFC(const double* x, double* c, double* objGrad, double* jac)	override;
	
evalnSelectionFunctions.h 		Functions for hyperparameter selection - header
evalnSelectionFunctions.cpp		Functions for hyperparameter selection - implementation
	|
	+-> void select_RF_SVM_LC();


F. Authors
----------
This software implementation has been developed as part of the Master Thesis from David Wilson Romero Guzman,
under supervision of Dr.-Ing. Ronny Hänsch.




	
