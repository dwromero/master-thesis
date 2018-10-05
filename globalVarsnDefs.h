#pragma once

#ifndef GLOBALVARSDEFS_H
#define GLOBALVARSDEFS_H

//STD Dependencies
#include <map>
#include <list>
#include <algorithm>
#include <random>
#include <thread>
#include <functional>
#include <bitset>
#include <limits>
#include <chrono>
#include <time.h>
#include <utility>

//OpenCV Dependencies
#include<opencv2/opencv.hpp> 
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include<opencv2/ml.hpp>
#include<opencv2/dpm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv/highgui.h>

//MINLP-Optimizer Dependencies
#include "KTRSolver.h"
#include "KTRProblem.h"

///////////////////////////   Paths and Filenames   /////////////////////////////////
const std::string models_root = "C:/Users/David Romero/source/repos/Thesis/Thesis/";

const std::string OR_GT_train = models_root + "PASCAL_train_annotations.txt";
const std::string OR_GT_test = models_root + "PASCAL_test_annotations.txt";

const std::string SC_aux = models_root + "spatial_envelope_256x256_static_8outdoorcategories/";

const std::string DPM_MODELS_PATH = "C:/Users/David Romero/Documents/MasterThesis/Object Detection/DPM_Models/";
const std::string OR_train_IMG_PATH = "C:/Users/David Romero/Documents/MasterThesis/Object Detection/Dataset/voc2006_trainval/VOCdevkit/VOC2006/PNGImages/";
const std::string OR_test_IMG_PATH = "C:/Users/David Romero/Documents/MasterThesis/Object Detection/Dataset/voc2006_test/VOCdevkit/VOC2006/PNGImages/";

const std::string OR_1_filename = models_root + "OR_structure.json";
const std::string SC_1_filename = models_root + "SC_structure.json";


const std::string test_SC_1_filename = models_root + "TEST_SC_structure.json";
const std::string test_OR_1_filename = models_root + "TEST_OR_structure.json";

const std::string SC_GISTDescr = models_root + "SC_GistDescriptors.txt";
const std::string SC_GT = models_root + "SC_GistLabels.txt";

const std::string SC_OR_GISTDescr = models_root + "OR_GistDescriptors.txt";
const std::string SC_OR_GT = models_root + "OR_GistLabels.txt";
const std::string SC_OR_GISTDescr_test = models_root + "OR_GistDescriptors_test.txt";
const std::string SC_OR_GT_test = models_root + "OR_GistLabels_test.txt";

const std::string SC_DPM_Detections = models_root + "SC_train_DPM.json";
const std::string OR_DPM_Detections_train = models_root + "OR_train_DPM.json";
const std::string OR_DPM_Detections_test = models_root + "OR_test_DPM.json";

const std::string OR_SC_featMap_filename = models_root + "OR_in_SCDataset_FeatureMap.json";
const std::string SC_OR_featMap_filename = models_root + "SC_in_ORDataset_FeatureMap.json";

const std::string test_OR_SC_featMap_filename = models_root + "TEST_OR_in_SCDataset_FeatureMap.json";
const std::string test_SC_OR_featMap_filename = models_root + "TEST_SC_in_ORDataset_FeatureMap.json";
//---------------------------------------------------------------------------------//

//////////////////////////////   Feat Reduction   ///////////////////////////////////

const std::string PCA_FE_SC_filename = models_root + "PCA_FE_SC_it";
const std::string PCA_FE_OR_filename = models_root + "PCA_FE_OR";

const std::string _pca_LDA_FE_SC_filename = models_root + "_pca_LDA_FE_SC_it";
const std::string _pca_LDA_FE_OR_filename = models_root + "_pca_LDA_FE_OR";
const std::string _lda_LDA_FE_SC_filename = models_root + "_lda_LDA_FE_SC_it";
const std::string _lda_LDA_FE_OR_filename = models_root + "_lda_LDA_FE_OR";

//---------------------------------------------------------------------------------//

//Task labels:
const std::string OR_label = "OR";
const std::string SC_label = "SC";

//Program Flow Controllers
bool calculateDPMs = false;

bool constructStructure = false;
bool calculateTestSet = false;

bool train1stLyr = false;
bool train2ndLyr = false;

bool trainSelectRFSVMLC = false;

int max_iters = 5;
int init_iter = 2;

//bool EFE_CCM = false;
bool CCM = false;
bool PCA_FE_CCM = false;
bool LDA_FE_CCM = false;
bool train_LDA_EFE = false;

//PCA_EFE_CCM Control Variables
bool trainPCA = false;

//ML Parameters:
int no_trees = 512;
float norm_factor = 1 / (float)no_trees;

//Classifier savefile-names --- Selection of best 2nd Layer Classifier
const std::string firstLyrSC_Class_filename = models_root + "firstLyrSC_svm_it";
const std::string secLyrSC_Class_filename = models_root + "secLyrSC_RF_it";

const std::string firstLyrOR_Class_filename = models_root + "firstLyrOR_";
const std::string secLyrOR_Class_filename = models_root + "secLyrOR_";

const std::string secLyrSC_SVM_filename = models_root + "secLyrSC_svm_it";
const std::string secLyrOR_SVM_filename = models_root + "secLyrOR_";

const std::string secLyrSC_LC_filename = models_root + "secLyrLC_svm_it";
const std::string secLyrOR_LC_filename = models_root + "secLyrLC_";

//////////////////////////   Structs and Typedefs   /////////////////////////////////

typedef std::map<std::string, std::map<std::string, int>> Task_1;

struct FeatureMap {

	//Constructors
	FeatureMap() = default;
	FeatureMap(cv::Mat features, cv::Mat labels, std::vector<std::string> localizer) :
		features(features), labels(labels), localizer(localizer) {}

	//Members
	cv::Mat_<float> features;
	cv::Mat_<int> labels;
	std::vector<std::string> localizer;
};
//---------------------------------------------------------------------------------//

//Define HOG Descriptor
cv::HOGDescriptor hog(
	cv::Size(64, 64), //winsize
	cv::Size(16, 16), //blocksize
	cv::Size(8, 8), //blockstride
	cv::Size(8, 8), //cellsize
	9,
	1, -1, 0, 0.2, 1, 64, false);
const int descriptorSize = hog.getDescriptorSize() + 1; //HOG Descriptor Size + DPM Detection Score

//Load DPM Detectors  --------------- This structure defines the object classes considered in the model.
//Based on the PASCAL Models of Felzenswalb.
std::map<int, std::string> dpmDetector_filenames{
	{ 1, "car.xml" },
{ 2, "person.xml" },
{ 3, "horse.xml" },
{ 4, "cow.xml" }
};

//SC_variables
cv::Size res_HeatMapSize = cv::Size(16, 16);
int class_no_SC = 8;
int flatMapSize = res_HeatMapSize.width * res_HeatMapSize.height;
int cols_extraFeat;
int secLyr_SC_cols;
int _SC_offset_noOdds;
int _SC_offset;


//OR_variables
int firstLyr_OR_featSize_cols;
int _OR_offset; //firstLyr_OR_featSize_cols + class_no
int secLyr_OR_featSize_cols;

//FeatRed
int flatDimension = dpmDetector_filenames.size()*flatMapSize;


#endif // !GLOBALVARSDEFS_H
