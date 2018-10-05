
#include "globalVarsnDefs.h" //Contains general required includes and definitions
#include "fileStorage.h" //Contains all functions related with storage and information consistency
#include "structConstruct.h" //Contains all functions with which we construct our data structures
#include "ml.h" //Contains all functions related with machine learning
#include "optProb.h" //Contains all the functions related with the optimization problem of the algorithm
#include "featReduction.h" //Contains functions related to feature reduction.

#include "evalnSelectionFunctions.h" //Functions for evaluation and selection of parameters

int main() {

	//Seed rand()
	srand(time(NULL));

	//select_RF_SVM_LC();

	//Get DPM-Detections from the datasets based on the considered object classes and save it in hard-drive
	if (calculateDPMs) {

		std::cout << "[info]\t Obtaining DPM Detections from data set" << std::endl;
		std::cout << "[info]\t    Object Recognition: Training" << std::endl;

		std::map<int, cv::Ptr<cv::dpm::DPMDetector>> dpmDetectors;
		for (auto &dpm_file : dpmDetector_filenames) {
			dpmDetectors[dpm_file.first] = cv::dpm::DPMDetector::create(std::vector<std::string>(1, DPM_MODELS_PATH + dpm_file.second),
				std::vector<std::string>(1, std::to_string(dpm_file.first)));
		} //Construct the detectors from saved files.

		std::map<int, std::map<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>>> dpm_detections;
		getDPMDetectionsFromDatabase(OR_train_IMG_PATH, dpmDetectors, dpm_detections); //Detect all objs in database
		fs::saveLoadDPMDetections(OR_DPM_Detections_train, fs::MODEL::SAVE, dpm_detections); //Save the detected DPM in the dataset
	}

	if (calculateDPMs) {
		std::cout << "[info]\t    Object Recognition: Test" << std::endl;

		std::map<int, cv::Ptr<cv::dpm::DPMDetector>> dpmDetectors;
		for (auto &dpm_file : dpmDetector_filenames) {
			dpmDetectors[dpm_file.first] = cv::dpm::DPMDetector::create(std::vector<std::string>(1, DPM_MODELS_PATH + dpm_file.second),
				std::vector<std::string>(1, std::to_string(dpm_file.first)));
		} //Construct the detectors from saved files.

		std::map<int, std::map<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>>> dpm_detections;
		getDPMDetectionsFromDatabase(OR_test_IMG_PATH, dpmDetectors, dpm_detections); //Detect all objs in database
		fs::saveLoadDPMDetections(OR_DPM_Detections_test, fs::MODEL::SAVE, dpm_detections); //Save the detected DPM in the dataset
	}

	if (calculateDPMs) {
		std::cout << "[info]\t    Scene Categorization" << std::endl;

		std::map<int, cv::Ptr<cv::dpm::DPMDetector>> dpmDetectors;
		for (auto &dpm_file : dpmDetector_filenames) {
			dpmDetectors[dpm_file.first] = cv::dpm::DPMDetector::create(std::vector<std::string>(1, DPM_MODELS_PATH + dpm_file.second),
				std::vector<std::string>(1, std::to_string(dpm_file.first)));
		} //Construct the detectors from saved files.

		std::map<int, std::map<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>>> dpm_detections;
		getDPMDetectionsFromDatabase(SC_aux, dpmDetectors, dpm_detections); //Detect all objs in database
		fs::saveLoadDPMDetections(SC_DPM_Detections, fs::MODEL::SAVE, dpm_detections); //Save the detected DPM in the dataset
	}

	std::cout << "[info]\t Constructing / Loading Datasets" << std::endl;
	/****************************************************************************************\
	*                   Load Datasets / Construct Data Structures                            *
	\****************************************************************************************/
	// Initially, the datasets need to be read in the program and the corresponding required 
	//structures must be created.
	/*
	Our software implementation mainly relies on two data structures. A Task_1 structure and a
	FeatureMap structure (see definition in "globalVarsnDefs.h"). The Task_1 structure keeps an
	overview on all the samples of the dataset and relates it with the feature vector matrix of
	the tasks included in the Holistic Scene Understanding formulation. The FeatureMap structure,
	signalized by an "_2" at the end of the variable name, is composed of the feature vector matrix
	and the corresponding labels for the considered task.

	Example: SC_1: Relates to the Scene Categorization Task and contains an overview over all the
	images contained in the SC data set. Additionally, it contains the samples encountered for 
	converse tasks in the same dataset. SC_1 is composed of a map<filename, list_of_samples>, in
	which filename corresponds to every single image in the dataset and list_of_samples provides
	a list of all the encountered samples relevant for all the holistic scene understanding tasks.
	At the same time, list_of_samples is composed of a map<task_label, index>, which relates each
	task with a corresponding index in the feature vector matrix. 
	In our work an image contains a list_of_samples with OR_$index$ and SC task_labels. OR_$index$
	relates to candidate windows for objects of the class $index$ encountered in the image and SC
	provides the descriptor of the image for the scene classification task.
	
	An "_1" structure is related to multiple "_2" structures. In our work, they are composed of a
	FeatureMap "SC_2" containing the feature vector matrix of the SC task and a map<index, featureMap>
	"OR_2" containing the feature vector matrix corresponding to the $index$ class.
	*/

	std::cout << res_HeatMapSize << std::endl;

	std::cout << "[info]\t    Loading Scene Categorization" << std::endl;
	Task_1 SC_1;
	FeatureMap firstLyr_SC_2;
	
	if (constructStructure) {
		Task_1 SC_1_train;
		Task_1 SC_1_test;
		FeatureMap firstLyr_SC_2_train;
		FeatureMap firstLyr_SC_2_test;

		fs::readGISTfromFile(SC_GT, SC_GISTDescr, SC_1, firstLyr_SC_2);

		//Select a subset of the datasets to do training faster (DEBUG)
		ml::divideTrainTestSet(SC_1_train, firstLyr_SC_2_train, SC_1_test, firstLyr_SC_2_test, SC_1, firstLyr_SC_2,
			ml::TRAIN_SUBSET::TRAIN_SAMPLES, 100);

		SC_1 = SC_1_train;
		firstLyr_SC_2 = firstLyr_SC_2_train;
		//saveLoadSet("SC_structure.json", MODEL::SAVE, SC_1_train, firstLyr_SC_2_train);
		// Since there are posterior modifications in the SC_1 Structure, it is not save here.
	}
	else fs::saveLoadSet(SC_1_filename, fs::MODEL::LOAD, SC_1, firstLyr_SC_2);

	std::cout << "[info]\t    Loading Object Recognition" << std::endl;
	Task_1 OR_1;
	std::map<int, FeatureMap> firstLyr_OR_2;

	//construct or load structure
	if (constructStructure) {

		//Load DPM Detections:
		std::map<int, std::map<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>>> dpm_detections_OR;
		fs::saveLoadDPMDetections(OR_DPM_Detections_train, fs::MODEL::LOAD, dpm_detections_OR); //Load the detected DPM in the dataset

		//Load Groundtruth files:
		std::map<int, std::map<std::string, std::vector<cv::Rect>>> groundTruth_OR; //map classID -> map filename -> vector of true ROIs in image
		fs::readGroundTruthFile(OR_GT_train, groundTruth_OR); //Load the groundtruthfile

		//Construct Dataset:
		int threadCount = dpm_detections_OR.size();
		cv::parallel_for_(cv::Range(0, threadCount), [&](const cv::Range& range) {
			for (int r = range.start; r < range.end; r++) {
				getFeatureMapfromDPMDetections(dpm_detections_OR.at(r + 1), groundTruth_OR.at(r + 1), OR_1, firstLyr_OR_2[r + 1], r + 1, 0.5);
			}
		});
		//Select a subset of the datasets to do training faster (DEBUG) -- OPTIONAL!
		//divideTrainTestSet(OR_1_train, firstLyr_OR_2_train, OR_1_test, firstLyr_OR_2_test, OR_1, firstLyr_OR_2,
		//	TRAIN_SUBSET::TRAIN_SAMPLES, 100);

		//OR_1 = OR_1_train;
		//firstLyr_OR_2 = firstLyr_OR_2_train;

		//fs::saveLoadSet(OR_1_filename, fs::MODEL::SAVE, OR_1, firstLyr_OR_2);
		// Since there are posterior modifications in the SC_1 Structure, it is not saved here.
	}
	else fs::saveLoadSet(OR_1_filename, fs::MODEL::LOAD, OR_1, firstLyr_OR_2);

	//Now the datastructures need to be modified to include the converse tasks
	std::cout << "[info]\t    Loading Object Recognition on Scene Categorization Dataset" << std::endl;
	
	std::map<int, FeatureMap> firstLyr_OR_SC_2;
	if (constructStructure) {
		//Load DPM Detections
		std::map<int, std::map<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>>> dpm_detections_SC;
		fs::saveLoadDPMDetections(SC_DPM_Detections, fs::MODEL::LOAD, dpm_detections_SC); //Load the detected DPM in the dataset

		for (int index = 1; index <= dpm_detections_SC.size(); index++) {
			int size_counter = 0;
			for (auto & img_path : SC_1) {

				auto img_path_pos = std::find_if(dpm_detections_SC.at(index).begin(), dpm_detections_SC.at(index).end(),
					[&](std::pair<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>> str) {
					return(str.first.find(img_path.first) != std::string::npos);
				}); //Find the detections that correspond to the filename.

				if (img_path_pos == dpm_detections_SC.at(index).end()) continue; //NO DPM found in that image

				cv::Mat img = cv::imread(img_path_pos->first); //Load the image
				for (auto &roi_detect : img_path_pos->second) {
					std::vector<float> hog_descriptor;

					cv::Mat sub_img = paddedROI(img, roi_detect.rect); //Get the subimage corresponding to the ROI
					cv::resize(sub_img, sub_img, hog.winSize); //Resize to 64x64 (Constant Size)
					hog.compute(sub_img, hog_descriptor); //Compute the descriptor of the ROI
					hog_descriptor.push_back(roi_detect.score); //And add the score of the DPM detection to the vector --> DescriptorSize = hog.descriptorSize()+1

					//Construct the string descriptor of the ROI
					std::string rect_str = std::to_string(roi_detect.rect.x) + ";" + std::to_string(roi_detect.rect.y)
						+ ";" + std::to_string(roi_detect.rect.width) + ";" + std::to_string(roi_detect.rect.height);

					std::string task_roi_ref = OR_label + "_" + std::to_string(index) + "&" + rect_str;

					//And add the entry to the structures
					img_path.second[task_roi_ref] = size_counter;
					firstLyr_OR_SC_2[index].features.push_back(hog_descriptor);
					firstLyr_OR_SC_2[index].labels.push_back(0);
					firstLyr_OR_SC_2[index].localizer.push_back(img_path.first + "&" + task_roi_ref);
					size_counter++;
				}
			}
			firstLyr_OR_SC_2[index].features = firstLyr_OR_SC_2[index].features.reshape(0,
				(firstLyr_OR_SC_2[index].features.rows*firstLyr_OR_SC_2[index].features.cols) / descriptorSize); //Reshape feature Mat.
		}

		fs::saveLoadSet(OR_SC_featMap_filename, fs::MODEL::SAVE, firstLyr_OR_SC_2);
		saveLoadSet(SC_1_filename, fs::MODEL::SAVE, SC_1, firstLyr_SC_2);
	}
	else fs::saveLoadSet(OR_SC_featMap_filename, fs::MODEL::LOAD, firstLyr_OR_SC_2);

	std::cout << "[info]\t    Loading Scene Classification on Object Recognition Dataset" << std::endl;
	
	FeatureMap firstLyr_SC_OR_2;
	if (constructStructure) {
		Task_1 SC_OR_aux;
		FeatureMap firstLyr_SC_OR_aux;
		fs::readGISTfromFile(SC_OR_GT, SC_OR_GISTDescr, SC_OR_aux, firstLyr_SC_OR_aux);

		int size_counter = 0;
		for (auto &img_path : OR_1) {

			auto img_path_pos = std::find_if(SC_OR_aux.begin(), SC_OR_aux.end(),
				[&](std::pair<std::string, std::map<std::string, int>> str) {
				return(img_path.first.find(str.first) != std::string::npos);
			}); //Find the position of the string that correspond to the filename.

			if (img_path_pos == SC_OR_aux.end()) continue; //NO entry found

			int row = img_path_pos->second.at(SC_label);
			firstLyr_SC_OR_2.features.push_back(firstLyr_SC_OR_aux.features.row(row).clone());
			firstLyr_SC_OR_2.labels.push_back(firstLyr_SC_OR_aux.labels.row(row).clone());
			firstLyr_SC_OR_2.localizer.push_back(img_path.first + "&" + SC_label);

			img_path.second[SC_label] = size_counter;
			size_counter++;
		}
		fs::saveLoadSet(SC_OR_featMap_filename, fs::MODEL::SAVE, firstLyr_SC_OR_2);
		fs::saveLoadSet(OR_1_filename, fs::MODEL::SAVE, OR_1, firstLyr_OR_2);
	}
	else fs::saveLoadSet(SC_OR_featMap_filename, fs::MODEL::LOAD, firstLyr_SC_OR_2);

	std::cout << "[info]\t Loading TEST SETS:" << std::endl;
	std::cout << "[info]\t    Scene Classification" << std::endl;

	Task_1 test_SC_1;
	FeatureMap test_firstLyr_SC_2;
	std::map<int, FeatureMap> test_firstLyr_OR_SC_2;
	if (calculateTestSet) {

		FeatureMap test_first_SC_2_aux;
		fs::readGISTfromFile(SC_GT, SC_GISTDescr, test_SC_1, test_first_SC_2_aux);

		int counter = 0;
		auto it = test_SC_1.begin();

		while (it != test_SC_1.end()) {
			if (SC_1.find(it->first) != SC_1.end()) {
				it = test_SC_1.erase(it);
				continue;
			}
			//if not in train  dataset, add
			for (auto & sample : it->second) {
				test_firstLyr_SC_2.features.push_back(test_first_SC_2_aux.features.row(sample.second).clone());
				test_firstLyr_SC_2.labels.push_back(test_first_SC_2_aux.labels.row(sample.second).clone());
				sample.second = counter;
				counter++;
			}
			it++;
		}

		std::map<int, std::map<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>>> dpm_detections_SC;
		fs::saveLoadDPMDetections(SC_DPM_Detections, fs::MODEL::LOAD, dpm_detections_SC); //Load the detected DPM in the dataset

		for (int index = 1; index <= dpm_detections_SC.size(); index++) {
			int size_counter = 0;
			for (auto & img_path : test_SC_1) {

				auto img_path_pos = std::find_if(dpm_detections_SC.at(index).begin(), dpm_detections_SC.at(index).end(),
					[&](std::pair<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>> str) {
					return(str.first.find(img_path.first) != std::string::npos);
				}); //Find the detections that correspond to the filename.

				if (img_path_pos == dpm_detections_SC.at(index).end()) continue; //NO DPM found in that image

				cv::Mat img = cv::imread(img_path_pos->first); //Load the image
				for (auto &roi_detect : img_path_pos->second) {
					std::vector<float> hog_descriptor;

					cv::Mat sub_img = paddedROI(img, roi_detect.rect); //Get the subimage corresponding to the ROI
					cv::resize(sub_img, sub_img, hog.winSize); //Resize to 64x64 (Constant Size)
					hog.compute(sub_img, hog_descriptor); //Compute the descriptor of the ROI
					hog_descriptor.push_back(roi_detect.score); //And add the score of the DPM detection to the vector --> DescriptorSize = hog.descriptorSize()+1

					//Construct the string descriptor of the ROI
					std::string rect_str = std::to_string(roi_detect.rect.x) + ";" + std::to_string(roi_detect.rect.y)
						+ ";" + std::to_string(roi_detect.rect.width) + ";" + std::to_string(roi_detect.rect.height);

					std::string task_roi_ref = OR_label + "_" + std::to_string(index) + "&" + rect_str;

					//And add the entry to the structures
					img_path.second[task_roi_ref] = size_counter;
					test_firstLyr_OR_SC_2[index].features.push_back(hog_descriptor);
					test_firstLyr_OR_SC_2[index].labels.push_back(0);
					test_firstLyr_OR_SC_2[index].localizer.push_back(img_path.first + "&" + task_roi_ref);
					size_counter++;
				}
			}
			test_firstLyr_OR_SC_2[index].features = test_firstLyr_OR_SC_2[index].features.reshape(0,
				(test_firstLyr_OR_SC_2[index].features.rows*test_firstLyr_OR_SC_2[index].features.cols) / descriptorSize); //Reshape feature Mat.
		}

		std::cout << "Size structure : " << test_firstLyr_SC_2.features.size() << "and counter:" << counter << std::endl;

		fs::saveLoadSet(test_OR_SC_featMap_filename, fs::MODEL::SAVE, test_firstLyr_OR_SC_2);
		fs::saveLoadSet(test_SC_1_filename, fs::MODEL::SAVE, test_SC_1, test_firstLyr_SC_2);
	}
	else {
		fs::saveLoadSet(test_OR_SC_featMap_filename, fs::MODEL::LOAD, test_firstLyr_OR_SC_2);
		fs::saveLoadSet(test_SC_1_filename, fs::MODEL::LOAD, test_SC_1, test_firstLyr_SC_2);
	}

	std::cout << "[info]\t    Object Recognition" << std::endl;
	//Object Recognition
	Task_1 test_OR_1;
	std::map<int, FeatureMap> test_firstLyr_OR_2;
	FeatureMap test_firstLyr_SC_OR_2;

	if (calculateTestSet) {

		//Load DPM Detections:
		std::map<int, std::map<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>>> dpm_detections_OR;
		fs::saveLoadDPMDetections(OR_DPM_Detections_test, fs::MODEL::LOAD, dpm_detections_OR); //Load the detected DPM in the dataset

		//Load Groundtruth files:
		std::map<int, std::map<std::string, std::vector<cv::Rect>>> groundTruth_OR; //map classID -> map filename -> vector of true ROIs in image
		fs::readGroundTruthFile(OR_GT_test, groundTruth_OR); //Load the groundtruthfile

		//Construct Dataset:
		int threadCount = dpm_detections_OR.size();
		cv::parallel_for_(cv::Range(0, threadCount), [&](const cv::Range& range) {
			for (int r = range.start; r < range.end; r++) {
				getFeatureMapfromDPMDetections(dpm_detections_OR.at(r + 1), groundTruth_OR.at(r + 1), test_OR_1, test_firstLyr_OR_2[r + 1], r + 1, 0.5);
			}
		});

		Task_1 SC_OR_aux;
		FeatureMap firstLyr_SC_OR_aux;
		fs::readGISTfromFile(SC_OR_GT_test, SC_OR_GISTDescr_test, SC_OR_aux, firstLyr_SC_OR_aux);

		int size_counter = 0;
		for (auto &img_path : test_OR_1) {

			auto img_path_pos = std::find_if(SC_OR_aux.begin(), SC_OR_aux.end(),
				[&](std::pair<std::string, std::map<std::string, int>> str) {
				return(img_path.first.find(str.first) != std::string::npos);
			}); //Find the position of the string that correspond to the filename.

			if (img_path_pos == SC_OR_aux.end()) continue; //NO entry found

			int row = img_path_pos->second.at(SC_label);
			test_firstLyr_SC_OR_2.features.push_back(firstLyr_SC_OR_aux.features.row(row).clone());
			test_firstLyr_SC_OR_2.labels.push_back(firstLyr_SC_OR_aux.labels.row(row).clone());
			test_firstLyr_SC_OR_2.localizer.push_back(img_path.first + "&" + SC_label);

			img_path.second[SC_label] = size_counter;
			size_counter++;
		}

		std::cout << "counter of SC_OR" << size_counter << " and size" << test_firstLyr_SC_OR_2.features.size() << std::endl;

		fs::saveLoadSet(test_SC_OR_featMap_filename, fs::MODEL::SAVE, test_firstLyr_SC_OR_2);
		fs::saveLoadSet(test_OR_1_filename, fs::MODEL::SAVE, test_OR_1, test_firstLyr_OR_2);
	}
	else {
		fs::saveLoadSet(test_SC_OR_featMap_filename, fs::MODEL::LOAD, test_firstLyr_SC_OR_2);
		fs::saveLoadSet(test_OR_1_filename, fs::MODEL::LOAD, test_OR_1, test_firstLyr_OR_2);
	}

	/****************************************************************************************\
	*                   Initialization : GT Independence between layers                      *
	\****************************************************************************************/
	// A copy of the X_2 for each classifier is created for the second layer to devinculate the
	//labels and features of the 1st layer with those of the 2nd.

	std::cout << "[info]\t Initialization : Creating Independence between layers" << std::endl;

	flatMapSize = res_HeatMapSize.height * res_HeatMapSize.width;
	cols_extraFeat = firstLyr_OR_2.size() * flatMapSize + class_no_SC;

	std::cout << "[info]\t    Creating Independence between layers - Scene Classification" << std::endl;
	
	FeatureMap secLyr_SC_2;
	independizeLayers(secLyr_SC_2, firstLyr_SC_2, cols_extraFeat);

	FeatureMap test_secLyr_SC_2;
	independizeLayers(test_secLyr_SC_2, test_firstLyr_SC_2, cols_extraFeat);

	std::cout << "[info]\t    Creating Independence between layers - Object Recognition" << std::endl;

	std::map<int, FeatureMap> secLyr_OR_2;
	independizeLayers(secLyr_OR_2, firstLyr_OR_2, cols_extraFeat);

	std::map<int, FeatureMap> test_secLyr_OR_2;
	independizeLayers(test_secLyr_OR_2, test_firstLyr_OR_2, cols_extraFeat);

	//Definition of additional variables

	//SC
	secLyr_SC_cols = secLyr_SC_2.features.cols;
	_SC_offset_noOdds = firstLyr_SC_2.features.cols;
	_SC_offset = _SC_offset_noOdds + class_no_SC;

	//OR
	firstLyr_OR_featSize_cols = firstLyr_OR_2.begin()->second.features.cols;
	_OR_offset = firstLyr_OR_featSize_cols + class_no_SC;
	secLyr_OR_featSize_cols = secLyr_OR_2.begin()->second.features.cols;

	/****************************************************************************************\
	*                   Initialization : Dimension -IDs for 2nd Layer Extra-features         *
	\****************************************************************************************/
	std::cout << "[info]\t Initialization : Determine Heatmap positions in the second layer feature vector" << std::endl;

	std::cout << "[info]\t    Determine Heatmap positions in the second layer feature vector - Scene Categorization" << std::endl;
		
	std::map<std::string, std::map<std::string, std::vector<int>>> SC_flatCoords;
	for (auto & filename : SC_1) {
		gen_flatCoords(SC_flatCoords[filename.first], SC_aux + filename.first, filename.second, _SC_offset);
	}

	std::map<std::string, std::map<std::string, std::vector<int>>> test_SC_flatCoords;
	for (auto & filename : test_SC_1) {
		gen_flatCoords(test_SC_flatCoords[filename.first], SC_aux + filename.first, filename.second, _SC_offset);
	}

	std::cout << "[info]\t    Determine Heatmap positions in the second layer feature vector - Object Recognition" << std::endl;

	std::map<std::string, std::map<std::string, std::vector<int>>> OR_flatCoords;
	for (auto & filename : OR_1) {
		gen_flatCoords(OR_flatCoords[filename.first], filename.first, filename.second, _OR_offset);
	}

	std::map<std::string, std::map<std::string, std::vector<int>>> test_OR_flatCoords;
	for (auto & filename : test_OR_1) {
		gen_flatCoords(test_OR_flatCoords[filename.first], filename.first, filename.second, _OR_offset);
	}
	//TODO - Persistence on gen_flatCoords

	/****************************************************************************************\
	*                           TRAINING				                                     *
	\****************************************************************************************/
	/*
	The training procedure is composed of two big steps, namely feed-forward and feedback steps.
	In the feed-forward, all classifiers (and feature reduction formulations) are trained (obtain
	parameters)	while in the feedback, an optimization problem is solved, which finds the best
	assignments for the first layer labels to improve performance.
	*/

	std::cout << "[info]\t Training" << std::endl;

	for (int iter = init_iter; iter <= max_iters; iter++) {

		std::cout << "[info]\t Iteration No: " << iter << std::endl;

		/****************************************************************************************\
		*                           TRAINING: Feed - Forward                                     *
		\****************************************************************************************/

		//////////////////        Training of 1st lyr Classifiers          /////////////////////////
		//M - Step: Feed-forward Step: Calculate the parameters of the classifiers, given the labels:

		cv::Mat newLabels_iprev_SC;
		std::map<int, cv::Mat> newLabels_iprev_OR_SC;

		cv::Mat newLabels_iprev_SC_OR;
		std::map<int, cv::Mat> newLabels_iprev_OR;

		if (iter != 1) {
			//SC
			cv::FileStorage fs_SC("newLabels_SC_1.json", cv::FileStorage::READ);

			fs_SC["newLabels_SC"] >> newLabels_iprev_SC;
			for (auto &val : firstLyr_OR_SC_2)
				fs_SC["newLabels_OR_SC_" + std::to_string(val.first)] >> newLabels_iprev_OR_SC[val.first];

			fs_SC.release();

			//OR
			cv::FileStorage fs_OR("newLabels_OR_1.json", cv::FileStorage::READ);

			fs_OR["newLabels_SC_OR"] >> newLabels_iprev_SC_OR;
			for (auto &val : firstLyr_OR_SC_2)
				fs_OR["newLabels_OR_" + std::to_string(val.first)] >> newLabels_iprev_OR[val.first];

			fs_OR.release();
		} //TODO send to fs namespace

		std::cout << "[info]\t M - Step: Feed-forward Step: Calculate the parameters of the classifiers, given the labels" << std::endl;
		
		std::cout << "[info]\t 1st Layer - Training : Scene Classification - SVM" << std::endl;

		cv::Ptr<cv::ml::SVM> firstLyr_SC_svm;

		if (true) { //Keep scope isolated

			auto tr_rf_start = std::chrono::system_clock::now();

			std::string SC_filename = fs::generateName(firstLyrSC_Class_filename, iter);

			if (train1stLyr) {

				firstLyr_SC_svm = cv::ml::SVM::create();
				cv::Mat trainSet_feat;
				cv::Mat trainSet_labels;

				if (iter != 1) {
					trainSet_feat = firstLyr_SC_2.features.clone();
					trainSet_labels = newLabels_iprev_SC.clone();
					cv::vconcat(trainSet_feat, firstLyr_SC_OR_2.features, trainSet_feat);
					cv::vconcat(trainSet_labels, newLabels_iprev_SC_OR, trainSet_labels);
				}
				else {
					trainSet_feat = firstLyr_SC_2.features;
					trainSet_labels = firstLyr_SC_2.labels;
				}
				ml::svmTrain(firstLyr_SC_svm, trainSet_feat, trainSet_labels, cv::ml::SVM::RBF);
				firstLyr_SC_svm->save(SC_filename);
			}
			else firstLyr_SC_svm = cv::Algorithm::load<cv::ml::SVM>(SC_filename);

			//test results
			double count = 0;

			cv::Mat testResponse_svm;
			ml::inference(firstLyr_SC_svm, testResponse_svm, test_firstLyr_SC_2.features);

			auto rf_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - tr_rf_start);
			
			ml::evaluate(testResponse_svm, count, test_firstLyr_SC_2.labels, class_no_SC);

			std::cout << "[info]\t    excecution time is " << rf_duration.count() << "ms for " << test_firstLyr_SC_2.features.rows << "samples" << std::endl;
		}

		std::cout << "[info]\t 1st Layer - Training : Object Recognition - SVM" << std::endl;

		std::map<int, cv::Ptr<cv::ml::SVM>> firstLyr_OR_svm;
		std::map<int, cv::Mat> firstLyr_OR_probs;

		for (int index = 1; index <= firstLyr_OR_2.size(); index++) {

			auto tr_rf_start = std::chrono::system_clock::now();

			std::string OR_filename = fs::generateName(firstLyrOR_Class_filename, index, "svm_it", iter);

			if (train1stLyr) {
				firstLyr_OR_svm[index] = cv::ml::SVM::create();
				cv::Mat trainSet_feat;
				cv::Mat trainSet_labels;

				if (iter != 1) {
					trainSet_feat = firstLyr_OR_2.at(index).features.clone();
					trainSet_labels = newLabels_iprev_OR.at(index).clone();
					cv::vconcat(trainSet_feat, firstLyr_OR_SC_2.at(index).features, trainSet_feat);
					cv::vconcat(trainSet_labels, newLabels_iprev_OR_SC.at(index), trainSet_labels);
				}
				else {
					trainSet_feat = firstLyr_OR_2.at(index).features;
					trainSet_labels = firstLyr_OR_2.at(index).labels;
				}

				std::cout << trainSet_labels.size() << " " << firstLyr_OR_2.at(index).labels.size() << std::endl;

				ml::svmTrainWeigthed(firstLyr_OR_svm.at(index), trainSet_feat, trainSet_labels, cv::ml::SVM::RBF);
				firstLyr_OR_svm.at(index)->save(OR_filename);
			}
			else firstLyr_OR_svm[index] = cv::Algorithm::load<cv::ml::SVM>(OR_filename);

			std::cout << "Object Recognition - Class No: " << index << std::endl;
			cv::Mat trainResponse_svm;
			ml::svmInference_RAW(firstLyr_OR_svm[index], trainResponse_svm, firstLyr_OR_probs[index], firstLyr_OR_2[index].features, index);

			double count = 0.0;

			cv::Mat testResponse_svm;
			ml::inference(firstLyr_OR_svm[index], testResponse_svm, test_firstLyr_OR_2[index].features);

			auto rf_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - tr_rf_start);

			std::cout << "[info]\t    excecution time is " << rf_duration.count() << "ms for " 
				<< test_firstLyr_OR_2[index].features.rows + firstLyr_OR_2[index].features.rows << "samples"<< std::endl;

			ml::evaluate(testResponse_svm, count, test_firstLyr_OR_2[index].labels, index);
		}

		std::cout << "[info]\t 2nd Layer : Generate 1stLayer Outputs for 2nd Layer" << std::endl;

		std::cout << "[info]\t    For 2nd Layer Scene Classification:" << std::endl;

		std::cout << "[info]\t       For 2nd Layer Scene Classification - Odd Ratios" << std::endl;

		auto SC_odd_start = std::chrono::system_clock::now();

		cv::Mat firstLyrSC_oddsPredictions;

		ml::predict_LogOdds(firstLyrSC_oddsPredictions, firstLyr_SC_svm, firstLyr_SC_2.features, class_no_SC);
		firstLyrSC_oddsPredictions.copyTo(
			secLyr_SC_2.features(cv::Rect(firstLyr_SC_2.features.cols, 0, class_no_SC, firstLyr_SC_2.features.rows)));

		cv::Mat test_firstLyrSC_oddsPredictions;
		ml::predict_LogOdds(test_firstLyrSC_oddsPredictions, firstLyr_SC_svm, test_firstLyr_SC_2.features, class_no_SC);
		test_firstLyrSC_oddsPredictions.copyTo(
			test_secLyr_SC_2.features(cv::Rect(firstLyr_SC_2.features.cols, 0, class_no_SC, test_firstLyr_SC_2.features.rows)));

		auto SC_odd_duration = std::chrono::duration_cast<std::chrono::milliseconds>
			(std::chrono::system_clock::now() - SC_odd_start);

		std::cout << "[info]\t    excecution time is " << SC_odd_duration.count() << "ms for "
			<< firstLyr_SC_2.features.rows + test_firstLyr_SC_2.features.rows << "samples" << std::endl;


		std::cout << "[info]\t       For 2nd Layer Scene Classification: Heat Maps" << std::endl;

		auto SC_heat_start = std::chrono::system_clock::now();

		std::map<int, cv::Mat> firstLyr_OR_SC_probs;
		std::map<int, cv::Mat> inference_OR_SC;

		heatmaps_SC(secLyr_SC_2, firstLyr_OR_SC_probs, inference_OR_SC, firstLyr_OR_svm, firstLyr_OR_SC_2, SC_1, SC_flatCoords);

		std::map<int, cv::Mat> test_firstLyr_OR_SC_probs;
		std::map<int, cv::Mat> test_inference_OR_SC;

		heatmaps_SC(test_secLyr_SC_2, test_firstLyr_OR_SC_probs, test_inference_OR_SC, firstLyr_OR_svm, test_firstLyr_OR_SC_2,
			test_SC_1, test_SC_flatCoords);

		auto SC_heat_duration = std::chrono::duration_cast<std::chrono::milliseconds>
			(std::chrono::system_clock::now() - SC_heat_start);
		std::cout << "[info]\t    excecution time is " << SC_heat_duration.count() << " ms" << std::endl;

		std::cout << "[info]\t    For 2nd Layer Object Recognition:" << std::endl;

		std::cout << "[info]\t       For 2nd Layer Object Recognition: Odd-Ratios" << std::endl;

		auto OR_odd_start = std::chrono::system_clock::now();

		cv::Mat firstLyrOR_oddsPredictions;
		ml::logOdds_OR(firstLyrOR_oddsPredictions, secLyr_OR_2, firstLyr_SC_svm, firstLyr_SC_OR_2, firstLyr_OR_featSize_cols, OR_1);
		
		cv::Mat test_firstLyrOR_oddsPredictions;
		ml::logOdds_OR(test_firstLyrOR_oddsPredictions, test_secLyr_OR_2, firstLyr_SC_svm, test_firstLyr_SC_OR_2,
			firstLyr_OR_featSize_cols, test_OR_1);

		auto OR_odd_duration = std::chrono::duration_cast<std::chrono::milliseconds>
			(std::chrono::system_clock::now() - OR_odd_start);

		std::cout << "[info]\t    excecution time is " << OR_odd_duration.count() << "ms for "
			<< firstLyr_SC_OR_2.features.rows + test_firstLyr_SC_OR_2.features.rows << "samples" << std::endl;

		std::cout << "[info]\t       For 2nd Layer Object Recognition: Heat Maps" << std::endl;

		auto OR_heat_start = std::chrono::system_clock::now();

		std::map<int, cv::Mat> inference_OR;
		heatmaps_OR(secLyr_OR_2, inference_OR, firstLyr_OR_svm, firstLyr_OR_2, OR_1, OR_flatCoords,
			secLyr_OR_featSize_cols, firstLyr_OR_featSize_cols);

		std::map<int, cv::Mat> test_inference_OR;
		heatmaps_OR(test_secLyr_OR_2, test_inference_OR, firstLyr_OR_svm, test_firstLyr_OR_2, test_OR_1, test_OR_flatCoords,
			secLyr_OR_featSize_cols, firstLyr_OR_featSize_cols);

		auto OR_heat_duration = std::chrono::duration_cast<std::chrono::milliseconds>
			(std::chrono::system_clock::now() - OR_heat_start);
		std::cout << "[info]\t    excecution time is " << OR_heat_duration.count() << " ms" << std::endl;

		/*
		if (true) {
			fs::saveLoadSet("secLyr_SC.json", fs::MODEL::SAVE, secLyr_SC_2);
			fs::saveLoadSet("secLyr_OR.json", fs::MODEL::SAVE, secLyr_OR_2);

			fs::saveLoadSet("test_secLyr_SC.json", fs::MODEL::SAVE, test_secLyr_SC_2);
			fs::saveLoadSet("test_secLyr_OR.json", fs::MODEL::SAVE, test_secLyr_OR_2);
		}
		*/

		assert(!(PCA_FE_CCM && LDA_FE_CCM)); //Both can't be activated simultaneously

		if (PCA_FE_CCM) {

			auto tr_rf_start = std::chrono::system_clock::now();

			std::cout << "PCA-FE-CCM Enabled" << std::endl;
			std::cout << "[info]\t SC Dataset - Generate PCA's" << std::endl;

			cv::PCA PCA_FE_SC;
			std::string SC_PCA_name = fs::generateName(PCA_FE_SC_filename, iter);

			if (trainPCA) fr::trainPCA(PCA_FE_SC, secLyr_SC_2.features, SC_PCA_name);
			else fr::loadPCA(PCA_FE_SC, SC_PCA_name);

			fr::reduceDataset(PCA_FE_SC, secLyr_SC_2.features, test_secLyr_SC_2.features,
				fr::featRed_SC_params.first, fr::featRed_SC_params.second);

			std::cout << "[info]\t OR Dataset - Generate PCA's" << std::endl;

			std::map<int, cv::PCA> PCA_EFE_OR;

			for (auto &set : secLyr_OR_2) {

				std::cout << "[info]\t    Object Class No. " << set.first << std::endl;
				std::string OR_PCA_name = fs::generateName(PCA_FE_OR_filename, set.first, "_it", iter);

				PCA_EFE_OR[set.first];

				if (trainPCA) fr::trainPCA(PCA_EFE_OR[set.first], set.second.features, OR_PCA_name);
				else fr::loadPCA(PCA_EFE_OR[set.first], OR_PCA_name);

				std::pair<fr::BASE, float>* value = &fr::featRed_OR_params.at(set.first - 1);
				fr::reduceDataset(PCA_EFE_OR[set.first], secLyr_OR_2.at(set.first).features, 
					test_secLyr_OR_2.at(set.first).features, value->first, value->second);
			}

			auto rf_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - tr_rf_start);

			std::cout << "[info]\t    excecution time is " << rf_duration.count() << "ms" << std::endl;
		}

		if (LDA_FE_CCM) {

			auto tr_rf_start = std::chrono::system_clock::now();

			std::cout << "[info]\t LDA-FE-CCM Enabled" << std::endl;
			
			std::cout << "[info]\t SC Dataset" << std::endl;

			if (true) {//Keep scopes isolated

				cv::Mat _pca_feats;
				cv::Mat _lda_feats;
				cv::Mat _pca_feats_test;
				cv::Mat _lda_feats_test;

				fr::init_datasets_LDA_FE(secLyr_SC_2, test_secLyr_SC_2, _pca_feats, _lda_feats,
					_pca_feats_test, _lda_feats_test);

				std::cout << "[info]\t   Generate PCAs" << std::endl;

				cv::PCA _pca_LDA_FE_SC;
				std::string SC_PCA_name = fs::generateName(_pca_LDA_FE_SC_filename, iter);

				if (train_LDA_EFE) fr::trainPCA(_pca_LDA_FE_SC, _pca_feats, SC_PCA_name);
				else fr::loadPCA(_pca_LDA_FE_SC, SC_PCA_name);

				std::pair<fr::BASE, float> value = fr::featRed_SC_params;
				if (value.first == fr::BASE::NO_FEATS) value.second = value.second - flatMapSize;
				fr::reduceDataset(_pca_LDA_FE_SC, _pca_feats,
					_pca_feats_test, value.first, value.second);

				std::cout << "[info]\t   Generate LDAs" << std::endl;

				cv::LDA _lda_LDA_FE_SC;
				std::string SC_LDA_name = fs::generateName(_lda_LDA_FE_SC_filename, iter);

				if (train_LDA_EFE) fr::trainLDA_SC(_lda_LDA_FE_SC, SC_LDA_name, SC_1, _lda_feats, secLyr_SC_2);
				else _lda_LDA_FE_SC.load(SC_LDA_name);

				fr::applyLDA(_lda_LDA_FE_SC, _lda_feats);
				fr::applyLDA(_lda_LDA_FE_SC, _lda_feats_test);

				std::cout << _pca_feats.type() << " " << _lda_feats.type() << std::endl;

				std::cout << "[info]\t   Combine Responses" << std::endl;

				fr::comb_outputs_LDA_FE(secLyr_SC_2, test_secLyr_SC_2, _pca_feats, _lda_feats,
					_pca_feats_test, _lda_feats_test);
			}

			std::cout << "[info]\t OR Dataset" << std::endl;

			if (true) { //keep scopes isolated

				std::map<int, cv::Mat> _pca_feats;
				std::map<int, cv::Mat> _lda_feats;
				std::map<int, cv::Mat> _pca_feats_test;
				std::map<int, cv::Mat> _lda_feats_test;

				fr::init_datasets_LDA_FE(secLyr_OR_2, test_secLyr_OR_2, _pca_feats, _lda_feats,
					_pca_feats_test, _lda_feats_test);

				std::cout << "[info]\t   Generate PCAs" << std::endl;

				std::map<int, cv::PCA> _pca_LDA_FE_OR;

				for (auto &set : _pca_feats) {
					std::string OR_PCA_name = fs::generateName(_pca_LDA_FE_OR_filename, set.first, "_it", iter);

					_pca_LDA_FE_OR[set.first];

					if (train_LDA_EFE) fr::trainPCA(_pca_LDA_FE_OR[set.first], set.second, OR_PCA_name);
					else fr::loadPCA(_pca_LDA_FE_OR[set.first], OR_PCA_name);

					std::pair<fr::BASE, float> value = fr::featRed_OR_params.at(set.first - 1);
					if (value.first == fr::BASE::NO_FEATS) value.second = value.second - flatMapSize;
					fr::reduceDataset(_pca_LDA_FE_OR[set.first], _pca_feats.at(set.first), 
						_pca_feats_test.at(set.first), value.first, value.second);
				}

				std::cout << "[info]\t   Generate LDAs" << std::endl;

				std::map<int, cv::LDA> _lda_LDA_FE_OR;

				if (train_LDA_EFE) fr::trainLDA_OR(_lda_LDA_FE_OR, _lda_LDA_FE_OR_filename, OR_1, secLyr_OR_2,
					firstLyr_OR_probs, OR_flatCoords, iter);
				else fr::loadLDA_OR(_lda_LDA_FE_OR, _lda_LDA_FE_OR_filename, iter);

				fr::applyLDA(_lda_LDA_FE_OR, _lda_feats);
				fr::applyLDA(_lda_LDA_FE_OR, _lda_feats_test);
			
				std::cout << "[info]\t   Combine Responses" << std::endl;

				fr::comb_outputs_LDA_FE(secLyr_OR_2, test_secLyr_OR_2, _pca_feats, _lda_feats,
					_pca_feats_test, _lda_feats_test);
			}

			auto rf_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - tr_rf_start);

			std::cout << "[info]\t    excecution time is " << rf_duration.count() << "ms" << std::endl;
		}

		std::cout << "[info]\t 2nd Layer - Training : Scene Classification - RF" << std::endl;

		cv::Ptr<cv::ml::RTrees> secLyr_SC_RF;

		if (true) { //Keep scope isolated

			auto tr_rf_start = std::chrono::system_clock::now();

			std::string SC_filename;
			if(PCA_FE_CCM) SC_filename = fs::generateName(secLyrSC_Class_filename + "_PCA_EFE_", iter);
			else if(LDA_FE_CCM) SC_filename = fs::generateName(secLyrSC_Class_filename + "_LDA_EFE_", iter);
			else  SC_filename = fs::generateName(secLyrSC_Class_filename, iter);

			if (train2ndLyr) {
				secLyr_SC_RF = cv::ml::RTrees::create();
				ml::RFTrain(secLyr_SC_RF, secLyr_SC_2.features, secLyr_SC_2.labels);

				secLyr_SC_RF->save(SC_filename);
			}
			else secLyr_SC_RF = cv::Algorithm::load<cv::ml::RTrees>(SC_filename);

			double count = 0.0;

			cv::Mat testResponse_RF;
			ml::inference(secLyr_SC_RF, testResponse_RF, test_secLyr_SC_2.features);
			
			auto rf_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - tr_rf_start);

			std::cout << "[info]\t    excecution time is " << rf_duration.count() << "ms for " << test_secLyr_SC_2.features.rows << "samples" << std::endl;

			ml::evaluate(testResponse_RF, count, test_secLyr_SC_2.labels, class_no_SC);
		}

		std::cout << "[info]\t 2nd Layer - Training : Object Recognition - RF" << std::endl;

		std::map<int, cv::Ptr<cv::ml::RTrees>> secLyr_OR_RF;

		for (int index = 1; index <= dpmDetector_filenames.size(); index++) {

			auto tr_rf_start = std::chrono::system_clock::now();
			std::cout << "Object Recognition - Class No: " << index << std::endl;

			std::string OR_filename;
			if(PCA_FE_CCM) OR_filename = fs::generateName(secLyrOR_Class_filename + "_PCA_EFE_", index, "RF_it", iter);
			else if(LDA_FE_CCM) OR_filename = fs::generateName(secLyrOR_Class_filename + "_LDA_EFE_", index, "RF_it", iter);
			else OR_filename = fs::generateName(secLyrOR_Class_filename, index, "RF_it", iter);

			if (train2ndLyr) {

				secLyr_OR_RF[index] = cv::ml::RTrees::create();
				ml::RFTrain(secLyr_OR_RF[index], secLyr_OR_2.at(index).features, secLyr_OR_2.at(index).labels);

				secLyr_OR_RF[index]->save(OR_filename);
			}
			else secLyr_OR_RF[index] = cv::Algorithm::load<cv::ml::RTrees>(OR_filename);

			double count = 0.0;

			cv::Mat testResponse_RF;
			ml::inference(secLyr_OR_RF[index], testResponse_RF, test_secLyr_OR_2[index].features);

			auto rf_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - tr_rf_start);

			std::cout << "[info]\t    excecution time is " << rf_duration.count() << "ms for "
				<< test_secLyr_OR_2[index].features.rows << "samples" << std::endl;

			ml::evaluate(testResponse_RF, count, test_secLyr_OR_2[index].labels, index);
		}


		std::string exampleImg = "000576.png";

		for (auto & img : test_OR_1) {

			if (img.first.find(exampleImg) == std::string::npos) continue;

			for (auto & lbl : img.second) {
				int row = lbl.second;

				cv::Mat prediction;

				if (lbl.first == SC_label) {
					firstLyr_SC_svm->predict(test_firstLyr_SC_OR_2.features.row(row), prediction);
				}
				else {

					std::vector<std::string> split_ref;
					split(lbl.first, '&', split_ref); //split the class_id using & --> tasklabel_"ID" , ROI
					int class_no = std::stoi(split_ref.at(0).substr(split_ref.at(0).find('_') + 1)); //get the class id

					secLyr_OR_RF[class_no]->predict(test_secLyr_OR_2[class_no].features.row(lbl.second),prediction);
				}

				std::cout << lbl.first << " " << prediction << std::endl;
			}
		}
		
		if (CCM) break; //Up to here, the model corresponds to the CCM structure

		std::cout << "[info]\t E - Step: Feedback Step: Claculate the best possible labels, given the parameters" << std::endl;
		std::cout << "[info]\t argmax_{Y^{1,gt}} ( sum(i=1,n)(fun_inference_1_{i}) + fun_inference_2_{j} )" << std::endl;

		std::cout << "[info]\t E - Step: Scene Classification" << std::endl;

		cv::Mat newLabels_SC = cv::Mat::zeros(firstLyr_SC_2.labels.size(), CV_32S);

		std::map<int, cv::Mat> newLabels_OR_SC;
		for (auto &lbl : firstLyr_OR_SC_2)
			newLabels_OR_SC[lbl.first] = cv::Mat::zeros(lbl.second.labels.size(), CV_32S);

		for (auto & img_path : SC_1) {

			// ---------------- Calculate Probabilities and select the best:

			auto opti_start = std::chrono::system_clock::now();

			//Get probabilities from 1st layer

			//SC
			cv::Mat probabilities_SC;
			firstLyrSC_oddsPredictions.row(img_path.second.at(SC_label)).copyTo(probabilities_SC);

			//OR
			auto labels = img_path.second;
			labels.erase(SC_label);
			cv::Mat_<float> probabilities_OR;

			for (auto &label : labels) {
				std::vector<std::string> split_ref;
				split(label.first, '&', split_ref); //split the class_id using & --> tasklabel_"ID" , ROI
				int class_no = std::stoi(split_ref.at(0).substr(split_ref.at(0).find('_') + 1));
				probabilities_OR.push_back(firstLyr_OR_SC_probs.at(class_no).at<float>(label.second));
			}

			std::cout << "[info]\t Probabilities : " << std::endl;
			if (labels.size() != 0) {
				std::cout << "\t\tOR : [";
				for (auto &val : probabilities_OR) std::cout << val << " ";
				std::cout << "]" << std::endl;
			}
			std::cout << "\t\tSC : " << probabilities_SC << std::endl;

			//get Original Feature Vector and reset all heatMaps and oddRatio values
			cv::Mat feat_vector = secLyr_SC_2.features.row(img_path.second.at(SC_label)).clone();
			int right_label = secLyr_SC_2.labels.at<int>(img_path.second.at(SC_label));

			feat_vector(cv::Rect(_SC_offset_noOdds, 0, secLyr_SC_cols - _SC_offset_noOdds, 1)) = 0.0;

			std::cout << "Solving for " << img_path.first << std::endl;

			int nVars = img_path.second.size() + (class_no_SC - 1);

			opt::ProblemMINLP_SC* problem = new opt::ProblemMINLP_SC(nVars, 1,
				probabilities_SC, probabilities_OR, feat_vector, secLyr_SC_RF, SC_flatCoords.at(img_path.first),
				labels, right_label);

			knitro::KTRSolver solver(problem, KTR_GRADOPT_CENTRAL, KTR_HESSOPT_BFGS);

			solver.setParam(KTR_PARAM_MSENABLE, 1);
			solver.setParam(KTR_PARAM_MSMAXSOLVES, 64);
			solver.setParam("ms_savetol", 0.001);
			solver.setParam("par_numthreads", 8);
			solver.setParam("outlev", 5);
			solver.setParam("mip_relaxable", 0);
			solver.setParam("mip_method", 3);

			int solveStatus = solver.solve();

			if (solveStatus != 0) {
				std::cout << std::endl;
				std::cout << "Knitro failed to solve the problem, final status = ";
				std::cout << solveStatus << std::endl;
			}
			else {
				std::cout << std::endl << "Knitro successful, objective is = ";
				std::cout << solver.getObjValue() << std::endl;
				std::cout << "The solution is : ";
				for (auto &val : solver.getXValues()) std::cout << val << " ";
			} std::cout << std::endl;

			auto opti_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - opti_start);
			std::cout << "[info]\t Total optimization time: " << opti_duration.count() << "ms" << std::endl;

			//set the labels in the newLabels;
			std::vector<double> result = solver.getXValues();
			std::map<std::string, int>::iterator label_it = labels.begin();
			int j = 0;
			while (j < nVars) {

				if (j < class_no_SC) {
					if (result[j] == 1.0) {
						newLabels_SC.at<int>(img_path.second.at(SC_label)) = j + 1;
						j = (class_no_SC - 1);
					}
				}
				else {
					if (result[j] == 1.0) {

						std::vector<std::string> split_ref;
						split(label_it->first, '&', split_ref); //split the class_id using & --> tasklabel_"ID" , ROI
						int class_no = std::stoi(split_ref.at(0).substr(split_ref.at(0).find('_') + 1)); //get the class id

						newLabels_OR_SC.at(class_no).at<int>(label_it->second) = class_no;

					}
					label_it++;
				}
				j++;
			}
			delete problem;
		}

		//Save the obtained new_labels
		{
		cv::FileStorage fs("newLabels_SC_" + std::to_string(iter) + ".json", cv::FileStorage::WRITE);
		fs << "newLabels_SC" << newLabels_SC;
		for (auto &lblset : newLabels_OR_SC) {
			fs << "newLabels_OR_SC_" + std::to_string(lblset.first) << lblset.second;
		}
		fs.release();
		}

		std::cout << "[info]\t E - Step: Object Recognition" << std::endl;

		cv::Mat newLabels_SC_OR = cv::Mat::zeros(firstLyr_SC_OR_2.labels.size(), CV_32S);

		std::map<int, cv::Mat> newLabels_OR;
		for (auto &lbl : firstLyr_OR_2)
			newLabels_OR[lbl.first] = cv::Mat::zeros(lbl.second.labels.size(), CV_32S);

		for (auto & img_path : OR_1) {

			auto opti_start = std::chrono::system_clock::now();

			cv::Mat probabilities_SC;
			firstLyrOR_oddsPredictions.row(img_path.second.at(SC_label)).copyTo(probabilities_SC);

			//OR
			auto labels = img_path.second;
			labels.erase(SC_label);
			cv::Mat_<float> probabilities_OR;

			std::vector<std::pair<int, int>> feat_ids;

			std::map<int, FeatureMap> features_i; //FeatureMap that contains the appropiate labels and features
												  //of all the variables in this iteration

			for (auto &label : labels) {
				std::vector<std::string> split_ref;
				split(label.first, '&', split_ref); //split the class_id using & --> tasklabel_"ID" , ROI
				int class_no = std::stoi(split_ref.at(0).substr(split_ref.at(0).find('_') + 1));

				probabilities_OR.push_back(firstLyr_OR_probs.at(class_no).at<float>(label.second));

				//create structure of label_class and position of the sample.
				feat_ids.push_back(std::make_pair(class_no, label.second));

				//elements from the 2nd Layer
				features_i[class_no].features.push_back(secLyr_OR_2.at(class_no).features.row(label.second).clone());
				features_i[class_no].labels.push_back(secLyr_OR_2.at(class_no).labels.at<int>(label.second));
				features_i[class_no].localizer.push_back(std::to_string(label.second));
			}

			//Now reset the output of the 1st layer to 0 in the feature_layers of the 2nd lyr.
			for (auto &mat : features_i)
				mat.second.features(cv::Rect(firstLyr_OR_featSize_cols, 0, mat.second.features.cols - firstLyr_OR_featSize_cols, mat.second.features.rows)) = 0.0;

			std::cout << "[info]\t Probabilities : " << std::endl;
			if (labels.size() != 0) {
				std::cout << "\t\tOR : [";
				for (auto &val : probabilities_OR) std::cout << val << " ";
				std::cout << "]" << std::endl;
			}
			std::cout << "\t\tSC : " << probabilities_SC << std::endl;

			//Now we need the elements of the 2nd layer.
			std::vector<cv::Mat> feat_vectors;
			std::vector<int> right_labels;
			for (auto & id : feat_ids) {
				feat_vectors.push_back(secLyr_OR_2.at(id.first).features.row(id.second).clone());
				right_labels.push_back(secLyr_OR_2.at(id.first).labels.at<int>(id.second));
			}
			for (auto &mat : feat_vectors)
				mat(cv::Rect(firstLyr_OR_featSize_cols, 0, mat.cols - firstLyr_OR_featSize_cols, 1)) = 0.0;

			std::cout << "Solving for " << img_path.first << std::endl;

			int nVars = img_path.second.size() + (class_no_SC - 1);

			opt::ProblemMINLP_OR* problem = new opt::ProblemMINLP_OR(nVars, 1,
				probabilities_SC, probabilities_OR, feat_vectors, secLyr_OR_RF, OR_flatCoords.at(img_path.first),
				labels, right_labels, feat_ids, features_i);

			knitro::KTRSolver solver(problem, KTR_GRADOPT_CENTRAL, KTR_HESSOPT_BFGS);

			solver.setParam(KTR_PARAM_MSENABLE, 1);
			solver.setParam(KTR_PARAM_MSMAXSOLVES, 8);
			solver.setParam("ms_savetol", 0.001);
			solver.setParam("par_numthreads", 8);
			solver.setParam("outlev", 5);
			solver.setParam("mip_relaxable", 0);
			solver.setParam("mip_method", 3);

			int solveStatus = solver.solve();

			if (solveStatus != 0) {
				std::cout << std::endl;
				std::cout << "Knitro failed to solve the problem, final status = ";
				std::cout << solveStatus << std::endl;
			}
			else {
				std::cout << std::endl << "Knitro successful, objective is = ";
				std::cout << solver.getObjValue() << std::endl;
				std::cout << "The solution is : ";
				for (auto &val : solver.getXValues()) std::cout << val << " ";
			} std::cout << std::endl;

			auto opti_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - opti_start);
			std::cout << "[info]\t Total optimization time: " << opti_duration.count() << "ms" << std::endl;

			std::vector<double> result = solver.getXValues();
			std::vector<std::pair<int, int>>::iterator label_it = feat_ids.begin();

			int j = 0;
			while (j < nVars) {

				if (j < class_no_SC) {
					if (result[j] == 1.0) {
						newLabels_SC_OR.at<int>(img_path.second.at(SC_label)) = j + 1;
						j = (class_no_SC - 1);
					}
				}
				else {
					if (result[j] == 1.0) {
						newLabels_OR.at(label_it->first).at<int>(label_it->second) = label_it->first;
					}
					label_it++;
				}
				j++;
			}
			delete problem;
		}

		//Save the obtained new_labels
		{
			cv::FileStorage fs("newLabels_OR_" + std::to_string(iter) + ".json", cv::FileStorage::WRITE);
			fs << "newLabels_SC_OR" << newLabels_SC_OR;
			for (auto &lblset : newLabels_OR) {
				fs << "newLabels_OR_" + std::to_string(lblset.first) << lblset.second;
			}
			fs.release();
		}

		std::cout << "[info]\t End of Iteration" << std::endl;
	}
	system("pause");
	return 0;
}