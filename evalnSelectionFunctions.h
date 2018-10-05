#pragma once

#ifndef EVAL_SELECT_H
#define EVAL_SELECT_H

#include "globalVarsnDefs.h" //Contains general required includes and definitions
#include "fileStorage.h" //Contains all functions related with storage and information consistency
#include "structConstruct.h" //Contains all functions with which we construct our data structures
#include "ml.h" //Contains all functions related with machine learning

void select_RF_SVM_LC(){
	//This function evaluates the second classification layer with RF, SVM and LC for proper selection

	//function variables
	int iter = 1;

	FeatureMap secLyr_SC_2;
	FeatureMap test_secLyr_SC_2;

	std::map<int, FeatureMap> secLyr_OR_2;
	std::map<int, FeatureMap> test_secLyr_OR_2;

	//Load Datasets -- NOTE: Save the datasets before entering the second layer!
	fs::saveLoadSet("secLyr_SC.json", fs::MODEL::LOAD, secLyr_SC_2);
	fs::saveLoadSet("secLyr_OR.json", fs::MODEL::LOAD, secLyr_OR_2);

	fs::saveLoadSet("test_secLyr_SC.json", fs::MODEL::LOAD, test_secLyr_SC_2);
	fs::saveLoadSet("test_secLyr_OR.json", fs::MODEL::LOAD, test_secLyr_OR_2);
	
	std::cout << "[info]\t Random Forest" << std::endl;

	cv::Ptr<cv::ml::RTrees> secLyr_SC_RF;
	std::map<int, cv::Ptr<cv::ml::RTrees>> secLyr_OR_RF;

	//SC
	if (true) { //Keep scope isolated
		std::string SC_filename = fs::generateName(secLyrSC_Class_filename, iter);

		if (trainSelectRFSVMLC) {
			secLyr_SC_RF = cv::ml::RTrees::create();
			ml::RFTrain(secLyr_SC_RF, secLyr_SC_2.features, secLyr_SC_2.labels);

			secLyr_SC_RF->save(SC_filename);
		}
		else secLyr_SC_RF = cv::Algorithm::load<cv::ml::RTrees>(SC_filename);

		double count = 0.0;

		cv::Mat testResponse_RF;
		cv::Mat probResponse_RF;
		ml::inference(secLyr_SC_RF, testResponse_RF, test_secLyr_SC_2.features);
		ml::evaluate(testResponse_RF, count, test_secLyr_SC_2.labels, class_no_SC);
	}

	//OR
	for (int index = 1; index <= dpmDetector_filenames.size(); index++) {

		std::cout << "Object Recognition - Class No: " << index << std::endl;
		std::string OR_filename = fs::generateName(secLyrOR_Class_filename, index, "RF_it", iter);

		if (trainSelectRFSVMLC) {

			secLyr_OR_RF[index] = cv::ml::RTrees::create();
			ml::RFTrain(secLyr_OR_RF.at(index), secLyr_OR_2.at(index).features, secLyr_OR_2.at(index).labels);

			secLyr_OR_RF[index]->save(OR_filename);
		}
		else secLyr_OR_RF[index] = cv::Algorithm::load<cv::ml::RTrees>(OR_filename);

		double count = 0.0;

		cv::Mat testResponse_RF;
		ml::inference(secLyr_OR_RF[index], testResponse_RF, test_secLyr_OR_2[index].features);
		ml::evaluate(testResponse_RF, count, test_secLyr_OR_2[index].labels, index);
	}

	std::cout << "[info]\t Support Vector Machines" << std::endl;

	cv::Ptr<cv::ml::SVM> secLyr_SC_svm;
	std::map<int, cv::Ptr<cv::ml::SVM>> secLyr_OR_svm;

	//SC
	if (true) { //Keep scope isolated
		std::string SC_filename = fs::generateName(secLyrSC_SVM_filename, iter);

		if (trainSelectRFSVMLC) {
			secLyr_SC_svm = cv::ml::SVM::create();
			ml::svmTrain(secLyr_SC_svm, secLyr_SC_2.features, secLyr_SC_2.labels, cv::ml::SVM::RBF);
			secLyr_SC_svm->save(SC_filename);
		}
		else secLyr_SC_svm = cv::Algorithm::load<cv::ml::SVM>(SC_filename);

		double count = 0.0;

		cv::Mat testResponse_svm;
		ml::inference(secLyr_SC_svm, testResponse_svm, test_secLyr_SC_2.features);
		ml::evaluate(testResponse_svm, count, test_secLyr_SC_2.labels, class_no_SC);
	}

	//OR
	for (int index = 1; index <= dpmDetector_filenames.size(); index++) {

		std::cout << "Object Recognition - Class No: " << index << std::endl;
		std::string OR_filename = fs::generateName(secLyrOR_SVM_filename, index, "svm_it", iter);

		if (trainSelectRFSVMLC) {

			secLyr_OR_svm[index] = cv::ml::SVM::create();
			ml::svmTrainWeigthed(secLyr_OR_svm.at(index), secLyr_OR_2.at(index).features, secLyr_OR_2.at(index).labels, cv::ml::SVM::RBF);
			
			secLyr_OR_svm.at(index)->save(OR_filename);
		}
		else secLyr_OR_svm[index] = cv::Algorithm::load<cv::ml::SVM>(OR_filename);

		double count = 0.0;

		cv::Mat testResponse_svm;
		ml::inference(secLyr_OR_svm[index], testResponse_svm, test_secLyr_OR_2[index].features);
		ml::evaluate(testResponse_svm, count, test_secLyr_OR_2[index].labels, index);
	}

	std::cout << "[info]\t Logistic Classifier" << std::endl;

	cv::Ptr<cv::ml::LogisticRegression> secLyr_SC_LC;
	std::map<int, cv::Ptr<cv::ml::LogisticRegression>> secLyr_OR_LC;

	//SC
	if (true) { //Keep scope isolated
		std::string SC_filename = fs::generateName(secLyrSC_LC_filename, iter);

		if (true) {
			secLyr_SC_LC = cv::ml::LogisticRegression::create();
			ml::LCTrain(secLyr_SC_LC, secLyr_SC_2.features, secLyr_SC_2.labels);

			secLyr_SC_LC->save(SC_filename);
		}
		else secLyr_SC_LC = cv::Algorithm::load<cv::ml::LogisticRegression>(SC_filename);

		double count = 0.0;

		cv::Mat testResponse_LC;
		ml::inference(secLyr_SC_LC, testResponse_LC, test_secLyr_SC_2.features);
		ml::evaluate(testResponse_LC, count, test_secLyr_SC_2.labels, class_no_SC);
	}
	
	//OR
	for (int index = 1; index <= dpmDetector_filenames.size(); index++) {

		std::cout << "Object Recognition - Class No: " << index << std::endl;
		std::string OR_filename = fs::generateName(secLyrOR_LC_filename, index, "LC_it", iter);

		if (true) {

			secLyr_OR_LC[index] = cv::ml::LogisticRegression::create();
			ml::LCTrain(secLyr_OR_LC.at(index), secLyr_OR_2.at(index).features, secLyr_OR_2.at(index).labels);

			secLyr_OR_LC.at(index)->save(OR_filename);
		}
		else secLyr_OR_LC[index] = cv::Algorithm::load<cv::ml::LogisticRegression>(OR_filename);

		double count = 0.0;

		cv::Mat testResponse_LC;
		ml::inference(secLyr_OR_LC[index], testResponse_LC, test_secLyr_OR_2[index].features);
		ml::evaluate(testResponse_LC, count, test_secLyr_OR_2[index].labels, index);
	}
}



#endif // !EVAL_SELECT_H

