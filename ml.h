#pragma once

#ifndef ML_H
#define ML_H

#include "globalVarsnDefs.h"
#include "generalPurpose.h"

namespace ml {

	enum TRAIN_SUBSET {
		K_FOLD,
		TRAIN_SAMPLES,
		TRAIN_PERCENTAGE,
	};

	void divideTrainTestSet(FeatureMap& train, FeatureMap&test, const FeatureMap& src, TRAIN_SUBSET train_subset, int folds_or_samples) {

		std::map<int, std::vector<int>> sampleDistribution; //retrieve the distribution of the samples per label
		for (size_t i = 0; i < src.labels.rows; i++) sampleDistribution[src.labels.at<int>(i)].push_back(i);

		for (auto &label : sampleDistribution) std::cout << label.first << " " << label.second.size() << std::endl;
		std::cout << "samplDistr Size: " << sampleDistribution.size() << std::endl;

		std::map<int, std::vector<int>> folds;
		std::map<int, std::vector<int>> sampleDistribution_aux = sampleDistribution;
		int n_folds = 0;

		if (train_subset == TRAIN_SUBSET::K_FOLD) {
			n_folds = folds_or_samples;
			for (int i = 1; i < n_folds; i++) { //last fold is build on the remaining elements
				for (auto &element : sampleDistribution_aux) {
					int fold_elementNumber = (int)sampleDistribution[element.first].size() / n_folds;
					for (size_t j = 0; j < fold_elementNumber; j++) {
						int index = rand() % element.second.size();
						folds[i].push_back(element.second.at(index));
						element.second.erase(element.second.begin() + index);
					}
				}
			}
		}

		if (train_subset == TRAIN_SUBSET::TRAIN_SAMPLES) {
			n_folds = 2;
			for (int i = 1; i < n_folds; i++) { //last fold is build on the remaining elements
				for (auto &element : sampleDistribution_aux) {
					int fold_elementNumber = folds_or_samples;
					for (size_t j = 0; j < fold_elementNumber; j++) {
						int index = rand() % element.second.size();
						folds[i].push_back(element.second.at(index));
						element.second.erase(element.second.begin() + index);
					}
				}
			}
		}

		assert(n_folds != 0);
		for (auto& element : sampleDistribution_aux) folds[n_folds].insert(folds[n_folds].end(), element.second.begin(), element.second.end());
		for (auto &fold : folds) std::cout << fold.first << " " << fold.second.size() << std::endl;

		int test_fold = n_folds;

		for (size_t row = 0; row < src.labels.rows; row++) {
			if (std::find(folds[test_fold].begin(), folds[test_fold].end(), row) != folds[test_fold].end()) { //Found

				test.labels.push_back(src.labels.row(row).clone());
				test.features.push_back(src.features.row(row).clone());

				//std::string localizer = src.localizer.at(row);
				//test.localizer.push_back(localizer);
			}
			else { //Not found
				train.labels.push_back(src.labels.row(row).clone());
				train.features.push_back(src.features.row(row).clone());

				//std::string localizer = src.localizer.at(row);
				//train.localizer.push_back(localizer);
			}
		}
	}

	//Divides the dataset into a train and test subdatasets.
	void divideTrainTestSet(Task_1& task_1_train, FeatureMap& task_2_train, Task_1& task_1_test, FeatureMap& task_2_test,
		const Task_1& task_1, const FeatureMap& task_2, TRAIN_SUBSET train_subset, int folds_or_samples) {

		std::map<int, std::vector<int>> sampleDistribution; //retrieve the distribution of the samples per label
		for (size_t i = 0; i < task_2.labels.rows; i++) sampleDistribution[task_2.labels.at<int>(i)].push_back(i);

		for (auto &label : sampleDistribution) std::cout << label.first << " " << label.second.size() << std::endl;
		std::cout << "samplDistr Size: " << sampleDistribution.size() << std::endl;

		std::map<int, std::vector<int>> folds;
		std::map<int, std::vector<int>> sampleDistribution_aux = sampleDistribution;
		int n_folds = 0;

		if (train_subset == TRAIN_SUBSET::K_FOLD) {
			n_folds = folds_or_samples;
			for (int i = 1; i < n_folds; i++) { //last fold is build on the remaining elements
				for (auto &element : sampleDistribution_aux) {
					int fold_elementNumber = (int)sampleDistribution[element.first].size() / n_folds;
					for (size_t j = 0; j < fold_elementNumber; j++) {
						int index = rand() % element.second.size();
						folds[i].push_back(element.second.at(index));
						element.second.erase(element.second.begin() + index);
					}
				}
			}
		}

		if (train_subset == TRAIN_SUBSET::TRAIN_SAMPLES) {
			n_folds = 2;
			for (int i = 1; i < n_folds; i++) { //last fold is build on the remaining elements
				for (auto &element : sampleDistribution_aux) {
					int fold_elementNumber = folds_or_samples;
					for (size_t j = 0; j < fold_elementNumber; j++) {
						int index = rand() % element.second.size();
						folds[i].push_back(element.second.at(index));
						element.second.erase(element.second.begin() + index);
					}
				}
			}
		}

		assert(n_folds != 0);
		for (auto& element : sampleDistribution_aux) folds[n_folds].insert(folds[n_folds].end(), element.second.begin(), element.second.end());
		for (auto &fold : folds) std::cout << fold.first << " " << fold.second.size() << std::endl;

		//std::string task_label = //task_1.begin()->second.begin()->first;
		int test_fold = n_folds;

		for (size_t row = 0; row < task_2.labels.rows; row++) {
			if (std::find(folds[test_fold].begin(), folds[test_fold].end(), row) != folds[test_fold].end()) { //Found

				task_2_test.labels.push_back(task_2.labels.row(row).clone());
				task_2_test.features.push_back(task_2.features.row(row).clone());

				std::string localizer = task_2.localizer.at(row);
				task_2_test.localizer.push_back(localizer);

				auto position = localizer.find('&'); //Detects the first & corresponding to the division filename --> tasklabel
				task_1_test[localizer.substr(0, position)][localizer.substr(position + 1)] = task_2_test.features.rows - 1;//row;
			}
			else { //Not found
				task_2_train.labels.push_back(task_2.labels.row(row).clone());
				task_2_train.features.push_back(task_2.features.row(row).clone());

				std::string localizer = task_2.localizer.at(row);
				task_2_train.localizer.push_back(localizer);

				auto position = localizer.find('&'); //Detects the first & corresponding to the division filename --> tasklabel
				task_1_train[localizer.substr(0, position)][localizer.substr(position + 1)] = task_2_train.features.rows - 1;// row;
			}
		}
	}

	//Divides the dataset into a train and test subdatasets. 
	void divideTrainTestSet(Task_1& task_1_train, std::map<int, FeatureMap>& task_2_train,
		Task_1& task_1_test, std::map<int, FeatureMap>& task_2_test,
		const Task_1& task_1, const std::map<int, FeatureMap>& task_2, TRAIN_SUBSET train_subset, int folds_or_samples) {

		for (auto & index_it : task_2) {
			int index = index_it.first;
			divideTrainTestSet(task_1_train, task_2_train[index], task_1_test, task_2_test[index],
				task_1, task_2.at(index), train_subset, folds_or_samples);
		}
	}

	void svmTrain(cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& trainMat, const cv::Mat& trainLabels, cv::ml::SVM::KernelTypes kernel_type) {

		//set-up
		svm->setKernel(kernel_type);
		svm->setType(cv::ml::SVM::C_SVC);
		cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(trainMat, cv::ml::SampleTypes::ROW_SAMPLE,
			trainLabels);

		//train
		svm->trainAuto(tData);
		//printSVMParam(svm);
	}

	void svmTrainWeigthed(cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& trainMat, const cv::Mat& trainLabels, cv::ml::SVM::KernelTypes kernel_type) {

		//set-up
		svm->setKernel(kernel_type);
		svm->setType(cv::ml::SVM::C_SVC);
		cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(trainMat, cv::ml::SampleTypes::ROW_SAMPLE,
			trainLabels);

		//getWeigths
		int nonZeros = cv::countNonZero(trainLabels);
		float weight_neg = (float)(nonZeros / (double)trainLabels.rows);
		float weight_pos = 1.0f - weight_neg;

		cv::Mat weights;
		weights.push_back(weight_neg);
		weights.push_back(weight_pos);

		svm->setClassWeights(weights);

		std::cout << "Weights" << svm->getClassWeights() << std::endl;

		//train
		svm->trainAuto(tData, 10, svm->getDefaultGrid(cv::ml::SVM::C), svm->getDefaultGrid(cv::ml::SVM::GAMMA), svm->getDefaultGrid(cv::ml::SVM::P),
			svm->getDefaultGrid(cv::ml::SVM::NU), svm->getDefaultGrid(cv::ml::SVM::COEF), svm->getDefaultGrid(cv::ml::SVM::DEGREE), true);
		//printSVMParam(svm);
	}

	void RFTrain(cv::Ptr<cv::ml::RTrees>& rf, cv::Mat& trainMat, cv::Mat& trainLabels) {

		cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(trainMat,
			cv::ml::SampleTypes::ROW_SAMPLE, trainLabels);

		rf->setActiveVarCount((int)(trainMat.cols*0.05) + 1);
		rf->setMaxDepth(std::numeric_limits<int>::max());
		rf->setMinSampleCount(1);

		rf->setCalculateVarImportance(true);

		cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 512, 0.001f);
		rf->setTermCriteria(termCriteria);

		rf->train(tData);
	}

	void LCTrain(cv::Ptr<cv::ml::LogisticRegression>& lc, cv::Mat& trainMat, cv::Mat& trainLabels) {

		cv::Mat labels_float = trainLabels.clone();
		labels_float.convertTo(labels_float, CV_32FC1);

		//cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(trainMat,
		//	cv::ml::SampleTypes::ROW_SAMPLE, trainLabels);

		lc->setLearningRate(0.0001);
		lc->setIterations(1000);
		lc->setRegularization(cv::ml::LogisticRegression::REG_L2);
		lc->setTrainMethod(cv::ml::LogisticRegression::BATCH);
		lc->setMiniBatchSize(1);

		lc->train(trainMat, cv::ml::ROW_SAMPLE, labels_float);		
	}

	template<typename T>
	void inference(cv::Ptr<T>& classifier, cv::Mat& testResponse, const cv::Mat& testMat) {
		//inference function for cv::ml classifiers.
		classifier->predict(testMat, testResponse);
	}

	void svmInference_RAW(cv::Ptr<cv::ml::SVM>& svm, cv::Mat& testResponse, cv::Mat& prob_response, const cv::Mat& testMat, const int trueLbl) {

		svm->predict(testMat, prob_response, cv::ml::StatModel::RAW_OUTPUT);

		prob_response = (-1)*prob_response;

		testResponse.create(prob_response.size(), prob_response.type());
		for (int i = 0; i < prob_response.rows; i++) {
			testResponse.at<float>(i) = (prob_response.at<float>(i) >= 0) ? (float)trueLbl : (float)0;
		}

		prob_response = sigmoid(prob_response);
	}

	void RFInference_RAW(cv::Ptr<cv::ml::RTrees>& rf, cv::Mat& prob_response, const cv::Mat& testMat) {

		rf->getVotes(testMat, prob_response, 0);

		prob_response = prob_response(cv::Rect(0, 1, prob_response.cols, prob_response.rows - 1));
		prob_response.convertTo(prob_response, CV_32FC1);
		prob_response = (norm_factor) * prob_response;
	}

	void evaluate(const cv::Mat& testResponse, double& count, const cv::Mat& testLabels, int max_lbl) {

		cv::Mat matrix = cv::Mat::zeros(max_lbl + 1, max_lbl + 1, CV_32FC1);
		for (int i = 0; i < testResponse.rows; i++) {

			matrix.at<float>(testLabels.at<int>(i), testResponse.at<float>(i))++;

			if (testResponse.at<float>(i) == testLabels.at<int>(i)) {
				count++;
			}
		}
		std::cout << count / testResponse.rows << std::endl;
		std::cout << matrix << std::endl;
	}

	void predict_LogOdds(cv::Mat&oddsPrediction, cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& testFeatures, int N_class) {

		cv::Mat sv = svm->getSupportVectors();
		int sv_total = sv.rows;
		sv.convertTo(sv, 6);

		int i, j, dfi, k;

		for (int row = 0; row < testFeatures.rows; row++) {

			cv::Mat buffer(1, sv.rows, 6);
			cv::Mat testRow = testFeatures.row(row).clone();
			testRow.convertTo(testRow, 6);
			//std::cout << svm->getGamma() << std::endl;
			calc_rbf(svm->getGamma(), sv.rows, sv.cols, sv, testRow, buffer);  // apply kernel on data (CV_32F vector) and support vectors

			std::vector<int> votes(N_class, 0);
			double sum = 0;

			for (i = dfi = 0; i < N_class; i++)
			{
				for (j = i + 1; j < N_class; j++, dfi++)
				{
					//std::cout << dfi << " " << i << " " << j << std::endl;
					cv::Mat alpha, svidx;
					double rho = svm->getDecisionFunction(dfi, alpha, svidx);
					//std::cout << alpha << " " << alpha.type() << std::endl;
					//std::cout << svidx << " " << svidx.type() << " " << buffer.type() << std::endl;
					sum = -rho;
					int sv_count = svidx.cols;
					//std::cout << " NOW HERE " << std::endl;
					for (k = 0; k < sv_count; k++) {
						sum += alpha.at<double>(k)*buffer.at<double>(svidx.at<int>(k));
						//std::cout << alpha.at<double>(k) << " " << buffer.at<double>(svidx.at<int>(k)) << std::endl;
					}
					//std::cout << sum << std::endl;
					votes[sum > 0 ? i : j]++;
				}
			}

			oddsPrediction.push_back(votes);
		}
		oddsPrediction = oddsPrediction.reshape(0, testFeatures.rows);
		oddsPrediction.convertTo(oddsPrediction, CV_32FC1);

		int norm_factor = multinom_coeff(N_class, 2);
		oddsPrediction = oddsPrediction / norm_factor;
	}

	void logOdds_OR(cv::Mat& firstLyrOR_oddsPredictions, std::map<int, FeatureMap>& secLyr_OR_2, cv::Ptr<cv::ml::SVM>& firstLyr_SC_svm,
		const FeatureMap& firstLyr_SC_OR_2, int firstLyr_OR_featSize_cols, const Task_1& OR_1) {

		predict_LogOdds(firstLyrOR_oddsPredictions, firstLyr_SC_svm, firstLyr_SC_OR_2.features, class_no_SC); //Obtain logOdds for each img
		//Here, a direct concatenation is not possible, since the elements in the feature map are composed of ROIs.
		//Therefore, it is required to use an slightly more complicated concatenation.

		for (auto &img_path : OR_1) {
			const cv::Mat oddPrediction_i = firstLyrOR_oddsPredictions.row(img_path.second.at(SC_label));

			for (auto &class_lbl : img_path.second) {
				if (class_lbl.first.find(SC_label) != std::string::npos) continue; //If the label is the SC_label, next iter

				std::vector<std::string> split_ref;
				split(class_lbl.first, '&', split_ref); //split the class_id using & --> tasklabel_"ID" , ROI
				int class_no = std::stoi(split_ref.at(0).substr(split_ref.at(0).find('_') + 1)); //get the class id

				cv::Rect addRect(firstLyr_OR_featSize_cols, class_lbl.second, 8, 1);
				oddPrediction_i.copyTo(secLyr_OR_2.at(class_no).features(addRect));
			}
		}
	}

}




#endif // !ML_H
