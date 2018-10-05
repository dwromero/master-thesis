#pragma once

#ifndef OPT_PROB_H
#define OPT_PROB_H

#include "globalVarsnDefs.h" //Contains general required includes and definitions
#include "fileStorage.h" //Contains all functions related with storage and information consistency
#include "structConstruct.h" //Contains all functions with which we construct our data structures

namespace opt {

	float min_prob = 0.00000000000000000000001;
	float max_prob = 0.99999999999999999999999;

	class ProblemMINLP_SC : public knitro::KTRProblem {

		typedef std::map<std::string, std::vector<int>> flatCoordStruct_i;

	private:

		//Objective Properties
		void setObjectiveProperties() {
			setObjType(knitro::KTREnums::ObjectiveType::ObjGeneral);
			setObjGoal(knitro::KTREnums::ObjectiveGoal::Maximize);
			setObjFnType(knitro::KTREnums::FunctionType::Uncertain);
		}

		//Variable Bounds
		void setVariableProperties() {
			setVarLoBnds(0.0);
			setVarUpBnds(1.0); //1 inclusiv
			setVarTypes(knitro::KTREnums::VariableType::Integer);
		}

		//constraint properties
		void setConstraintProperties() {
			//set constraint types
			setConTypes(0, knitro::KTREnums::ConstraintType::ConGeneral);
			setConFnTypes(0, knitro::KTREnums::FunctionType::Uncertain);

			//set constraint bounds
			setConLoBnds(1.0);
			setConUpBnds(1.0);
		}

	public:
		//constructor: pass number of variables and constraints to base class
		//3vars, 2constraints
		ProblemMINLP_SC(int _nVars, int nConst, cv::Mat& _probabilities_SC, cv::Mat& _probabilities_OR, cv::Mat& _feat_Vector,
			cv::Ptr<cv::ml::RTrees>& _ptr_classifier, flatCoordStruct_i &_SC_flatCoords_i, std::map<std::string, int>& _labels,
			int trueLabel) :
			KTRProblem(_nVars, nConst), probabilities_SC(&_probabilities_SC), probabilities_OR(&_probabilities_OR), feat_vector(&_feat_Vector),
			ptr_classifier(&_ptr_classifier), nVars(_nVars), ptr_SC_flatCoords_i(&_SC_flatCoords_i), ptr_labels(&_labels),
			secLyrTrueLabel(trueLabel - 1) {

			//set probproperties in constructor
			setObjectiveProperties();
			setVariableProperties();
			setConstraintProperties();
		}

		const cv::Mat* probabilities_SC;
		const cv::Mat* probabilities_OR;
		flatCoordStruct_i* ptr_SC_flatCoords_i;
		cv::Mat* feat_vector;
		std::map<std::string, int>* ptr_labels;
		cv::Ptr<cv::ml::RTrees>* ptr_classifier;
		int nVars;
		int secLyrTrueLabel;


		//Obj n constraint evaluation function
		//overrides KTRIProblem class
		double evaluateFC(const double* x, double* c, double* objGrad, double* jac)
			override {

			//constraints
			c[0] = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];

			//functionEvaluation
			cv::Mat modifProb;
			cv::Mat modif_feat_vector = feat_vector->clone();
			std::map<std::string, int>::iterator label_it = ptr_labels->begin();

			int j = 0;
			while (j < nVars) {
				if (j < class_no_SC) { //We are looking at the odds of SC

					if (x[j] == 1.0) {
						modifProb.push_back(probabilities_SC->at<float>(j));
						modif_feat_vector.at<float>(j + _SC_offset_noOdds) = 1.0;
						j = (class_no_SC - 1);
					}
				}
				else { //We are looking at the heatmaps
					if (x[j] == 1) {
						modifProb.push_back(probabilities_OR->at<float>(j - class_no_SC));
						for (auto & elem : ptr_SC_flatCoords_i->at(label_it->first)) {
							modif_feat_vector.at<float>(elem) = 1.0;
						}
					}
					else {
						modifProb.push_back(1 - probabilities_OR->at<float>(j - class_no_SC));
					}
					label_it++;
				}
				j++;
			}

			cv::Mat secLyr_odds;
			ml::RFInference_RAW(*ptr_classifier, secLyr_odds, modif_feat_vector);

			modifProb.push_back(secLyr_odds.at<float>(secLyrTrueLabel));

			for (int i = 0; i < modifProb.rows; i++) {
				if (modifProb.at<float>(i) == 0.0f)
					modifProb.at<float>(i) = min_prob;
			}

			cv::log(modifProb, modifProb);

			double sum = cv::sum(modifProb)[0];

			return sum;
		}
	};

	class ProblemMINLP_OR : public knitro::KTRProblem {

		typedef std::map<std::string, std::vector<int>> flatCoordStruct_i;

	private:

		//Objective Properties
		void setObjectiveProperties() {
			setObjType(knitro::KTREnums::ObjectiveType::ObjGeneral);
			setObjGoal(knitro::KTREnums::ObjectiveGoal::Maximize);
			setObjFnType(knitro::KTREnums::FunctionType::Uncertain);
		}

		//Variable Bounds
		void setVariableProperties() {
			setVarLoBnds(0.0);
			setVarUpBnds(1.0); //1 inclusiv
			setVarTypes(knitro::KTREnums::VariableType::Integer);
		}

		//constraint properties
		void setConstraintProperties() {
			//set constraint types
			setConTypes(0, knitro::KTREnums::ConstraintType::ConGeneral);
			setConFnTypes(0, knitro::KTREnums::FunctionType::Uncertain);

			//set constraint bounds
			setConLoBnds(1.0);
			setConUpBnds(1.0);
		}

	public:
		//constructor: pass number of variables and constraints to base class
		//3vars, 2constraints
		ProblemMINLP_OR(int _nVars, int nConst, cv::Mat& _probabilities_SC, cv::Mat& _probabilities_OR, std::vector<cv::Mat>& _feat_vectors,
			std::map<int, cv::Ptr<cv::ml::RTrees>>& _ptr_classifiers, flatCoordStruct_i &_OR_flatCoords_i, std::map<std::string, int>& _labels,
			std::vector<int>& _trueLabels, std::vector<std::pair<int, int>>& _feat_ids, std::map<int, FeatureMap>& _features_i) :
			KTRProblem(_nVars, nConst), probabilities_SC(&_probabilities_SC), probabilities_OR(&_probabilities_OR), feat_vectors(&_feat_vectors),
			ptr_classifiers(&_ptr_classifiers), nVars(_nVars), ptr_OR_flatCoords_i(&_OR_flatCoords_i), ptr_labels(&_labels),
			trueLabels(&_trueLabels), feat_ids(&_feat_ids), ptr_features_i(&_features_i) {

			//set probproperties in constructor
			setObjectiveProperties();
			setVariableProperties();
			setConstraintProperties();
		}

		const cv::Mat* probabilities_SC;
		const cv::Mat* probabilities_OR;
		flatCoordStruct_i* ptr_OR_flatCoords_i;
		std::vector<cv::Mat>* feat_vectors;
		std::map<std::string, int>* ptr_labels;
		std::map<int, cv::Ptr<cv::ml::RTrees>>* ptr_classifiers;
		int nVars;
		std::vector<int>* trueLabels;
		std::vector<std::pair<int, int>>* feat_ids;
		std::map<int, FeatureMap>* ptr_features_i;


		//Obj n constraint evaluation function
		//overrides KTRIProblem class
		double evaluateFC(const double* x, double* c, double* objGrad, double* jac)
			override {

			//constraints
			c[0] = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];

			//functionEvaluation
			cv::Mat modifProb;
			//std::vector<cv::Mat> modif_feat_vectors = *feat_vectors;
			//cv::Mat modifVector = cv::Mat::zeros(modif_feat_vectors.begin()->size(), CV_32FC1);
			std::map<std::string, int>::iterator label_it = ptr_labels->begin();

			std::map<int, FeatureMap> features_i = *ptr_features_i;
			for (auto & mat : features_i)
				mat.second.features = ptr_features_i->at(mat.first).features.clone();

			cv::Mat modifVector = cv::Mat::zeros(1, features_i.begin()->second.features.cols, CV_32FC1);

			int j = 0;
			while (j < nVars) {
				if (j < class_no_SC) { //We are looking at the odds of SC

					if (x[j] == 1.0) {
						modifProb.push_back(probabilities_SC->at<float>(j));
						modifVector.at<float>(j + firstLyr_OR_featSize_cols) = 1.0;
						j = (class_no_SC - 1);
					}
				}
				else { //We are looking at the heatmaps
					if (x[j] == 1.0) {
						modifProb.push_back(probabilities_OR->at<float>(j - class_no_SC));
						for (auto & elem : ptr_OR_flatCoords_i->at(label_it->first)) {
							modifVector.at<float>(elem) = 1.0;
						}
					}
					else {
						modifProb.push_back(1.0f - probabilities_OR->at<float>(j - class_no_SC));
					}
					label_it++;
				}
				j++;
			}

			modifVector = modifVector(cv::Rect(firstLyr_OR_featSize_cols, 0, (modifVector.cols - firstLyr_OR_featSize_cols), 1));
			for (auto & mat : features_i)
				for (int i = 0; i < mat.second.features.rows; i++)
					modifVector.copyTo(mat.second.features(cv::Rect(firstLyr_OR_featSize_cols, i, (mat.second.features.cols - firstLyr_OR_featSize_cols), 1)));

			for (auto &mat : features_i) {

				cv::Mat prob_response;
				ml::RFInference_RAW(ptr_classifiers->at(mat.first), prob_response, mat.second.features);
				
				for (int i = 0; i < prob_response.rows; i++) {
					if (mat.second.labels.at<int>(i) != 0)
						modifProb.push_back(prob_response.at<float>(i,1));
					else
						modifProb.push_back(prob_response.at<float>(i,0));
				}
			}

			for (int i = 0; i < modifProb.rows; i++) {
				if (modifProb.at<float>(i) == 0.0f)
					modifProb.at<float>(i) = min_prob;
			}

			cv::log(modifProb, modifProb);

			double sum = cv::sum(modifProb)[0];

			return sum;
		}
	};
}

#endif // !OPT_PROB_H

