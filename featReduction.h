#pragma once

#ifndef FEAT_REDUCTION_H
#define FEAT_REDUCTION_H

#include "globalVarsnDefs.h" //Contains general required includes and definitions
#include "fileStorage.h" //Contains all functions related with storage and information consistency
#include "structConstruct.h" //Contains all functions with which we construct our data structures

namespace fr {

	enum BASE {
		RED_RATIO,
		NO_FEATS
	};

	std::pair<fr::BASE, float> featRed_SC_params{ //please refer to Document for the selection of this value
	std::make_pair(fr::BASE::RED_RATIO,0.34f)
	};
	std::vector<std::pair<fr::BASE, float>> featRed_OR_params{ //please refer to Document for the selection of this value
		std::make_pair(fr::BASE::NO_FEATS,1823),
		std::make_pair(fr::BASE::NO_FEATS,1866),
		std::make_pair(fr::BASE::NO_FEATS,1837),
		std::make_pair(fr::BASE::RED_RATIO,0.34f),
	};

	void reduceDataset(cv::PCA& pca, cv::Mat& train, cv::Mat& test, BASE base, float ratio_feats) {
		//Reduces the datasets to the given ratio or no_features
		int req_features;
		if (base == BASE::RED_RATIO) req_features = (int)((1 - ratio_feats)*train.cols);
		else if (base == BASE::NO_FEATS) req_features = (int)ratio_feats;

		int pca_eig_size = pca.eigenvalues.rows;
		req_features = (req_features < pca_eig_size) ? req_features : pca_eig_size;

		train = pca.project(train);
		train = train(cv::Rect(0, 0, req_features, train.rows));

		test = pca.project(test);
		test = test(cv::Rect(0, 0, req_features, test.rows));
	}

	void trainPCA(cv::PCA& pca, cv::Mat& train, std::string pca_filename) {
		pca(train, cv::Mat(), CV_PCA_DATA_AS_ROW, 0);
		cv::FileStorage fs(pca_filename, cv::FileStorage::WRITE);
		pca.write(fs);
		fs.release();
	}

	void loadPCA(cv::PCA& pca, std::string pca_filename) {
		cv::FileStorage fs(pca_filename, cv::FileStorage::READ);
		pca.read(fs.root());
		fs.release();
	}

	void trainLDA_SC(cv::LDA& lda, std::string lda_filename, Task_1& SC_1, cv::Mat& _lda_feats,
		FeatureMap& secLyr_SC_2) {

		cv::Mat labels_i = cv::Mat::zeros(1, flatMapSize, CV_32SC1);

		for (auto &img : SC_1) {
			int label = secLyr_SC_2.labels.at<int>(img.second.at(SC_label));
			cv::Mat detectMap_lbls = cv::Mat::zeros(1, flatMapSize, CV_32SC1);
			detectMap_lbls = label;

			cv::vconcat(labels_i, detectMap_lbls, labels_i);
		}

		labels_i = labels_i(cv::Rect(0, 1, labels_i.cols, labels_i.rows - 1)); //The first row is a 0 vector

		//Now we need to convert the map into rows. Resulting structure: (nx4) - (nx1)
		std::vector<cv::Mat> flat_aux;

		for (int i = 0; i < 4; i++) {
			cv::Mat channel_i;
			_lda_feats(cv::Rect(flatMapSize*i, 0, flatMapSize, _lda_feats.rows)).copyTo(channel_i);
			channel_i = channel_i.reshape(0, flatMapSize * _lda_feats.rows);
			flat_aux.push_back(channel_i);
		}
		cv::Mat detectMaps = flat_aux.begin()->clone();
		for (int i = 1; i < 4; i++) cv::hconcat(detectMaps, flat_aux.at(i), detectMaps);

		labels_i = labels_i.reshape(0, labels_i.rows * labels_i.cols);

		//check-up
		int nonzero_feat = cv::countNonZero(detectMaps);
		int nonzero_lbl = cv::countNonZero(labels_i);
		assert((nonzero_feat != 0) && (nonzero_lbl != 0));

		lda = cv::LDA(detectMaps, labels_i, 1); //hold 1!
		lda.save(lda_filename);
	}

	void trainLDA_OR(std::map<int,cv::LDA>& lda_map, std::string lda_filename_1stpart, Task_1& OR_1, 
		std::map<int, FeatureMap>& OR_2, std::map<int, cv::Mat>& OR_probs,
		std::map<std::string, std::map<std::string, std::vector<int>>>& OR_flatCoords, int iter) {
		//FUTURE WORK: Improvements might appear if instead of the 0-1 predictions of the first layer,
		//each window would become the real probability of appearing.
		//NOT IMPLEMENTED DUE TO TIME REASONS (Training time).
		//The routine is however almost entirely implemented. Just delete the row
		//prob = (prob >= 0.5f) ? 1.0 : 0.0; and change the output of the first layer classifiers.

		for (auto &set : OR_2) {

			int active_lbl = set.first;

			cv::Mat detectMaps = cv::Mat::zeros(1, flatDimension, CV_32FC1);
			cv::Mat labels_i = cv::Mat::zeros(1, flatMapSize, CV_32SC1);

			std::string label_obj = OR_label + "_" + std::to_string(active_lbl);

			for (auto &img : OR_1) {

				bool contain_flag = false;
				for (auto &lbl : img.second)
					if (lbl.first.find(label_obj) != std::string::npos) {
						//if found, there are instances of the class in that image. Hence, we require that img
						contain_flag = true;
						break;
					}
				if (contain_flag == false) continue;

				cv::Mat detectMap_img = cv::Mat::zeros(1, flatDimension, CV_32FC1);
				cv::Mat detectMap_lbls = cv::Mat::zeros(1, flatMapSize, CV_32SC1);

				for (auto &lbl : img.second) {

					if (lbl.first.find(OR_label) == std::string::npos) continue;

					int class_no = std::stoi(lbl.first.substr(lbl.first.find('_') + 1, 1));

					float prob = OR_probs.at(class_no).at<float>(lbl.second);
					prob = (prob >= 0.5f) ? 1.0 : 0.0;
					int label = OR_2.at(class_no).labels.at<int>(lbl.second);

					for (auto &val : OR_flatCoords.at(img.first).at(lbl.first)) {
						int val_modif = val - (firstLyr_OR_featSize_cols + class_no_SC);
						detectMap_img.at<float>(val_modif) = prob;
						if (label == active_lbl) {
							int val_modif_lbl = val_modif - (active_lbl - 1)*flatMapSize;
							detectMap_lbls.at<int>(val_modif_lbl) = label;
						}
					}
				}
				cv::vconcat(detectMaps, detectMap_img, detectMaps);
				cv::vconcat(labels_i, detectMap_lbls, labels_i);
			}

			detectMaps = detectMaps(cv::Rect(0, 1, detectMaps.cols, detectMaps.rows - 1)); //The first row is a 0 vector
			labels_i = labels_i(cv::Rect(0, 1, labels_i.cols, labels_i.rows - 1));

			//Now we need to convert the map into rows. Resulting structure: (nx4) - (nx1)
			std::vector<cv::Mat> flat_aux;

			for (int i = 0; i < 4; i++) {
				cv::Mat channel_i;
				detectMaps(cv::Rect(flatMapSize*i, 0, flatMapSize, detectMaps.rows)).copyTo(channel_i);
				channel_i = channel_i.reshape(0, flatMapSize * detectMaps.rows);
				flat_aux.push_back(channel_i);
			}
			detectMaps = flat_aux.begin()->clone();
			for (int i = 1; i < 4; i++) cv::hconcat(detectMaps, flat_aux.at(i), detectMaps);

			labels_i = labels_i.reshape(0, labels_i.rows * labels_i.cols);

			//check-up
			int nonzero_feat = cv::countNonZero(detectMaps);
			int nonzero_lbl = cv::countNonZero(labels_i);
			assert((nonzero_feat != 0) && (nonzero_lbl != 0));

			std::string lda_filename = fs::generateName(lda_filename_1stpart, active_lbl, "_it", iter);

			lda_map[active_lbl] = cv::LDA(detectMaps, labels_i, 0);
			lda_map[active_lbl].save(lda_filename);
		}
	}

	void loadLDA_OR(std::map<int, cv::LDA>& lda_map, std::string lda_filename_1stpart, int iter) {
		for (auto &value : dpmDetector_filenames) {
			int active_lbl = value.first;
			std::string lda_filename = fs::generateName(lda_filename_1stpart, active_lbl, "_it", iter);

			lda_map[active_lbl];
			lda_map[active_lbl].load(lda_filename);
		}
	}

	void init_datasets_LDA_FE(FeatureMap& secLyr_SC_2, FeatureMap& test_secLyr_SC_2,
		cv::Mat& _pca_feats, cv::Mat& _lda_feats, cv::Mat& _pca_feats_test, cv::Mat& _lda_feats_test) {

		int cols = secLyr_SC_2.features.cols;
		int rows = secLyr_SC_2.features.rows;
		int rows_test = test_secLyr_SC_2.features.rows;

		cv::Rect _pca_rect = cv::Rect(0, 0, _SC_offset, rows);
		cv::Rect _lda_rect = cv::Rect(_SC_offset, 0, cols - _SC_offset, rows);
		cv::Rect _pca_rect_test = cv::Rect(0, 0, _SC_offset, rows_test);
		cv::Rect _lda_rect_test = cv::Rect(_SC_offset, 0, cols - _SC_offset, rows_test);

		_pca_feats = secLyr_SC_2.features(_pca_rect).clone();
		_lda_feats = secLyr_SC_2.features(_lda_rect).clone();
		_pca_feats_test = test_secLyr_SC_2.features(_pca_rect_test).clone();
		_lda_feats_test = test_secLyr_SC_2.features(_lda_rect_test).clone();
	}

	void init_datasets_LDA_FE(std::map<int, FeatureMap>& secLyr_OR_2, std::map<int, FeatureMap>& test_secLyr_OR_2,
		std::map<int, cv::Mat>& _pca_feats,	std::map<int, cv::Mat>& _lda_feats,
		std::map<int, cv::Mat>& _pca_feats_test, std::map<int, cv::Mat>& _lda_feats_test) {

		for (auto &set : secLyr_OR_2) {
			int active_lbl = set.first;

			int cols = set.second.features.cols;
			int rows = set.second.features.rows;
			int rows_test = test_secLyr_OR_2.at(active_lbl).features.rows;

			_pca_feats[active_lbl];
			_lda_feats[active_lbl];
			_pca_feats_test[active_lbl];
			_lda_feats_test[active_lbl];

			cv::Rect _pca_rect = cv::Rect(0, 0, firstLyr_OR_featSize_cols, rows);
			cv::Rect _lda_rect = cv::Rect(firstLyr_OR_featSize_cols, 0, cols - firstLyr_OR_featSize_cols, rows);
			cv::Rect _pca_rect_test = cv::Rect(0, 0, firstLyr_OR_featSize_cols, rows_test);
			cv::Rect _lda_rect_test = cv::Rect(firstLyr_OR_featSize_cols, 0, cols - firstLyr_OR_featSize_cols, rows_test);

			_pca_feats[active_lbl] = set.second.features(_pca_rect).clone();
			_lda_feats[active_lbl] = set.second.features(_lda_rect).clone();
			_pca_feats_test[active_lbl] = test_secLyr_OR_2.at(active_lbl).features(_pca_rect_test).clone();
			_lda_feats_test[active_lbl] = test_secLyr_OR_2.at(active_lbl).features(_lda_rect_test).clone();
		}

	}

	void applyLDA(cv::LDA& lda, cv::Mat& feats) {

		std::vector<cv::Mat> flat_aux;
		for (int i = 0; i < 4; i++) {
			cv::Rect channel_i_rect(flatMapSize*i, 0, flatMapSize, feats.rows);
			cv::Mat channel_i;
			feats(channel_i_rect).copyTo(channel_i);
			channel_i = channel_i.reshape(0, flatMapSize * feats.rows);
			flat_aux.push_back(channel_i);
		}
		cv::Mat detectMaps = flat_aux.begin()->clone();
		for (int i = 1; i < 4; i++) cv::hconcat(detectMaps, flat_aux.at(i), detectMaps); //OK

		cv::Mat transformedMap = lda.project(detectMaps);
		transformedMap = transformedMap.reshape(0, (transformedMap.cols * transformedMap.rows) / flatMapSize);
		transformedMap.convertTo(transformedMap, CV_32FC1);

		//feats.release(); required?
		feats = transformedMap.clone();
	}

	void applyLDA(std::map<int, cv::LDA>& lda_map, std::map<int, cv::Mat>& feats_map) {
		for (auto &featMap : feats_map) {
			applyLDA(lda_map[featMap.first], featMap.second);
		}
	}

	void comb_outputs_LDA_FE(FeatureMap& secLyr_SC_2, FeatureMap& test_secLyr_SC_2,
		cv::Mat& _pca_feats, cv::Mat& _lda_feats, cv::Mat& _pca_feats_test, cv::Mat& _lda_feats_test) {

		cv::Mat newFeatMap;
		cv::hconcat(_pca_feats, _lda_feats, newFeatMap);
		secLyr_SC_2.features = newFeatMap.clone();
		
		newFeatMap = _pca_feats_test.clone();
		cv::hconcat(_pca_feats_test, _lda_feats_test, newFeatMap);
		test_secLyr_SC_2.features = newFeatMap.clone();
	}

	void comb_outputs_LDA_FE(std::map<int, FeatureMap>& secLyr_OR_2, 
		std::map<int, FeatureMap>& test_secLyr_OR_2, std::map<int, cv::Mat>& _pca_feats, 
		std::map<int, cv::Mat>& _lda_feats,	std::map<int, cv::Mat>& _pca_feats_test, 
		std::map<int, cv::Mat>& _lda_feats_test) {

		for (auto &set : dpmDetector_filenames) {
			int active_lbl = set.first;
			cv::Mat newFeatMap;

			cv::hconcat(_pca_feats.at(active_lbl), _lda_feats.at(active_lbl), newFeatMap);
			secLyr_OR_2.at(active_lbl).features = newFeatMap.clone();
			newFeatMap.release();

			cv::hconcat(_pca_feats_test.at(active_lbl), _lda_feats_test.at(active_lbl), newFeatMap);
			test_secLyr_OR_2.at(active_lbl).features = newFeatMap.clone();
		}
	}
}

#endif // !FEAT_REDUCTION_H

