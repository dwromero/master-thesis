#pragma once

#ifndef STRUCT_CONST_H
#define STRUCT_CONST_H

#include "globalVarsnDefs.h"
#include "generalPurpose.h"
#include "ml.h"

// jaccard similarity coefficient
// overlap/covered area, in range [0,1]
const double jaccardScore(const cv::Rect roi1, const cv::Rect roi2)
{
	return 1.*(roi1 & roi2).area() / (roi1 | roi2).area();
}

cv::Mat paddedROI(const cv::Mat& input, const cv::Rect& rect) {
	//Obtain a padded ROI for cases when the ROI goes out of the frame of the original image.

	int top_left_x = rect.x,
		top_left_y = rect.y,
		width = rect.width,
		height = rect.height;

	int bottom_right_x = top_left_x + width;
	int bottom_right_y = top_left_y + height;

	cv::Mat output;
	if (top_left_x < 0 || top_left_y < 0 || bottom_right_x > input.cols || bottom_right_y > input.rows) { //Padding required

		int border_left = 0, border_right = 0, border_top = 0, border_bottom = 0;

		if (top_left_x < 0) {
			width = width + top_left_x;
			border_left = -1 * top_left_x;
			top_left_x = 0;
		}
		if (top_left_y < 0) {
			height = height + top_left_y;
			border_top = -1 * top_left_y;
			top_left_y = 0;
		}
		if (bottom_right_x > input.cols) {
			width = width - (bottom_right_x - input.cols);
			border_right = bottom_right_x - input.cols;
		}
		if (bottom_right_y > input.rows) {
			height = height - (bottom_right_y - input.rows);
			border_bottom = bottom_right_y - input.rows;
		}

		cv::Rect R(top_left_x, top_left_y, width, height);
		cv::copyMakeBorder(input(R), output, border_top, border_bottom, border_left, border_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	}
	else {
		// no padding required
		cv::Rect R(top_left_x, top_left_y, width, height);
		output = input(R);
	}
	return output;
}

void getDPMDetectionsFromDatabase(std::string datasetPath, std::map<int, cv::Ptr<cv::dpm::DPMDetector>>& dpmDetectors,
	std::map<int, std::map<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>>>& dpm_detections) {
	// The function goes through the images in the dataset, generating detections based on the map of dpmDetectors.
	// The dpmDetectors map is constituted as: map<int (class_ID), DPMDetector (class_Detector) >
	// The generated dpm_detections map is contituted as: map <int (class_ID), map< string (filename) , vector<detctions> (detections in file with $filename$) > > 
	// The implementation creates a thread per class to distribute the processing. THis is why the first key of the dpm_detections map is defined as the class_ID.

	std::vector<cv::String> dataset_imgs;
	cv::glob(datasetPath, dataset_imgs, false);

	//Remove all elements from the list that do not correspond to images
	std::vector<cv::String> img_extensions = { ".jpg",".png",".jpeg" };
	std::vector<cv::String>::iterator dataset_imgsit = dataset_imgs.begin();

	while (dataset_imgsit != dataset_imgs.end()) {
		cv::String current = dataset_imgsit->toLowerCase();
		bool found = false;
		for (auto &ext : img_extensions) {
			if (current.find(ext) != cv::String::npos) {
				found = true; break;
			}
		}
		if (found) ++dataset_imgsit;
		else dataset_imgsit = dataset_imgs.erase(dataset_imgsit);
	}

	for (auto & img_d : dataset_imgs) std::cout << img_d << std::endl;
	std::cout << dataset_imgs.size() << std::endl;

	//dataset_imgs.erase(dataset_imgs.begin());

	//for(auto & img_d : dataset_imgs) std::cout << img_d << std::endl;

	//std::vector<cv::String> try_out = std::vector<cv::String>(dataset_imgs.begin(), dataset_imgs.begin() + 2);
	//std::cout << try_out.size() << std::endl;

	int threadCount = dpmDetectors.size(); //std::cout << threadCount << std::endl;

	cv::parallel_for_(cv::Range(0, threadCount), [&](const cv::Range& range) {
		//std::cout << range.start << " " << range.end << std::endl;
		for (int r = range.start; r < range.end; r++) {
			auto detector = dpmDetectors.at(r + 1);
			for (auto &img_path : dataset_imgs) {
				cv::Mat img = cv::imread(img_path);
				std::cout << img_path << std::endl;
				std::vector<cv::dpm::DPMDetector::ObjectDetection> detections;
				detector->detect(img, detections);
				if (!detections.empty()) dpm_detections[r + 1][img_path] = detections;
			}
		}
	});
}

void getFeatureMapfromDPMDetections(const std::map<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>>& cl_i_dpmDetections,
	const std::map<std::string, std::vector<cv::Rect>>& class_i_gt, Task_1& task_1, FeatureMap& task_2,
	int class_id, double jaccardThresh = 0.5) {

	int size_counter = 0; // count the number of added instances to the featureSet and labelSet.
	for (auto & img_path : cl_i_dpmDetections) {

		auto gt_found = std::find_if(class_i_gt.begin(), class_i_gt.end(), [&](std::pair<std::string, std::vector<cv::Rect>> str) {
			int size = str.first.size();
			std::string search_str = str.first.substr(size - 10); //Lenght of the image_name in VOC Dataset
			return (img_path.first.find(search_str) != std::string::npos);
		}); //Check if the path in the dpm_detections is found in the groundTruth as well.

		bool imgpath_in_gt = (gt_found != class_i_gt.end());
		cv::Mat img = cv::imread(img_path.first); //Load the image

		for (auto & roi_detect : img_path.second) {
			int label; //Define label of the instance based on its jaccard score if there's any ground truth instance in the file
			if (imgpath_in_gt) {
				double jaccard = 0;
				for (auto & roi_gt : gt_found->second) {
					double jaccardScr = jaccardScore(roi_gt, roi_detect.rect);
					if (jaccardScr > jaccard) jaccard = jaccardScr;
				}
				label = (jaccard >= 0.5) ? class_id : 0;
			}
			else label = 0; //If no instance in gt, assign 0.
							//std::cout << label << std::endl;

			std::vector<float> hog_descriptor;

			cv::Mat sub_img = paddedROI(img, roi_detect.rect); //Get the subimage corresponding to the ROI
			cv::resize(sub_img, sub_img, hog.winSize); //Resize to 64x64 (Constant Size)
			hog.compute(sub_img, hog_descriptor); //Compute the descriptor of the ROI
			hog_descriptor.push_back(roi_detect.score); //And add the score of the DPM detection to the vector --> DescriptorSize = hog.descriptorSize()+1

			std::string rect_str = std::to_string(roi_detect.rect.x) + ";" + std::to_string(roi_detect.rect.y)
				+ ";" + std::to_string(roi_detect.rect.width) + ";" + std::to_string(roi_detect.rect.height);

			std::string task_roi_ref = OR_label + "_" + std::to_string(class_id) + "&" + rect_str; //to_string(roi_detect.rect);

			//std::string file_roi_reference = img_path.first + "&" + to_string(roi_detect.rect);

			task_1[img_path.first][task_roi_ref] = size_counter;
			task_2.features.push_back(hog_descriptor);
			task_2.labels.push_back(label);
			task_2.localizer.push_back(img_path.first + "&" + task_roi_ref);
			size_counter++;
		}
	}
	task_2.features = task_2.features.reshape(0,
		(task_2.features.rows*task_2.features.cols) / descriptorSize); //Reshape feature Mat.
}

void independizeLayers(FeatureMap& secLyr, const FeatureMap & firstLyr, int cols_extraFeat) {
	secLyr.labels = firstLyr.labels.clone();
	secLyr.features = firstLyr.features.clone();
	secLyr.localizer = firstLyr.localizer;
	cv::hconcat(secLyr.features, cv::Mat::zeros(cv::Size(cols_extraFeat, secLyr.features.rows), CV_32FC1), secLyr.features);
}

void independizeLayers(std::map<int, FeatureMap>& secLyr, const std::map<int, FeatureMap>& firstLayer, int cols_extraFeat) {
	for (auto & featMap : firstLayer) {
		secLyr[featMap.first];
		independizeLayers(secLyr.at(featMap.first), featMap.second, cols_extraFeat);
	}
}

void gen_flatCoords(std::map<std::string, std::vector<int>>& flatCoords, const std::string filename,
	const std::map<std::string, int> ROI_strVect, int _offset) {

	cv::Mat img = cv::imread(filename);
	cv::Rect imgWhole = cv::Rect(0, 0, img.cols, img.rows);
	int flat_size = res_HeatMapSize.width * res_HeatMapSize.height;

	for (auto & ROI_string : ROI_strVect) {

		int class_no_pos = ROI_string.first.find(OR_label); //Find the OR_label
		if (class_no_pos == std::string::npos) continue; //If no label of OR, take next label

		std::vector<std::string> split_ref;
		split(ROI_string.first, '&', split_ref); //split the class_id using & --> tasklabel_"ID" , ROI
		int class_no = std::stoi(split_ref.at(0).substr(split_ref.at(0).find('_') + 1)); //get the class id

		std::vector<std::string> split_rect;
		split(split_ref.at(1), ';', split_rect); //Reconstruct the ROI
		cv::Rect reconstructed(std::stoi(split_rect.at(0)), std::stoi(split_rect.at(1)),
			std::stoi(split_rect.at(2)), std::stoi(split_rect.at(3)));

		reconstructed = (reconstructed & imgWhole);

		//Resize the ROI:

		//Resizing:
		double scaleFactor_x = res_HeatMapSize.width / (double)img.cols;
		double scaleFactor_y = res_HeatMapSize.height / (double)img.rows;

		//ResizedROISize:
		cv::Size resSize(getIntApprox(scaleFactor_x * reconstructed.width),
			getIntApprox(scaleFactor_y * reconstructed.height));

		if ((resSize.height == 0) || (resSize.width == 0)) {
			flatCoords[ROI_string.first] == std::vector<int>();
			continue; //If any part of the size becomes 0, the ROI just disappears.
		}
		//Define the ROI Starting Point:
		cv::Point2f in_ROIstart(reconstructed.x, reconstructed.y);

		//Define the Scaling base-point
		cv::Point2f resized_midPoint((res_HeatMapSize.width - 1.0) / 2.0, (res_HeatMapSize.height - 1.0) / 2.0);
		cv::Point2f in_midPoint((img.cols - 1.0) / 2.0, (img.rows - 1.0) / 2.0);

		cv::Point2f mid_toROICorner = in_ROIstart - in_midPoint;
		cv::Point2f res_mid_toROICorner((float)scaleFactor_x * mid_toROICorner.x, (float)scaleFactor_y * mid_toROICorner.y);

		cv::Point2f res_startPoint = resized_midPoint + res_mid_toROICorner;
		cv::Rect res_rect(getIntApprox(res_startPoint.x), getIntApprox(res_startPoint.y),
			resSize.width, resSize.height);

		int rect_x = res_rect.x;
		int rect_y = res_rect.y;
		int res_img_width = res_HeatMapSize.width;

		int offset = (class_no - 1) * flat_size + _offset;

		for (int h_i = 0; h_i < res_rect.height; h_i++) {
			for (int w_i = 0; w_i < res_rect.width; w_i++) {
				flatCoords[ROI_string.first].push_back(
					(rect_y + h_i) * res_img_width + rect_x + w_i + offset);
			}
		}

		//To check if correct:
		//cv::Mat checking = cv::Mat::zeros(res_HeatMapSize, CV_32FC1);
		//checking(res_rect) = 1;
		//std::cout << "BEFORE : \n" << checking << std::endl;

		//cv::Mat checking_after = cv::Mat::zeros(1, res_HeatMapSize.height * res_HeatMapSize.width, CV_32FC1);
		//for (auto & val : flatCoords.at(ROI_string.first)) {
		//	checking_after.at<float>(val - offset) = 1;
		//}

		//checking_after = checking_after.reshape(0, res_HeatMapSize.height);
		//std::cout << "AFTER : \n" << checking_after << std::endl;
		//int i = 0;
	}
}

void heatmaps_SC(FeatureMap& secLyr_SC_2, std::map<int, cv::Mat>& OR_SC_probs, std::map<int, cv::Mat>& OR_SC_inference,
	std::map<int, cv::Ptr<cv::ml::SVM>>& firstLyr_OR_svm, const std::map<int, FeatureMap>& firstLyr_OR_SC_2,
	const Task_1& SC_1, const std::map<std::string, std::map<std::string, std::vector<int>>>& SC_flatCoords) {

	for (auto & svm : firstLyr_OR_svm)
		ml::svmInference_RAW(svm.second, OR_SC_inference[svm.first], OR_SC_probs[svm.first], firstLyr_OR_SC_2.at(svm.first).features, svm.first);

	//Create the HeatMaps for each image.
	//In order to append the estimations to the original dataset, an additional dataset with dimensions
	//flatten_heat_map columns and orig_feature_map rows is created and iteratively modified.

	for (auto &img_path : SC_1) {

		int row_in_featMap = img_path.second.at(SC_label); //location to which the heatmap should be added

		for (auto &class_label : img_path.second) {

			int class_no_pos = class_label.first.find(OR_label); //Find the OR_label
			if (class_no_pos == std::string::npos) continue; //If no label of OR, take next label

			std::vector<std::string> split_ref;
			split(class_label.first, '&', split_ref); //split the class_id using & --> tasklabel_"ID" , ROI

			int class_no = std::stoi(split_ref.at(0).substr(split_ref.at(0).find('_') + 1)); //get the class id

			if (OR_SC_inference.at(class_no).at<float>(class_label.second) == 0) continue; //If label is 0, no procesing required.

			for (auto &index : SC_flatCoords.at(img_path.first).at(class_label.first)) {
				secLyr_SC_2.features.at<float>(row_in_featMap, index) = 1.0;
			}
		}
	}

}

void heatmaps_OR(std::map<int, FeatureMap>& secLyr_OR_2, std::map<int, cv::Mat>& OR_inference,
	std::map<int, cv::Ptr<cv::ml::SVM>>& firstLyr_OR_svm, const std::map<int, FeatureMap>& firstLyr_OR_2,
	const Task_1& OR_1, const std::map<std::string, std::map<std::string, std::vector<int>>>& OR_flatCoords,
	int secLyr_OR_featSize_cols, int firstLyr_OR_featSize_cols) {

	for (auto &svm : firstLyr_OR_svm)
		ml::inference(svm.second, OR_inference[svm.first], firstLyr_OR_2.at(svm.first).features);

	for (auto & img_path : OR_1) {

		cv::Mat flattenFeats =
			cv::Mat::zeros(cv::Size(secLyr_OR_featSize_cols, 1), CV_32FC1);

		for (auto & class_label : img_path.second) {

			int class_no_pos = class_label.first.find(OR_label); //Find the OR_label
			if (class_no_pos == std::string::npos) continue; //If no label of OR, take next label

			std::vector<std::string> split_ref;
			split(class_label.first, '&', split_ref); //split the class_id using & --> tasklabel_"ID" , ROI

			int class_no = std::stoi(split_ref.at(0).substr(split_ref.at(0).find('_') + 1)); //get the class id

			if (OR_inference.at(class_no).at<float>(class_label.second) == 0) continue; //If label is 0, no procesing required.

			for (auto &index : OR_flatCoords.at(img_path.first).at(class_label.first)) {
				flattenFeats.at<float>(0, index) = 1.0;
			}
		}

		cv::Rect subMat_rect = cv::Rect(firstLyr_OR_featSize_cols + class_no_SC, 0,
			flattenFeats.cols - (firstLyr_OR_featSize_cols + class_no_SC), 1);

		flattenFeats = flattenFeats(subMat_rect);

		//Append the generated heatMaps to the original features
		for (auto & class_label : img_path.second) {

			int class_no_pos = class_label.first.find(OR_label); //Find the OR_label
			if (class_no_pos == std::string::npos) continue; //If no label of OR, take next label

			std::vector<std::string> split_ref;
			split(class_label.first, '&', split_ref); //split the class_id using & --> tasklabel_"ID" , ROI

			int class_no = std::stoi(split_ref.at(0).substr(split_ref.at(0).find('_') + 1));

			cv::Rect subMat_rect_modif = subMat_rect;
			subMat_rect_modif.y = class_label.second;

			flattenFeats.copyTo(secLyr_OR_2.at(class_no).features(subMat_rect_modif));
		}
	}

}

#endif // !STRUCT_CONST_H
