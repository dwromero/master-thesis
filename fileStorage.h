#pragma once

#ifndef FILESTORAGE_H
#define FILESTORAGE_H

#include "globalVarsnDefs.h"
#include "generalPurpose.h"

namespace fs {

	enum MODEL {
		SAVE,
		LOAD
	};

	//DO NOT USE DIRECTLY, Use saveLoad&Routine& instead -- checked
	void _inner_saveTask_1(cv::FileStorage& fs, Task_1& task_1) {
		fs << "task_1" << "[";
		for (auto & img_path : task_1) {
			fs << "{:" << "img_path" << img_path.first << "second" << "[";
			for (auto & task_id : img_path.second) {
				fs << "{:" << "task_id" << task_id.first << "index" << task_id.second << "}";
			}
			fs << "]" << "}";
		}
		fs << "]";
	}

	//DO NOT USE DIRECTLY, Use saveLoad&Routine& instead -- checked
	void _inner_loadTask_1(cv::FileStorage& fs, Task_1& task_1) {
		cv::FileNode img_path = fs["task_1"];
		cv::FileNodeIterator img_path_it = img_path.begin(), img_path_end = img_path.end();

		for (; img_path_it != img_path_end; ++img_path_it) {
			cv::FileNode class_id = (*img_path_it)["second"];
			cv::FileNodeIterator class_id_it = class_id.begin(), class_id_end = class_id.end();

			std::map<std::string, int> taskid_index;
			for (; class_id_it != class_id_end; ++class_id_it) {
				taskid_index[(std::string)(*class_id_it)["task_id"]] = (int)(*class_id_it)["index"];
			}
			task_1[(std::string)(*img_path_it)["img_path"]] = taskid_index;
		}
	}

	//DO NOT USE DIRECTLY, Use saveLoad&Routine& instead -- checked
	void _inner_saveFeatureMap(cv::FileStorage& fs, FeatureMap& task_2) {
		fs << "task_2" << "[";
		fs << "{:" << "features" << task_2.features << "labels" << task_2.labels << "localizer" << "[";
		for (auto & value : task_2.localizer) fs << "{:" << "value" << value << "}";
		fs << "]" << "}" << "]";
	}

	//DO NOT USE DIRECTLY, Use saveLoad&Routine& instead -- checked
	void _inner_loadFeatureMap(cv::FileStorage& fs, FeatureMap& task_2) {
		cv::FileNode task_2_node = fs["task_2"];
		cv::FileNodeIterator task_2_it = task_2_node.begin(), task_2_end = task_2_node.end();

		for (; task_2_it != task_2_end; ++task_2_it) {
			(*task_2_it)["features"] >> task_2.features;
			(*task_2_it)["labels"] >> task_2.labels;

			std::vector<std::string> localizer_vector;
			cv::FileNode loc_node = (*task_2_it)["localizer"];
			cv::FileNodeIterator loc_node_it = loc_node.begin(), loc_node_end = loc_node.end();
			for (; loc_node_it != loc_node_end; ++loc_node_it) {
				localizer_vector.push_back((std::string)(*loc_node_it)["value"]);
			}
			task_2.localizer = localizer_vector;
		}
	}

	//DO NOT USE DIRECTLY, Use saveLoad&Routine& instead -- checked
	void _inner_saveFeatureMap(cv::FileStorage& fs, std::map<int, FeatureMap>& task_2) {
		fs << "task_2" << "[";
		for (auto &class_id : task_2) {
			fs << "{:" << "class_id" << class_id.first <<"second" << "[";
			fs << "{:" << "features" << class_id.second.features << "labels" << class_id.second.labels << "localizer" << "[";
			for (auto & value : class_id.second.localizer) fs << "{:" << "value" << value << "}";
			fs << "]" << "}" << "]" << "}";
		}
		fs << "]";
	}

	//DO NOT USE DIRECTLY, Use saveLoad&Routine& instead -- checked
	void _inner_loadFeatureMap(cv::FileStorage& fs, std::map<int, FeatureMap>& task_2) {

		cv::FileNode task_2_node = fs["task_2"];
		cv::FileNodeIterator task_2_it = task_2_node.begin(), task_2_end = task_2_node.end(); //Change for multiple sets

		for (; task_2_it != task_2_end; ++task_2_it) {

			cv::FileNode featMaps = (*task_2_it)["second"];
			cv::FileNodeIterator featMaps_it = featMaps.begin(), featMaps_end = featMaps.end();

			for (; featMaps_it != featMaps_end; ++featMaps_it) {

				FeatureMap featmap_aux;

				(*featMaps_it)["features"] >> featmap_aux.features;
				(*featMaps_it)["labels"] >> featmap_aux.labels;

				std::vector<std::string> localizer_vector;
				cv::FileNode loc_node = (*featMaps_it)["localizer"];
				cv::FileNodeIterator loc_node_it = loc_node.begin(), loc_node_end = loc_node.end();
				for (; loc_node_it != loc_node_end; ++loc_node_it) {
					localizer_vector.push_back((std::string)(*loc_node_it)["value"]);
				}
				featmap_aux.localizer = localizer_vector;
				task_2[(int)(*task_2_it)["class_id"]] = featmap_aux;
			}
		}
	}

	//Saves or loads a Task_1 Structure.
	void saveLoadSet(std::string filename, MODEL saveLoad, Task_1& task_1) {
		if (saveLoad == MODEL::SAVE) {
			cv::FileStorage fs(filename, cv::FileStorage::WRITE);
			_inner_saveTask_1(fs, task_1);
			fs.release();
		}
		if (saveLoad == MODEL::LOAD) {
			cv::FileStorage fs(filename, cv::FileStorage::READ);
			_inner_loadTask_1(fs, task_1);
			fs.release();
		}
	}

	//Saves or loads a FeatureMap.
	void saveLoadSet(std::string filename, MODEL saveLoad, FeatureMap& task_2) {
		if (saveLoad == MODEL::SAVE) {
			cv::FileStorage fs(filename, cv::FileStorage::WRITE);
			_inner_saveFeatureMap(fs, task_2);
			fs.release();
		}
		if (saveLoad == MODEL::LOAD) {
			cv::FileStorage fs(filename, cv::FileStorage::READ);
			_inner_loadFeatureMap(fs, task_2);
			fs.release();
		}
	}

	//Saves or loads a std::map<int,FeatureMap>.
	void saveLoadSet(std::string filename, MODEL saveLoad, std::map<int, FeatureMap>& task_2) {
		if (saveLoad == MODEL::SAVE) {
			cv::FileStorage fs(filename, cv::FileStorage::WRITE);
			_inner_saveFeatureMap(fs, task_2);
			fs.release();
		}
		if (saveLoad == MODEL::LOAD) {
			cv::FileStorage fs(filename, cv::FileStorage::READ);
			_inner_loadFeatureMap(fs, task_2);
			fs.release();
		}
	}

	//Saves or loads a composite of a Task_1 and a FeatureMap
	void saveLoadSet(std::string filename, MODEL saveLoad, Task_1& task_1, FeatureMap& task_2) {

		if (saveLoad == MODEL::SAVE) {
			cv::FileStorage fs(filename, cv::FileStorage::WRITE);
			_inner_saveTask_1(fs, task_1);
			_inner_saveFeatureMap(fs, task_2);
			fs.release();
		}
		if (saveLoad == MODEL::LOAD) {
			cv::FileStorage fs(filename, cv::FileStorage::READ);
			_inner_loadTask_1(fs, task_1);
			_inner_loadFeatureMap(fs, task_2);
			fs.release();
		}
	}

	//Saves or loads a composite of a Task_1 and a std::map<int,FeatureMap>
	void saveLoadSet(std::string filename, MODEL saveLoad, Task_1& task_1, std::map<int, FeatureMap>& task_2) {

		if (saveLoad == MODEL::SAVE) {
			cv::FileStorage fs(filename, cv::FileStorage::WRITE);
			_inner_saveTask_1(fs, task_1);
			_inner_saveFeatureMap(fs, task_2);
			fs.release();
		}
		if (saveLoad == MODEL::LOAD) {
			cv::FileStorage fs(filename, cv::FileStorage::READ);
			_inner_loadTask_1(fs, task_1);
			_inner_loadFeatureMap(fs, task_2);
			fs.release();
		}
	}

	void saveLoadDPMDetections(std::string detections_filename, MODEL saveLoad,
		std::map<int, std::map<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>>>& dpm_detections) {
		// The detections_filename must be a .yalm'or .json file

		if (saveLoad == MODEL::SAVE) {

			cv::FileStorage fs(detections_filename, cv::FileStorage::WRITE);
			fs << "detections_structure" << "[";
			for (auto & class_id : dpm_detections) {
				fs << "{:" << "class_id" << class_id.first << "second" << "[";
				for (auto & img_path : class_id.second) {
					fs << "{:" << "img_path" << img_path.first << "second" << "[";
					for (auto & detection : img_path.second)
						fs << "{:" << "classID" << detection.classID << "rect" << detection.rect << "score" << detection.score << "}";
					fs << "]" << "}";
				}
				fs << "]" << "}";
			}
			fs << "]";
			fs.release();
		}

		if (saveLoad == MODEL::LOAD) {

			cv::FileStorage fs(detections_filename, cv::FileStorage::READ);
			cv::FileNode dpm_class = fs["detections_structure"];
			cv::FileNodeIterator itclass = dpm_class.begin(), itclass_end = dpm_class.end();

			for (; itclass != itclass_end; ++itclass) {
				cv::FileNode dpm_path = (*itclass)["second"];
				cv::FileNodeIterator itpath = dpm_path.begin(), itpath_end = dpm_path.end();

				std::map<std::string, std::vector<cv::dpm::DPMDetector::ObjectDetection>> path_detections;
				for (; itpath != itpath_end; ++itpath) {

					cv::FileNode dpm_detect = (*itpath)["second"];
					cv::FileNodeIterator it_detect = dpm_detect.begin(), it_detect_end = dpm_detect.end();

					std::vector<cv::dpm::DPMDetector::ObjectDetection> detections;
					for (; it_detect != it_detect_end; ++it_detect) {
						cv::Rect roi;
						(*it_detect)["rect"] >> roi;
						detections.push_back(cv::dpm::DPMDetector::ObjectDetection(roi, (float)(*it_detect)["score"], (int)(*it_detect)["classID"]));
					}
					path_detections[(std::string)(*itpath)["img_path"]] = detections;
				}
				dpm_detections[(int)(*itclass)["class_id"]] = path_detections;
			}
			fs.release();
		}
	}

	void readGroundTruthFile(std::string filename, std::map<int, std::map<std::string, std::vector<cv::Rect>>> &records_map) {
		// Annotation structure:  img_filename;leftCol;topRow;rightCol;bottomRow

		// open ground truth file
		std::ifstream gtFile;
		gtFile.open(filename);

		while (gtFile.good())
		{
			// read record (whole line)
			std::string line;
			std::vector<std::string> splitted;

			std::getline(gtFile, line);

			if (line.empty()) continue;

			// split line into elements
			split(line, ';', splitted);
			//for (auto &spl : splitted) std::cout << spl << std::endl;
			assert(splitted.size() == 6);

			// write elements into record
			cv::Point leftTopCorner(std::stoul(splitted.at(1)), std::stoul(splitted.at(2)));
			cv::Point botRightCorner(std::stoul(splitted.at(3)), std::stoul(splitted.at(4)));
			int label = std::stoi(splitted.at(5));
			std::string filename = splitted.at(0);

			records_map[label][filename].push_back(cv::Rect(leftTopCorner, botRightCorner));
		}
	}

	//Reads the GIST Descriptor and labels from the corresponding .txt files into the respective structures
	void readGISTfromFile(std::string filename_labels, std::string filename_descriptors, Task_1& task_1,
		FeatureMap& task_2) {


		// open ground truth file
		std::ifstream gtFile;
		gtFile.open(filename_labels);

		int linecounter = 0;
		while (gtFile.good())
		{
			// read record (whole line)
			std::string line;
			std::vector<std::string> splitted;

			std::getline(gtFile, line);

			if (line.empty()) continue;

			// split line into elements
			split(line, ';', splitted);
			//for (auto &spl : splitted) std::cout << spl << std::endl;
			assert(splitted.size() == 2); //File is composed of two elements.

			task_1[splitted.at(0)][SC_label] = linecounter; //add index of row to SC_1
			task_2.labels.push_back(std::stoi(splitted.at(1))); //and the label to SC_2.second
			task_2.localizer.push_back(splitted.at(0) + "&" + SC_label);
			linecounter++;
		}
		gtFile.close();

		// open ground truth file
		std::ifstream descrFile;
		descrFile.open(filename_descriptors);
		while (descrFile.good())
		{
			std::string line;
			std::vector<std::string> splitted;

			std::getline(descrFile, line);
			if (line.empty()) continue;

			// split line into elements
			split(line, ';', splitted);
			assert(splitted.size() == 512); //Gist Descriptor Size

			std::vector<float> descr;
			for (auto & value : splitted) descr.push_back(std::stof(value));
			task_2.features.push_back(descr);
		}
		descrFile.close();
		task_2.features = task_2.features.reshape(0, (task_2.features.rows*task_2.features.cols) / 512); //Reshape the cv::Mat
	}

	std::string generateName(std::string first_part, int index, std::string second_part, int iter) {
		return first_part + std::to_string(index) + second_part + std::to_string(iter) + ".json";
	}

	std::string generateName(std::string first_part, int iter) {
		return first_part + std::to_string(iter) + ".json";
	}
}

#endif // !FILESTORAGE_H

