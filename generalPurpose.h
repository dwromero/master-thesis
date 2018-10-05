#pragma once

#ifndef GENPURPOSE_H
#define GENPURPOSE_H

#include "globalVarsnDefs.h"

// this method is suitable for splitting a string into a vector of substrings, divided by a delimiter character
// source: http://stackoverflow.com/questions/236129/split-a-string-in-c
// usage: create a vector of strings, then call split(string, delimiter, vector);
void split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
}

int factorial(int n, int n_min) {
	//calculates (n)!/(n-n_min)!

	assert((n > 0) && (n_min >= 0) && (n > n_min));

	double _factorial = 1;
	while (n > n_min) {
		_factorial *= n; --n;
	}
	return _factorial;
}

int factorial(int n) {
	//calculates (n)!

	return factorial(n, 0);
}

int multinom_coeff(int n, int k) {
	// Calculates (n)!/(k!(n-k)!)
	return factorial(n, (n - k)) / factorial(k);
}

//Approximate to int-coord:
int getIntApprox(double x) {
	double rest = (double)(x - (int)x);
	return (rest >= 0.5) ? (int)((int)x + 1) : (int)((int)x);
}

cv::Mat sigmoid(const cv::Mat& input) {
	cv::Mat output;
	cv::exp(-input, output);
	return 1.0 / (1.0 + output);
}

//Performs the space transformation using the RBF Kernel
void calc_rbf(double gamma, int vec_count, int var_count, cv::Mat& sv, cv::Mat& other, cv::Mat& results) {

	int j, k;
	gamma = -gamma;

	for (j = 0; j < vec_count; j++) {

		cv::Mat sample = sv.row(j).clone();
		double s = 0;

		for (k = 0; k <= var_count - 4; k += 4) {

			double t0 = sample.at<double>(k) - other.at<double>(k);
			double t1 = sample.at<double>(k + 1) - other.at<double>(k + 1);

			s += t0 * t0 + t1 * t1;
			t0 = sample.at<double>(k + 2) - other.at<double>(k + 2);
			t1 = sample.at<double>(k + 3) - other.at<double>(k + 3);
			s += t0 * t0 + t1 * t1;
		}

		for (; k < var_count; k++) {

			double t0 = sample.at<double>(k) - other.at<double>(k);
			s += t0 * t0;
		}
		results.at<double>(j) = /*(float)*/(s*gamma);
	}
	if (vec_count > 0) {
		cv::exp(results, results);
	}
}

#endif // !GENPURPOSE_H
