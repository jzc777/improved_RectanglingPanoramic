#pragma once
#ifndef ADDSEAM_H
#define ADDSEAM_H

#include <vector>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define INF 1111111111

using namespace std;
using namespace cv;

void compute_energy(Mat& img, Mat& output, Mat& mask);
void update_energy(Mat& img, Mat& output, Mat& mask, int& st, int& en, int* to);
int* find_seam(Mat& img, int& dir, int& st, int& en);
void add_seam(Mat& img, int* to, int dir, Mat& mask, int& st, int& en, Mat& disimg);
void localwrap(Mat& inputimg, Mat& orimask, Mat& displacement, Mat& new_img);
void compute_length(Mat& bor, int to, int& len, int& dir, int& st, int& en);
void direction(Mat& mask, int& dir, int& st, int& en);


#endif // ADDSEAM_H
