#pragma once
#ifndef GLOEL_WARP_H
#define GLOEL_WARP_H

#include <vector>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <opencv2/core/eigen.hpp>

const double pi = 3.1415926;
using namespace std;
using namespace cv;
using namespace Eigen;
void drawGridmask(Mat& ygrid, Mat& xgrid, int rows, int cols, Mat& gridmask);
void drawGrid(Mat& gridmask, Mat& img, Mat& outimage);
void quadVertex(int yy, int xx, Mat& ygrid, Mat& xgrid, Mat& vx, Mat& vy);
void intersection(Mat& segment1, Mat& segment2, int& intersectionFlag, Mat& intersectionPoint);
int checkIsIn(Mat& vy, Mat& vx, int pstx, int psty, int pendx, int pendy);
void trans_mat(Mat& vx, Mat& vy, Mat& p, Mat& TP);
void getLinTrans(float pst_y, float pst_x, Mat& yVq, Mat& xVq, Mat& T, int& to);
void global_warp(Mat& inputimg, Mat& displacement, Mat& mask, Mat& outimg,double lambl,double& draw);
void BlockDiag(Mat& input1, Mat& input2, Mat& output);
#endif // GLOEL_WARP_H
