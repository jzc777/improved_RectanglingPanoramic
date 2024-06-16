#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <GL/glut.h>
#include "localwarp.h"
#include "globalwarp.h"
using namespace cv;
using namespace std;
using namespace std::chrono;

//去除小的噪声点或者孔洞，面积小于 100 像素的轮廓，使它们变成黑色（背景）-hd
void quzao(Mat& src){
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(src, contours, hierarchy, cv::RETR_LIST, CHAIN_APPROX_NONE);
	if (!contours.empty() && !hierarchy.empty()){
		for (int idx = 0; idx < contours.size(); idx++){
			if (contours[idx].size() < 100){
				drawContours(src, contours, idx, Scalar::all(0), cv::FILLED, 8);
			}
		}
	}
}

//生成前景和背景的掩码-hd
int FG_mask(Mat& rgbImg, int yuzhi, Mat& mask){
	Mat grayImg;
	cvtColor(rgbImg, grayImg, cv::COLOR_BGR2GRAY);
	int rows = rgbImg.rows;
	int cols = rgbImg.cols;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (grayImg.at<uchar>(i, j) > yuzhi ) {
				mask.at<uchar>(i, j) = 1;//背景
			}else{
				mask.at<uchar>(i, j) = 0;//前景
			}
		}
	}
	quzao(mask);

	//边界统一设置为背景
	for (int i = 0; i < mask.rows; i++){
		mask.at<uchar>(i, 0) = 1;
		mask.at<uchar>(i, mask.cols - 1) = 1;
	}
	for (int i = 0; i < mask.cols; i++){
		mask.at<uchar>(0, i) = 1;
		mask.at<uchar>(mask.rows - 1, i) = 1;
	}
	//滤波（去小噪，填小洞-hd
	filter2D(mask, mask, mask.depth(), Mat::ones(7, 7, CV_8UC1));
	filter2D(mask, mask, mask.depth(), Mat::ones(2, 2, CV_8UC1));
	

	//归一化处理掩码，大于1的像素归为1，其余的归为0
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++) {
			if (mask.at<uchar>(i, j) > 1) {
				mask.at<uchar>(i, j) = 1;
			}else {
				mask.at<uchar>(i, j) = 0;
			}
		}
	}
	return 0;
}

int main(int argc, char* argv[]){
	std::string imagePath = "E:/VSwork/NKU/SeamCarving/images/1.jpg";
	Mat img = imread(imagePath, 1);
	if (img.empty()) {
		cout << "no image!";
		return -1;
	}
	double lambl = 3;
	if (imagePath == "E:/VSwork/NKU/SeamCarving/images/5.png") {
		lambl = 10;
	}

	Mat grayImg;
	Mat new_img;
	Mat outimg;
	Mat inputimg;
	Mat orimask;
	int col = img.cols;
	int row = img.rows;
	double draw;
	double scale = sqrt((double)210000 / (img.cols * img.rows));//放缩部分 应该可以后续修改 就2

	auto st = high_resolution_clock::now();
	resize(img, inputimg, Size(col * scale, row * scale), 0, 0, cv::INTER_CUBIC);//按scale比例缩小
	cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	Mat mask(Size(img.cols, img.rows), CV_8UC1);//初始全0的掩码

	resize(mask, orimask, Size(col * scale, row * scale), 0, 0, cv::INTER_CUBIC);//将初始掩码 mask 缩小到与缩小后的图像 inputimg 相同的大小
	Mat displacement = Mat::zeros(Size(inputimg.cols, inputimg.rows), CV_32FC2);
	imshow("Input image", inputimg);
	
	FG_mask(inputimg, 251, orimask);
	localwrap(inputimg, orimask, displacement, new_img);
	//全局扭曲
	global_warp(inputimg, displacement, orimask, outimg, lambl,draw);
	if (outimg.empty()) {
		cout << "Error: outimg is empty." << endl;
		return -1;
	}
	if (draw < 0.7) {
		float stretchFactor = 0.8 / draw;
		resize(outimg, outimg, Size(), stretchFactor, 1.0, cv::INTER_CUBIC);
	}else {
		resize(outimg, outimg, Size(col, row), 0, 0, cv::INTER_CUBIC);//将变形后的图像 outimg 缩放回原始图像的大小。
	}
	
	imshow("Result", outimg);
    cv::imwrite("E:/VSwork/NKU/SeamCarving/images/result.jpg", outimg);
	// 结束计时
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - st);

	// 打印运行时间，精确到小数点后两位
	cout << fixed << setprecision(2);
	cout << "Total execution time: " << duration.count() << " ms" << endl;

	waitKey(0);
	return 0;
}
