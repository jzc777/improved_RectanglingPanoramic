#include "globalwarp.h"
//将网格绘制在图像上-hd
void drawGrid(Mat& gridmask, Mat& img, Mat& outimage){
    outimage = img.clone();
    for (int i = 0; i < gridmask.rows; i++){
        for (int j = 0; j < gridmask.cols; j++){
            if (gridmask.at<float>(i, j) == 1){
                outimage.at<Vec3b>(i, j)[0] = 0;
                outimage.at<Vec3b>(i, j)[1] = 255;
                outimage.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }
}

//生成带网格线的掩码图
void drawGridmask(Mat& ygrid, Mat& xgrid, int rows, int cols, Mat& gridmask){
    int xgridN = ygrid.cols;
    int ygridN = ygrid.rows;
    Mat outmask = Mat::zeros(rows, cols, CV_32FC1);
    double m;
    for (int y = 0; y < ygridN; y++) {
        for (int x = 0; x < xgridN; x++) {
            if (y != 0) {
                if (ygrid.at<float>(y, x) != ygrid.at<float>(y - 1, x)) {
                    for (int i = ygrid.at<float>(y - 1, x); i <= ygrid.at<float>(y, x); i++) {
                        m = double(xgrid.at<float>(y, x) - xgrid.at<float>(y - 1, x)) /(ygrid.at<float>(y, x) - ygrid.at<float>(y - 1, x));
                        outmask.at<float>(i, int(xgrid.at<float>(y - 1, x) +int(m * (i - ygrid.at<float>(y - 1, x))))) = 1;
                    }
                }
            }
            if (x != 0) {
                if (xgrid.at<float>(y, x) != xgrid.at<float>(y, x - 1)) {
                    for (int j = xgrid.at<float>(y, x - 1); j <= xgrid.at<float>(y, x); j++) {
                        m = double(ygrid.at<float>(y, x) - ygrid.at<float>(y, x - 1)) /(xgrid.at<float>(y, x) - xgrid.at<float>(y, x - 1));
                        outmask.at<float>(int(ygrid.at<float>(y, x - 1) +int(m * (j - xgrid.at<float>(y, x - 1)))), j) = 1;
                    }
                }
            }
        }
    }
    gridmask = outmask.clone();
}

//获取四个顶点的坐标-hd
void quadVertex(int y, int x, Mat& ygrid, Mat& xgrid, Mat& vx, Mat& vy) {
    // 从xgrid中提取x坐标并存储在vx中
    vx.at<float>(0, 0) = xgrid.at<float>(y, x);
    vx.at<float>(0, 1) = xgrid.at<float>(y, x + 1);
    vx.at<float>(1, 0) = xgrid.at<float>(y + 1, x);
    vx.at<float>(1, 1) = xgrid.at<float>(y + 1, x + 1);

    // 从ygrid中提取y坐标并存储在vy中
    vy.at<float>(0, 0) = ygrid.at<float>(y, x);
    vy.at<float>(0, 1) = ygrid.at<float>(y, x + 1);
    vy.at<float>(1, 0) = ygrid.at<float>(y + 1, x);
    vy.at<float>(1, 1) = ygrid.at<float>(y + 1, x + 1);
}


// 计算两条线段的交点 -hd
void intersection(Mat& segment1, Mat& segment2, int& intersectionFlag, Mat& intersectionPoint) {
    // 计算segment1和segment2的参数
    int deltaY1 = (int)segment1.at<float>(1, 1) - (int)segment1.at<float>(0, 1);
    int deltaY2 = (int)segment2.at<float>(1, 1) - (int)segment2.at<float>(0, 1);
    int deltaX1 = (int)segment1.at<float>(0, 0) - (int)segment1.at<float>(1, 0);
    int deltaX2 = (int)segment2.at<float>(0, 0) - (int)segment2.at<float>(1, 0);
    int const1 = (int)segment1.at<float>(1, 0) * (int)segment1.at<float>(0, 1) - (int)segment1.at<float>(1, 1) * (int)segment1.at<float>(0, 0);
    int const2 = (int)segment2.at<float>(1, 0) * (int)segment2.at<float>(0, 1) - (int)segment2.at<float>(1, 1) * (int)segment2.at<float>(0, 0);

    // 构建方程矩阵
    Mat coefficientMatrix = (Mat_<float>(2, 2) << deltaY1, deltaX1, deltaY2, deltaX2);
    Mat constantMatrix = (Mat_<float>(2, 1) << -const1, -const2);
    // 计算交点
    intersectionPoint = coefficientMatrix.inv() * constantMatrix;
    intersectionPoint = intersectionPoint.t();
    // 判断交点是否在线段内
    intersectionFlag = 0; // 0表示交点不在两条线段内
    if ((intersectionPoint.at<float>(0, 0) - segment1.at<float>(0, 0)) * (intersectionPoint.at<float>(0, 0) - segment1.at<float>(1, 0)) <= 0) {
        if ((intersectionPoint.at<float>(0, 0) - segment2.at<float>(0, 0)) * (intersectionPoint.at<float>(0, 0) - segment2.at<float>(1, 0)) <= 0) {
            if ((intersectionPoint.at<float>(0, 1) - segment1.at<float>(0, 1)) * (intersectionPoint.at<float>(0, 1) - segment1.at<float>(1, 1)) <= 0) {
                if ((intersectionPoint.at<float>(0, 1) - segment2.at<float>(0, 1)) * (intersectionPoint.at<float>(0, 1) - segment2.at<float>(1, 1)) <= 0) {
                    intersectionFlag = 1;
                }
            }
        }
    }
}

//检查点是否在给定的网格顶点范围内
int checkIsIn(Mat& vy, Mat& vx, int pstx, int psty, int pendx, int pendy) {
    int min_x = min(pstx, pendx);
    int min_y = min(psty, pendy);
    int max_x = max(pstx, pendx);
    int max_y = max(psty, pendy);
    if ((min_x < vx.at<float>(0, 0) && min_x < vx.at<float>(1, 0)) || max_x > vx.at<float>(0, 1) && max_x > vx.at<float>(1, 1)){
        return 0;
    }else if ((min_y < vy.at<float>(0, 0) && min_y < vy.at<float>(0, 1)) || (max_y > vy.at<float>(1, 0) && max_y > vy.at<float>(1, 1))){
        return 0;
    }else{
        return 1;
    }
}

// 计算变换矩阵，将点 p 从一个网格变换到另一个网格
void trans_mat(Mat& vx, Mat& vy, Mat& p, Mat& TP) {
    // 将网格顶点坐标存储在 quad 矩阵中
    Mat quad(2, 4, CV_32FC1);
    quad.at<float>(0, 0) = vx.at<float>(0, 0);
    quad.at<float>(1, 0) = vy.at<float>(0, 0);
    quad.at<float>(0, 1) = vx.at<float>(0, 1);
    quad.at<float>(1, 1) = vy.at<float>(0, 1);
    quad.at<float>(0, 2) = vx.at<float>(1, 0);
    quad.at<float>(1, 2) = vy.at<float>(1, 0);
    quad.at<float>(0, 3) = vx.at<float>(1, 1);
    quad.at<float>(1, 3) = vy.at<float>(1, 1);

    // 创建单位矩阵和零矩阵
    Mat identity_matrix = Mat::eye(4, 4, CV_32FC1);
    Mat zero_matrix_2x2 = Mat::zeros(2, 2, CV_32FC1);
    Mat zero_matrix_4x1 = Mat::zeros(4, 1, CV_32FC1);

    // 构建变换矩阵 Vq
    Mat Vq_part1, Vq_part2, Vq_combined;
    hconcat(identity_matrix, quad.t(), Vq_part1);
    hconcat(quad, zero_matrix_2x2, Vq_part2);
    vconcat(Vq_part1, Vq_part2, Vq_part1);
    vconcat(zero_matrix_4x1, p, Vq_combined);

    // 计算变换矩阵的系数
    Mat coefficients = Vq_part1.inv() * Vq_combined;
    Mat TT = coefficients.rowRange(0, 4);

    // 检查变换是否正确
    if (norm(quad * TT - p) > 0.0001) {
        cout << "error" << endl;
    }

    // 构建最终的变换矩阵 T
    Mat T(2, 8, CV_32FC1);
    T.at<float>(0, 0) = TT.at<float>(0, 0);
    T.at<float>(1, 0) = 0;
    T.at<float>(0, 1) = 0;
    T.at<float>(1, 1) = TT.at<float>(0, 0);
    T.at<float>(0, 2) = TT.at<float>(1, 0);
    T.at<float>(1, 2) = 0;
    T.at<float>(0, 3) = 0;
    T.at<float>(1, 3) = TT.at<float>(1, 0);
    T.at<float>(0, 4) = TT.at<float>(2, 0);
    T.at<float>(1, 4) = 0;
    T.at<float>(0, 5) = 0;
    T.at<float>(1, 5) = TT.at<float>(2, 0);
    T.at<float>(0, 6) = TT.at<float>(3, 0);
    T.at<float>(1, 6) = 0;
    T.at<float>(0, 7) = 0;
    T.at<float>(1, 7) = TT.at<float>(3, 0);

    // 将结果存储在 TP 矩阵中
    TP = T.clone();
}

//计算线性变换矩阵，将点 (pst_x, pst_y) 从一个网格变换到另一个网格。
void getLinTrans(float pst_y, float pst_x, Mat& yVq, Mat& xVq, Mat& T, int& to){
    Mat V(8, 1, CV_32FC1);
    Mat v1(2, 1, CV_32FC1), v2(2, 1, CV_32FC1), v3(2, 1, CV_32FC1), v4(2, 1, CV_32FC1);
    V.at<float>(0, 0) = xVq.at<float>(0, 0);
    V.at<float>(1, 0) = yVq.at<float>(0, 0);
    V.at<float>(2, 0) = xVq.at<float>(0, 1);
    V.at<float>(3, 0) = yVq.at<float>(0, 1);
    V.at<float>(4, 0) = xVq.at<float>(1, 0);
    V.at<float>(5, 0) = yVq.at<float>(1, 0);
    V.at<float>(6, 0) = xVq.at<float>(1, 1);
    V.at<float>(7, 0) = yVq.at<float>(1, 1);
    v1.at<float>(0, 0) = xVq.at<float>(0, 0);
    v1.at<float>(1, 0) = yVq.at<float>(0, 0);
    v2.at<float>(0, 0) = xVq.at<float>(0, 1);
    v2.at<float>(1, 0) = yVq.at<float>(0, 1);
    v3.at<float>(0, 0) = xVq.at<float>(1, 0);
    v3.at<float>(1, 0) = yVq.at<float>(1, 0);
    v4.at<float>(0, 0) = xVq.at<float>(1, 1);
    v4.at<float>(1, 0) = yVq.at<float>(1, 1);
    Mat v21 = v2 - v1, v31 = v3 - v1, v41 = v4 - v1;
    Mat p(2, 1, CV_32FC1);
    p.at<float>(0, 0) = pst_x;  p.at<float>(1, 0) = pst_y;
    Mat p1 = p - v1;
    double a1 = v31.at<float>(0, 0), a2 = v21.at<float>(0, 0),          //x
        a3 = v41.at<float>(0, 0) - v31.at<float>(0, 0) - v21.at<float>(0, 0);
    double b1 = v31.at<float>(1, 0), b2 = v21.at<float>(1, 0),      //y
        b3 = v41.at<float>(1, 0) - v31.at<float>(1, 0) - v21.at<float>(1, 0);
    double px = p1.at<float>(0, 0), py = p1.at<float>(1, 0);
    Mat tvec, mat_t;
    double t1n, t2n;
    double a, b, c;
    if (a3 == 0 && b3 == 0){
        hconcat(v31, v21, mat_t);
        tvec = mat_t.inv() * p1;
        t1n = tvec.at<float>(0, 0);
        t2n = tvec.at<float>(1, 0);
    }else{
        a = (b2 * a3 - a2 * b3);
        b = (-a2 * b1 + b2 * a1 + px * b3 - a3 * py);
        c = px * b1 - py * a1;
        if (a == 0){
            t2n = -c / b;
        }else{
            if ((b * b - 4 * a * c) > 0){
                t2n = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
            }else{
                t2n = (-b - 0) / (2 * a);
            }
        }
        if (abs(a1 + t2n * a3) <= 0.0000001){
            t1n = (py - t2n * b2) / (b1 + t2n * b3);
        }else{
            t1n = (px - t2n * a2) / (a1 + t2n * a3);
        }
    }
    Mat m1 = v1 + t1n * (v3 - v1);
    Mat m4 = v2 + t1n * (v4 - v2);
    Mat ptest = m1 + t2n * (m4 - m1);
    double v1w = 1 - t1n - t2n + t1n * t2n;
    double v2w = t2n - t1n * t2n;
    double v3w = t1n - t1n * t2n;
    double v4w = t1n * t2n;
    Mat out(2, 8, CV_32FC1);
    out.at<float>(0, 0) = v1w;
    out.at<float>(1, 0) = 0;
    out.at<float>(0, 1) = 0;
    out.at<float>(1, 1) = v1w;
    out.at<float>(0, 2) = v2w;
    out.at<float>(1, 2) = 0;
    out.at<float>(0, 3) = 0;
    out.at<float>(1, 3) = v2w;
    out.at<float>(0, 4) = v3w;
    out.at<float>(1, 4) = 0;
    out.at<float>(0, 5) = 0;  
    out.at<float>(1, 5) = v3w;
    out.at<float>(0, 6) = v4w; 
    out.at<float>(1, 6) = 0;
    out.at<float>(0, 7) = 0; 
    out.at<float>(1, 7) = v4w;
    T = out.clone();
    if (norm(T * V - p) > 0.01){
        to = 1;
    }
}

//构建块对角矩阵。它接收两个输入矩阵 input1 和 input2，放左上角和右下角。
void BlockDiag(Mat& input1, Mat& input2, Mat& output){
    if (input1.type() == CV_8UC1) {
        Mat out = Mat::zeros(input1.rows + input2.rows, input1.cols + input2.cols, CV_8UC1);
        for (int i = 0; i < input1.rows; i++) {
            for (int j = 0; j < input1.cols; j++) {
                out.at<uchar>(i, j) = input1.at<uchar>(i, j);
            }
        }
        for (int i = 0; i < input2.rows; i++) {
            for (int j = 0; j < input2.cols; j++) {
                out.at<uchar>(i + input1.rows, j + input1.cols) = input2.at<uchar>(i, j);
            }
        }
        output = out;
    }else if (input1.type() == CV_32FC1) {
        Mat out = Mat::zeros(input1.rows + input2.rows, input1.cols + input2.cols, CV_32FC1);
        for (int i = 0; i < input1.rows; i++) {
            for (int j = 0; j < input1.cols; j++) {
                out.at<float>(i, j) = input1.at<float>(i, j);
            }
        }
        for (int i = 0; i < input2.rows; i++) {
            for (int j = 0; j < input2.cols; j++) {
                out.at<float>(i + input1.rows, j + input1.cols) = input2.at<float>(i, j);
            }
        }
        output = out;
    }else if (input1.type() == CV_32SC1) {
        Mat out = Mat::zeros(input1.rows + input2.rows, input1.cols + input2.cols, CV_32SC1);
        for (int i = 0; i < input1.rows; i++) {
            for (int j = 0; j < input1.cols; j++) {
                out.at<int>(i, j) = input1.at<int>(i, j);
            }
        }
        for (int i = 0; i < input2.rows; i++) {
            for (int j = 0; j < input2.cols; j++) {
                out.at<int>(i + input1.rows, j + input1.cols) = input2.at<int>(i, j);
            }
        }
        output = out;
    }
}

