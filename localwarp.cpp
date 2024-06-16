#include "localwarp.h"
//#include <GL/glut.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "globalwarp.h"
#include <iostream>
using namespace cv;
using namespace std;

//能量计算-hd
void compute_energy(Mat& img, Mat& energy, Mat& mask) {
    Mat dx, dy, gradient_magnitude;
    Sobel(img, dx, CV_64F, 1, 0, 3);  // 计算 x 方向的梯度
    Sobel(img, dy, CV_64F, 0, 1, 3);  // 计算 y 方向的梯度
    magnitude(dx, dy, gradient_magnitude);  // 计算梯度幅值

    // 初始化能量图为0
    energy = Mat::zeros(img.rows, img.cols, CV_64F);

    for (int i = 1; i < img.rows - 1; i++) {  // 注意这里从1到img.rows-1，避免越界
        for (int j = 1; j < img.cols - 1; j++) {  // 同样从1到img.cols-1，避免越界
            double CL = abs(img.at<double>(i, j + 1) - img.at<double>(i, j - 1)) + abs(img.at<double>(i - 1, j) - img.at<double>(i, j - 1));
            double CU = abs(img.at<double>(i, j + 1) - img.at<double>(i, j - 1));
            double CR = abs(img.at<double>(i, j + 1) - img.at<double>(i, j - 1)) + abs(img.at<double>(i - 1, j) - img.at<double>(i, j + 1));

            // 累积能量更新，每个像素的能量等于其梯度幅值加上最小代价
            double min_energy = gradient_magnitude.at<double>(i, j) + min({ CL, CU, CR });
            energy.at<double>(i, j) = min_energy;

            // 掩码处理
            if (mask.at<uchar>(i, j) != 0) {
                energy.at<double>(i, j) += 100000000; //将掩码区域的能量设置为非常大的值
            }
        }
    }
}

// 更新累积代价矩阵的函数-hd
void update_accumulative_energy(Mat& energy, Mat& cumulative_energy) {
    int rows = energy.rows;
    int cols = energy.cols;
    cumulative_energy = energy.clone();

    for (int i = 1; i < rows; i++) {
        for (int j = 1; j < cols - 1; j++) {
            double CL = abs(energy.at<double>(i, j + 1) - energy.at<double>(i, j - 1)) + abs(energy.at<double>(i - 1, j) - energy.at<double>(i, j - 1));
            double CU = abs(energy.at<double>(i, j + 1) - energy.at<double>(i, j - 1));
            double CR = abs(energy.at<double>(i, j + 1) - energy.at<double>(i, j - 1)) + abs(energy.at<double>(i - 1, j) - energy.at<double>(i, j + 1));
            cumulative_energy.at<double>(i, j) += min({ cumulative_energy.at<double>(i - 1, j - 1) + CL,cumulative_energy.at<double>(i - 1, j) + CU,cumulative_energy.at<double>(i - 1, j + 1) + CR });
        }
    }
}
//更新能量-hd
void update_energy(Mat& img, Mat& output, Mat& mask, int& start, int& end, int* to) {
    int W = img.cols;
    int H = img.rows;
    Mat out = output.clone();
    for (int i = start; i <= end; i++) {
        for (int j = W - 1; j >= to[i] - 1 && j >= 0; j--) {
            if (j > to[i]) {
                out.at<double>(i, j) = out.at<double>(i, j - 1);
            }else {
                Vec3b z = { 0, 0, 0 };
                Vec3b l = (j > 0) ? img.at<Vec3b>(i, j - 1) : z;
                Vec3b r = (j < W - 1) ? img.at<Vec3b>(i, j + 1) : z;
                Vec3b u = (i > 0) ? img.at<Vec3b>(i - 1, j) : z;
                Vec3b d = (i < H - 1) ? img.at<Vec3b>(i + 1, j) : z;
                int val = sqrt((l[0] - r[0]) * (l[0] - r[0]) + (l[1] - r[1]) * (l[1] - r[1])) + sqrt((l[2] - r[2]) * (l[2] - r[2]) + (u[0] - d[0]) * (u[0] - d[0])) + sqrt((u[1] - d[1]) * (u[1] - d[1]) + (u[2] - d[2]) * (u[2] - d[2]));
                out.at<double>(i, j) = val;
            }
            if (mask.at<uchar>(i, j) != 0) {//掩码区域背景赋值 超大能量
                out.at<double>(i, j) = 10000000;
            }
        }
    }
    output = out.clone();
}

// 计算图像中的最小能量接缝-hd
int* find_seam(Mat& img, int& dir, int& start, int& end, vector<Point>& seam_path) {
    int H = img.rows;
    int W = img.cols;
    if (dir == 2 || dir == 3) {
        int t = start;
        start = end;
        end = t;
        start = H - start - 1;
        end = H - end - 1;
    }
    int** dp = new int* [H];
    for (int i = 0; i < H; i++)
        dp[i] = new int[W];
    for (int i = 0; i < W; i++) dp[start][i] = (int)img.at<double>(start, i);
#pragma omp parallel for
    for (int i = start + 1; i <= end; i++) {
        for (int j = 0; j < W; j++) {
            if (j == 0) {
                dp[i][j] = min(dp[i - 1][j], dp[i - 1][j + 1]);
            }
            else if (j == W - 1) {
                dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1]);
            }
            else {
                dp[i][j] = min(min(dp[i - 1][j - 1], dp[i - 1][j]), dp[i - 1][j + 1]);
            }
            dp[i][j] += (int)img.at<double>(i, j);
        }
    }
    int* to = new int[H];
    int minn = INF, Vq = -1;
    for (int i = 0; i < W; i++)
        if (dp[end][i] < minn) {
            minn = dp[end][i];
            Vq = i;
        }
    to[end] = Vq;
    Point pos(end, Vq);
    seam_path.push_back(pos);  // 保存路径点
    while (pos.x > start) {
        int x = pos.x;
        int y = pos.y;
        int res = dp[x][y] - (int)img.at<double>(x, y);
        if (y == 0) {
            if (res == dp[x - 1][y]) {
                pos = Point(x - 1, y);
            }
            else if (res == dp[x - 1][y + 1]) {
                pos = Point(x - 1, y + 1);
            }
        }
        else if (y == W - 1) {
            if (res == dp[x - 1][y]) {
                pos = Point(x - 1, y);
            }
            else if (res == dp[x - 1][y - 1]) {
                pos = Point(x - 1, y - 1);
            }
        }
        else {
            if (res == dp[x - 1][y]) {
                pos = Point(x - 1, y);
            }
            else if (res == dp[x - 1][y + 1]) {
                pos = Point(x - 1, y + 1);
            }
            else if (res == dp[x - 1][y - 1]) {
                pos = Point(x - 1, y - 1);
            }
        }
        to[pos.x] = pos.y;
        seam_path.push_back(pos);  // 保存路径点
    }
    delete[] dp;
    return to;
}

//添加缝隙-hd
void add_seam(Mat& img, int* to, int dir, Mat& mask, int& start, int& end, Mat& displacement) {
    int W = img.cols;
    int H = img.rows;

    // 插入像素并更新掩码
    for (int i = start; i <= end; i++) {
        // 对缝隙位置的像素进行插值
        for (int k = 0; k < 3; k++) {//左侧与缝隙的平均值
            img.at<Vec3b>(i, to[i])[k] = (img.at<Vec3b>(i, to[i] - 1)[k] + img.at<Vec3b>(i, to[i])[k]) / 2;
        }
        // 更新掩码，标记插入的像素
        if (mask.at<uchar>(i, to[i]) == 0) {
            mask.at<uchar>(i, to[i]) = 2;//掩码值为 0：表示背景区域。 2表示表示新插入的背景像素
        }
        else if (mask.at<uchar>(i, to[i]) == 1) {
            mask.at<uchar>(i, to[i]) = 3;
        }//掩码值为 1：表示前景区域。 3表示表示新插入的前景像素
    }

    // 插入新的像素，将后面的像素向右移动一位
    for (int i = start; i <= end; i++) {
        for (int j = W - 1; j > to[i]; j--) {
            img.at<Vec3b>(i, j) = img.at<Vec3b>(i, j - 1);
            mask.at<uchar>(i, j) = mask.at<uchar>(i, j - 1);
            // 根据插入方向更新位移图
            Vec2f dis;
            if (dir == 1) {  // 右边界
                dis[0] = 0; //dis[0]代表x轴，dis[1]代表y轴
                dis[1] = 1;
            }
            else if (dir == 2) {  // 下边界
                dis[0] = 1;
                dis[1] = 0;
            }
            else if (dir == 3) {  // 左边界
                dis[0] = 0; dis[1] = -1;
            }
            else {  // 上边界
                dis[0] = -1;
                dis[1] = 0;
            }
            displacement.at<Vec2f>(i, j) += dis;//计算偏移量
        }
    }
}

//计算最长边界段-hd
void compute_length(Mat& bor, int to, int& len, int& dir, int& start, int& end) {
    int dif, l = 0, r = 0;
    if (to == 1 || to == 3) {
        for (int i = 0; i <= bor.rows; i++) {
            if (bor.at<uchar>(i, 0) == 2) {
                bor.at<uchar>(i, 0) = 1;
            }
            else if (bor.at<uchar>(i, 0) == 3) {
                bor.at<uchar>(i, 0) == 1;
            }
            if (i == 0) {
                dif = bor.at<uchar>(i, 0);
            }
            else if (i == bor.rows) {
                dif = -bor.at<uchar>(i - 1, 0);
            }
            else {
                dif = bor.at<uchar>(i, 0) - bor.at<uchar>(i - 1, 0);
            }
            if (dif == 1) {
                l = i;
            }
            else if (dif == -1) {
                r = i - 1;
                if (r - l + 1 > len) {
                    len = r - l + 1;
                    dir = to;
                    start = l;
                    end = r;
                }
            }
        }
    }
    else {
        for (int i = 0; i <= bor.cols; i++) {
            if (bor.at<uchar>(0, i) == 2) {
                bor.at<uchar>(0, i) = 1;
            }
            else if (bor.at<uchar>(0, i) == 3) {
                bor.at<uchar>(0, i) = 1;
            }
            if (i == 0) {
                dif = bor.at<uchar>(0, i);
            }
            else if (i == bor.cols) {
                dif = -bor.at<uchar>(0, i - 1);
            }
            else {
                dif = bor.at<uchar>(0, i) - bor.at<uchar>(0, i - 1);
            }
            if (dif == 1) {
                l = i;
            }
            if (dif == -1) {
                r = i - 1;
                if (r - l + 1 > len) {
                    len = r - l + 1;
                    dir = to;
                    start = l;
                    end = r;
                }
            }
        }
    }
}
//查找四周最短-hd
void direction(Mat& mask, int& dir, int& start, int& end) {
    int W = mask.cols;
    int H = mask.rows;
    int len = 0;
    dir = 0;
    start = 0;
    end = 0;
    for (int i = 1; i <= 4; i++) {
        if (i == 1) {
            Mat bor = mask.col(W - 1).clone();  // 获取右边界
            compute_length(bor, 1, len, dir, start, end);
        }
        else if (i == 2) {
            Mat bor = mask.row(H - 1).clone();  // 获取下边界
            compute_length(bor, 2, len, dir, start, end);
        }
        else if (i == 3) {
            Mat bor = mask.col(0).clone();  // 获取左边界
            compute_length(bor, 3, len, dir, start, end);
        }
        else {
            Mat bor = mask.row(0).clone();  // 获取上边界
            compute_length(bor, 4, len, dir, start, end);
        }
    }
    if (len < 20) {
        dir = 0;
    } //如果长度小于20，太短不进行处理
}
void rotate_image(Mat& img, int to, bool inverse = false) {
    if (to == 2) {
        if (inverse) {
            flip(img, img, 0);
            transpose(img, img);
        }
        else {
            transpose(img, img);
            flip(img, img, 0);
        }
    }
    if (to == 3) {
        flip(img, img, -1); // 无需区分inverse，因为对角线翻转相同
    }
    if (to == 4) {
        if (inverse) {
            flip(img, img, 1);
            transpose(img, img);
        }
        else {
            transpose(img, img);
            flip(img, img, 1);
        }
    }
}

void localwrap(Mat& inputimg, Mat& orimask, Mat& displacement, Mat& new_img) {
    int dir = 0, start = 0, end = 0;
    Mat grad, mask;
    Mat energy, cumulative_energy;
    mask = orimask.clone();
    Mat img = inputimg.clone();
    compute_energy(img, grad, mask);
    update_accumulative_energy(energy, cumulative_energy);
    vector<Point> seam_path;  // 保存路径的点
    while (1) {
        dir = 1;
        direction(mask, dir, start, end);
        if (dir == 0) {
            break;
        }
        rotate_image(img, dir);
        rotate_image(grad, dir);
        rotate_image(mask, dir);
        rotate_image(displacement, dir);

        int* to = find_seam(grad, dir, start, end, seam_path);  // 传递 seam_path 以保存路径
        add_seam(img, to, dir, mask, start, end, displacement);
        update_energy(img, grad, mask, start, end, to);
        update_accumulative_energy(energy, cumulative_energy);
        delete[] to;
        rotate_image(img, dir, true);
        rotate_image(grad, dir, true);
        rotate_image(mask, dir, true);
        rotate_image(displacement, dir, true);
    }

    // 放置网格
    int cols = img.cols;
    int rows = img.rows;
    int x_num = 20;
    int y_num = 20;

    Mat xgrid(y_num, x_num, CV_32FC1);
    Mat ygrid(y_num, x_num, CV_32FC1);
    int x = 0, y = 0;
    for (double i = 0; i < rows; i += 1.0 * (rows - 1) / (y_num - 1)) {
        for (double j = 0; j < cols; j += 1.0 * (cols - 1) / (x_num - 1)) {
            xgrid.at<float>(x, y) = (int)j;
            ygrid.at<float>(x, y) = (int)i;
            y++;
        }
        x++;
        y = 0;
    }
    Mat gridmask, imageGrided;
    drawGridmask(ygrid, xgrid, rows, cols, gridmask);
    drawGrid(gridmask, img, imageGrided);
    cv::imshow("After Seam Carving", imageGrided);
    //imshow("After Seam Carving1", img);
    new_img = img.clone();
}

