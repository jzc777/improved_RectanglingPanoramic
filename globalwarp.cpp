#include "globalwarp.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include "localwarp.h"
#include "globalwarp.h"
using namespace cv;
using namespace std;
//函数是实现全局变形的核心函数。它接受输入图像 img，位移图 displacement，掩码 mask，以及输出图像 output。
void global_warp(Mat& img, Mat& displacement, Mat& mask, Mat& output,double lambl,double& draw){
    if (img.empty() || displacement.empty() || mask.empty()) {
        std::cerr << "一个或多个输入矩阵为空!" << std::endl;
        return;
    }
    if (displacement.rows != img.rows || displacement.cols != img.cols) {
        std::cerr << "位移矩阵的维度与输入图像不匹配!" << std::endl;
        return;
    }
    if (mask.rows != img.rows || mask.cols != img.cols) {
        std::cerr << "掩码的维度与输入图像不匹配!" << std::endl;
        return;
    }

    int cols = img.cols;
    int rows = img.rows;
    //初始化网格，将图像划分成20*20的网格，并初始化网格坐标-hd
    int x_num = 20;
    int y_num = 20;
    Mat xgrid(y_num, x_num, CV_32FC1);
    Mat ygrid(y_num, x_num, CV_32FC1);
    for (int x = 0; x < y_num; x++) {
        for (int y = 0; y < x_num; y++) {
            float i = static_cast<float>(x) * (rows - 1) / (y_num - 1);
            float j = static_cast<float>(y) * (cols - 1) / (x_num - 1);
            xgrid.at<float>(x, y) = static_cast<int>(j);
            ygrid.at<float>(x, y) = static_cast<int>(i);
        }
    }
    
    //计算变形后的网格，displacement 存储了每个像素的位移矢量，通过减去位移矢量，计算出变形后的网格顶点位置
#pragma omp parallel for collapse(2)
    Mat warp_xgrid = xgrid.clone();
    Mat warp_ygrid = ygrid.clone();
    // 在访问warp_xgrid和warp_ygrid之前检查维度
    if (warp_xgrid.rows != warp_ygrid.rows || warp_xgrid.cols != warp_ygrid.cols) {
        std::cerr << "warp_xgrid和warp_ygrid的维度不匹配!" << std::endl;
        return;
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < warp_xgrid.rows; i++){
        for (int j = 0; j < warp_xgrid.cols; j++){
            warp_xgrid.at<float>(i, j) = xgrid.at<float>(i, j) - displacement.at<Vec2f>(ygrid.at<float>(i, j), xgrid.at<float>(i, j))[1];//[1]代表y方向位移
            warp_ygrid.at<float>(i, j) = ygrid.at<float>(i, j) - displacement.at<Vec2f>(ygrid.at<float>(i, j), xgrid.at<float>(i, j))[0];//[0]代表x方向位移
        }
    }

    // 计算变形后的网格的纵向高度
    float minTransformedHeight = FLT_MAX;
    for (int j = 0; j < warp_ygrid.cols; j++) {
        float minY = FLT_MAX;
        float maxY = -FLT_MAX;
        for (int i = 0; i < warp_ygrid.rows; i++) {
            float y = warp_ygrid.at<float>(i, j);
            if (y > maxY) {
                maxY = y;
            }
            if (y < minY) {
                minY = y;
            }
        }
        float height = maxY - minY;
        if (height < minTransformedHeight) {
            minTransformedHeight = height;
        }
    }
    draw = minTransformedHeight / rows;
    //cout << draw;

    //绘制变形后的网格
        Mat gridmask1,imageGrided1;
        drawGridmask(warp_ygrid,warp_xgrid,rows,cols,gridmask1);
        drawGrid(gridmask1,img,imageGrided1);
        cv::imshow("Mesh Warped Backward",imageGrided1);

        for (int i = 0; i < warp_xgrid.rows; i++) {
            for (int j = 0; j < warp_xgrid.cols; j++) {
                int idx_y = ygrid.at<float>(i, j);
                int idx_x = xgrid.at<float>(i, j);
                if (idx_y < 0 || idx_y >= displacement.rows || idx_x < 0 || idx_x >= displacement.cols) {
                    std::cerr << "位移矩阵索引越界：idx_y = " << idx_y << ", idx_x = " << idx_x << std::endl;
                    continue;
                }
                warp_xgrid.at<float>(i, j) = xgrid.at<float>(i, j) - displacement.at<Vec2f>(idx_y, idx_x)[1];
                warp_ygrid.at<float>(i, j) = ygrid.at<float>(i, j) - displacement.at<Vec2f>(idx_y, idx_x)[0];
            }
        }

    int gridrows = 19;
    int gridcols = 19;
    Mat** Es1 = new Mat * [gridrows];
    for (int i = 0; i < gridrows; i++) {
        Es1[i] = new Mat[gridcols];
    }
    //形状保存Es Shape Preservation-hd
    Mat Aq(8, 4, CV_32FC1);
    Mat Vq(4, 2, CV_32FC1);
    for (int i = 0; i < gridrows; i++){
        for (int j = 0; j < gridcols; j++){
            Mat I = Mat::eye(8, 8, CV_32FC1);
            Vq.at<float>(0, 0) = warp_xgrid.at<float>(i, j);
            Vq.at<float>(0, 1) = warp_ygrid.at<float>(i, j);
            Vq.at<float>(1, 0) = warp_xgrid.at<float>(i, j + 1);
            Vq.at<float>(1, 1) = warp_ygrid.at<float>(i, j + 1);
            Vq.at<float>(2, 0) = warp_xgrid.at<float>(i + 1, j);
            Vq.at<float>(2, 1) = warp_ygrid.at<float>(i + 1, j);
            Vq.at<float>(3, 0) = warp_xgrid.at<float>(i + 1, j + 1);
            Vq.at<float>(3, 1) = warp_ygrid.at<float>(i + 1, j + 1);
            // 填充Aq矩阵
            for (int k = 0; k < 4; k++) { //通过Aq间接得到了Vq存储的值
                int row = k * 2;
                Aq.at<float>(row, 0) = Vq.at<float>(k, 0);
                Aq.at<float>(row, 1) = -Vq.at<float>(k, 1);
                Aq.at<float>(row, 2) = 1;
                Aq.at<float>(row, 3) = 0;
                Aq.at<float>(row + 1, 0) = Vq.at<float>(k, 1);
                Aq.at<float>(row + 1, 1) = Vq.at<float>(k, 0);
                Aq.at<float>(row + 1, 2) = 0;
                Aq.at<float>(row + 1, 3) = 1;
            }
            Es1[i][j] = Aq * (Aq.t() * Aq).inv() * Aq.t() - I;//保存Es

        }
    }
    //形状保存第一部分结束-hd
    int quadrows = 19;
    int quadcols = 19;
    Mat S;//用于存储最终的形状保持能量项矩阵。
    int S_hd = 0;
    int Si_hd = 0;
    for (int i = 0; i < quadrows; i++) {
        Mat Si;//用于存储一行网格单元的形状保持能量项矩阵。
        Si_hd = 0;
        for (int j = 0; j < quadcols; j++) {
            if (Si_hd == 0) {   //如果 Si_hd 为 0，则直接将 Es[i][j] 赋值给 Si，并将 Si_dir 设为 1，表示已赋值
                Si = Es1[i][j];
                Si_hd++;
            }
            else {
                BlockDiag(Si, Es1[i][j], Si); //如果 Si_hd不为 0，则将 Es[i][j] 与当前的 Si 矩阵进行块对角拼接，结果仍存储在 Si 中。
            }
        }
        if (S_hd == 0) { //如果 S_hd 为 0，则直接将 Si 赋值给 S，并将 S_dir 设为 1，表示初始矩阵已赋值。
            S = Si;
            S_hd++;
        }
        else {//如果 S_hd 不为 0，则将 Si 与当前的 S 矩阵进行块对角拼接，结果仍存储在 S 中。
            BlockDiag(S, Si, S);
        }
    }

    //边界约束-hd
    int total = x_num * y_num;
    std::vector<float> B_mat(total * 2, 0);  // 用于保存边界顶点的期望位置
    std::vector<float> BW(total * 2, 0); // 用于保存边界点的约束，类似于权重
    //乘以2是因为要存储x和y两个值

    // 设置左边界和上边界的约束
    for (int i = 0; i < total * 2; i += x_num * 2) {
        B_mat[i] = 1;
        BW[i] = 1;
    }
    for (int i = 1; i < x_num * 2; i += 2) {
        B_mat[i] = 1;
        BW[i] = 1;
    }
    // 设置右边界的约束
    for (int i = x_num * 2 - 2; i < total * 2; i += x_num * 2) {
        B_mat[i] = img.cols;//希望紧贴右边
        BW[i] = 1;
    }
    // 设置下边界的约束
    for (int i = total * 2 - x_num * 2 + 1; i < total * 2; i += 2) {
        B_mat[i] = img.rows;//希望紧贴下边
        BW[i] = 1;
    }

    // 将一维数组转换为 OpenCV 矩阵
    //加权后的边界约束矩阵 B
    Mat B(total * 2, 1, CV_32FC1, B_mat.data());//包含了边界顶点期望位置的矩阵，每个顶点位置乘以相应的权重
    Mat BI(total * 2, 1, CV_32FC1, BW.data());//权重信息
    //边界约束能量矩阵EB
    Mat EB = Mat::diag(BI);//对角矩阵，包含了边界点的权重信息。这个矩阵用于在优化过程中保持边界点的位置

    Mat img_gray;
    cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    //line preservation 直线段保存EL
    //LSD线段检测和处理，原理是通过像素的梯度信息计算的
    vector<Vec4f>lines;
    Ptr<LineSegmentDetector>lsd = createLineSegmentDetector(LSD_REFINE_STD);
    lsd->detect(img_gray, lines);
    Mat drawnLines(img_gray);
    lsd->drawSegments(drawnLines, lines);
    cv::imshow("LSD work", drawnLines);//打印LSD的结果
    //网格分割初始化
    int line_num = lines.size();
    int num[20][20];
    memset(num, 0, sizeof(num));
    Mat** LineInfo = new Mat * [19];
    for (int i = 0; i < 19; i++) {
        LineInfo[i] = new Mat[x_num - 1];
    }
    //存储直线段的信息
    // 遍历所有检测到的直线段
    for (int i = 0; i < line_num; i++) {
        Mat line1(2, 2, CV_32SC1); // 创建一个2x2矩阵存储直线段的起点和终点坐标
        line1.at<int>(0, 1) = lines[i][0];
        line1.at<int>(0, 0) = lines[i][1];
        line1.at<int>(1, 1) = lines[i][2];
        line1.at<int>(1, 0) = lines[i][3];
        // 检查直线段的起点和终点是否在掩码区域内，如果是则跳过
        if ((mask.at<uchar>(line1.at<int>(0, 0), line1.at<int>(0, 1)) == 1) ||(mask.at<uchar>(line1.at<int>(1, 0), line1.at<int>(1, 1)) == 1)) {
            continue;
        }

        // 根据位移矢量更新直线段的起点坐标
        int outy1 = line1.at<int>(0, 0) + displacement.at<Vec2f>(line1.at<int>(0, 0), line1.at<int>(0, 1))[1];
        int outx1 = line1.at<int>(0, 1) + displacement.at<Vec2f>(line1.at<int>(0, 0), line1.at<int>(0, 1))[0];
        // 计算网格单元的宽度和高度
        float gw = (img.cols - 1) / (gridcols - 1);
        float gh = (img.rows - 1) / (gridrows - 1);

        // 确定直线段起点所在的网格单元
        int stgrid_y = 1.0 * outy1 / gh;
        int stgrid_x = 1.0 * outx1 / gw;
        int now_x = stgrid_x; // 当前网格单元的x坐标
        int now_y = stgrid_y; // 当前网格单元的y坐标
        Mat vx(2, 2, CV_32FC1); // 存储网格顶点的x坐标
        Mat vy(2, 2, CV_32FC1); // 存储网格顶点的y坐标
        int dir[4][2] = { {1, 0}, {0, 1}, {-1, 0}, {0, -1} }; // 网格单元的四个方向

        Mat pst(1, 2, CV_32FC1); // 存储直线段的起点
        Mat pen(1, 2, CV_32FC1); // 存储直线段的终点
        Mat pnow(1, 2, CV_32FC1); // 当前处理的点
        pst.at<float>(0, 0) = line1.at<int>(0, 0);
        pst.at<float>(0, 1) = line1.at<int>(0, 1);
        pen.at<float>(0, 0) = line1.at<int>(1, 0);
        pen.at<float>(0, 1) = line1.at<int>(1, 1);
        pnow = pen.clone();
        int to;
        Mat p(2, 2, CV_32FC1);
        int last = -1, count = 0;

        // 遍历网格单元，更新直线段信息
        while (true) {
            count++;
            if (count > 1) {
                break;
            }
            int hd = 0;//判断是否找到交点
            if (now_y >= 19 || now_x >= x_num - 1 || now_x < 0 || now_y < 0) {
                std::cerr << "索引越界：now_x = " << now_x << ", now_y = " << now_y << std::endl;
                break;
            }
            // 获取当前网格单元的顶点坐标
            quadVertex(now_y, now_x, warp_ygrid, warp_xgrid, vx, vy);

            // 检查直线段是否在当前网格单元内
            int isin = checkIsIn(vy, vx, pst.at<float>(0, 1), pst.at<float>(0, 0), pen.at<float>(0, 1), pen.at<float>(0, 0));
            if (isin == 0) {
                // 如果不在当前网格单元内，则计算交点并更新网格单元
                Mat quad(2, 2, CV_32FC4);
                quad.at<Vec4f>(0, 0)[0] = vy.at<float>(0, 1);
                quad.at<Vec4f>(0, 1)[0] = vx.at<float>(0, 1);
                quad.at<Vec4f>(1, 0)[0] = vy.at<float>(1, 1);
                quad.at<Vec4f>(1, 1)[0] = vx.at<float>(1, 1);
                quad.at<Vec4f>(0, 0)[1] = vy.at<float>(1, 1);
                quad.at<Vec4f>(0, 1)[1] = vx.at<float>(1, 1);
                quad.at<Vec4f>(1, 0)[1] = vy.at<float>(1, 0); 
                quad.at<Vec4f>(1, 1)[1] = vx.at<float>(1, 0);
                quad.at<Vec4f>(0, 0)[2] = vy.at<float>(1, 0); 
                quad.at<Vec4f>(0, 1)[2] = vx.at<float>(1, 0);
                quad.at<Vec4f>(1, 0)[2] = vy.at<float>(0, 0); 
                quad.at<Vec4f>(1, 1)[2] = vx.at<float>(0, 0);
                quad.at<Vec4f>(0, 0)[3] = vy.at<float>(0, 0); 
                quad.at<Vec4f>(0, 1)[3] = vx.at<float>(0, 0);
                quad.at<Vec4f>(1, 0)[3] = vy.at<float>(0, 1); 
                quad.at<Vec4f>(1, 1)[3] = vx.at<float>(0, 1);
                Mat line_now;
                vconcat(pst, pen, line_now);
                for (int k = 0; k < 4; k++) {
                    if (abs(last - k) == 2) {
                        continue;
                    }
                    Mat quad1(2, 2, CV_32FC1);
                    for (int i = 0; i < 2; i++) {
                        for (int j = 0; j < 2; j++) {
                            quad1.at<float>(i, j) = quad.at<Vec4f>(i, j)[k];
                        }
                    }
                    intersection(quad1, line_now, to, p);
                    if (to == 1) {
                        last = k;
                        hd = 1;
                        now_x += dir[k][0];
                        now_y += dir[k][1];
                        pnow = p.clone();
                        break;
                    }
                }
            }
            // 将直线段信息存储到当前网格单元
            Mat mat_t;
            Mat count_zero(1, 1, CV_32FC1);
            count_zero.at<float>(0, 0) = 0;
            hconcat(pst, pen, mat_t);
            hconcat(mat_t, count_zero, mat_t);

            if (now_x > x_num || now_y > y_num || now_x < 0 || now_y < 0) {
                break;
            }
            if (num[now_y][now_x] == 0) {
                LineInfo[now_y][now_x] = mat_t.clone();
                num[now_y][now_x]++;
            }else {
                vconcat(LineInfo[now_y][now_x], mat_t, LineInfo[now_y][now_x]);
            }

            pst = pnow.clone();
            pnow = pen.clone();

            if (isin == 1) {
                break;
            }
            if (hd == 0) {
                break;
            }
        }
    }
    //EL到此结束

    // 网格初始化和线段分配

    int quadID;
    int topleftverterID;
    Mat Q = Mat::zeros(8 * quadrows * quadcols, 2 * y_num * x_num, CV_32FC1);
    for (int i = 0; i < quadrows; i++){
        for (int j = 0; j < quadcols; j++){//初始化Q矩阵
            quadID = (i * quadcols + j) * 8;
            topleftverterID = (i * x_num + j) * 2;
            Q.at<float>(quadID, topleftverterID) = 1;
            Q.at<float>(quadID, topleftverterID + 1) = 0;
            Q.at<float>(quadID + 1, topleftverterID) = 0;
            Q.at<float>(quadID + 1, topleftverterID + 1) = 1;
            Q.at<float>(quadID + 2, topleftverterID + 2) = 1;
            Q.at<float>(quadID + 2, topleftverterID + 3) = 0;
            Q.at<float>(quadID + 3, topleftverterID + 2) = 0;
            Q.at<float>(quadID + 3, topleftverterID + 3) = 1;
            Q.at<float>(quadID + 4, topleftverterID + x_num * 2) = 1;
            Q.at<float>(quadID + 4, topleftverterID + x_num * 2 + 1) = 0;
            Q.at<float>(quadID + 5, topleftverterID + x_num * 2) = 0;
            Q.at<float>(quadID + 5, topleftverterID + x_num * 2 + 1) = 1;
            Q.at<float>(quadID + 6, topleftverterID + x_num * 2 + 2) = 1;
            Q.at<float>(quadID + 6, topleftverterID + x_num * 2 + 3) = 0;
            Q.at<float>(quadID + 7, topleftverterID + x_num * 2 + 2) = 0;
            Q.at<float>(quadID + 7, topleftverterID + x_num * 2 + 3) = 1;
        }
    }

    
    
    //角度盒子θ-hd
    double delta =pi/50;//划分成50份
    vector<double>quad_theta[19][19]; // 存储每个网格单元的平均角度θ
    vector<int>quad_bin[19][19]; // 存储每个网格单元的线段对应的盒子编号
    // 遍历每个网格单元，计算线段角度并分配到盒子
    for (int i = 0; i < quadrows; i++){
        for (int j = 0; j < quadcols; j++){
            // 获取线段的代码
            Mat quadseg = LineInfo[i][j];
            int lineN = quadseg.rows;
            quad_bin[i][j].clear();
            quad_theta[i][j].clear();
            for (int k = 0; k < lineN; k++){
                // 获取线段的起点和终点
                int pst_y = quadseg.at<float>(k, 0);
                int pst_x = quadseg.at<float>(k, 1);
                int pen_y = quadseg.at<float>(k, 2);
                int pen_x = quadseg.at<float>(k, 3);

                // 计算线段的倾斜角度
                double angle;
                if (pst_x == pen_x){
                    angle = pi / 2;   //90度
                }else {
                    angle = atan(double(pst_y - pen_y) / (pst_x - pen_x));//向量e的表达，atan函数求得角度
                }

                if (angle < 0){
                    angle += 2 * pi;
                }

                // 得到θ，盒子分配
                int theta = (int)((angle + pi/2)/delta);

                // 将线段的角度和盒子编号存储起来
                quad_theta[i][j].push_back(angle);
                quad_bin[i][j].push_back(theta);
            }
        }
    }

    
    Mat R(2, 2, CV_32FC1);
    Mat pst(2, 1, CV_32FC1);
    Mat pen(2, 1, CV_32FC1);
    Mat vx(2, 2, CV_32FC1);
    Mat vy(2, 2, CV_32FC1);
    Mat** EL1 = new Mat * [quadrows];
    for (int i = 0; i < quadrows; i++) { 
        EL1[i] = new Mat[quadcols];
    }

    //优化循环
    int iteration = 1;//迭代次数
    double NL;
    int jump[20][20][110];
    Mat new_xgrid(y_num, x_num, CV_32FC1);
    Mat new_ygrid(y_num, x_num, CV_32FC1);
    for (int i = 0; i < iteration; i++){//循环包括线段信息更新，构建优化矩形，优化EL
        NL = 0;
        memset(jump, 0, sizeof(jump));
        int EL_dir[120][120];
        vector<Mat>TT[120][120];
        memset(EL_dir, 0, sizeof(EL_dir));
        // 1. 更新线段信息
        // 遍历每个网格单元，更新直线段信息
        for (int i = 0; i < quadrows; i++){  
            for (int j = 0; j < quadcols; j++){
                int lineN = LineInfo[i][j].rows;
                NL += lineN;
                for (int k = 0; k < lineN; k++){
                    quadVertex(i, j, warp_ygrid, warp_xgrid, vx, vy);
                    pst.at<float>(0, 0) = LineInfo[i][j].at<float>(k, 0);//起点xy
                    pst.at<float>(1, 0) = LineInfo[i][j].at<float>(k, 1);
                    pen.at<float>(0, 0) = LineInfo[i][j].at<float>(k, 2);//终点xy
                    pen.at<float>(1, 0) = LineInfo[i][j].at<float>(k, 3);
                    Mat T1, T2;
                    int flgg = 0;
                    getLinTrans(pst.at<float>(0, 0), pst.at<float>(1, 0), vy, vx, T1, flgg);
                    getLinTrans(pen.at<float>(0, 0), pen.at<float>(1, 0), vy, vx, T2, flgg);
                    TT[i][j].push_back(T1);
                    TT[i][j].push_back(T2);

                    //方向向量e
                    Mat e(2, 1, CV_32FC1);
                    e.at<float>(0, 0) = pen.at<float>(1, 0) - pst.at<float>(1, 0);
                    //旋转矩阵R-hd
                    double theta = LineInfo[i][j].at<float>(k, 4);
                    R.at<float>(0, 0) = cos(theta); 
                    R.at<float>(0, 1) = -sin(theta);
                    R.at<float>(1, 0) = sin(theta);
                    R.at<float>(1, 1) = cos(theta);
                    
                    e.at<float>(1, 0) = pen.at<float>(0, 0) - pst.at<float>(0, 0);
                    Mat I = Mat::eye(2, 2, CV_32FC1);
                    Mat C = (R * e * (e.t() * e).inv() * e.t() * R.t() - I) * (T2 - T1);//C*V

                    if (EL_dir[i][j] == 0){
                        EL1[i][j] = C;
                        EL_dir[i][j]++;
                    }else {
                        vconcat(EL1[i][j], C, EL1[i][j]);
                    }
                }
            }
        }

        //构建一个块对角矩阵 L，它包含了每个网格单元的线段保持能量项矩阵
        Mat L;
        int L_hd = 0,Li_hd = 0,n,m = 0;
        for (int i = 0; i < quadrows; i++) {
            Li_hd = 0; //被重置为0，用于标记当前行的第一个有效网格单元 是否已处理
            n = 0; //被重置为0，用于计算当前行中无效网格单元的总列数。
            Mat Li;
            for (int j = 0; j < quadcols; j++) {
                int lineN = LineInfo[i][j].rows;
                if (lineN == 0) {  //为0代表没有有效直线段
                    if (Li_hd != 0) {
                        Mat x = Mat::zeros(Li.rows, 8, CV_32FC1);
                        hconcat(Li, x, Li); //右拼接一个全0矩阵
                    }else {
                        n = n + 8;//累加无效网格单元的总列数
                    }
                }else {
                    if (Li_hd == 0) { //根据 n 的值初始化 Li
                        if (n != 0) {
                            Li = Mat::zeros(EL1[i][j].rows, n, CV_32FC1);
                            hconcat(Li, EL1[i][j], Li);
                        }else {
                            Li = EL1[i][j].clone();
                        }
                        Li_hd++;
                    }else {
                        BlockDiag(Li, EL1[i][j], Li);
                    }
                }
            }
            if (L_hd == 0 && Li_hd == 0) { //整行没有有效网格单元，累加无效网格单元的列数m 
                m = m + n;
            }else if (L_hd == 0 && Li_hd != 0) {
                if (m != 0) {
                    L = Mat::zeros(Li.rows, m, CV_32FC1);
                    hconcat(L, Li, L);
                }else {
                    L = Li;
                }
                L_hd++;
            }else {
                if (Li_hd == 0) {
                    Li = Mat::zeros(L.rows, n, CV_32FC1);
                    hconcat(L, Li, L);
                }else {
                    BlockDiag(L, Li, L);
                }
            }
        }
        double N = quadrows*quadcols;
        //固定θ角度 优化 V（动点）
        double lambl = 3;
        double lambB = 100000000;
        Mat E;
//组合能量项到总能量函数中
        MatrixXd S_matrix, Q_matrix, L_matrix, EB_matrix, Es, EL, EB_t;
        cv2eigen(S, S_matrix);
        cv2eigen(Q, Q_matrix);
        cv2eigen(L, L_matrix);
        cv2eigen(EB, EB_matrix);
        SparseMatrix<double> S1 = S_matrix.sparseView();
        SparseMatrix<double> Q1 = Q_matrix.sparseView();
        SparseMatrix<double> L1 = L_matrix.sparseView();
        SparseMatrix<double> EB1 = EB_matrix.sparseView();
        Es = (1.0 / N) * S1 * Q1;// 计算形状保持矩阵 S
        EL = (lambl / NL) * L1 * Q1; // 计算线段保持矩阵 L
        EB_t = lambB * EB1; // 计算边界约束矩阵 EB

        // 组合优化矩阵
        Mat z1, z2, z3;
        eigen2cv(Es, z1);
        eigen2cv(EL, z2);
        eigen2cv(EB_t, z3);
        Mat Z; //总能量Z
        vconcat(z1, z2, Z);
        vconcat(Z, z3, Z);
//构建总能量矩阵并求解
        //将边界约束纳入优化目标中
        cv::vconcat(Mat::zeros(Z.rows - B.rows, 1, CV_32FC1), lambB * B, E);//纵向合并
        MatrixXd Z_matrix, E_matrix, A_matrix, b_matrix;//
        cv2eigen(Z, Z_matrix);
        cv2eigen(E, E_matrix);
        SparseMatrix<double> Z1 = Z_matrix.sparseView();//拼接后的总能量矩阵（稀疏矩阵）
        SparseMatrix<double> E1 = E_matrix.sparseView();//这是加权后的边界约束矩阵 B 转换为稀疏矩阵的形式
        A_matrix = Z1.transpose() * Z1;//这是优化目标的系数矩阵，表示总能量函数中的二次项。
        b_matrix = Z1.transpose() * E1;//这是优化目标的常数项

        //通过求解优化问题，得到新顶点的位置
        SparseMatrix<double> A = A_matrix.sparseView();
        SparseLU<SparseMatrix<double>> solver;
        solver.compute(A);
        // 优化求解，得到新的网格顶点位置
        MatrixXd x_matrix = solver.solve(b_matrix); //通过俩solver，得到线性方程 Ax = b，并求解用于更新网格顶点
        cv::Mat new1;
        eigen2cv(x_matrix, new1);

        // 更新网格顶点位置
        int sum1 = 0;//sum1 用于统计每个像素位置的累加次数
        for (int i = 0; i < y_num; i++){
            for (int j = 0; j < x_num; j++){
                new_xgrid.at<float>(i, j) = (int)new1.at<double>(sum1, 0) - 1;
                new_ygrid.at<float>(i, j) = (int)new1.at<double>(sum1 + 1, 0) - 1;
                sum1 += 2;
            }
        }

        //固定 V 网格优化 θ角度
        //优化过程中约束线段保持一致的倾斜角度
        double bin_num[51];
        double dire_sum[51]; //每个bin中的总角度变化量
        for (int i = 0; i < quadrows; i++){
            for (int j = 0; j < quadcols; j++){
                int lineN = LineInfo[i][j].rows;
                quadVertex(i, j, new_ygrid, new_xgrid, vx, vy);//获取方块顶点坐标
                for (int k = 0; k < lineN; k++){
                    if (jump[i][j][k]) continue;
                    Mat T1 = TT[i][j][k * 2];
                    Mat T2 = TT[i][j][k * 2 + 1];
                    Mat V(8, 1, CV_32FC1);
                    V.at<float>(0, 0) = vx.at<float>(0, 0);
                    V.at<float>(1, 0) = vy.at<float>(0, 0);
                    V.at<float>(2, 0) = vx.at<float>(0, 1);
                    V.at<float>(3, 0) = vy.at<float>(0, 1);
                    V.at<float>(4, 0) = vx.at<float>(1, 0);
                    V.at<float>(5, 0) = vy.at<float>(1, 0);
                    V.at<float>(6, 0) = vx.at<float>(1, 1);
                    V.at<float>(7, 0) = vy.at<float>(1, 1);
                    Mat st = T1 * V;
                    Mat en = T2 * V;
                    double oritheta = quad_theta[i][j][k];
                    double theta = atan((st.at<float>(1, 0) - en.at<float>(1, 0)) / (st.at<float>(0, 0) - en.at<float>(0, 0)));
                    //优化过程中的线段倾斜角度的更新
                    double change = theta - oritheta;//角度的变化量
                    if (isnan(change)) {
                        continue;
                    }
                    if (change > pi / 2) {
                        change -= pi;
                    }
                    if (change < -pi / 2) {
                        change += pi;
                    }
                    int bin = quad_bin[i][j][k];//线段所在的盒子编号
                    bin_num[bin]++;// 增加盒子中的线段计数
                    dire_sum[bin] += change;// 累加角度变化量
                }
            }
        }
        for (int i = 0; i < 50; i++){
            if (bin_num[i] == 0) {
                dire_sum[i] = 0;
            }else {
                dire_sum[i] = 1.0 * dire_sum[i] / bin_num[i];
            }
        }
        // 在优化循环中，更新线段的倾斜角度-line preservation
        for (int i = 0; i < quadrows; i++){
            for (int j = 0; j < quadcols; j++){
                int lineN = LineInfo[i][j].rows;
                // 更新每个网格单元中线段的倾斜角度
                for (int k = 0; k < lineN; k++){
                    LineInfo[i][j].at<float>(k, 4) = dire_sum[quad_bin[i][j][k]];
                }
            }
        }

        Mat gridmask2,imageGrided2;
        drawGridmask(new_ygrid,new_xgrid,rows,cols,gridmask2);
        drawGrid(gridmask2,img,imageGrided2);
        cv::imshow("Adjusted grid",imageGrided2);
    }

    //4生成最后的图像
    Mat vx1(2, 2, CV_32FC1);
    Mat vx2(2, 2, CV_32FC1);
    Mat vy1(2, 2, CV_32FC1);
    Mat vy2(2, 2, CV_32FC1);
    // 创建用于存储输出图像和计数器的矩阵
    Mat outimg(img.rows, img.cols, CV_32SC3);
    int** sum = new int* [img.rows];
    for (int i = 0; i < img.rows; i++) {
        sum[i] = new int[img.cols];
        for (int j = 0; j < img.cols; j++) {
            sum[i][j] = 0;
        }
    }

#pragma omp parallel for collapse(2) reduction(+:sx, sy)
    // 遍历每个网格单元，应用变换并累加像素值
    for (int i = 0; i < quadrows; i++){
        for (int j = 0; j < quadcols; j++){
            // 获取当前网格单元的顶点
            quadVertex(i, j, new_ygrid, new_xgrid, vx1, vy1);
            Mat Vq(8, 1, CV_32FC1);//8行一列
            //第一个顶点xy
            Vq.at<float>(0, 0) = vx1.at<float>(0, 0);
            Vq.at<float>(1, 0) = vy1.at<float>(0, 0);
            Vq.at<float>(2, 0) = vx1.at<float>(0, 1);
            Vq.at<float>(3, 0) = vy1.at<float>(0, 1);
            Vq.at<float>(4, 0) = vx1.at<float>(1, 0);
            Vq.at<float>(5, 0) = vy1.at<float>(1, 0);
            //第四个顶点xy
            Vq.at<float>(6, 0) = vx1.at<float>(1, 1);
            Vq.at<float>(7, 0) = vy1.at<float>(1, 1);
            
            // 获取原始网格单元的顶点
            quadVertex(i, j, warp_ygrid, warp_xgrid, vx2, vy2);
            Mat V2(8, 1, CV_32FC1);
            V2.at<float>(0, 0) = vx2.at<float>(0, 0);
            V2.at<float>(1, 0) = vy2.at<float>(0, 0);
            V2.at<float>(2, 0) = vx2.at<float>(0, 1);
            V2.at<float>(3, 0) = vy2.at<float>(0, 1);
            V2.at<float>(4, 0) = vx2.at<float>(1, 0);
            V2.at<float>(5, 0) = vy2.at<float>(1, 0);
            V2.at<float>(6, 0) = vx2.at<float>(1, 1);
            V2.at<float>(7, 0) = vy2.at<float>(1, 1);
            
            // 计算当前网格单元的边界和长度
            double minx = min(min(Vq.at<float>(0, 0), Vq.at<float>(2, 0)), min(Vq.at<float>(4, 0), Vq.at<float>(6, 0)));
            double maxx = max(max(Vq.at<float>(0, 0), Vq.at<float>(2, 0)), max(Vq.at<float>(4, 0), Vq.at<float>(6, 0)));
            double miny = min(min(Vq.at<float>(1, 0), Vq.at<float>(3, 0)), min(Vq.at<float>(5, 0), Vq.at<float>(7, 0)));
            double maxy = max(max(Vq.at<float>(1, 0), Vq.at<float>(3, 0)), max(Vq.at<float>(5, 0), Vq.at<float>(7, 0)));
            double lenx = maxx - minx;
            double leny = maxy - miny;
            double lx = 1.0 / (2 * lenx);
            double ly = 1.0 / (2 * leny);
            
            // 遍历网格单元内的像素，进行插值
            for (double y = 0; y < 1; y += ly){
                for (double x = 0; x < 1; x += lx){
#pragma omp parallel for
                    double k1 = 1 - y - x + y * x;
                    double k2 = x - y * x;
                    double k3 = y - y * x;
                    double k4 = y * x;
                    Mat T(2, 8, CV_32FC1);
                    T.at<float>(0, 0) = k1;
                    T.at<float>(1, 0) = 0;
                    T.at<float>(0, 1) = 0;
                    T.at<float>(1, 1) = k1;
                    T.at<float>(0, 2) = k2;
                    T.at<float>(1, 2) = 0;
                    T.at<float>(0, 3) = 0;
                    T.at<float>(1, 3) = k2;
                    T.at<float>(0, 4) = k3;
                    T.at<float>(1, 4) = 0;
                    T.at<float>(0, 5) = 0;
                    T.at<float>(1, 5) = k3;
                    T.at<float>(0, 6) = k4;
                    T.at<float>(1, 6) = 0;
                    T.at<float>(0, 7) = 0;
                    T.at<float>(1, 7) = k4;
                    // 应用变换矩阵T，获取新旧网格单元顶点坐标
                    Mat afterTrans = T * Vq;
                    Mat beforeTrans = T * V2;
                    int x1 = (int)afterTrans.at<float>(0, 0);
                    int y1 = (int)afterTrans.at<float>(1, 0);
                    int x2 = (int)beforeTrans.at<float>(0, 0);
                    int y2 = (int)beforeTrans.at<float>(1, 0);

                    // 检查坐标是否在边界内
                    if (y1 < 0 || x1 < 0 || y2 < 0 || x2 < 0) { 
                        continue;
                    }
                    if (y1 >= img.rows || x1 >= img.cols || y2 >= img.rows || x2 >= img.cols) {
                        continue;
                    }

                    // 累加像素值
                    outimg.at<Vec3i>(y1, x1) += img.at<Vec3b>(y2, x2);
                    sum[y1][x1]++;
                }
            }
        }
    }
    // 归一化（防止超255）输出图像的像素值
#pragma omp parallel for collapse(2)
    for (int i = 0; i < img.rows; i++){
        for (int j = 0; j < img.cols; j++){
            if (sum[i][j] == 0) {
                continue;
            }
            outimg.at<Vec3i>(i, j)[0] /= sum[i][j];
            outimg.at<Vec3i>(i, j)[1] /= sum[i][j];
            outimg.at<Vec3i>(i, j)[2] /= sum[i][j];
        }
    }
    // 释放累加数组
    delete(sum);
    // 调整图像大小并复制到输出图像
    outimg.convertTo(outimg, CV_8U);
    output = outimg.clone();

    
}
