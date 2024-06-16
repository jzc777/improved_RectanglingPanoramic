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
//������ʵ��ȫ�ֱ��εĺ��ĺ���������������ͼ�� img��λ��ͼ displacement������ mask���Լ����ͼ�� output��
void global_warp(Mat& img, Mat& displacement, Mat& mask, Mat& output,double lambl,double& draw){
    if (img.empty() || displacement.empty() || mask.empty()) {
        std::cerr << "һ�������������Ϊ��!" << std::endl;
        return;
    }
    if (displacement.rows != img.rows || displacement.cols != img.cols) {
        std::cerr << "λ�ƾ����ά��������ͼ��ƥ��!" << std::endl;
        return;
    }
    if (mask.rows != img.rows || mask.cols != img.cols) {
        std::cerr << "�����ά��������ͼ��ƥ��!" << std::endl;
        return;
    }

    int cols = img.cols;
    int rows = img.rows;
    //��ʼ�����񣬽�ͼ�񻮷ֳ�20*20�����񣬲���ʼ����������-hd
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
    
    //������κ������displacement �洢��ÿ�����ص�λ��ʸ����ͨ����ȥλ��ʸ������������κ�����񶥵�λ��
#pragma omp parallel for collapse(2)
    Mat warp_xgrid = xgrid.clone();
    Mat warp_ygrid = ygrid.clone();
    // �ڷ���warp_xgrid��warp_ygrid֮ǰ���ά��
    if (warp_xgrid.rows != warp_ygrid.rows || warp_xgrid.cols != warp_ygrid.cols) {
        std::cerr << "warp_xgrid��warp_ygrid��ά�Ȳ�ƥ��!" << std::endl;
        return;
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < warp_xgrid.rows; i++){
        for (int j = 0; j < warp_xgrid.cols; j++){
            warp_xgrid.at<float>(i, j) = xgrid.at<float>(i, j) - displacement.at<Vec2f>(ygrid.at<float>(i, j), xgrid.at<float>(i, j))[1];//[1]����y����λ��
            warp_ygrid.at<float>(i, j) = ygrid.at<float>(i, j) - displacement.at<Vec2f>(ygrid.at<float>(i, j), xgrid.at<float>(i, j))[0];//[0]����x����λ��
        }
    }

    // ������κ�����������߶�
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

    //���Ʊ��κ������
        Mat gridmask1,imageGrided1;
        drawGridmask(warp_ygrid,warp_xgrid,rows,cols,gridmask1);
        drawGrid(gridmask1,img,imageGrided1);
        cv::imshow("Mesh Warped Backward",imageGrided1);

        for (int i = 0; i < warp_xgrid.rows; i++) {
            for (int j = 0; j < warp_xgrid.cols; j++) {
                int idx_y = ygrid.at<float>(i, j);
                int idx_x = xgrid.at<float>(i, j);
                if (idx_y < 0 || idx_y >= displacement.rows || idx_x < 0 || idx_x >= displacement.cols) {
                    std::cerr << "λ�ƾ�������Խ�磺idx_y = " << idx_y << ", idx_x = " << idx_x << std::endl;
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
    //��״����Es Shape Preservation-hd
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
            // ���Aq����
            for (int k = 0; k < 4; k++) { //ͨ��Aq��ӵõ���Vq�洢��ֵ
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
            Es1[i][j] = Aq * (Aq.t() * Aq).inv() * Aq.t() - I;//����Es

        }
    }
    //��״�����һ���ֽ���-hd
    int quadrows = 19;
    int quadcols = 19;
    Mat S;//���ڴ洢���յ���״�������������
    int S_hd = 0;
    int Si_hd = 0;
    for (int i = 0; i < quadrows; i++) {
        Mat Si;//���ڴ洢һ������Ԫ����״�������������
        Si_hd = 0;
        for (int j = 0; j < quadcols; j++) {
            if (Si_hd == 0) {   //��� Si_hd Ϊ 0����ֱ�ӽ� Es[i][j] ��ֵ�� Si������ Si_dir ��Ϊ 1����ʾ�Ѹ�ֵ
                Si = Es1[i][j];
                Si_hd++;
            }
            else {
                BlockDiag(Si, Es1[i][j], Si); //��� Si_hd��Ϊ 0���� Es[i][j] �뵱ǰ�� Si ������п�Խ�ƴ�ӣ�����Դ洢�� Si �С�
            }
        }
        if (S_hd == 0) { //��� S_hd Ϊ 0����ֱ�ӽ� Si ��ֵ�� S������ S_dir ��Ϊ 1����ʾ��ʼ�����Ѹ�ֵ��
            S = Si;
            S_hd++;
        }
        else {//��� S_hd ��Ϊ 0���� Si �뵱ǰ�� S ������п�Խ�ƴ�ӣ�����Դ洢�� S �С�
            BlockDiag(S, Si, S);
        }
    }

    //�߽�Լ��-hd
    int total = x_num * y_num;
    std::vector<float> B_mat(total * 2, 0);  // ���ڱ���߽綥�������λ��
    std::vector<float> BW(total * 2, 0); // ���ڱ���߽���Լ����������Ȩ��
    //����2����ΪҪ�洢x��y����ֵ

    // ������߽���ϱ߽��Լ��
    for (int i = 0; i < total * 2; i += x_num * 2) {
        B_mat[i] = 1;
        BW[i] = 1;
    }
    for (int i = 1; i < x_num * 2; i += 2) {
        B_mat[i] = 1;
        BW[i] = 1;
    }
    // �����ұ߽��Լ��
    for (int i = x_num * 2 - 2; i < total * 2; i += x_num * 2) {
        B_mat[i] = img.cols;//ϣ�������ұ�
        BW[i] = 1;
    }
    // �����±߽��Լ��
    for (int i = total * 2 - x_num * 2 + 1; i < total * 2; i += 2) {
        B_mat[i] = img.rows;//ϣ�������±�
        BW[i] = 1;
    }

    // ��һά����ת��Ϊ OpenCV ����
    //��Ȩ��ı߽�Լ������ B
    Mat B(total * 2, 1, CV_32FC1, B_mat.data());//�����˱߽綥������λ�õľ���ÿ������λ�ó�����Ӧ��Ȩ��
    Mat BI(total * 2, 1, CV_32FC1, BW.data());//Ȩ����Ϣ
    //�߽�Լ����������EB
    Mat EB = Mat::diag(BI);//�ԽǾ��󣬰����˱߽���Ȩ����Ϣ����������������Ż������б��ֱ߽���λ��

    Mat img_gray;
    cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    //line preservation ֱ�߶α���EL
    //LSD�߶μ��ʹ���ԭ����ͨ�����ص��ݶ���Ϣ�����
    vector<Vec4f>lines;
    Ptr<LineSegmentDetector>lsd = createLineSegmentDetector(LSD_REFINE_STD);
    lsd->detect(img_gray, lines);
    Mat drawnLines(img_gray);
    lsd->drawSegments(drawnLines, lines);
    cv::imshow("LSD work", drawnLines);//��ӡLSD�Ľ��
    //����ָ��ʼ��
    int line_num = lines.size();
    int num[20][20];
    memset(num, 0, sizeof(num));
    Mat** LineInfo = new Mat * [19];
    for (int i = 0; i < 19; i++) {
        LineInfo[i] = new Mat[x_num - 1];
    }
    //�洢ֱ�߶ε���Ϣ
    // �������м�⵽��ֱ�߶�
    for (int i = 0; i < line_num; i++) {
        Mat line1(2, 2, CV_32SC1); // ����һ��2x2����洢ֱ�߶ε������յ�����
        line1.at<int>(0, 1) = lines[i][0];
        line1.at<int>(0, 0) = lines[i][1];
        line1.at<int>(1, 1) = lines[i][2];
        line1.at<int>(1, 0) = lines[i][3];
        // ���ֱ�߶ε������յ��Ƿ������������ڣ������������
        if ((mask.at<uchar>(line1.at<int>(0, 0), line1.at<int>(0, 1)) == 1) ||(mask.at<uchar>(line1.at<int>(1, 0), line1.at<int>(1, 1)) == 1)) {
            continue;
        }

        // ����λ��ʸ������ֱ�߶ε��������
        int outy1 = line1.at<int>(0, 0) + displacement.at<Vec2f>(line1.at<int>(0, 0), line1.at<int>(0, 1))[1];
        int outx1 = line1.at<int>(0, 1) + displacement.at<Vec2f>(line1.at<int>(0, 0), line1.at<int>(0, 1))[0];
        // ��������Ԫ�Ŀ�Ⱥ͸߶�
        float gw = (img.cols - 1) / (gridcols - 1);
        float gh = (img.rows - 1) / (gridrows - 1);

        // ȷ��ֱ�߶�������ڵ�����Ԫ
        int stgrid_y = 1.0 * outy1 / gh;
        int stgrid_x = 1.0 * outx1 / gw;
        int now_x = stgrid_x; // ��ǰ����Ԫ��x����
        int now_y = stgrid_y; // ��ǰ����Ԫ��y����
        Mat vx(2, 2, CV_32FC1); // �洢���񶥵��x����
        Mat vy(2, 2, CV_32FC1); // �洢���񶥵��y����
        int dir[4][2] = { {1, 0}, {0, 1}, {-1, 0}, {0, -1} }; // ����Ԫ���ĸ�����

        Mat pst(1, 2, CV_32FC1); // �洢ֱ�߶ε����
        Mat pen(1, 2, CV_32FC1); // �洢ֱ�߶ε��յ�
        Mat pnow(1, 2, CV_32FC1); // ��ǰ����ĵ�
        pst.at<float>(0, 0) = line1.at<int>(0, 0);
        pst.at<float>(0, 1) = line1.at<int>(0, 1);
        pen.at<float>(0, 0) = line1.at<int>(1, 0);
        pen.at<float>(0, 1) = line1.at<int>(1, 1);
        pnow = pen.clone();
        int to;
        Mat p(2, 2, CV_32FC1);
        int last = -1, count = 0;

        // ��������Ԫ������ֱ�߶���Ϣ
        while (true) {
            count++;
            if (count > 1) {
                break;
            }
            int hd = 0;//�ж��Ƿ��ҵ�����
            if (now_y >= 19 || now_x >= x_num - 1 || now_x < 0 || now_y < 0) {
                std::cerr << "����Խ�磺now_x = " << now_x << ", now_y = " << now_y << std::endl;
                break;
            }
            // ��ȡ��ǰ����Ԫ�Ķ�������
            quadVertex(now_y, now_x, warp_ygrid, warp_xgrid, vx, vy);

            // ���ֱ�߶��Ƿ��ڵ�ǰ����Ԫ��
            int isin = checkIsIn(vy, vx, pst.at<float>(0, 1), pst.at<float>(0, 0), pen.at<float>(0, 1), pen.at<float>(0, 0));
            if (isin == 0) {
                // ������ڵ�ǰ����Ԫ�ڣ�����㽻�㲢��������Ԫ
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
            // ��ֱ�߶���Ϣ�洢����ǰ����Ԫ
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
    //EL���˽���

    // �����ʼ�����߶η���

    int quadID;
    int topleftverterID;
    Mat Q = Mat::zeros(8 * quadrows * quadcols, 2 * y_num * x_num, CV_32FC1);
    for (int i = 0; i < quadrows; i++){
        for (int j = 0; j < quadcols; j++){//��ʼ��Q����
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

    
    
    //�ǶȺ��Ӧ�-hd
    double delta =pi/50;//���ֳ�50��
    vector<double>quad_theta[19][19]; // �洢ÿ������Ԫ��ƽ���ǶȦ�
    vector<int>quad_bin[19][19]; // �洢ÿ������Ԫ���߶ζ�Ӧ�ĺ��ӱ��
    // ����ÿ������Ԫ�������߶νǶȲ����䵽����
    for (int i = 0; i < quadrows; i++){
        for (int j = 0; j < quadcols; j++){
            // ��ȡ�߶εĴ���
            Mat quadseg = LineInfo[i][j];
            int lineN = quadseg.rows;
            quad_bin[i][j].clear();
            quad_theta[i][j].clear();
            for (int k = 0; k < lineN; k++){
                // ��ȡ�߶ε������յ�
                int pst_y = quadseg.at<float>(k, 0);
                int pst_x = quadseg.at<float>(k, 1);
                int pen_y = quadseg.at<float>(k, 2);
                int pen_x = quadseg.at<float>(k, 3);

                // �����߶ε���б�Ƕ�
                double angle;
                if (pst_x == pen_x){
                    angle = pi / 2;   //90��
                }else {
                    angle = atan(double(pst_y - pen_y) / (pst_x - pen_x));//����e�ı�atan������ýǶ�
                }

                if (angle < 0){
                    angle += 2 * pi;
                }

                // �õ��ȣ����ӷ���
                int theta = (int)((angle + pi/2)/delta);

                // ���߶εĽǶȺͺ��ӱ�Ŵ洢����
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

    //�Ż�ѭ��
    int iteration = 1;//��������
    double NL;
    int jump[20][20][110];
    Mat new_xgrid(y_num, x_num, CV_32FC1);
    Mat new_ygrid(y_num, x_num, CV_32FC1);
    for (int i = 0; i < iteration; i++){//ѭ�������߶���Ϣ���£������Ż����Σ��Ż�EL
        NL = 0;
        memset(jump, 0, sizeof(jump));
        int EL_dir[120][120];
        vector<Mat>TT[120][120];
        memset(EL_dir, 0, sizeof(EL_dir));
        // 1. �����߶���Ϣ
        // ����ÿ������Ԫ������ֱ�߶���Ϣ
        for (int i = 0; i < quadrows; i++){  
            for (int j = 0; j < quadcols; j++){
                int lineN = LineInfo[i][j].rows;
                NL += lineN;
                for (int k = 0; k < lineN; k++){
                    quadVertex(i, j, warp_ygrid, warp_xgrid, vx, vy);
                    pst.at<float>(0, 0) = LineInfo[i][j].at<float>(k, 0);//���xy
                    pst.at<float>(1, 0) = LineInfo[i][j].at<float>(k, 1);
                    pen.at<float>(0, 0) = LineInfo[i][j].at<float>(k, 2);//�յ�xy
                    pen.at<float>(1, 0) = LineInfo[i][j].at<float>(k, 3);
                    Mat T1, T2;
                    int flgg = 0;
                    getLinTrans(pst.at<float>(0, 0), pst.at<float>(1, 0), vy, vx, T1, flgg);
                    getLinTrans(pen.at<float>(0, 0), pen.at<float>(1, 0), vy, vx, T2, flgg);
                    TT[i][j].push_back(T1);
                    TT[i][j].push_back(T2);

                    //��������e
                    Mat e(2, 1, CV_32FC1);
                    e.at<float>(0, 0) = pen.at<float>(1, 0) - pst.at<float>(1, 0);
                    //��ת����R-hd
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

        //����һ����ԽǾ��� L����������ÿ������Ԫ���߶α������������
        Mat L;
        int L_hd = 0,Li_hd = 0,n,m = 0;
        for (int i = 0; i < quadrows; i++) {
            Li_hd = 0; //������Ϊ0�����ڱ�ǵ�ǰ�еĵ�һ����Ч����Ԫ �Ƿ��Ѵ���
            n = 0; //������Ϊ0�����ڼ��㵱ǰ������Ч����Ԫ����������
            Mat Li;
            for (int j = 0; j < quadcols; j++) {
                int lineN = LineInfo[i][j].rows;
                if (lineN == 0) {  //Ϊ0����û����Чֱ�߶�
                    if (Li_hd != 0) {
                        Mat x = Mat::zeros(Li.rows, 8, CV_32FC1);
                        hconcat(Li, x, Li); //��ƴ��һ��ȫ0����
                    }else {
                        n = n + 8;//�ۼ���Ч����Ԫ��������
                    }
                }else {
                    if (Li_hd == 0) { //���� n ��ֵ��ʼ�� Li
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
            if (L_hd == 0 && Li_hd == 0) { //����û����Ч����Ԫ���ۼ���Ч����Ԫ������m 
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
        //�̶��ȽǶ� �Ż� V�����㣩
        double lambl = 3;
        double lambB = 100000000;
        Mat E;
//��������������������
        MatrixXd S_matrix, Q_matrix, L_matrix, EB_matrix, Es, EL, EB_t;
        cv2eigen(S, S_matrix);
        cv2eigen(Q, Q_matrix);
        cv2eigen(L, L_matrix);
        cv2eigen(EB, EB_matrix);
        SparseMatrix<double> S1 = S_matrix.sparseView();
        SparseMatrix<double> Q1 = Q_matrix.sparseView();
        SparseMatrix<double> L1 = L_matrix.sparseView();
        SparseMatrix<double> EB1 = EB_matrix.sparseView();
        Es = (1.0 / N) * S1 * Q1;// ������״���־��� S
        EL = (lambl / NL) * L1 * Q1; // �����߶α��־��� L
        EB_t = lambB * EB1; // ����߽�Լ������ EB

        // ����Ż�����
        Mat z1, z2, z3;
        eigen2cv(Es, z1);
        eigen2cv(EL, z2);
        eigen2cv(EB_t, z3);
        Mat Z; //������Z
        vconcat(z1, z2, Z);
        vconcat(Z, z3, Z);
//�����������������
        //���߽�Լ�������Ż�Ŀ����
        cv::vconcat(Mat::zeros(Z.rows - B.rows, 1, CV_32FC1), lambB * B, E);//����ϲ�
        MatrixXd Z_matrix, E_matrix, A_matrix, b_matrix;//
        cv2eigen(Z, Z_matrix);
        cv2eigen(E, E_matrix);
        SparseMatrix<double> Z1 = Z_matrix.sparseView();//ƴ�Ӻ������������ϡ�����
        SparseMatrix<double> E1 = E_matrix.sparseView();//���Ǽ�Ȩ��ı߽�Լ������ B ת��Ϊϡ��������ʽ
        A_matrix = Z1.transpose() * Z1;//�����Ż�Ŀ���ϵ�����󣬱�ʾ�����������еĶ����
        b_matrix = Z1.transpose() * E1;//�����Ż�Ŀ��ĳ�����

        //ͨ������Ż����⣬�õ��¶����λ��
        SparseMatrix<double> A = A_matrix.sparseView();
        SparseLU<SparseMatrix<double>> solver;
        solver.compute(A);
        // �Ż���⣬�õ��µ����񶥵�λ��
        MatrixXd x_matrix = solver.solve(b_matrix); //ͨ����solver���õ����Է��� Ax = b����������ڸ������񶥵�
        cv::Mat new1;
        eigen2cv(x_matrix, new1);

        // �������񶥵�λ��
        int sum1 = 0;//sum1 ����ͳ��ÿ������λ�õ��ۼӴ���
        for (int i = 0; i < y_num; i++){
            for (int j = 0; j < x_num; j++){
                new_xgrid.at<float>(i, j) = (int)new1.at<double>(sum1, 0) - 1;
                new_ygrid.at<float>(i, j) = (int)new1.at<double>(sum1 + 1, 0) - 1;
                sum1 += 2;
            }
        }

        //�̶� V �����Ż� �ȽǶ�
        //�Ż�������Լ���߶α���һ�µ���б�Ƕ�
        double bin_num[51];
        double dire_sum[51]; //ÿ��bin�е��ܽǶȱ仯��
        for (int i = 0; i < quadrows; i++){
            for (int j = 0; j < quadcols; j++){
                int lineN = LineInfo[i][j].rows;
                quadVertex(i, j, new_ygrid, new_xgrid, vx, vy);//��ȡ���鶥������
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
                    //�Ż������е��߶���б�Ƕȵĸ���
                    double change = theta - oritheta;//�Ƕȵı仯��
                    if (isnan(change)) {
                        continue;
                    }
                    if (change > pi / 2) {
                        change -= pi;
                    }
                    if (change < -pi / 2) {
                        change += pi;
                    }
                    int bin = quad_bin[i][j][k];//�߶����ڵĺ��ӱ��
                    bin_num[bin]++;// ���Ӻ����е��߶μ���
                    dire_sum[bin] += change;// �ۼӽǶȱ仯��
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
        // ���Ż�ѭ���У������߶ε���б�Ƕ�-line preservation
        for (int i = 0; i < quadrows; i++){
            for (int j = 0; j < quadcols; j++){
                int lineN = LineInfo[i][j].rows;
                // ����ÿ������Ԫ���߶ε���б�Ƕ�
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

    //4��������ͼ��
    Mat vx1(2, 2, CV_32FC1);
    Mat vx2(2, 2, CV_32FC1);
    Mat vy1(2, 2, CV_32FC1);
    Mat vy2(2, 2, CV_32FC1);
    // �������ڴ洢���ͼ��ͼ������ľ���
    Mat outimg(img.rows, img.cols, CV_32SC3);
    int** sum = new int* [img.rows];
    for (int i = 0; i < img.rows; i++) {
        sum[i] = new int[img.cols];
        for (int j = 0; j < img.cols; j++) {
            sum[i][j] = 0;
        }
    }

#pragma omp parallel for collapse(2) reduction(+:sx, sy)
    // ����ÿ������Ԫ��Ӧ�ñ任���ۼ�����ֵ
    for (int i = 0; i < quadrows; i++){
        for (int j = 0; j < quadcols; j++){
            // ��ȡ��ǰ����Ԫ�Ķ���
            quadVertex(i, j, new_ygrid, new_xgrid, vx1, vy1);
            Mat Vq(8, 1, CV_32FC1);//8��һ��
            //��һ������xy
            Vq.at<float>(0, 0) = vx1.at<float>(0, 0);
            Vq.at<float>(1, 0) = vy1.at<float>(0, 0);
            Vq.at<float>(2, 0) = vx1.at<float>(0, 1);
            Vq.at<float>(3, 0) = vy1.at<float>(0, 1);
            Vq.at<float>(4, 0) = vx1.at<float>(1, 0);
            Vq.at<float>(5, 0) = vy1.at<float>(1, 0);
            //���ĸ�����xy
            Vq.at<float>(6, 0) = vx1.at<float>(1, 1);
            Vq.at<float>(7, 0) = vy1.at<float>(1, 1);
            
            // ��ȡԭʼ����Ԫ�Ķ���
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
            
            // ���㵱ǰ����Ԫ�ı߽�ͳ���
            double minx = min(min(Vq.at<float>(0, 0), Vq.at<float>(2, 0)), min(Vq.at<float>(4, 0), Vq.at<float>(6, 0)));
            double maxx = max(max(Vq.at<float>(0, 0), Vq.at<float>(2, 0)), max(Vq.at<float>(4, 0), Vq.at<float>(6, 0)));
            double miny = min(min(Vq.at<float>(1, 0), Vq.at<float>(3, 0)), min(Vq.at<float>(5, 0), Vq.at<float>(7, 0)));
            double maxy = max(max(Vq.at<float>(1, 0), Vq.at<float>(3, 0)), max(Vq.at<float>(5, 0), Vq.at<float>(7, 0)));
            double lenx = maxx - minx;
            double leny = maxy - miny;
            double lx = 1.0 / (2 * lenx);
            double ly = 1.0 / (2 * leny);
            
            // ��������Ԫ�ڵ����أ����в�ֵ
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
                    // Ӧ�ñ任����T����ȡ�¾�����Ԫ��������
                    Mat afterTrans = T * Vq;
                    Mat beforeTrans = T * V2;
                    int x1 = (int)afterTrans.at<float>(0, 0);
                    int y1 = (int)afterTrans.at<float>(1, 0);
                    int x2 = (int)beforeTrans.at<float>(0, 0);
                    int y2 = (int)beforeTrans.at<float>(1, 0);

                    // ��������Ƿ��ڱ߽���
                    if (y1 < 0 || x1 < 0 || y2 < 0 || x2 < 0) { 
                        continue;
                    }
                    if (y1 >= img.rows || x1 >= img.cols || y2 >= img.rows || x2 >= img.cols) {
                        continue;
                    }

                    // �ۼ�����ֵ
                    outimg.at<Vec3i>(y1, x1) += img.at<Vec3b>(y2, x2);
                    sum[y1][x1]++;
                }
            }
        }
    }
    // ��һ������ֹ��255�����ͼ�������ֵ
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
    // �ͷ��ۼ�����
    delete(sum);
    // ����ͼ���С�����Ƶ����ͼ��
    outimg.convertTo(outimg, CV_8U);
    output = outimg.clone();

    
}
