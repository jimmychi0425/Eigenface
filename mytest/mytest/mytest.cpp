#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
	//测试图像的文件名
	char inputfile[100];
	//model文件名
	char xmlfile[100];
	if (argc == 3)
	{
		sprintf_s(inputfile, argv[1]);
		sprintf_s(xmlfile, argv[2]);
	}
	else
	{
		sprintf_s(inputfile, "D:\\att\\s1\\3.pgm");
		sprintf_s(xmlfile, "D:\\eigenface.xml");
	}
	//读取特征脸文件
	FileStorage fs(xmlfile, FileStorage::READ);
	Mat eigenface;
	fs["eigenface"] >> eigenface;
	//设定训练集的人脸个数
	int N = 40;
	char filename[100];
	//以样本中一张图获取图片的行数和列数
	Mat test = imread("D:\\att\\s1\\1.pgm", CV_LOAD_IMAGE_GRAYSCALE);
	//从文件夹中读取被训练的人脸并将其存入vector中
	vector<Mat> imageset;
	for (int k = 1; k <= N; k++)
	{
		sprintf_s(filename, "D:\\att\\s%d\\1.pgm", k);
	    Mat image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		equalizeHist(image, image);
		Mat output = Mat::zeros(test.rows*test.cols, 1, CV_32FC1);
	    for (int i = 0; i < image.rows; i++)
		{
			for (int j = 0; j < image.cols; j++)
			{
				output.at<float>(i*image.cols + j, 0) = image.at<uchar>(i, j);
			}
		}
		imageset.push_back(output);
	}
	//读入测试图片并进行直方图均衡化处理
	Mat input = imread(inputfile, CV_LOAD_IMAGE_GRAYSCALE);
	equalizeHist(input, input);
	Mat VecInput(input.rows*input.cols, 1, CV_32FC1);
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			VecInput.at<float>(i*input.cols + j, 0) = input.at<uchar>(i, j);
		}
	}
	//将测试图像投影到特征脸空间
	Mat Finput = eigenface.t()*VecInput;
	normalize(Finput, Finput, 255, 0, NORM_MINMAX);
	//计算测试图像和训练图像之间的欧式距离
	Mat oushi = Mat::zeros(1, N, CV_32FC1);
	double oushijuli = 0;
	float oushijuli2 = 0;
	Mat temp(test.cols*test.rows, 1, CV_32FC1);
	for (int i = 0; i < N; i++)
	{
		temp = eigenface.t()*imageset[i];
		normalize(temp, temp, 255, 0, NORM_MINMAX);
	    oushijuli = norm(Finput, temp, NORM_L2);
		oushijuli2 = (float)oushijuli;
	    oushi.at<float>(0, i) = oushijuli2;
	}
	//通过minMaxLoc函数来找到欧式距离最小的图片
	double min, max;
	Point min_loc, max_loc;
	minMaxLoc(oushi, &min, &max, &min_loc, &max_loc);
	sprintf_s(filename, "D:\\att\\s%d\\1.pgm", min_loc.x + 1);
	//识别结果加入在输入人脸上输出
	char text[100];
	sprintf_s(text, "person %d", min_loc.x + 1);
	cout << input.cols << input.rows << endl;
	putText(input, text, Point(18, 100), CV_FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 255, 0));
	namedWindow("input");
	imshow("input", input);
	//显示最像输入图像的一张人脸
	Mat result = imread(filename);
	namedWindow("result");
	imshow("result", result);
	waitKey();
	return 0;
}