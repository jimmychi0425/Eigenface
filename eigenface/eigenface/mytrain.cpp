#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	//能量百分比a
	double a;
	//model文件名
	char xmlfile[100];
	if (argc == 3)
	{
		sscanf_s(argv[1], "%lf", &a);
		sprintf_s(xmlfile, argv[2]);
	}
	else
	{
		a = 0.99;
		sprintf_s(xmlfile, "D:\\eigenface.xml");
	}
	//设定训练集的人脸个数
	int N = 40;
	char filename[100];
	//以样本中一张图获取图片的行数和列数
	Mat test = imread("D:\\att\\s1\\1.pgm", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imageset(test.rows*test.cols, N, CV_32FC1);
	//从文件夹中读取被训练的人脸并将其存入imageset中
	for (int k = 1; k <= N; k++)
	{
		sprintf_s(filename, "D:\\att\\s%d\\1.pgm", k);
		Mat image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		equalizeHist(image, image);
		for (int i = 0; i < image.rows; i++)
		{
			for (int j = 0; j < image.cols; j++)
			{
				imageset.at<float>(i*image.cols + j, k - 1) = image.at<uchar>(i, j);
			}
		}
	}
	
	//计算平均脸
	Mat avgface = Mat::zeros(test.rows*test.cols, 1, CV_32FC1);
	for (int i = 0; i < imageset.rows; i++)
	{
		for (int j = 0; j < imageset.cols; j++)
		{
			avgface.at<float>(i, 0) = avgface.at<float>(i, 0) + imageset.at<float>(i, j);
		}
		avgface.at<float>(i, 0) = 1 / N*avgface.at<float>(i, 0);
	}
	
	//原集合减去平均脸
	Mat DeAvgImageset = imageset.clone();
	for (int i = 0; i < DeAvgImageset.rows; i++)
	{
		for (int j = 0; j < DeAvgImageset.cols; j++)
		{
			DeAvgImageset.at<float>(i, j) = DeAvgImageset.at<float>(i, j) - avgface.at<float>(i, 0);
		}
	}
	
	//求取N*N的协方差矩阵并计算其特征值特征向量
	Mat covarmat = DeAvgImageset.t()*DeAvgImageset;
	Mat eigenVector;
	Mat eigenValues;
	eigen(covarmat, eigenValues, eigenVector);
	eigenVector = eigenVector.t();
	//计算所有特征值的总和
	double sum = 0;
	for (int i = 0; i < eigenValues.rows; i++)
	{
		sum = sum + eigenValues.at<float>(i, 0);
	}

	//依靠能量百分比确定是前多少个特征向量被留下
	int k;
	double ksum = 0;
	for ( k = 0; k < eigenValues.rows; k++)
	{
		ksum = ksum + eigenValues.at<float>(k, 0);
		if (ksum >= (a*sum))
			break;
	}
	Mat selected = eigenVector.colRange(0, k).clone();
	Mat eigenface = DeAvgImageset*selected;
	normalize(eigenface, eigenface, 255, 0, NORM_MINMAX);
	//展示前10张特征脸拼接起来的图片
	Mat show(2 * test.rows, 5 * test.cols, CV_8UC1);
	for (int i = 0; i < test.rows; i++)
	{
		for (int j = 0; j < test.cols; j++)
		{
			show.at<uchar>(i, j) = eigenface.at<float>(i*test.cols + j, 0);
			show.at<uchar>(i, test.cols + j) = eigenface.at<float>(i*test.cols + j, 1);
			show.at<uchar>(i, 2 * test.cols + j) = eigenface.at<float>(i*test.cols + j, 2);
			show.at<uchar>(i, 3 * test.cols + j) = eigenface.at<float>(i*test.cols + j, 3);
			show.at<uchar>(i, 4 * test.cols + j) = eigenface.at<float>(i*test.cols + j, 4);
		}
	}
	for (int i = 0; i < test.rows; i++)
	{
		for (int j = 0; j < test.cols; j++)
		{
			show.at<uchar>(test.rows + i, j) = eigenface.at<float>(i*test.cols + j, 5);
			show.at<uchar>(test.rows + i, test.cols + j) = eigenface.at<float>(i*test.cols + j, 6);
			show.at<uchar>(test.rows + i, 2 * test.cols + j) = eigenface.at<float>(i*test.cols + j, 7);
			show.at<uchar>(test.rows + i, 3 * test.cols + j) = eigenface.at<float>(i*test.cols + j, 8);
			show.at<uchar>(test.rows + i, 4 * test.cols + j) = eigenface.at<float>(i*test.cols + j, 9);
		}
	}
	imshow("show", show);
	//将特征脸的矩阵mat保存到xml文件中。
	FileStorage fs(xmlfile, FileStorage::WRITE);
	fs << "eigenface" << eigenface;
	fs.release();
	waitKey();
	return 0;
}