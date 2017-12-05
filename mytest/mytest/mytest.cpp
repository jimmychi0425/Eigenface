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
	//����ͼ����ļ���
	char inputfile[100];
	//model�ļ���
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
	//��ȡ�������ļ�
	FileStorage fs(xmlfile, FileStorage::READ);
	Mat eigenface;
	fs["eigenface"] >> eigenface;
	//�趨ѵ��������������
	int N = 40;
	char filename[100];
	//��������һ��ͼ��ȡͼƬ������������
	Mat test = imread("D:\\att\\s1\\1.pgm", CV_LOAD_IMAGE_GRAYSCALE);
	//���ļ����ж�ȡ��ѵ�����������������vector��
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
	//�������ͼƬ������ֱ��ͼ���⻯����
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
	//������ͼ��ͶӰ���������ռ�
	Mat Finput = eigenface.t()*VecInput;
	normalize(Finput, Finput, 255, 0, NORM_MINMAX);
	//�������ͼ���ѵ��ͼ��֮���ŷʽ����
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
	//ͨ��minMaxLoc�������ҵ�ŷʽ������С��ͼƬ
	double min, max;
	Point min_loc, max_loc;
	minMaxLoc(oushi, &min, &max, &min_loc, &max_loc);
	sprintf_s(filename, "D:\\att\\s%d\\1.pgm", min_loc.x + 1);
	//ʶ�����������������������
	char text[100];
	sprintf_s(text, "person %d", min_loc.x + 1);
	cout << input.cols << input.rows << endl;
	putText(input, text, Point(18, 100), CV_FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 255, 0));
	namedWindow("input");
	imshow("input", input);
	//��ʾ��������ͼ���һ������
	Mat result = imread(filename);
	namedWindow("result");
	imshow("result", result);
	waitKey();
	return 0;
}