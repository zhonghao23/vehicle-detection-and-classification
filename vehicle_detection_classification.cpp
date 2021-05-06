#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/bgsegm.hpp"
#include <opencv2/saliency.hpp>
#include <opencv2/saliency/saliencySpecializedClasses.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <fstream>
#include <string>
#include <utility>
#include <stdlib.h>

using namespace std;
using namespace cv;
using namespace saliency;
using namespace cv::bgsegm;
using namespace dnn;

const int STEP = 8;
const int GABOR_R = 8;
const float WEIGHT_I = 0.2f; 
const float WEIGHT_LAB = 0.2f; 
const float WEIGHT_HSV = 0.3f;
const float WEIGHT_O = 0.3f;

double max(double a, double b, double c) {
	return ((a > b) ? (a > c ? a : c) : (b > c ? b : c));
}
double min(double a, double b, double c) {
	return ((a < b) ? (a < c ? a : c) : (b < c ? b : c));
}

void GaussianSmooth(const vector<double>& inputImg, const int& width, const int& height, const vector<double>& kernel, vector<double>& smoothImg)
{
	int center = int(kernel.size()) / 2;

	int sz = width * height;
	smoothImg.clear();
	smoothImg.resize(sz);
	vector<double> tempim(sz);
	int rows = height;
	int cols = width;

	int index(0);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			double kernelsum(0);
			double sum(0);
			for (int cc = (-center); cc <= center; cc++) {
				if (((c + cc) >= 0) && ((c + cc) < cols)) {
					sum += inputImg[r * cols + (c + cc)] * kernel[center + cc];
					kernelsum += kernel[center + cc];
				}
			}
			tempim[index] = sum / kernelsum;
			index++;
		}
	}
	index = 0;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			double kernelsum(0);
			double sum(0);
			for (int rr = (-center); rr <= center; rr++) {
				if (((r + rr) >= 0) && ((r + rr) < rows)) {
					sum += tempim[(r + rr) * cols + c] * kernel[center + rr];
					kernelsum += kernel[center + rr];
				}
			}
			smoothImg[index] = sum / kernelsum;
			index++;
		}
	}
}

void Normalize(const vector<double>& input, const int& width, const int& height, vector<double>& output, const int& Normrange2) {
	double maxval(0);
	double minval(DBL_MAX);
	int i(0);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (maxval < input[i]) maxval = input[i];
			if (minval > input[i]) minval = input[i];
			i++;
		}
	}
	double range = maxval - minval;
	if (0 == range) range = 1;
	i = 0;
	output.clear();
	output.resize(width * height);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			output[i] = ((Normrange2 * (input[i] - minval)) / range);
			i++;
		}
	}

}

void RGB2HSV(const vector<vector<uint>>& ubuff, vector<double>& hvec, vector<double>& svec, vector<double> & vvec) {
	int sz = int(ubuff.size());
	hvec.resize(sz);
	svec.resize(sz);
	vvec.resize(sz);

	for (int j = 0; j < sz; j++) {
		double h, s, v;
		int sR = ubuff[j][2];
		int sG = ubuff[j][1];
		int sB = ubuff[j][0];

		double R = sR / 255.0;
		double G = sG / 255.0;
		double B = sB / 255.0;

		double cmax = max(R, G, B);
		double cmin = min(R, G, B);
		double diff = cmax - cmin;
		if (cmax == cmin)
			h = 0;
		else if (cmax == R)
			h = fmod((60 * ((G - B) / diff) + 360), 360.0);
		else if (cmax == G)
			h = fmod((60 * ((B - R) / diff) + 120), 360.0);
		else if (cmax == B)
			h = fmod((60 * ((R - G) / diff) + 240), 360.0);
		if (cmax == 0)
			s = 0;
		else
			s = (diff / cmax) * 100;
		v = cmax * 100;

		hvec[j] = h;
		svec[j] = s;
		vvec[j] = v;
	}
}

void calcHSVColourMap(const vector<vector<uint> >& inputimg, const int& width, const int& height, vector<double>& salmap, const bool& normflag) {
	int sz = width * height;
	salmap.clear();
	salmap.resize(sz);

	vector<double> hvec(0), svec(0), vvec(0);
	RGB2HSV(inputimg, hvec, svec, vvec);

	double avgh(0), avgs(0), avgv(0);
	for (int i = 0; i < sz; i++) {
		avgh += hvec[i];
		avgs += svec[i];
		avgv += vvec[i];
	}

	avgh /= sz;
	avgs /= sz;
	avgv /= sz;

	vector<double> shvec(0), ssvec(0), svvec(0);

	vector<double> kernel(0);
	kernel.push_back(1.0);
	kernel.push_back(2.0);
	kernel.push_back(1.0);

	GaussianSmooth(hvec, width, height, kernel, shvec);
	GaussianSmooth(svec, width, height, kernel, ssvec);
	GaussianSmooth(vvec, width, height, kernel, svvec);

	for (int i = 0; i < sz; i++) {
		salmap[i] = (shvec[i] - avgh) * (shvec[i] - avgh) +	
			(ssvec[i] - avgs) * (ssvec[i] - avgs) +	
			(svvec[i] - avgv) * (svvec[i] - avgv);
	}

	if (normflag == true) {
		vector<double> normalized(0);
		Normalize(salmap, width, height, normalized, 255);
		swap(salmap, normalized);
	}
}

void calcVColourMap(const vector<vector<uint> >& inputimg, const int& width, const int& height, vector<double>& salmap, const bool& normflag) {
	int sz = width * height;
	salmap.clear();
	salmap.resize(sz);

	vector<double> hvec(0), svec(0), vvec(0);
	RGB2HSV(inputimg, hvec, svec, vvec);

	double avgh(0), avgs(0), avgv(0);
	for (int i = 0; i < sz; i++) {
		avgh += hvec[i];
		avgs += svec[i];
		avgv += vvec[i];
	}

	avgh /= sz;
	avgs /= sz;
	avgv /= sz;

	vector<double> shvec(0), ssvec(0), svvec(0);

	vector<double> kernel(0);
	kernel.push_back(1.0);
	kernel.push_back(2.0);
	kernel.push_back(1.0);

	GaussianSmooth(hvec, width, height, kernel, shvec);
	GaussianSmooth(svec, width, height, kernel, ssvec);
	GaussianSmooth(vvec, width, height, kernel, svvec);

	for (int i = 0; i < sz; i++) {
		salmap[i] = (svvec[i] - avgv) * (svvec[i] - avgv);
	}

	if (normflag == true) {
		vector<double> normalized(0);
		Normalize(salmap, width, height, normalized, 255);
		swap(salmap, normalized);
	}
}

Mat getHSVColourMap(Mat frame)
{
	//Get colour saliency map
	std::vector<vector<uint>>array(frame.cols * frame.rows, vector<uint>
		(3, 0));

	for (int y = 0; y < frame.rows; y++) {
		for (int x = 0; x < frame.cols; x++) {
			Vec3b color = frame.at<Vec3b>(Point(x, y));
			array[frame.cols * y + x][0] = color[0]; array[frame.cols * y + x]
				[1] = color[1]; array[frame.cols * y + x][2] = color[2];
		}
	}

	vector<double> salmap; bool normflag = true;
	calcHSVColourMap(array, frame.size().width, frame.size().height, salmap, normflag);
	Mat output = Mat(frame.rows, frame.cols, CV_8UC1);
	int k = 0;
	for (int y = 0; y < frame.rows; y++) {
		for (int x = 0; x < frame.cols; x++) {
			output.at<uchar>(Point(x, y)) = int(salmap[k]);
			k++;
		}
	}
	return output;
}

void RGB2LAB2(const vector<vector<uint> >& ubuff, vector<double>& lvec, vector<double>& avec, vector<double>& bvec)
{
	int sz = int(ubuff.size());
	lvec.resize(sz);
	avec.resize(sz);
	bvec.resize(sz);

	for (int j = 0; j < sz; j++) {
		int sR = ubuff[j][2];
		int sG = ubuff[j][1];
		int sB = ubuff[j][0];
		//------------------------
		// sRGB to XYZ conversion
		// (D65 illuminant assumption)
		//------------------------
		double R = sR / 255.0;
		double G = sG / 255.0;
		double B = sB / 255.0;

		double r, g, b;
		if (R <= 0.04045)    r = R / 12.92;
		else                r = pow((R + 0.055) / 1.055, 2.4);
		if (G <= 0.04045)    g = G / 12.92;
		else                g = pow((G + 0.055) / 1.055, 2.4);
		if (B <= 0.04045)    b = B / 12.92;
		else                b = pow((B + 0.055) / 1.055, 2.4);

		double X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
		double Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
		double Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
		//------------------------
		// XYZ to LAB conversion
		//------------------------
		double epsilon = 0.008856;  //actual CIE standard
		double kappa = 903.3;     //actual CIE standard

		double Xr = 0.950456;   //reference white
		double Yr = 1.0;        //reference white
		double Zr = 1.088754;   //reference white

		double xr = X / Xr;
		double yr = Y / Yr;
		double zr = Z / Zr;

		double fx, fy, fz;
		if (xr > epsilon)    fx = pow(xr, 1.0 / 3.0);
		else                fx = (kappa * xr + 16.0) / 116.0;
		if (yr > epsilon)    fy = pow(yr, 1.0 / 3.0);
		else                fy = (kappa * yr + 16.0) / 116.0;
		if (zr > epsilon)    fz = pow(zr, 1.0 / 3.0);
		else                fz = (kappa * zr + 16.0) / 116.0;

		lvec[j] = 116.0 * fy - 16.0;
		avec[j] = 500.0 * (fx - fy);
		bvec[j] = 200.0 * (fy - fz);
	}
}

void calcLABColourMap(const vector<vector<uint> >& inputimg, const int& width, const int& height, vector<double>& salmap, const bool& normflag) {
	int sz = width * height;
	salmap.clear();
	salmap.resize(sz);

	vector<double> lvec(0), avec(0), bvec(0);
	RGB2LAB2(inputimg, lvec, avec, bvec);

	double avgl(0), avga(0), avgb(0);
	for (int i = 0; i < sz; i++) {
		avgl += lvec[i];
		avga += avec[i];
		avgb += bvec[i];
	}

	avgl /= sz;
	avga /= sz;
	avgb /= sz;

	vector<double> slvec(0), savec(0), sbvec(0);

	vector<double> kernel(0);
	kernel.push_back(1.0);
	kernel.push_back(2.0);
	kernel.push_back(1.0);

	GaussianSmooth(lvec, width, height, kernel, slvec);
	GaussianSmooth(avec, width, height, kernel, savec);
	GaussianSmooth(bvec, width, height, kernel, sbvec);

	for (int i = 0; i < sz; i++) {
		salmap[i] = (slvec[i] - avgl) * (slvec[i] - avgl) +
			(savec[i] - avga) * (savec[i] - avga) +
			(sbvec[i] - avgb) * (sbvec[i] - avgb);
	}

	if (normflag == true) {
		vector<double> normalized(0);
		Normalize(salmap, width, height, normalized, 255);
		swap(salmap, normalized);
	}
}

Mat getLABColourMap(Mat frame)
{
	//Get colour saliency map
	std::vector<vector<uint>>array(frame.cols * frame.rows, vector<uint>
		(3, 0));

	for (int y = 0; y < frame.rows; y++) {
		for (int x = 0; x < frame.cols; x++) {
			Vec3b color = frame.at<Vec3b>(Point(x, y));
			array[frame.cols * y + x][0] = color[0];
			array[frame.cols * y + x][1] = color[1];
			array[frame.cols * y + x][2] = color[2];
		}
	}

	vector<double> salmap; bool normflag = true;
	calcLABColourMap(array, frame.size().width, frame.size().height, salmap, normflag);
	Mat output = Mat(frame.rows, frame.cols, CV_8UC1);
	int k = 0;
	for (int y = 0; y < frame.rows; y++) {
		for (int x = 0; x < frame.cols; x++) {
			output.at<uchar>(Point(x, y)) = int(salmap[k]);
			k++;
		}
	}
	return output;
}

Mat operateCenterSurround(const Mat& center, const Mat& surround)
{
	Mat csmap(center.size(), center.type());
	resize(surround, csmap, csmap.size());
	csmap = abs(csmap - center);
	return csmap;
}

vector<Mat> buildCenterSurroundPyramid(const vector<Mat>& pyramid)
{

	vector<Mat> cspyr(6);
	cspyr[0] = operateCenterSurround(pyramid[2], pyramid[5]);
	cspyr[1] = operateCenterSurround(pyramid[2], pyramid[6]);
	cspyr[2] = operateCenterSurround(pyramid[3], pyramid[6]);
	cspyr[3] = operateCenterSurround(pyramid[3], pyramid[7]);
	cspyr[4] = operateCenterSurround(pyramid[4], pyramid[7]);
	cspyr[5] = operateCenterSurround(pyramid[4], pyramid[8]);
	return cspyr;
}

void normalizeRange(Mat& image)
{
	double minval, maxval;
	minMaxLoc(image, &minval, &maxval);

	image -= minval;
	if (minval < maxval)
		image /= maxval - minval;
}

void trimPeaks(Mat& image, int step)
{
	const int w = image.cols;
	const int h = image.rows;

	const double M = 1.0;
	normalizeRange(image);
	double m = 0.0;
	for (int y = 0; y < h - step; y += step)
		for (int x = 0; x < w - step; x += step)
		{
			Mat roi(image, Rect(x, y, step, step));
			double minval = 0.0, maxval = 0.0;
			minMaxLoc(roi, &minval, &maxval);
			m += maxval;
		}
	m /= (w / step - (w%step ? 0 : 1))*(h / step - (h%step ? 0 : 1));
	image *= (M - m)*(M - m);
}

Mat calcITTISaliencyMap(const Mat& image1)
{
	const Mat_<Vec3f> image = image1 / 255.0f;
	Mat image0 = image1;
	const Size ksize = Size(GABOR_R + 1 + GABOR_R, GABOR_R + 1 + GABOR_R);
	const double sigma = GABOR_R / CV_PI;
	const double lambda = GABOR_R + 1;
	const double deg = CV_PI / 8.0;
	//8 orientation gabor kernel
	Mat gabor001 = getGaborKernel(ksize, sigma, deg * 0, lambda, 1.0, 0.0, CV_32F);
	Mat gabor002 = getGaborKernel(ksize, sigma, deg * 1, lambda, 1.0, 0.0, CV_32F);
	Mat gabor003 = getGaborKernel(ksize, sigma, deg * 2, lambda, 1.0, 0.0, CV_32F);
	Mat gabor004 = getGaborKernel(ksize, sigma, deg * 3, lambda, 1.0, 0.0, CV_32F);
	Mat gabor005 = getGaborKernel(ksize, sigma, deg * 4, lambda, 1.0, 0.0, CV_32F);
	Mat gabor006 = getGaborKernel(ksize, sigma, deg * 5, lambda, 1.0, 0.0, CV_32F);
	Mat gabor007 = getGaborKernel(ksize, sigma, deg * 6, lambda, 1.0, 0.0, CV_32F);
	Mat gabor008 = getGaborKernel(ksize, sigma, deg * 7, lambda, 1.0, 0.0, CV_32F);


	const int NUM_SCALES = 9;
	vector<Mat> pyramidI(NUM_SCALES);	//INTENSITY
	//vector<Mat> pyramidRG(NUM_SCALES);	//RG
	//vector<Mat> pyramidBY(NUM_SCALES);	//BY
	//8 Orientation Pyramid
	vector<Mat> pyramid001(NUM_SCALES);
	vector<Mat> pyramid002(NUM_SCALES);
	vector<Mat> pyramid003(NUM_SCALES);
	vector<Mat> pyramid004(NUM_SCALES);
	vector<Mat> pyramid005(NUM_SCALES);
	vector<Mat> pyramid006(NUM_SCALES);
	vector<Mat> pyramid007(NUM_SCALES);
	vector<Mat> pyramid008(NUM_SCALES);

	vector<Mat> pyramidLAB(NUM_SCALES);	//LAB
	vector<Mat> pyramidHSV(NUM_SCALES);	//HSV

	Mat scaled = image;
	for (int s = 0; s < NUM_SCALES; ++s)
	{
		const int w = scaled.cols;
		const int h = scaled.rows;

		//INTENSITY--------------------------------------------------------
		vector<Mat_<float> > colors;
		split(scaled, colors);
		Mat_<float> imageI = (colors[0] + colors[1] + colors[2]) / 3.0f;
		pyramidI[s] = imageI;
		//imshow("intensity", pyramidI[s]);
		//waitKey();

		//RGB--------------------------------------------------------------
		//double minval, maxval;
		//minMaxLoc(imageI, &minval, &maxval);
		//Mat_<float> r(h, w, 0.0f);
		//Mat_<float> g(h, w, 0.0f);
		//Mat_<float> b(h, w, 0.0f);
		//for (int j = 0; j < h; ++j)
		//	for (int i = 0; i < w; ++i)
		//	{
		//		if (imageI(j, i) < 0.1f*maxval)
		//			continue;
		//		r(j, i) = colors[2](j, i) / imageI(j, i);
		//		g(j, i) = colors[1](j, i) / imageI(j, i);
		//		b(j, i) = colors[0](j, i) / imageI(j, i);
		//	}

		////Generation of hue map(negative value is clamped to 0)
		//Mat R = max(0.0f, r - (g + b) / 2);
		//Mat G = max(0.0f, g - (b + r) / 2);
		//Mat B = max(0.0f, b - (r + g) / 2);
		//Mat Y = max(0.0f, (r + g) / 2 - abs(r - g) / 2 - b);
		//pyramidRG[s] = R - G;
		//pyramidBY[s] = B - Y;
		//imshow("rg", pyramidRG[s]);
		//imshow("by", pyramidBY[s]);
		//waitKey();

		//LAB-------------------------------------------------------------------------------
		vector<vector<uint>>array(image0.cols * image0.rows, vector<uint>(3, 0));

		for (int y = 0; y < image0.rows; y++) {
			for (int x = 0; x < image0.cols; x++) {
				Vec3b color = image0.at<Vec3b>(Point(x, y));
				array[image0.cols * y + x][0] = color[0];
				array[image0.cols * y + x][1] = color[1];
				array[image0.cols * y + x][2] = color[2];
			}
		}

		vector<double> salmap; bool normflag = true;
		calcLABColourMap(array, image0.size().width, image0.size().height, salmap, normflag);
		
		
		Mat output = Mat(image0.rows, image0.cols, CV_8UC1);
		int k = 0;
		for (int y = 0; y < image0.rows; y++) {
			for (int x = 0; x < image0.cols; x++) {
				output.at<uchar>(Point(x, y)) = int(salmap[k]);
				k++;
			}
		}
		Mat_<float> LABcolourMap;
		output.convertTo(LABcolourMap, CV_32F, 1.0 / 255.0);
		pyramidLAB[s] = LABcolourMap;
		//imshow("lab", LABcolourMap);
		//waitKey();


		//HSV---------------------------------------------------------------------------------------
		vector<vector<uint>>array1(image0.cols * image0.rows, vector<uint>(3, 0));

		for (int y = 0; y < image0.rows; y++) {
			for (int x = 0; x < image0.cols; x++) {
				Vec3b color1 = image0.at<Vec3b>(Point(x, y));
				array1[image0.cols * y + x][0] = color1[0];
				array1[image0.cols * y + x][1] = color1[1];
				array1[image0.cols * y + x][2] = color1[2];
			}
		}

		vector<double> salmap1; bool normflag1 = true;
		calcHSVColourMap(array1, image0.size().width, image0.size().height, salmap1, normflag1);

		Mat output1 = Mat(image0.rows, image0.cols, CV_8UC1);
		int k1 = 0;
		for (int y = 0; y < image0.rows; y++) {
			for (int x = 0; x < image0.cols; x++) {
				output1.at<uchar>(Point(x, y)) = int(salmap1[k1]);
				k1++;
			}
		}
		Mat_<float> HSVcolourMap;
		output1.convertTo(HSVcolourMap, CV_32F, 1.0 / 255.0);
		pyramidHSV[s] = HSVcolourMap;
		//imshow("pyramidHSV", pyramidHSV[s]);
		//waitKey();

		//V--------------------------------------------------------------------
		vector<vector<uint>>array2(image0.cols * image0.rows, vector<uint>(3, 0));

		for (int y = 0; y < image0.rows; y++) {
			for (int x = 0; x < image0.cols; x++) {
				Vec3b color2 = image0.at<Vec3b>(Point(x, y));
				array2[image0.cols * y + x][0] = color2[0];
				array2[image0.cols * y + x][1] = color2[1];
				array2[image0.cols * y + x][2] = color2[2];
			}
		}

		vector<double> salmap2; bool normflag2 = true;
		calcVColourMap(array2, image0.size().width, image0.size().height, salmap2, normflag2);

		Mat output2 = Mat(image0.rows, image0.cols, CV_8UC1);
		int k2 = 0;
		for (int y = 0; y < image0.rows; y++) {
			for (int x = 0; x < image0.cols; x++) {
				output2.at<uchar>(Point(x, y)) = int(salmap2[k2]);
				k2++;
			}
		}
		Mat_<float> VcolourMap;
		output2.convertTo(VcolourMap, CV_32F, 1.0 / 255.0);

		//V-ORIENTATION--------------------------------------------------------------------
		////Direction map generation (obtained by intensity instead of gray map)
		filter2D(VcolourMap, pyramid001[s], -1, gabor001);
		filter2D(VcolourMap, pyramid002[s], -1, gabor002);
		filter2D(VcolourMap, pyramid003[s], -1, gabor003);
		filter2D(VcolourMap, pyramid004[s], -1, gabor004);
		filter2D(VcolourMap, pyramid005[s], -1, gabor005);
		filter2D(VcolourMap, pyramid006[s], -1, gabor006);
		filter2D(VcolourMap, pyramid007[s], -1, gabor007);
		filter2D(VcolourMap, pyramid008[s], -1, gabor008);

		pyrDown(scaled, scaled);
		pyrDown(image0, image0);
	}


	vector<Mat> cspyrI = buildCenterSurroundPyramid(pyramidI);
	//vector<Mat> cspyrRG = buildCenterSurroundPyramid(pyramidRG);
	//vector<Mat> cspyrBY = buildCenterSurroundPyramid(pyramidBY);

	vector<Mat> cspyr001 = buildCenterSurroundPyramid(pyramid001);
	vector<Mat> cspyr002 = buildCenterSurroundPyramid(pyramid002);
	vector<Mat> cspyr003 = buildCenterSurroundPyramid(pyramid003);
	vector<Mat> cspyr004 = buildCenterSurroundPyramid(pyramid004);
	vector<Mat> cspyr005 = buildCenterSurroundPyramid(pyramid005);
	vector<Mat> cspyr006 = buildCenterSurroundPyramid(pyramid006);
	vector<Mat> cspyr007 = buildCenterSurroundPyramid(pyramid007);
	vector<Mat> cspyr008 = buildCenterSurroundPyramid(pyramid008);
	
	vector<Mat> cspyrLAB = buildCenterSurroundPyramid(pyramidLAB);
	vector<Mat> cspyrHSV = buildCenterSurroundPyramid(pyramidHSV);


	Mat_<float> temp(image.size());
	Mat_<float> conspI(image.size(), 0.0f);
	//Mat_<float> conspC(image.size(), 0.0f);

	Mat_<float> consp001(image.size(), 0.0f);
	Mat_<float> consp002(image.size(), 0.0f);
	Mat_<float> consp003(image.size(), 0.0f);
	Mat_<float> consp004(image.size(), 0.0f);
	Mat_<float> consp005(image.size(), 0.0f);
	Mat_<float> consp006(image.size(), 0.0f);
	Mat_<float> consp007(image.size(), 0.0f);
	Mat_<float> consp008(image.size(), 0.0f);
	Mat_<float> conspLAB(image.size(), 0.0f);
	Mat_<float> conspHSV(image.size(), 0.0f);

	for (int t = 0; t<int(cspyrI.size()); ++t)
	{

		trimPeaks(cspyrI[t], STEP); resize(cspyrI[t], temp, image.size()); conspI += temp;

		//trimPeaks(cspyrRG[t], STEP); resize(cspyrRG[t], temp, image.size()); conspC += temp;
		//trimPeaks(cspyrBY[t], STEP); resize(cspyrBY[t], temp, image.size()); conspC += temp;

		trimPeaks(cspyr001[t], STEP); resize(cspyr001[t], temp, image.size()); consp001 += temp;
		trimPeaks(cspyr002[t], STEP); resize(cspyr002[t], temp, image.size()); consp002 += temp;
		trimPeaks(cspyr003[t], STEP); resize(cspyr003[t], temp, image.size()); consp003 += temp;
		trimPeaks(cspyr004[t], STEP); resize(cspyr004[t], temp, image.size()); consp004 += temp;
		trimPeaks(cspyr005[t], STEP); resize(cspyr005[t], temp, image.size()); consp005 += temp;
		trimPeaks(cspyr006[t], STEP); resize(cspyr006[t], temp, image.size()); consp006 += temp;
		trimPeaks(cspyr007[t], STEP); resize(cspyr007[t], temp, image.size()); consp007 += temp;
		trimPeaks(cspyr008[t], STEP); resize(cspyr008[t], temp, image.size()); consp008 += temp;

		trimPeaks(cspyrLAB[t], STEP); 
		resize(cspyrLAB[t], temp, image.size()); 
		conspLAB += temp;

		trimPeaks(cspyrHSV[t], STEP);
		resize(cspyrHSV[t], temp, image.size());
		conspHSV += temp;
	}

	trimPeaks(consp001, STEP);
	trimPeaks(consp002, STEP);
	trimPeaks(consp003, STEP);
	trimPeaks(consp004, STEP);
	trimPeaks(consp005, STEP);
	trimPeaks(consp006, STEP);
	trimPeaks(consp007, STEP);
	trimPeaks(consp008, STEP);
	Mat_<float> conspO = consp001 + consp002 + consp003 + consp004 + consp005 + consp006 + consp007 + consp008;

	trimPeaks(conspI, STEP);
	//trimPeaks(conspC, STEP);
	trimPeaks(conspO, STEP);

	trimPeaks(conspLAB, STEP);
	trimPeaks(conspHSV, STEP);

	Mat saliency = WEIGHT_I * conspI + WEIGHT_LAB * conspLAB + WEIGHT_O * conspO + WEIGHT_HSV * conspHSV;
	normalizeRange(saliency);

	namedWindow("Intensity", WINDOW_AUTOSIZE);
	imshow("Intensity", conspI);
	namedWindow("LAB color", WINDOW_AUTOSIZE);
	imshow("LAB color", conspLAB);
	namedWindow("Orientation", WINDOW_AUTOSIZE);
	imshow("Orientation", conspO);
	namedWindow("HSV color", WINDOW_AUTOSIZE);
	imshow("HSV color", conspHSV);

	return saliency;
}

Mat K_Means(Mat Input, int K) {

	Mat samples(Input.rows * Input.cols, 1, CV_32F);
	for (int y = 0; y < Input.rows; y++)
		for (int x = 0; x < Input.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x * Input.rows, z) = Input.at<uchar>(y, x);

	Mat labels;
	int attempts = 5, cluster_idx;
	Mat centers;
	int countarray[4] = { 0,0,0,0 };

	kmeans(samples, K, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

	Mat new_image(Input.size(), CV_8UC3);
	Mat Binary = Mat::zeros(Input.rows, Input.cols, CV_8UC1);

	for (int y = 0; y < Input.rows; y++) {
		for (int x = 0; x < Input.cols; x++)
		{
			cluster_idx = labels.at<int>(y + x * Input.rows, 1);

			switch (cluster_idx) {
			case 0: //green
				new_image.at<Vec3b>(y, x)[0] = 43;
				new_image.at<Vec3b>(y, x)[1] = 155;
				new_image.at<Vec3b>(y, x)[2] = 155;
				countarray[0]++;
				break;
			case 1:  //yellow
				new_image.at<Vec3b>(y, x)[0] = 0;
				new_image.at<Vec3b>(y, x)[1] = 211;
				new_image.at<Vec3b>(y, x)[2] = 255;
				countarray[1]++;
				break;
			case 2:  //purple
				new_image.at<Vec3b>(y, x)[0] = 153;
				new_image.at<Vec3b>(y, x)[1] = 51;
				new_image.at<Vec3b>(y, x)[2] = 102;
				countarray[2]++;
				break;
			case 3: //blue
				new_image.at<Vec3b>(y, x)[0] = 255;
				new_image.at<Vec3b>(y, x)[1] = 255;
				new_image.at<Vec3b>(y, x)[2] = 1;
				countarray[3]++;
				break;
			}
		}
	}

	imshow("Color coded", new_image);

	int ascending[4] = { countarray[0],countarray[1], countarray[2], countarray[3] };
	sort(ascending, ascending + 4);
	int maxcluster, secondmaxcluster;
	for (int k = 0; k < 4; k++)
	{
		if (countarray[k] == ascending[3])
			maxcluster = k;
		else if (countarray[k] == ascending[2])
			secondmaxcluster = k;
	}

	for (int x = 0; x < Input.rows; x++) {
		for (int y = 0; y < Input.cols; y++)
		{
			int cluster_idx = labels.at<int>(x + y * Input.rows, 0);
			if (cluster_idx == maxcluster || cluster_idx == secondmaxcluster) { //black
				Binary.at<uchar>(x, y) = 0;
			}
			else { //white
				Binary.at<uchar>(x, y) = 255;
			}
		}
	}
	return Binary;
}

Mat vertical_erosion(Mat image) { //decrease whiteness
	int nb = 8;
	Mat Erosion = Mat::zeros(image.size(), CV_8UC1);
	for (int i = nb; i < image.rows - nb; i++) { //height
		for (int j = nb; j < image.cols - nb; j++) { //width
			Erosion.at<uchar>(i, j) = 255;	
			for (int jj = -nb; jj <= nb; jj++) {
				if (image.at<uchar>(i, j + jj) == 0)
				{
					Erosion.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	return Erosion;
}

Mat vertical_dilation(Mat image) {
	int nb = 20; //12
	Mat Dilation = Mat::zeros(image.size(), CV_8UC1);
	for (int i = nb; i < image.rows - nb; i++) {
		for (int j = nb; j < image.cols - nb; j++) {
			for (int ii = -nb; ii <= nb; ii++) {
				if (image.at<uchar>(i + ii, j) == 255)
				{
					Dilation.at<uchar>(i, j) = 255;
				}
			}
		}
	}
	return Dilation;
}

Mat horizontal_dilation(Mat image) {
	int nb = 12; 
	Mat Dilation = Mat::zeros(image.size(), CV_8UC1);
	for (int i = nb; i < image.rows - nb; i++) {
		for (int j = nb; j < image.cols - nb; j++) {
			for (int jj = -nb; jj <= nb; jj++) {
				if (image.at<uchar>(i, j + jj) == 255)
				{
					Dilation.at<uchar>(i, j) = 255;
				}
			}
		}
	}
	return Dilation;
}

Mat bigNoiseFilter(Mat image) {

	Mat Blob = image.clone();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(Blob, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

	Rect rect;
	Scalar black = CV_RGB(0, 0, 0);
	double area, rectWidth, rectHeight, rectArea, check;
	for (size_t j = 0; j < contours.size(); j++)
	{
		rect = boundingRect(contours[j]);
		rectWidth = rect.width;
		rectHeight = rect.height;
		//rectArea = rectWidth * rectHeight;
		//area = contourArea(contours[j]); //get area of segment
		//check = area / rectArea;
		if ( rect.width > image.cols * 0.7 || rect.y < image.rows * 0.25 || rect.y > image.rows * 0.80)
		{
			drawContours(Blob, contours, j, black, -1, 8, hierarchy);
		}
	}
	return Blob;
}

Mat smallNoiseFilter(Mat image) {

	Mat Blob = image.clone();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(Blob, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

	Rect rect;
	Scalar black = CV_RGB(0, 0, 0);
	double area, rectWidth, rectHeight, rectArea, check;
	for (size_t j = 0; j < contours.size(); j++)
	{
		rect = boundingRect(contours[j]);
		rectWidth = rect.width;
		rectHeight = rect.height;
		//rectArea = rectWidth * rectHeight;
		//area = contourArea(contours[j]); //get area of segment
		//check = area / rectArea;
		if ((rect.width < image.cols * 0.08 && rect.height < image.rows * 0.07))
		{
			drawContours(Blob, contours, j, black, -1, 8, hierarchy);
		}
	}
	return Blob;
}

Rect drawRedBox(Mat image) {
	Rect roi = Rect(10, 250, 1260, 430);
	rectangle(image, roi, Scalar(0, 0, 255), 2, 8, 0);
	return roi;
}

void drawGreenBox(Mat image, int x, int y, int width, int height) {
	Rect box = Rect(x, y, width, height);
	rectangle(image, box, Scalar(0, 255, 0), 2, 8, 0);
}

vector<Mat> finalSegment(Mat filtered_image, Mat original_image, vector<vector<int>>& contours_coordinates) {
	Mat clean_image = original_image.clone();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	Mat firstSegment = Mat::zeros(filtered_image.size(), CV_8UC1);
	findContours(filtered_image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	for (size_t i = 0; i < contours.size(); i++) {
		Rect firstRect = boundingRect(contours[i]);
		rectangle(firstSegment, firstRect, Scalar(255, 0, 0), 2, 8, 0);
	}

	vector<vector<Point>> contours2;
	vector<Vec4i> hierarchy2;
	vector<Mat> cropped_img;
	Rect roi = drawRedBox(original_image); //Rect(10, 250, 1260, 430);

	findContours(firstSegment, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	for (size_t i = 0; i < contours2.size(); i++) {
		Rect overlapped = boundingRect(contours2[i]);
		int midx = overlapped.x + (overlapped.width / 2);
		int midy = overlapped.y + (overlapped.height / 2);
		if ((midx > roi.x && midx < (roi.x + roi.width)) && (midy > roi.y && midy < (roi.y + roi.height))) {
			cropped_img.push_back(clean_image(overlapped));
			vector<int> coordinates;
			coordinates.push_back(overlapped.x);
			coordinates.push_back(overlapped.y);
			contours_coordinates.push_back(coordinates);
			rectangle(original_image, overlapped, Scalar(0, 255, 0), 2, 8, 0);
		}
	}
	return cropped_img;
}

void classifyImage(Mat contours, Mat clean_image, vector<vector<int>> contours_coordinates, int coordinate_count) {
	string model_file = "C:\\Users\\zhong\\Desktop\\googleNet\\bvlc_googlenet.caffemodel";
	string config_file = "C:\\Users\\zhong\\Desktop\\googleNet\\bvlc_googlenet.prototxt";
	string class_file = "C:\\Users\\zhong\\Desktop\\googleNet\\classification_classes_ILSVRC2012.txt";

	Mat original_contours = contours.clone();
	Net net = readNet(model_file, config_file);
	if (net.empty()) {
		cout << "ERR: There are no layers in the network\n";
	}
	
	fstream fs(class_file.c_str(), fstream::in);
	if (!fs.is_open()) {
		cout << "ERR: Cannot load the class names\n";
	}
	vector<string> classes;
	string line;
	while (getline(fs, line)) {
		classes.push_back(line);
	}
	fs.close();

	float width = 224;
	float height = 224;
	float scale_w = width / contours.size().width;
	float scale_h = height / contours.size().height;
	resize(contours, contours, Size(224,224), scale_w, scale_h);

	Mat blob = blobFromImage(contours, 1, Size(224, 224), Scalar(104, 117, 123));
	if (blob.empty()) {
		cout << "ERR: Cannot create blob\n";
	}

	net.setInput(blob);

	Mat prob = net.forward();

	vector<vector<string>> main_classes;
	vector<string> car; //0
	car.push_back("van");
	car.push_back("car");
	car.push_back("limo");
	car.push_back("wagon");
	vector<string> truck; //1
	truck.push_back("truck");
	truck.push_back("lorry");
	truck.push_back("tow");
	truck.push_back("tractor");
	vector<string> motor; //2
	motor.push_back("motor");
	motor.push_back("scooter");
	main_classes.push_back(car);
	main_classes.push_back(truck);
	main_classes.push_back(motor);

	Mat sorted_idx;
	sortIdx(prob, sorted_idx, SORT_EVERY_ROW + SORT_DESCENDING);
	float total_car_prob = 0.0, total_truck_prob = 0.0, total_motor_prob = 0.0;
	for (int i = 0; i < 5; ++i) {
		//split string
		vector<string> results;
		vector<string> final_results;
		string classes_result = classes[sorted_idx.at<int>(i)];
		stringstream s_stream(classes_result);
		for (string s; s_stream >> s;) {
			results.push_back(s);
		}
		for (string s : results) {
			s.erase(remove(s.begin(), s.end(), ','), s.end());
			final_results.push_back(s);
		}
		bool found = false;
		for (int j = 0; j < main_classes.size(); j++) {
			for (int k = 0; k < main_classes[j].size(); k++) {
				if (found == false) {		
					for (string m : final_results) {
						if (m == main_classes[j][k]) {
							found = true;
							break;
						}
					}
					if (found == true) {
						if (j == 0) {
							cout << "car" << endl;
							cout << classes[sorted_idx.at<int>(i)] << " - ";
							cout << "probability: " << prob.at<float>(sorted_idx.at<int>(i)) << endl;
							total_car_prob += prob.at<float>(sorted_idx.at<int>(i));
							break;
						}
						else if (j == 1) {
							cout << "truck" << endl;
							cout << classes[sorted_idx.at<int>(i)] << " - ";
							cout << "probability: " << prob.at<float>(sorted_idx.at<int>(i)) << endl;
							total_truck_prob += prob.at<float>(sorted_idx.at<int>(i));
							break;
						}
						else if (j == 2) {
							cout << "motor" << endl;
							cout << classes[sorted_idx.at<int>(i)] << " - ";
							cout << "probability: " << prob.at<float>(sorted_idx.at<int>(i)) << endl;
							total_motor_prob += prob.at<float>(sorted_idx.at<int>(i));
							break;
						}
					}
				}
			}
		}
	}
	if ((total_car_prob > 0.1) && (total_car_prob >= total_truck_prob) && (total_car_prob >= total_motor_prob)) {
		cout << "Final Result: Car Total Probability: " << total_car_prob << endl;
		string label = format("Car %.4f", total_car_prob);
		int left = contours_coordinates[coordinate_count][0];	//x
		int top = contours_coordinates[coordinate_count][1] - 5;	//y
		drawGreenBox(clean_image, left, top + 5, original_contours.size().width, original_contours.size().height);
		putText(clean_image, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
	}
	else if ((total_truck_prob > 0.1) && (total_truck_prob >= total_car_prob) && (total_truck_prob >= total_motor_prob)) {
		cout << "Final Result: Truck Total Probability: " << total_truck_prob << endl;
		string label = format("Truck %.4f", total_truck_prob);
		int left = contours_coordinates[coordinate_count][0];	//x
		int top = contours_coordinates[coordinate_count][1] - 5;	//y
		drawGreenBox(clean_image, left, top + 5, original_contours.size().width, original_contours.size().height);
		putText(clean_image, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
	}
	else if ((total_motor_prob > 0.1) && (total_motor_prob >= total_car_prob) && (total_motor_prob >= total_truck_prob)) {
		cout << "Final Result: Motor Total Probability: " << total_motor_prob << endl;
		string label = format("Motor %.4f", total_motor_prob);
		int left = contours_coordinates[coordinate_count][0];	//x
		int top = contours_coordinates[coordinate_count][1] - 5;	//y
		drawGreenBox(clean_image, left, top + 5, original_contours.size().width, original_contours.size().height);
		putText(clean_image, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
	}
	cout << "===========================================" << endl;
}

const string path = "C:\\Users\\zhong\\Desktop\\road_images\\road_image%4d.jpg";

int main()
{
	VideoCapture capture(path);
	if (!capture.isOpened()) {
		cout << "Unable to Open" << endl;
		return 0;
	}
	while (true) {
		Mat frame, fgMask;
		capture >> frame;	//read image into frame
		if (frame.empty()) {
			break;
		}
		resize(frame, frame, Size(1280,720));	//resize to show full image
		Mat original_image = frame.clone();	//for final display
		imshow("Original Image", original_image);
		Mat clean_image = frame.clone();
		GaussianBlur(frame, frame, Size(11, 11), 0);

		//didnt use
		Mat LABcolourMap = getLABColourMap(frame);

		//-----------------ITTI's SALIENCY 
		Mat saliency = calcITTISaliencyMap(frame);
		imshow("ITTI's Saliency", saliency);
		Mat ucharSaliency;
		saliency.convertTo(ucharSaliency, CV_8UC1, 255, 0);

		//-----------------K-MEANS CLUSTERING
		int clusters = 4;
		Mat binary_image = K_Means(ucharSaliency, clusters);
		imshow("Binary Image (K-Means)", binary_image);
		
		//-----------------VERTICAL EROSION TO SEPARATE NOISES FROM SKY
		Mat erode_image = vertical_erosion(binary_image);
		imshow("Vertical Eroded Image", erode_image);

		//-----------------BIG NOISE FILTER
		Mat filtered_image = bigNoiseFilter(erode_image); 
		imshow("Big Noise Filtered Image", filtered_image);

		//-----------------VERTICAL DILATION TO CONNECT BLOBS
		Mat dilate_image = vertical_dilation(filtered_image);
		imshow("Vertical Dilated Image", dilate_image);

		//-----------------HORIZONTAL DILATION TO CONNECET BLOBS
		Mat dilate_image2 = horizontal_dilation(dilate_image);
		imshow("Horizontal Dilated Image", dilate_image2);

		//-----------------SMALL NOISE FILTER
		Mat filtered_image2 = smallNoiseFilter(dilate_image2);
		imshow("Small Noise Filtered Image", filtered_image2);

		//-----------------REGION FILLING TO FILL HOLES
		Mat floodfilled_image = filtered_image2.clone();
		floodFill(floodfilled_image, cv::Point(0, 0), Scalar(255));
		Mat floodfilled_image_inv;
		bitwise_not(floodfilled_image, floodfilled_image_inv);
		
		Mat floodFilled = (filtered_image2 | floodfilled_image_inv);
		imshow("Flood Filled", floodFilled);
		
		//-----------------SEGMENTATION
		vector<vector<int>> contours_coordinates;
		vector<Mat> cropped_images = finalSegment(floodFilled, original_image, contours_coordinates);
		imshow("original", original_image);

		//-----------------CLASSIFICATION
		int coor_counter = 0;
		Rect drawing = drawRedBox(clean_image);
		for (Mat img : cropped_images) {
			classifyImage(img, clean_image, contours_coordinates, coor_counter);
			coor_counter++;
		}
		imshow("Final Result", clean_image);
		waitKey();
	}
}
