// CVAssignment1CPP.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <sstream>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

const float calibrationSquareDimension = 0.023f;
const Size cheesBoardDImensions = Size(6,9);

void createKnownBoardPositions(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
	for (int i = 0; i < boardSize.height; i++)
	{
		for (int j = 0; j < boardSize.width; j++)
		{
			corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
		}
	}
}

void getChessBoardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults)
{
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		vector<Point2f> pointBuffer;
		bool found = findChessboardCorners(*iter, Size(9, 6), pointBuffer, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

		if (found)
		{
			allFoundCorners.push_back(pointBuffer);
		}

		if (showResults)
		{
			drawChessboardCorners(*iter, Size(9, 6), pointBuffer, found);
			imshow("Looking for corners", *iter);
			waitKey(0);
		}
	}
}

void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squaredEdgeLength, Mat& cameraMatrix, Mat& distanceCoeff, vector<Mat>& rVectors, vector<Mat>& tVectors)
{
	vector<vector<Point2f>> checkerboardImageSpacePoints;
	getChessBoardCorners(calibrationImages, checkerboardImageSpacePoints, false);

	vector<vector<Point3f>> worldSpacePoints(1);

	createKnownBoardPositions(boardSize, squaredEdgeLength, worldSpacePoints[0]);
	worldSpacePoints.resize(checkerboardImageSpacePoints.size(), worldSpacePoints[0]);

	
	distanceCoeff = Mat::zeros(8, 1, CV_64F);
	calibrateCamera(worldSpacePoints, checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoeff, rVectors, tVectors);
}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients, vector<Mat> rVectors, vector<Mat> tVectors)
{
	ofstream outStream(name);
	if (outStream)
	{
		uint16_t rows = cameraMatrix.rows;
		uint16_t cols = cameraMatrix.cols;

		outStream << rows << endl;
		outStream << cols << endl;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				double value = cameraMatrix.at<double>(r, c);
				outStream << value << endl;
			}
		}

		rows = distanceCoefficients.rows;
		cols = distanceCoefficients.cols;

		outStream << rows << endl;
		outStream << cols << endl;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				double value = distanceCoefficients.at<double>(r, c);
				outStream << value << endl;
			}
		}

		outStream << rVectors.size() << endl;
		for (int i = 0; i < rVectors.size(); i++)
		{
			rows = rVectors[i].rows;
			cols = rVectors[i].cols;

			outStream << rows << endl;
			outStream << cols << endl;

			for (int r = 0; r < rows; r++)
			{
				for (int c = 0; c < cols; c++)
				{
					double value = rVectors[i].at<double>(r, c);
					outStream << value << endl;
				}
			}
		}

		outStream << tVectors.size() << endl;
		for (int i = 0; i < tVectors.size(); i++)
		{
			rows = tVectors[i].rows;
			cols = tVectors[i].cols;

			outStream << rows << endl;
			outStream << cols << endl;

			for (int r = 0; r < rows; r++)
			{
				for (int c = 0; c < cols; c++)
				{
					double value = tVectors[i].at<double>(r, c);
					outStream << value << endl;
				}
			}
		}

		outStream.close();
		return true;
	}
	return false;
}

bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoeficcients, vector<Mat>& rVectors, vector<Mat>& tVectors)
{
	ifstream inStream(name);
	if (inStream)
	{
		uint16_t rows = cameraMatrix.rows;
		uint16_t cols = cameraMatrix.cols;

		inStream >> rows;
		inStream >> cols;

		cameraMatrix = Mat(Size(rows, cols),CV_64F);

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				double read = 0.0f;
				inStream >> read;
				cameraMatrix.at<double>(r, c) = read;
				cout << cameraMatrix.at<double>(r, c) << endl;
			}
		}
		//Distance Coefficients

		inStream >> rows;
		inStream >> cols;

		distanceCoeficcients = Mat::zeros(rows, cols, CV_64F);

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				double read = 0.0f;
				inStream >> read;
				distanceCoeficcients.at<double>(r, c) = read;
				cout << distanceCoeficcients.at<double>(r, c) << endl;
			}
		}
		int arraySize = 0;
		inStream >> arraySize;
		cout << arraySize << endl;
		for (int i = 0; i < arraySize; i++)
		{
			inStream >> rows;
			inStream >> cols;
			Mat tempRVector = Mat::zeros(rows, cols, CV_64F);
			for (int r = 0; r < rows; r++)
			{
				for (int c = 0; c < cols; c++)
				{
					double read = 0.0f;
					inStream >> read;
					tempRVector.at<double>(r, c) = read;
					
				}
			}
			rVectors.push_back(tempRVector);
		}

		inStream >> arraySize;
		cout << arraySize << endl;
		for (int i = 0; i < arraySize; i++)
		{
			inStream >> rows;
			inStream >> cols;
			Mat tempTVector = Mat::zeros(rows, cols, CV_64F);
			for (int r = 0; r < rows; r++)
			{
				for (int c = 0; c < cols; c++)
				{
					double read = 0.0f;
					inStream >> read;
					tempTVector.at<double>(r, c) = read;
					
				}
			}
			tVectors.push_back(tempTVector);
		}

		inStream.close();
		return true;

	}

	return false;
}

// Display an Image
int main(int argc, char** argv)
{
	int imagesTaken = 0;
	Mat frame;
	Mat drawToFrame;

	Mat cameraMatrix = Mat::eye(3,3,CV_64F);

	Mat distanceCoefficients;

	vector<Mat> savedImages;

	vector<vector<Point2f>> markerCorners, rejectedCandidates;

	vector<Mat> rVectors, tVectors; //rotation and translation vectors

	VideoCapture vid(0);

	bool displayStuff = false;

	if (!vid.isOpened())
	{
		return 0;
	}
	int framesPS = 24;
	namedWindow("Webcam", WINDOW_AUTOSIZE);

	while (true)
	{
		if (!vid.read(frame)) break;

		vector<Vec2f> foundPoints;
		bool found = false;

		found = findChessboardCorners(frame, cheesBoardDImensions, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		frame.copyTo(drawToFrame);

		if (found) {
			drawChessboardCorners(drawToFrame, cheesBoardDImensions, foundPoints, found);
			imshow("Webcam", drawToFrame);
		}
		else
			imshow("Webcam", frame);

		char character = waitKey(1000 / framesPS);

		switch (character) 
		{
		case ' ': //Spacebar key
			//saving an image via spacebar
			if (found)
			{
				Mat temp;
				frame.copyTo(temp);
				savedImages.push_back(temp);
				string path = "Images\\Image_" + std::to_string(imagesTaken) + ".jpg";
				imwrite(path, temp);
				imagesTaken++;
				cout << imagesTaken << endl;
			}
			break;
		case 13: //Enter key
			//start calibration
			if (savedImages.size() > 15)
			{
				cout << "Starting calibration sequence" << endl;
				cameraCalibration(savedImages, cheesBoardDImensions, calibrationSquareDimension, cameraMatrix, distanceCoefficients, rVectors, tVectors);
				saveCameraCalibration("cameracalibrationFile", cameraMatrix, distanceCoefficients, rVectors, tVectors);
				cout << "Done!" << endl;
			}
			else
			{
				cout << "I need at least 16 images, so far I have " << imagesTaken << endl;
			}
			break;
		case 27: //Escape key
			//exit
			return 0;
			break;
		
		case 101: //e key
			//load camera matrix from file
			cout << "Loading calibration file" << endl;
			if(loadCameraCalibration("cameracalibrationFile", cameraMatrix, distanceCoefficients, rVectors, tVectors))
				cout << "Done!" << endl;
			else
				cout << "Failed no file found!" << endl;
			break;
		case 102: //f key
		//show axiss on saved images
			cout << "Drawing axis on images" << endl;
			for (int i = 0; i < rVectors.size(); i++)
			{
				string path = "Images\\Image_" + std::to_string(i) + ".jpg";
				Mat image = imread(path);
				drawFrameAxes(image, cameraMatrix, distanceCoefficients, rVectors[i], tVectors[i], 0.5f, 3);
				path = "AxisImages\\Image_" + std::to_string(i) + ".jpg";
				imwrite(path, image);
			}
			cout << "Done!" << endl;
			break;
		case 112: //p key
	//switch boolean that shows stuff on screen
			displayStuff = !displayStuff;
			cout << "switching dispalying stuff" << endl;
			break;
		}
	}
	return 0;
}
