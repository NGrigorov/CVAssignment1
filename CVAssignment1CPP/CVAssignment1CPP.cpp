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
			corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0));
		}
	}
}

bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoeficcients)
{
		//set up a FileStorage object to read camera params from file
		FileStorage fs;
		fs.open(name, FileStorage::READ);
		// read camera matrix and distortion coefficients from file
		Mat intrinsics, distortion;
		fs["camera_matrix"] >> cameraMatrix;
		fs["distortion_coefficients"] >> distanceCoeficcients;
		// close the input file
		fs.release();
		return true;
}

// Display an Image
int main(int argc, char** argv)
{

	Mat frame;
	Mat gray;

	Mat cameraMatrix = Mat::eye(3,3,CV_64F);

	Mat distanceCoeff = Mat::zeros(8, 1, CV_64F);

	vector<Mat> savedImages;

	vector<Mat> rVectors, tVectors; //rotation and translation vectors for images

	Mat rVecs = Mat(Size(3, 1), CV_64F); //Fo camerafeed
	Mat tVecs = Mat(Size(3, 1), CV_64F);


	VideoCapture vid(0);
	vector<Point3f> localCornerPosition;
	createKnownBoardPositions(cheesBoardDImensions, calibrationSquareDimension, localCornerPosition);

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

		cvtColor(frame, gray, COLOR_BGR2GRAY);
		found = findChessboardCorners(gray, cheesBoardDImensions, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		

		if (found) {
			drawChessboardCorners(frame, cheesBoardDImensions, foundPoints, found);
			solvePnP(localCornerPosition, foundPoints, cameraMatrix, distanceCoeff, rVecs, tVecs);
			drawFrameAxes(frame, cameraMatrix, distanceCoeff, rVecs, tVecs, 0.15f);
			imshow("Webcam", frame);
		}
		else
			imshow("Webcam", gray);

		char character = waitKey(1000 / framesPS);

		switch (character) 
		{
		case 27: //Escape key
			//exit
			return 0;
			break;
		
		case 101: //e key
			//load camera matrix from file
			cout << "Loading calibration file" << endl;
			if(loadCameraCalibration("out_camera_data.xml", cameraMatrix, distanceCoeff))
				cout << "Done!" << endl;
			else
				cout << "Failed no file found!" << endl;
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
