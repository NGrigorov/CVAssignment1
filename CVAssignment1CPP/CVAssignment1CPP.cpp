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

//Constant board size
const float calibrationSquareDimension = 0.023f;
const Size cheesBoardDImensions = Size(6,9);


//Generates local board positions.
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

//Loads camera calibration from file
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

//Draws world axis
void drawWorldAxis(Mat rVecs, Mat tVecs, Mat cameraMatrix, Mat distanceCoeff, Mat frame, float size)
{

	vector<Point3f> framePoints;
	//generate points in the reference frame
	framePoints.push_back(Point3d(0.0, 0.0, 0.0));
	framePoints.push_back(Point3d(0.1*size, 0.0, 0.0));
	framePoints.push_back(Point3d(0.0, 0.1 * size, 0.0));
	framePoints.push_back(Point3d(0.0, 0.0, -0.1 * size));

	vector<Point2f> imageFramePoints;
	
	//Projects 3d points to 2d
	projectPoints(framePoints, rVecs, tVecs, cameraMatrix, distanceCoeff, imageFramePoints);

	//DRAWING
	//Draws XYZ lines in different colors
	line(frame, imageFramePoints[0], imageFramePoints[1], CV_RGB(255, 0, 0), 3);
	line(frame, imageFramePoints[0], imageFramePoints[2], CV_RGB(0, 255, 0), 3);
	line(frame, imageFramePoints[0], imageFramePoints[3], CV_RGB(0, 0, 255), 3);
}

//Random roll
int roll(int min, int max)
{
	// x is in [0,1[
	double x = rand() / static_cast<double>(RAND_MAX + 1);

	// [0,1[ * (max - min) + min is in [min,max[
	int that = min + static_cast<int>(x * (max - min));

	return that;
}

//Draws a cube with specific size at the world origin
void drawSquare(Mat rVecs, Mat tVecs, Mat cameraMatrix, Mat distanceCoeff, Mat frame, float size, double speedX)
{
	//Create Cube
	vector<Point3f> SquareframePointsBottom;
	int random = roll(1, 4);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 2; k > 0; k--){
				Point3f temp = Point3f(i * size * 0.1 * random, j * size * 0.1 * random, k * size * -0.1 * random);
				SquareframePointsBottom.push_back(temp);
			}
		}
	}
		
	vector<Point2f> SquareimageFramePointsBT; //Holds 2d points projected from 3d points
	projectPoints(SquareframePointsBottom, rVecs, tVecs, cameraMatrix, distanceCoeff, SquareimageFramePointsBT);

	//drawContours doesnt like floats, so we use ints instead :#
	vector<Point> squareFramePointsBT;
	squareFramePointsBT.push_back(SquareimageFramePointsBT[0]);
	squareFramePointsBT.push_back(SquareimageFramePointsBT[1]);
	squareFramePointsBT.push_back(SquareimageFramePointsBT[2]);
	squareFramePointsBT.push_back(SquareimageFramePointsBT[3]);
	squareFramePointsBT.push_back(SquareimageFramePointsBT[4]);
	squareFramePointsBT.push_back(SquareimageFramePointsBT[5]);
	squareFramePointsBT.push_back(SquareimageFramePointsBT[6]);
	squareFramePointsBT.push_back(SquareimageFramePointsBT[7]);

	//We use this  to make a shape
	vector<vector<Point>> cubeHull(1);
	// Initialize the contour with the convex hull points
	convexHull(squareFramePointsBT, cubeHull[0]);
	
	//Totally accurate aninmation
	int randomR = roll(0, 3);
	int randomG = roll(0, 3);
	int randomB = roll(0, 3);

	drawContours(frame, cubeHull, 0, CV_RGB(100*randomR, 125*randomG, 50*randomB), -1);

}

// Main functions
int main(int argc, char** argv)
{
	//Main camera-feed
	Mat frame;
	//grayscale to perform calculations faster
	Mat gray;

	Mat cameraMatrix = Mat::eye(3,3,CV_64F);
	Mat distanceCoeff = Mat::zeros(8, 1, CV_64F);

	Mat rVecs = Mat(Size(3, 1), CV_64F); //Fo camerafeed
	Mat tVecs = Mat(Size(3, 1), CV_64F);

	
	vector<Point3f> localCornerPosition;
	createKnownBoardPositions(cheesBoardDImensions, calibrationSquareDimension, localCornerPosition);

	bool displayStuff = true;

	VideoCapture vid(0);

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

		//Change camera colour feed to grayscale.
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		//Finds chessboard corners from camera feed.
		found = findChessboardCorners(gray, cheesBoardDImensions, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
		

		if (found) {
			//Draws chessboard patern if the chessboard is found
			drawChessboardCorners(frame, cheesBoardDImensions, foundPoints, found);
			//creates rotation and translation vectors 
			solvePnP(localCornerPosition, foundPoints, cameraMatrix, distanceCoeff, rVecs, tVecs);

			if (displayStuff) {
				//
				drawWorldAxis(rVecs, tVecs, cameraMatrix, distanceCoeff, frame, 1);
				//
				drawSquare(rVecs, tVecs, cameraMatrix, distanceCoeff, frame, 0.5f, 5);
			}
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
