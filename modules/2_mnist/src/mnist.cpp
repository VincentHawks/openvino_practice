#include "mnist.hpp"
#include <fstream>

using namespace cv;

inline int readInt(std::ifstream& ifs) {
    int val;
    ifs.read((char*)&val, 4);
    // Integers in file are high endian which requires swap
    std::swap(((char*)&val)[0], ((char*)&val)[3]);
    std::swap(((char*)&val)[1], ((char*)&val)[2]);
    return val;
}

void loadImages(const std::string& filepath,
                std::vector<Mat>& images) {
    std::ifstream ifs(filepath.c_str(), std::ios::binary);
    CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

    int magicNum = readInt(ifs);
    CV_CheckEQ(magicNum, 2051, "");

    int numImages = readInt(ifs);

    // TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
    // at http://yann.lecun.com/exdb/mnist/
	
	int rows = readInt(ifs);
	int cols = readInt(ifs);

	Mat image_buffer(rows, cols, CV_8U);
	uint8_t pixel_buffer;
	int checksum;
	for (int i = 0; i < numImages; i++)
	{
		checksum = 0;
		for (int y = 0; y < rows; y++)
		{
			for (int x = 0; x < cols; x++)
			{
				ifs.read((char*)&pixel_buffer, 1);
				image_buffer.at<uint8_t>(Point(x, y)) = pixel_buffer;
				checksum += (int)pixel_buffer;
			}
		}
		assert(checksum != 0 && checksum != 255 * rows * cols);
		images.push_back(image_buffer);
	}
	ifs.close();
}

void loadLabels(const std::string& filepath,
                std::vector<int>& labels) {
    std::ifstream ifs(filepath.c_str(), std::ios::binary);
    CV_CheckEQ(ifs.is_open(), true, filepath.c_str());

    int magicNum = readInt(ifs);
    CV_CheckEQ(magicNum, 2049, "");

    int numLabels = readInt(ifs);

    // TODO: follow "FILE FORMATS FOR THE MNIST DATABASE" specification
    // at http://yann.lecun.com/exdb/mnist/

	unsigned char label_buffer;
	for (int i = 0; i < numLabels; i++)
	{
		ifs.read((char*)&label_buffer, 1);
		assert(label_buffer > -1 && label_buffer < 10);
		labels.push_back((int)label_buffer);
	}
	ifs.close();
}

void prepareSamples(const std::vector<cv::Mat>& images, cv::Mat& samples) {
	std::cout << "Preparing samples..." << std::endl;
	int rows = images.at(0).rows;
	int cols = images.at(0).cols;
	samples = Mat(images.size(), rows * cols, CV_8UC1);
	for (int i = 0; i < images.size(); i++)
	{
		for (int y = 0; y < rows; y++)
		{
			for (int x = 0; x < cols; x++)
			{
				samples.at<uint8_t>(i, rows * y + x) = images[i].at<uint8_t>(Point(x, y));
			}
		}
	}
	samples.convertTo(samples, CV_32FC3);
}

Ptr<ml::KNearest> train(const std::vector<cv::Mat>& images,
                        const std::vector<int>& labels) {
    //CV_Error(Error::StsNotImplemented, "train");
	std::cout << "Training in progress..." << std::endl;
	Mat samples;
	prepareSamples(images, samples);
	Ptr<ml::KNearest> model = ml::KNearest::create();
	model->train(samples, ml::ROW_SAMPLE, labels);
	return model;
}

float validate(Ptr<ml::KNearest> model,
               const std::vector<cv::Mat>& images,
               const std::vector<int>& labels) {
    //CV_Error(Error::StsNotImplemented, "validate");
	std::cout << "Validation..." << std::endl;
	Mat samples, results;
	prepareSamples(images, samples);

	model->findNearest(samples, 7, results);

	float successful_predictions = 0;
	std::cout << results << std::endl;
	for (int i = 0; i < results.cols; i++)
	{
		if (round(results.at<float>(i)) == (float)labels[i]) successful_predictions++;
	}
	return successful_predictions / (float)images.size();
}

int predict(Ptr<ml::KNearest> model, const Mat& image) {
    // TODO: resize image to 28x28 (cv::resize)
	Mat test_image;
	//resize(image, test_image, );

    // TODO: convert image from BGR to HSV (cv::cvtColor)

    // TODO: get Saturate component (cv::split)

    // TODO: prepare input - single row FP32 Mat

    // TODO: make a prediction by the model

    CV_Error(Error::StsNotImplemented, "predict");
}
