// By Ratnesh Kumar, PhD from INRIA, France
// A simple demo/tutorial code for basic OpenCV based background subtraction.

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp> //for morphological operations.
#include <boost/filesystem.hpp> //for saving files into a folder : create directory, checking etc,

#include <iostream>
#include <string>
#include <vector>
#include <cassert>


#include "CImg.h"

using namespace std;
using namespace cv;
using namespace cimg_library;
//Usually mixing namespaces is not a very good idea, however for this small application I take this liberty.
//Ideally if the software is large we should have one .cpp for .h for keeping things as separated as possible.

typedef CImg<unsigned char> UCCimg; //to make life easy with template writing again and again

//a simplification for reating directory
void create_directory(const string s1)
{
    boost::filesystem::path p1(s1);
    bool is_created = boost::filesystem::create_directory(p1);
    if (!is_created && (!boost::filesystem::is_directory(p1)))
    {
        std::cout << "\n" << s1 <<" Not created, Please check authority issues "<< endl;
        exit(0);
    }
}

//a function to save volume to images
template<typename T> void save_a_volume_image_to_single_frames(const CImg<T> &volume, const string opt_folder)
{
    const int w = volume.width(), h = volume.height(), nr_c = volume.spectrum();
    cimg_forZ(volume, z) //z: number of images in this volume
    {
        CImg<T> frame(w, h, 1, nr_c);
        cimg_forXYC(frame, x, y, c)
        {
            frame(x, y, c) = volume(x, y, z, c);
        }
        //save this frame
        char num[100];
        sprintf(num, "%05d", z);
        string fname = opt_folder + "/" + num + ".jpg";
        frame.save_jpeg(fname.c_str());
    }
}

void save_opencv_image(const string output_folder, const int id, const Mat &frame)
{
    char num[100];
    sprintf(num, "%05d", id);
    string fname = output_folder + "/" + num + ".jpg";
    imwrite(fname, frame);
}

//dialtion of erosion -> Morph Open
void postprocess_using_morphological_operations(Mat &fg)
{
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(3, 3));
    // an elliptical structure element which is used to process each pixel
    morphologyEx(fg, fg, MORPH_OPEN, kernel); // an opening operation
}

void compute_mean_image(const string inp, Mat &mean_img)
{
    VideoCapture capture(inp);
    //Compute the mean image
    const int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    const int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    //
    const int num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
    mean_img = Mat::zeros(height, width, CV_32FC3); //Initialize the mean image
    for (int i = 0; i< num_frames; i++) //for a video a while loop can be used
    {
        Mat frame;
        capture >> frame; // capture the current frame
        //convert to the float format so that I can add to mean_img
        Mat f_frame;
        frame.convertTo(f_frame, CV_32FC3);
        mean_img += f_frame; //polymorphism example !
    }
    //
    mean_img /= float(num_frames);
    capture.release();
}

//Option 1: mean image based subtraction: Although median will be more robust,
//but options 2, 3 (MOG) is a better way to remove BG
void remove_mean_image_from_video(const string inp, const string output_folder,
                                  const float thresh = 20, bool disp = false)
{
    Mat mean_img;
    compute_mean_image(inp, mean_img);
    VideoCapture capture(inp);
    const int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    const int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    const int num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
    for (int i = 0; i < num_frames; i++) //for a video a while loop can be used
    {
        Mat frame, f_frame, gray, dst, copy;
        capture >> frame; // capture the current frame
        frame.convertTo(f_frame, CV_32FC3);
        frame.copyTo(copy);
        f_frame -= mean_img;
        f_frame = abs(f_frame);
        cvtColor(f_frame, gray, CV_RGB2GRAY);
        threshold(gray, dst, thresh, 255, 0);
        //thresh for thresholding: An important parameter, which is nevertheles difficult to set
        dst.convertTo(dst, CV_8UC1); //mask format
        Mat opt = Mat::zeros(height, width, CV_32FC3);
        copy.copyTo(opt, dst);// apply mask
        if (disp)
        {
            imshow("mean_based_subtraction", opt);
            waitKey(0);
        }
        //save this frame onto another folder
        save_opencv_image(output_folder, i, opt);
    }
    capture.release();
}

struct MOG_params //wraps all params for both MOG
{
    bool morph;
    float lR; //learning rate
    int nmixtures; //number of gaussians
    int history; //length of background
    double noiseSigma; // Noise standard deviation
    float bgR; //background ratio
    float varThreshold;
    bool bShadowDetection;
    MOG_params() //I prefer to provide default values although this will be over ridden by the user,
    //but later when we add some params ...Default constructor is helpful
    {
        morph = true;
        lR = 0.001;
        nmixtures =5;
        history = 10;
        noiseSigma = 1.0;
        bgR = 0.7;
        varThreshold = 100;
        bShadowDetection = false;
    }
};


//compute MOG based FG model : DOXYGEN format can be used for input/output.
//I am just ignoring now because its simple enough
void simple_mog_based_fg(const string inp, const MOG_params& params,
                         const string output_folder, const bool disp = false)
{
    VideoCapture capture(inp);
    Ptr<BackgroundSubtractor> MOG; //automatic destruction is called (smart pointer, deletion based on RefCount)
    MOG = new BackgroundSubtractorMOG(params. history, params.
                                      nmixtures, params. bgR, params.noiseSigma);
    //MOG : You can also use MOG2 Z.Zivkovic
    //
    Mat frame, fg;
    const int num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
    const int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    const int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    for (int i = 0; i < num_frames; i++) //for a video a while loop can be used
    {
        capture >> frame; // capture the current frame
        MOG->operator()(frame, fg, params.lR); //apply tp learn fg mask ..// const is the learning rate
        if (disp)
        {
            imshow("F", frame);
            waitKey(0);
        }
        if (params.morph)
        {
            postprocess_using_morphological_operations(fg);
            imshow("FG Mask MOG_OPENED", fg);
            waitKey(0);
        }
        Mat opt = Mat::zeros(height, width, CV_8UC3); //output after BG removal
        frame.copyTo(opt, fg);// apply mask
        if (disp)
        {
            imshow("FG", opt);
            waitKey(0);
        }
        //save
        save_opencv_image(output_folder, i, opt);
    }
    capture.release();
}

//compute MOG based FG model : Zoran Zivkovic
void mog_based_fg(const string inp, const MOG_params &params, const string output_folder, const bool disp = false)
{
    cout <<" You chose to run Zoran's MOG "<< endl;
    VideoCapture capture(inp);
    Ptr<BackgroundSubtractor> MOG; //MOG : You can also use MOG2 Z.Zivkovic
    MOG= new BackgroundSubtractorMOG2(); //constructor for MOG
    MOG -> set("history", params.history);
    MOG -> set("nmixtures", params.nmixtures);
    MOG -> set("varThreshold", params.varThreshold);
    MOG -> set("detectShadows", params.bShadowDetection);
    //
    Mat frame, fg;
    const int num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
    const int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    const int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    for (int i = 0; i< num_frames; i++) //for a video a while loop can be used
    {
        capture >> frame; // capture the current frame
        MOG->operator()(frame, fg, params.lR); //apply tp learn fg mask ..// const is the learning rate
        if (disp)
        {
            imshow("F", frame);
            waitKey(0);
        }
        if (params.morph)
        {
            postprocess_using_morphological_operations(fg);
            imshow("FG Mask MOG_OPENED", fg);
            waitKey(0);
        }
        if (params.bShadowDetection) //shadows are marked with 127,
          //so if shadow detection is used, then I should remove 127 from mask
        {
            //shadow =127 in mask
            //make this to zero
            fg -= 127; // Again, overloading made our life easy.
        }
        Mat opt = Mat::zeros(height, width, CV_8UC3);
        frame.copyTo(opt, fg);// apply mask
        if (disp)
        {
            imshow("FG", opt);
            waitKey(0);
        }
        //save
        save_opencv_image(output_folder, i, opt);
    }
    capture.release();
}

//2d directional derivative along time-direction.
// sigma is the variation across time
void cimg_processing_3D_data(const string inp, const string opt, const bool disp =false,
                             const bool morph = false, const int sigma =5 , const int thresh = 10)
{
    // This is an elementary step to background removal:
    //It will only work is movement is fast and motion is perpendicular to optical axis
    // I keep this function to show you a simple line removal (time)

    // This function is slightly memory intensive:
    //Can be made more efficient by dropping the volume image and loading again when needed.
    VideoCapture capture(inp);
    const int num_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
    const int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    const int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    // init a Cimg Image
    UCCimg volume(width, height, num_frames, 3, 0);
    //3 for default channels: I can put a check, but we're in colorful world :-)
    for (int i = 0; i < num_frames; i++) //for a video a while loop can be used
    {
        Mat frame;
        capture >> frame; // capture the current frame
        //convert this to cimg
        cimg_forXYC(volume, x, y, c) //c is for num channels: CImg has nice macros
        {
            volume(x, y, i, c) = frame.at<Vec3b>(y, x)[c];
        }
    }
    capture.release();
    // Now I have all images in cimg format
    if (disp) volume.display("Input"); // I have to investigate why color looks weird : TODO

    // Now I will remove lines parallel to time axis.
    const int order = 2; // always second derivative
    UCCimg vol_img = volume.get_deriche(sigma, order, 'z'); // a memory intensive step.
    //----------- Lines below are for morphological open processing:
    if (morph)
    {
        vol_img.erode(3); //structural element of size 3
        vol_img.dilate(3);
    }
    //-----------
    //threshold
    vol_img.threshold(thresh);
    //mask
    cimg_forXYZ(vol_img, x, y, z)
    {
        int th = 0;
        cimg_forC(vol_img, c) //loop over channels
        {
            th += vol_img(x, y, z, c);
        }
        //if th =0, then mask that pixel out.
        if (th == 0)
        {
            cimg_forC(vol_img, c)
            {
                volume(x, y, z, c) = 0;
            }
        }
    }
    vol_img.clear(); //clearing mask
    if (disp) volume.display("output-FG");
    //save output
    save_a_volume_image_to_single_frames(volume, opt);
}

void rgb2gray(const UCCimg &im, UCCimg &out)
{
  out.assign(im.width(), im.height(), 1, 1, 0);
  cimg_forXY(im, x, y)
    {
      out(x, y) = (im(x, y, 0) + im(x, y, 1) +im(x, y, 2))/3;
    }
}
void remove_mean(UCCimg &img)
{
  float mean = img.mean();
  cimg_forXY(img, x, y)
    {
      img(x, y) -= mean;
    }
}

int main(int argc, char *argv[])
{
    // How to use
    cimg_usage("Default arguments are provided: Please choose -h while running ");
    //string inp = cimg_option("-input", "/home/fratnesh/Dropbox/img_small/imgs/%05d.jpg",
    //"Input Folder Location for Image Files");
    string inp = cimg_option("-input", "/home/fratnesh/Dropbox/img_small/%05d.jpg",
                             "Input Folder Location for Image Files");
    string opt = cimg_option("-output", "/home/fratnesh/opt", "Input Folder Location for Image Files");
    const int option = cimg_option("-bg_option", 2,
                                   "1- mean image subtraction, 2- MOG, 3- MOG by Zoran, 4- temporal gradient ");
    const int nmixtures = cimg_option("-nmixtures", 5, "Number of gaussians for MOG");
    const float bgR = cimg_option("-bgR", 0.08, "Background Ratio for MOG (option 1)");
    const float lR = cimg_option("-lR", 0.005,
                                 "learning rate for MOG: 0 means only first frame as BG i.e. no learning");
    bool is_morph_proc = cimg_option("-morph", true, "whether to perform morphological processing");
    const bool is_shadow_det = cimg_option("-shadow_det", false,
                                           "Shadow detection for option 3"); //shadows are 127 in mask
    const bool disp = cimg_option("-disp_frames", false, "display frames one by one during bg process");
    const float history = cimg_option("-history", 100, "history for MOG");
    const float noise_std = cimg_option("-noise_std", 0.0, "standard deviation for noise (option 1)");
    const float var_thresh = cimg_option("-var_thresh", 100.0, "variance threh for Zoran (option 3)");
    const float opt1_thresh = cimg_option("-mean_thresh", 25.0, "threshold for mean removal, (option 1)");

    // create output directory to save files
    create_directory(opt);
    // Init MOG_params
    MOG_params params;
    {
        //set params from user input
        params.nmixtures = nmixtures;
        params.bgR = bgR;
        params.bShadowDetection = is_shadow_det;
        params.history = history;
        params.lR = lR;
        params.morph = is_morph_proc;
        params.noiseSigma = noise_std;
        params.varThreshold = var_thresh;
    }//
    switch (option) {
    case 1:
        {
            string fldr = opt + "/option_1/";
            create_directory(fldr);
            remove_mean_image_from_video(inp, fldr, opt1_thresh, disp);
            break;
        }
    case 2:
        {
            string fldr2 = opt + "/option_2/";
            create_directory(fldr2);
            simple_mog_based_fg(inp, params, fldr2, disp);
            break;
        }
    case 3:
        {
            string fldr3 = opt + "/option_3/";
            create_directory(fldr3);
            mog_based_fg(inp, params, fldr3, disp);
            break;
        }
    case 4:
        {
            string fldr4 = opt + "/option_4/";
            create_directory(fldr4);
            cimg_processing_3D_data(inp, fldr4, disp, is_morph_proc);
            break;
        }
    default:
        {
            string fldr2 = opt + "/option_2/";
            create_directory(fldr2);
            simple_mog_based_fg(inp, params, fldr2, disp);
            break;
        }
    }//end case
    return 0;
}
