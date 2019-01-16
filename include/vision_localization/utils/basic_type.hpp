#ifndef VISION_LOCALIZATION_BASIC_TYPE
#define VISION_LOCALIZATION_BASIC_TYPE

#include <opencv2/opencv.hpp>
#include <ros/ros.h>

enum LaneLineType
{
    DETECTION = 0,
    TRACKING = 1,
};

struct LaneLine
{
    // basic value
    int id;//-1:init value; 0:left lane; 1:left second lane...
    int direct;//-1;left lane; 1:right lane
    LaneLineType type;
    double dist_to_car;


    // parameter
    std::vector<cv::Point2d> pts_ipm; //!< inliers coordinates
    std::vector<cv::Point2d> pts_img;

    cv::Vec4d vec_ipm; //!< (vec_ipm[0], vec_ipm[1]) is the unit normal vector of the line
                       //!< (vec_imp[2], vec_ipm[3]) is a point on the line

    cv::Vec4d vec_img; //!< (vec_img[0], vec_img[1]) is the unit normal vector of the line
                       //!< (vec_img[2], vec_img[3]) is a point on the line

    double theta_ipm;
    double radium_ipm;
    double theta_img;
    double radium_img;

    cv::Point2d pt_start_ipm; //!< start point of the line in the IPM image
    cv::Point2d pt_end_ipm;   //!< end point of the line in the IPM image
    cv::Point2d pt_start_img; //!< start point of the line in the original image
    cv::Point2d pt_end_img;   //!< end point of the line in the original image

    double len_ipm; //!< length of the line in pixels in the IPM image
    double len_img; //!< length of the line in pixels in the original image

    std::vector<double> occupied; // indicate which row in the IPM image this line occupies. 1 if occupied, 0 otherwise
};

struct LaneLineTracker
{
    LaneLine lane_line;
    int missing_times;
    int appear_times;
};

struct CameraInfo
{
    double fx;    //!< focal length in x-axis direction
    double fy;    //!< focal length in y-axis direction
    double cx;    //!< x coordinate of the principle point
    double cy;    //!< y coordinate of the principle point

    double pitch; //!< rotation around x-axis
    double yaw;   //!< rotation around z-axis
    double roll;  //!< rotation around y-axis

    // for debug purpose only
    std::string asString() const;
};

std::string CameraInfo::asString() const
{
    std::stringstream ss;

    ss << "====================\n";
    ss << " CameraInfo parameters\n";
    ss << "====================\n";

    ss << "fx: " << fx << "\n";
    ss << "fy: " << fy << "\n";
    ss << "cx: " << cx << "\n";
    ss << "cy: " << cy << "\n";
    ss << "pitch: " << pitch << "\n";
    ss << "yaw: " << yaw << "\n";
    ss << "roll: " << roll << "\n";
    ss << "--------------------\n";

    return ss.str();
}

#endif