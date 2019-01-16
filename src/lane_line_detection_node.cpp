#include "lane_line_detection.hpp"

static
void getParam(
        ros::NodeHandle& nh,
        LaneLineDetectionConfig& config,
        std::string& image_list_filename,
        std::string& image_dir
)
{
#define MY_READ(param) \
    if(!nh.getParam(#param, config.param))  \
    {\
        ROS_ERROR_STREAM("Failed to read " << #param); \
        exit(-1); \
    }

    MY_READ(camera_parameter_filename);
    MY_READ(ipm_cnn_bw_threshold);
    MY_READ(ipm_filter_bw_threshold);
    MY_READ(ransac_contour_success_probability);
    MY_READ(ransac_contour_max_outlier_ratio);
    MY_READ(ransac_contour_threshold);
    MY_READ(ransac_contour_minimum_required_points);
    MY_READ(ransac_contour_constrain_line_direction_ipm);
    MY_READ(ransac_contour_constrain_line_length_ipm);
    MY_READ(ransac_contour_minimum_line_length_ipm);
    MY_READ(merge_line_max_difference_r_ipm);

#undef MY_READ

    if(!nh.getParam("image_list_filename", image_list_filename))
    {
        ROS_ERROR("Failed to read the image list!");
        exit(-1);
    }

    if(!nh.getParam("image_dir", image_dir))
    {
        ROS_ERROR("Failed to read the image directory!");
        exit(-1);
    }
}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "lane_line_detection_node");
    ros::NodeHandle nh("~");
    LaneLineDetectionConfig config;

    std::string image_list_filename;
    std::string image_dir;

    getParam(nh, config, image_list_filename, image_dir);

    ROS_INFO_STREAM(config.paramAsString());
    ROS_INFO_STREAM("Image list filename: " << image_list_filename);
    ROS_INFO_STREAM("Image dir: " << image_dir);
    ROS_INFO_STREAM("Config: \n" << config.paramAsString());

    LaneLineDetection lane_line_detection(config);
    ROS_INFO_STREAM(lane_line_detection.paramAsString());

    std::vector<std::string> img_list(0);
    std::vector<std::string> img_cnn_list(0);

    std::ifstream f_in(image_list_filename.c_str(), std::ios::in);
    if (!f_in.is_open())
    {
        ROS_ERROR_STREAM("Cannot open file: " << image_list_filename);
        return -1;
    }
#if 0
    ROS_INFO("exited!");
    return 0;
#endif

    while (!f_in.eof())
    {
        std::string img_name;
        f_in >> img_name;
        img_list.push_back(img_name);
    }
    f_in.close();

    //// do lane detection
    for (size_t i = 0; i < img_list.size(); i+= 1)
    {
        //if ( i != 6) continue;
        ROS_WARN("img %d: %s", i, img_list[i].c_str());
        std::string img_filename = cv::format("%s/%s",
                                              image_dir.c_str(),
                                              img_list[i].c_str());

        std::string img_cnn_filename = cv::format("%s/%s%s",
                                                  image_dir.c_str(),
                                                  "0111_night_ipm_",
                                                  img_list[i].c_str());

        ROS_INFO_STREAM("Image filename: " << img_filename);
        ROS_INFO_STREAM("Image cnn filename: " << img_cnn_filename);

        cv::Mat img = cv::imread(img_filename);
        cv::Mat img_cnn = cv::imread(img_cnn_filename);

        CV_Assert(!img.empty());
        CV_Assert(!img_cnn.empty());

        cv::imshow("img_cnn color", img_cnn);

        lane_line_detection.doLaneLineDetection(img, img_cnn, img_list[i]);

        cv::waitKey(0);
    }

}