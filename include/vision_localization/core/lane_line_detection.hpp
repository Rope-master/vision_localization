#ifndef VISION_LOCALIZATION_LANE_LINE_DETECTION
#define VISION_LOCALIZATION_LANE_LINE_DETECTION

#include "ipm.hpp"
#include "basic_type.hpp"
#include <yaml-cpp/yaml.h>
#include "ransac.hpp"

class HoughTF
{
public:
    int doInit(
            const int& im_w,
            const int& im_h,
            const double& pos_x,
            const double& pos_y
    )
    {
        theta_interval_ = 1;
        radium_interval_ = 2;
        theta_num_ = 180 / theta_interval_;
        radium_num_ = sqrt(im_w * im_w + im_h * im_h) * 2;

        //// init the theta, cos and sin
        std::vector<double> theta_cos_mat(theta_num_);
        std::vector<double> theta_sin_mat(theta_num_);
        theta_mat_.reserve(theta_num_);
        for (size_t i = 0; i < theta_num_; ++i)
        {
            theta_mat_[i] = i * CV_PI / theta_num_;
            theta_cos_mat[i] = cos(theta_mat_[i]);
            theta_sin_mat[i] = sin(theta_mat_[i]);
        }

        //// init the radium
        radium_mat_.reserve(im_h);
        int radium_shift_ = radium_num_ >> 1;
        for (size_t i = 0; i < im_h; ++i)
        {
            radium_mat_[i].reserve(im_w);
            for (size_t j = 0; j < im_w; ++j)
            {
                radium_mat_[i][j].reserve(theta_num_);
                for (size_t k = 0; k < theta_num_; ++k)
                {
                    radium_mat_[i][j][k] = int(((j - pos_x) * theta_cos_mat[k] + (i - pos_y) * theta_sin_mat[k])
                                               / radium_interval_) + radium_shift_;
                }
            }
        }

        return 0;
    }

public:
    double radium_interval_; // e.g., 1
    double theta_interval_; // e.g., 1
    int radium_num_; // e.g., 600
    int theta_num_; // e.g., 180
    std::vector<double> theta_mat_;
    std::vector<std::vector<std::vector<int> > > radium_mat_;
};

struct LaneLineDetectionConfig
{
    std::string camera_parameter_filename;              //!< camera parameter filename
    int ipm_cnn_bw_threshold;                           //!< threshold used to binarization of the ipm_cnn image
    float ipm_filter_bw_threshold;                      //!< threshold used to binarization of ipm_filter, i.e., the gradient image

    float ransac_contour_success_probability;           //!< success probability used in RANSAC line fitting
    float ransac_contour_max_outlier_ratio;             //!< outlier ratio used in RANSAC line fitting
    float ransac_contour_threshold;                     //!< the threshold in pixel used in RANSAC line fitting
    float ransac_contour_minimum_required_points;       //!< minimum required number of points to do RANSAC line fitting
    int ransac_contour_constrain_line_direction_ipm;    //!< 1 to remove lines with directions outside the given region.
                                                        //!< 0 to keep all detected lines
    int ransac_contour_constrain_line_length_ipm;       //!< 1 to remove line in small length
                                                        //!< 0 to keep detected lines in small length
    int ransac_contour_minimum_line_length_ipm;         //!< minimum line length in pixel in the IPM image
                                                        //!< not used if ransac_contour_constrain_line_length_ipm is 0
    int merge_line_max_difference_r_ipm;

    std::string paramAsString() const
    {
        std::stringstream ss;

        ss << "====================\n";
        ss << " LaneDetectionConfig parameters\n";
        ss << "====================\n";
        ss << "camera parameter filename: " << camera_parameter_filename << "\n";
        ss << "ipm_cnn_bw_threshold: " << ipm_cnn_bw_threshold << "\n";
        ss << "ipm_filter_bw_threshold: " << ipm_filter_bw_threshold << "\n";
        ss << "ransac_contour_success_probability: " << ransac_contour_success_probability << "\n";
        ss << "ransac_contour_max_outlier_ratio: " << ransac_contour_max_outlier_ratio << "\n";
        ss << "ransac_contour_threshold: " << ransac_contour_threshold << "\n";
        ss << "ransac_contour_minimum_required_points: " << ransac_contour_minimum_required_points << "\n";
        ss << "ransac_contour_constrain_line_direction_ipm: " << ransac_contour_constrain_line_direction_ipm<< "\n";
        ss << "ransac_contour_constrain_line_length_ipm: " << ransac_contour_constrain_line_length_ipm<< "\n";
        ss << "ransac_contour_minimum_line_length_ipm: " << ransac_contour_minimum_line_length_ipm << "\n";
        ss << "merge_line_max_difference_r_ipm: " << merge_line_max_difference_r_ipm << "\n";

        ss << "--------------------\n";

        return ss.str();
    }
};


class LaneLineDetection
{
public:
    explicit LaneLineDetection(
            const LaneLineDetectionConfig& config
    );
    int doLaneLineDetection(
            const cv::Mat& img,
            const cv::Mat& img_cnn,
            const std::string& file_name
    );

public:
    std::string paramAsString() const;

public:
    //====================
    //  getters
    //--------------------
    CameraInfo getCameraInfo() const {return camera_info_;}
    const IpmTF& getIpmTF() const {return ipm_tf_;}
    cv::Rect getRoiOut() const {return roi_out_;}
    const LaneLineDetectionConfig& getConfig() const {return config_;}

private:
    // priority-1
    int initializeWithContours();
    int initializeWithHough();
    int mergeWithDist();
    int mergeWithCluster();
    int refine();
    int showImage(
            const cv::Mat& im,
            const int& mode,
            const std::string& win_name,
            const int& sleep_time
    );

    // priority-2
    int filterImage();
    cv::Mat getHoughImage(
            const cv::Mat& im_filter
    );
    LaneLine fromRthetaToLaneline(
            const cv::Point2d& pt
    );
    cv::Point2d refineLoc(
            const cv::Mat& im,
            cv::Point& pt
    );
    int collectLanePoints(
            const cv::Mat& im_binary,
            const cv::Mat& im_weight,
            const cv::Rect& rt,
            std::vector<cv::Point2d>& pts
    );
    LaneLine fromPtsToLaneline(
            const std::vector<cv::Point2d>& pts,
            const cv::Rect& rt
    );
    LaneLine mergeTwoLaneLines(
            const LaneLine& l1,
            const LaneLine& l2
    );

    // utils
    int initLaneLine(
            LaneLine& lane_line
    );
    static bool sortByYaxisAsc(
            const cv::Point2d& pt1,
            const cv::Point2d& pt2
    );

public:
    std::vector<LaneLine> lane_lines_; //!< detected lane lines
    cv::Mat img_;       //!< original image
    cv::Mat img_cnn_;   //!<
    cv::Mat img_mask_;  //!< IPM image mask
    cv::Mat ipm_;       //!< IPM image
    cv::Mat ipm_gray_;  //!< the grayscale version of ipm_
    cv::Mat ipm_cnn_;
    cv::Mat ipm_filter_; //!< the gradient magnitude of ipm_ (after filtering) with type CV_64F and value range [0,1]
    cv::Mat ipm_mask_;
    cv::Mat ipm_weight_;
    std::string file_name_;

private:
    //// parameter
    cv::Rect roi_in_;
    cv::Rect roi_out_;
    cv::Point2d car_pos_;
    cv::Point2d vanish_pt_;
    double max_dist_to_vanish_pt_;

    //// key modules
    HoughTF hough_tf_;
    IpmTF ipm_tf_;
    CameraInfo camera_info_;
    Ransac ransac_;

private:
    LaneLineDetectionConfig config_;
};


LaneLineDetection::LaneLineDetection(
        const LaneLineDetectionConfig& config
)
{
    config_ = config;
    //// deal with setup_path
    YAML::Node doc = YAML::LoadFile(config_.camera_parameter_filename);
    camera_info_.fx = doc["fx"].as<double>();
    camera_info_.fy = doc["fy"].as<double>();
    camera_info_.cx = doc["cx"].as<double>();
    camera_info_.cy = doc["cy"].as<double>();
    camera_info_.pitch = doc["pitch"].as<double>();
    camera_info_.yaw = doc["yaw"].as<double>();
    camera_info_.roll = doc["roll"].as<double>();
    roi_in_.x = doc["in_x1"].as<int>();
    roi_in_.width = doc["in_x2"].as<int>(); // TODO: (fangjun) fix it. width = x2-x1, and change IpmTF::doIpm
    roi_in_.y = doc["in_y1"].as<int>();
    roi_in_.height = doc["in_y2"].as<int>();
    roi_out_.width = doc["out_w"].as<int>();
    roi_out_.height = doc["out_h"].as<int>();
    roi_out_.x = roi_out_.y = 0;

    //// basic parameter
    vanish_pt_ = cv::Point2d(560, 360); // TODO: (fangjun) it should not be a constant value. center of the circle
    max_dist_to_vanish_pt_ = 100; // (fangjun): radius of the cirle
//    std::vector<float> measure_vals;
//    measure_vals.push_back(vanish_pt_.x);
//    measure_vals.push_back(vanish_pt_.y);
//    kalman_vp_.init(2, 2, measure_vals, 1e-5, 1e-5);
    car_pos_ = cv::Point2d(roi_out_.width >> 1, roi_out_.height); // assume the car is at the center

    //// init the key modules
    ipm_tf_.doIpm(camera_info_, roi_in_, roi_out_);
    hough_tf_.doInit(roi_out_.width, roi_out_.height,
        car_pos_.x, car_pos_.y);
}


int LaneLineDetection::filterImage()
{
    ROS_INFO("start to filter image");
    int kernel_w = 5;
    int kernel_h = 11;

    //// blur the image
    cv::Mat ipm_smooth;
    cv::cvtColor(ipm_, ipm_gray_, cv::COLOR_BGR2GRAY);
    cv::blur(ipm_gray_, ipm_smooth, cv::Size(kernel_w, kernel_h));

    ipm_filter_ = cv::Mat::zeros(roi_out_.height, roi_out_.width, CV_64F);
    const int x_shift = kernel_w;
    const int x_left_shift = x_shift + (x_shift >> 1);

    for (int i = 0; i < roi_out_.height; ++i)
    {
        uchar *p1 = ipm_smooth.ptr<uchar>(i);
        double *p2 = ipm_filter_.ptr<double>(i);

        // 2.1. scan each row
        for (int j = x_left_shift; j < roi_out_.width - x_left_shift; ++j)
        {
            int middle = p1[j];
            int left = p1[j - x_shift];
            int right = p1[j + x_shift];

            if (middle > left && middle > right)
            {
                p2[j] = (middle << 1) - left - right;
            }
        }
    }

    //// avoid the effect of the boundary
    ipm_mask_.convertTo(ipm_mask_, CV_64F);
    ipm_filter_ = ipm_filter_.mul(ipm_mask_);


//    double min_val;
//    double max_val;
//    cv::Point min_loc;
//    cv::Point max_loc;
//    cv::minMaxLoc(ipm_filter_, &min_val, &max_val, &min_loc, &max_loc);
//    std::cout << max_val << std::endl;

    //// normalize
    cv::threshold(ipm_filter_, ipm_filter_, 255.0, 255.0, CV_THRESH_TRUNC);
    cv::normalize(ipm_filter_, ipm_filter_, 1.0, 0.0, cv::NORM_MINMAX); // ipm_filter is of type CV_64F in the range [0,1]

    return 0;
}

cv::Mat LaneLineDetection::getHoughImage(
        const cv::Mat& im_filter
)
{
    ROS_INFO("start to get hough image");
    cv::Mat im_hough = cv::Mat::zeros(
            hough_tf_.theta_num_, hough_tf_.radium_num_, CV_64F);

    //// scan the im_filter
    for (size_t i = 0; i < im_filter.rows; ++i)
    {
        const double *p2 = im_filter.ptr<double>(i);
        for (size_t j = 0; j < im_filter.cols; ++j)
        {
            if (im_filter.at<double>(i, j) < 0.15)
            {
                continue;
            }

            for (int k = 0; k < hough_tf_.theta_num_; ++k)
            {
                im_hough.at<double>(k, hough_tf_.radium_mat_[i][j][k]) += p2[j];
            }
        }
    }

    //// re-assign the image
    cv::Mat im_hough_new = im_hough.clone();
    int theta_copy = hough_tf_.theta_num_ >> 1;
    im_hough_new(cv::Rect(0, 0, hough_tf_.radium_num_, theta_copy)).
            copyTo(im_hough(cv::Rect(0, theta_copy, hough_tf_.radium_num_, theta_copy)));
    cv::flip(im_hough_new(cv::Rect(0, theta_copy, hough_tf_.radium_num_, theta_copy)),
             im_hough(cv::Rect(0, 0, hough_tf_.radium_num_, theta_copy)), 1);

    cv::threshold(im_hough, im_hough, 255.0, 255.0, CV_THRESH_TRUNC);

//    cv::Mat im_hough_tmp;
//    cv::normalize(im_hough, im_hough_tmp, 1.0, 0.0, cv::NORM_MINMAX);
    cv::imshow("im_hough", im_hough / 255.0);
    cv::imshow("im_filter", im_filter);

    return im_hough;

}

LaneLine LaneLineDetection::fromRthetaToLaneline(
        const cv::Point2d& pt
)
{
    LaneLine lane_line;

    double theta = (pt.y - (hough_tf_.theta_num_ >> 1))
                   / hough_tf_.theta_num_ * CV_PI;
    double radium = (pt.x - (hough_tf_.radium_num_ >> 1)) * hough_tf_.radium_interval_;

    //// from rtheta to point
    double a = cos(theta), b = sin(theta);
    double x0 = a*radium, y0 = b*radium;

    lane_line.pt_start_ipm.y = -10000 + 0 - car_pos_.y;
    lane_line.pt_start_ipm.x = -(lane_line.pt_start_ipm.y - y0) * b / a + x0;
    lane_line.pt_end_ipm.y = roi_out_.height - car_pos_.y;
    lane_line.pt_end_ipm.x = -(lane_line.pt_end_ipm.y - y0) * b / a + x0;
    lane_line.pt_start_ipm += car_pos_;
    lane_line.pt_end_ipm += car_pos_;
    ransac_.xyToRtheta(lane_line.pt_start_ipm, lane_line.pt_end_ipm,
            car_pos_, lane_line.radium_ipm, lane_line.theta_ipm);

    //// deal with img
    lane_line.pt_start_img = ipm_tf_.ptTF(lane_line.pt_start_ipm,
                                          ipm_tf_.getIpmToImg());
    lane_line.pt_end_img = ipm_tf_.ptTF(lane_line.pt_end_ipm,
                                        ipm_tf_.getIpmToImg());

    std::vector<cv::Point2d> candidates;
    candidates.push_back(lane_line.pt_start_img);
    candidates.push_back(lane_line.pt_end_img);
    cv::fitLine(candidates, lane_line.vec_img, CV_DIST_L2, 0, 0.01, 0.01);
    double tmp = lane_line.vec_img.val[0];
    lane_line.vec_img.val[0] = -lane_line.vec_img.val[1];
    lane_line.vec_img.val[1] = tmp;

    return lane_line;
}

cv::Point2d LaneLineDetection::refineLoc(
        const cv::Mat& im,
        cv::Point& pt
)
{
    cv::Point2d pt_fusion = cv::Point2d(0, 0);
    double weight = 0.0;
    for (int i = pt.x - 1; i <= pt.x + 1; ++i)
    {
        for (int j = pt.y - 1; j <= pt.y + 1; ++j)
        {
            pt_fusion += cv::Point2d(im.at<double>(j, i) * i, im.at<double>(j, i) * j);
            weight += im.at<double>(j, i);
        }
    }

    pt_fusion.x /= weight;
    pt_fusion.y /= weight;

    return pt_fusion;
}

int LaneLineDetection::initializeWithHough()
{
    //// filter the image, get the hough image
    filterImage();
    cv::Mat im_hough = getHoughImage(ipm_filter_);

    //// find the maximal point as the line
    double min_val;
    double max_val;
    cv::Point min_loc;
    cv::Point max_loc;
    cv::minMaxLoc(im_hough, &min_val, &max_val, &min_loc, &max_loc);

    cv::Point2d max_loc_refined = refineLoc(im_hough, max_loc);
    std::cout << max_val << std::endl;
    if (max_val < 100)
    {
        return -1;
    }

    //// generate the lane and the corresponding
    LaneLine lane_line = fromRthetaToLaneline(max_loc_refined);
    lane_lines_.push_back(lane_line);
    int count = 0;
    lane_line.id = count++;
    std::cout << max_loc_refined << std::endl;

    int min_interval = 40;
    for (int shift = -10; shift <= 10; ++shift)
    {
        if (shift == 0)
        {
            continue;
        }

        int new_radium = max_loc.x + shift * min_interval;
        cv::Rect roi = cv::Rect(-10, -10, 10, 10);

        int x_min = std::max(0, new_radium + roi.x);
        int x_max = std::min(im_hough.cols, new_radium + roi.width);
        int y_min = std::max(0, max_loc.y + roi.y);
        int y_max = std::min(im_hough.rows, max_loc.y + roi.height);

        if (x_max < x_min + 20 || y_max < y_min + 20)
        {
            continue;
        }

        double min_val_other;
        double max_val_other;
        cv::Point min_loc_other;
        cv::Point max_loc_other;
        cv::minMaxLoc(im_hough(cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min)),
                      &min_val_other, &max_val_other, &min_loc_other, &max_loc_other);

        if (max_val_other < 50)
        {
            continue;
        }

        max_loc_other.x += x_min;
        max_loc_other.y += y_min;
        cv::Point2d max_loc_other_refined = refineLoc(im_hough, max_loc_other);
        if (fabs(max_loc_refined.y - max_loc_other_refined.y) * hough_tf_.theta_interval_ > 10)
        {
            continue;
        }


        LaneLine lane_line = fromRthetaToLaneline(max_loc_other_refined);
        lane_lines_.push_back(lane_line);
        lane_line.id = count++;
        std::cout << max_loc_other_refined << std::endl;
    }

    return 0;
}

int LaneLineDetection::initLaneLine(
        LaneLine& lane_line
)
{
    //// ipm calc
    double start_y = lane_line.pts_ipm[0].y;
    double end_y = lane_line.pts_ipm[lane_line.pts_ipm.size() - 1].y;

    // start_x
    double x = -lane_line.vec_ipm[1] / lane_line.vec_ipm[0] * (start_y - lane_line.vec_ipm[3]) + lane_line.vec_ipm[2];
    lane_line.pt_start_ipm = cv::Point2d(x, start_y);
    x = -lane_line.vec_ipm[1] / lane_line.vec_ipm[0] * (end_y - lane_line.vec_ipm[3]) + lane_line.vec_ipm[2];
    lane_line.pt_end_ipm = cv::Point2d(x, end_y);
    lane_line.len_ipm = sqrt((lane_line.pt_start_ipm.x - lane_line.pt_end_ipm.x) * (lane_line.pt_start_ipm.x - lane_line.pt_end_ipm.x)
                             + (lane_line.pt_start_ipm.y - lane_line.pt_end_ipm.y) * (lane_line.pt_start_ipm.y - lane_line.pt_end_ipm.y));

    ransac_.xyToRtheta(lane_line.pt_start_ipm, lane_line.pt_end_ipm,
                       car_pos_, lane_line.radium_ipm, lane_line.theta_ipm);


    //// img calc
    lane_line.pt_start_img = ipm_tf_.ptTF(lane_line.pt_start_ipm, ipm_tf_.getIpmToImg());
    lane_line.pt_end_img = ipm_tf_.ptTF(lane_line.pt_end_ipm, ipm_tf_.getIpmToImg());
    ransac_.xyToRtheta(lane_line.pt_start_img, lane_line.pt_end_img,
                       car_pos_, lane_line.radium_ipm, lane_line.theta_ipm);
    lane_line.len_img = sqrt((lane_line.pt_start_img.x - lane_line.pt_end_img.x) * (lane_line.pt_start_img.x - lane_line.pt_end_img.x)
                             + (lane_line.pt_start_img.y - lane_line.pt_end_img.y) * (lane_line.pt_start_img.y - lane_line.pt_end_img.y));
    std::vector<cv::Point2d> candidates;
    candidates.push_back(lane_line.pt_start_img);
    candidates.push_back(lane_line.pt_end_img);
    cv::fitLine(candidates, lane_line.vec_img, CV_DIST_L2, 0, 0.01, 0.01);
    double tmp = lane_line.vec_img.val[0];
    lane_line.vec_img.val[0] = -lane_line.vec_img.val[1];
    lane_line.vec_img.val[1] = tmp;

    return 0;
}

LaneLine LaneLineDetection::fromPtsToLaneline(
        const std::vector<cv::Point2d>& pts,
        const cv::Rect& rt
)
{
    //// init
    LaneLine lane_line;

    std::vector<cv::Point2d> inliers;
    std::vector<cv::Point2d> outliers;
    ransac_.fitLine(pts, lane_line.vec_ipm,
                    lane_line.pts_ipm, outliers);
    initLaneLine(lane_line);

    //// other para
    lane_line.occupied.resize(roi_out_.height, 0);
    for (size_t i = std::max(0, int(lane_line.pts_ipm[0].y)); i < std::min(roi_out_.height, int(lane_line.pts_ipm.back().y)); ++i)
    {
        lane_line.occupied[i] = 1;
    }


    return lane_line;
}

int LaneLineDetection::collectLanePoints(
        const cv::Mat& im_binary,
        const cv::Mat& im_weight,
        const cv::Rect& rt,
        std::vector<cv::Point2d>& pts
)
{

    for (int j = rt.y; j < rt.y + rt.height; ++j)
    {
        const uchar *p1 = im_binary.ptr<uchar>(j);
        const uchar *p2 = im_weight.ptr<uchar>(j);

        double sum_x_plus_weight = 0.0f;
        double sum_weight = 0.0f;

        for (int i = rt.x; i < rt.x + rt.width; ++i)
        {
            if (p1[i] == 128)
            {
                sum_weight += p2[i];
                sum_x_plus_weight += i * p2[i];
            }
        }

        if (sum_weight > 0.0)
        {
            pts.push_back(cv::Point2d(sum_x_plus_weight / sum_weight, j));
        }
    }

    return 0;
}

/**
 * Basic steps
 * 1. compute the gradient image, stored in ipm_filter_
 * 2. binarization of ipm_filter_, stored in ipm_bw_filter
 * 3. binarization of ipm_cnn, stored in ipm_bw_cnn
 * 4. compute the logic and of ipm_bw_filter and ipm_bw_cnn, stored in im_binary
 * 5. mulitply ipm_filter_ with ipm_cnn, stored in ipm_weight
 * 6. find the contours of im_binary
 * 7. for each found contour, use RANSAC to find a line inside the contour
 * 8. return the detected lines, stored in lane_lines_
 *
 */
int LaneLineDetection::initializeWithContours()
{
    //// filter image
    filterImage();

    //// get the binary and weight
    cv::Mat ipm_bw_filter;
    cv::threshold(ipm_filter_, ipm_bw_filter,
                  config_.ipm_filter_bw_threshold,
                  1.0, CV_THRESH_BINARY);
    ipm_bw_filter.convertTo(ipm_bw_filter, CV_8U);

    cv::Mat ipm_bw_cnn;
    if (ipm_cnn_.channels() > 1)
    {
        cv::cvtColor(ipm_cnn_, ipm_cnn_, cv::COLOR_BGR2GRAY); // since cv::threshold supports only single channel mat
    }

    cv::threshold(ipm_cnn_, ipm_bw_cnn,
                  config_.ipm_cnn_bw_threshold,
                  1, CV_THRESH_BINARY);

    cv::imshow("ipm_bw_cnn", ipm_bw_cnn*255);
    cv::Mat im_binary = ipm_bw_filter & ipm_bw_cnn; // (fangjun): ipm_bw_filter is either 0 or 1, ipm_bw_cnn is either 0 or 1
                                                    // (fangjun): therefore, im_binary is either 0 or 1
    //im_binary = ipm_bw_filter;
    cv::imshow("im_binary", im_binary*255);

    ipm_cnn_.convertTo(ipm_cnn_, CV_64F);
    ipm_weight_ = ipm_filter_.mul(ipm_cnn_); // (fangjun) ipm_filter_ is in [0,1]
    ipm_weight_.convertTo(ipm_weight_, CV_8U);
    cv::imshow("ipm_weight", ipm_weight_*255);

    //// use contour to initialize
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(im_binary, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    ransac_.doInit(config_.ransac_contour_success_probability,
                   config_.ransac_contour_max_outlier_ratio,
                   config_.ransac_contour_threshold);

    cv::Mat im_binary_tmp;
    size_t sz = contours.size();
    for (size_t i = 0; i < sz; ++i)
    {

        im_binary.copyTo(im_binary_tmp);
        cv::drawContours(im_binary_tmp, contours, i, cv::Scalar(128), CV_FILLED); // (fangjun): 128 is used inside collectLanePoints

        cv::Rect rt = cv::boundingRect(contours[i]);
        std::vector<cv::Point2d> pts;

        collectLanePoints(im_binary_tmp, ipm_weight_, rt, pts); // TODO: (fangjun) skip it if the height is less than 10, see the following threshold
//        std::sort(pts.begin(), pts.end(), sortByYaxisAsc);
        if (pts.size() < config_.ransac_contour_minimum_required_points)
        {
            //ROS_INFO_STREAM("skip: " << i);
            continue;
        }

        LaneLine lane_line = fromPtsToLaneline(pts, rt);

        lane_line.id = i;
        if (config_.ransac_contour_constrain_line_length_ipm)
        {
            if (lane_line.len_ipm < config_.ransac_contour_minimum_line_length_ipm)
            {
                continue;
            }
        }

        if (config_.ransac_contour_constrain_line_direction_ipm)
        {
            if ((-CV_PI / 4 < lane_line.theta_ipm && lane_line.theta_ipm < CV_PI / 4)
                || (-CV_PI * 3 / 4 > lane_line.theta_ipm)
                || (CV_PI * 3 / 4 < lane_line.theta_ipm))
            {
                //ROS_INFO_STREAM("skip: " << i << " due to theta_ipm");
                continue;
            }
        }

        lane_lines_.push_back(lane_line);
    }

    ROS_INFO_STREAM("There are " << lane_lines_.size() << " lines detected");

    return 0;
}

bool LaneLineDetection::sortByYaxisAsc(
        const cv::Point2d& pt1,
        const cv::Point2d& pt2)
{
    return pt1.y < pt2.y;
}

LaneLine LaneLineDetection::mergeTwoLaneLines(
        const LaneLine& l1,
        const LaneLine& l2
)
{
    //// merge pts and re-ransac
    LaneLine lane_line;
    lane_line.pts_ipm.insert(lane_line.pts_ipm.end(),
                             l1.pts_ipm.begin(), l1.pts_ipm.end());
    lane_line.pts_ipm.insert(lane_line.pts_ipm.end(),
                             l2.pts_ipm.begin(), l2.pts_ipm.end());
    lane_line.occupied = l1.occupied;
    for (int i = 0; i < roi_out_.height; ++i)
    {
        if (1 == l2.occupied[i])
        {
            lane_line.occupied[i] = 1;
        }
    }
    std::sort(lane_line.pts_ipm.begin(), lane_line.pts_ipm.end(), sortByYaxisAsc);
    std::vector<cv::Point2d> pts = lane_line.pts_ipm;
    std::vector<cv::Point2d> outliers;
    ransac_.fitLine(pts, lane_line.vec_ipm, lane_line.pts_ipm, outliers);

    initLaneLine(lane_line);
    lane_line.id = l1.id;
    return lane_line;
}

int LaneLineDetection::mergeWithDist()
{
    // sort with y-axis desc
//    std::sort(lane_lines_.begin(), lane_lines_.end(), sortByLenDesc);
    //ransac_.doInit(0.9999, 0.5, 3);
    ransac_.doInit(config_.ransac_contour_success_probability,
                   config_.ransac_contour_max_outlier_ratio,
                   config_.ransac_contour_threshold
    );


    // loop, merge two lane line
    while (true)
    {
        bool flag = false;
        for (size_t i = 0; i < lane_lines_.size(); ++i)
        {
            for (size_t j = i + 1; j < lane_lines_.size(); ++j)
            {

                if (fabs(lane_lines_[i].theta_ipm - lane_lines_[j].theta_ipm) <= 3 * CV_PI / 180
                    && fabs(lane_lines_[i].radium_ipm - lane_lines_[j].radium_ipm) <= config_.merge_line_max_difference_r_ipm)
                {

                    int sum = 0;
                    for (size_t ii = 0; ii < roi_out_.height; ++ii)
                    {
                        if (1 == lane_lines_[i].occupied[ii] && 1 == lane_lines_[j].occupied[ii])
                        {
                            ++sum;
                        }
                    }

                    if (sum > 0)
                    {
                        continue;
                    }

                    ROS_INFO("[MERGE] lane %d (theta=%f, radium=%f) + lane %d (theta=%f, radium=%f)",
                             lane_lines_[i].id, lane_lines_[i].theta_ipm * 180 / CV_PI, lane_lines_[i].radium_ipm,
                             lane_lines_[j].id, lane_lines_[j].theta_ipm * 180 / CV_PI, lane_lines_[j].radium_ipm);

                    lane_lines_[i] = mergeTwoLaneLines(lane_lines_[i], lane_lines_[j]);

                    ROS_INFO("= lane %d (theta=%f, radium=%f)",
                             lane_lines_[i].id, lane_lines_[i].theta_ipm * 180 / CV_PI, lane_lines_[i].radium_ipm);


                    lane_lines_.erase(lane_lines_.begin() + j);
                    --j;
                    flag = true;
                }
            }
        }

        if (!flag)
        {
            break;
        }
    }


    return 0;
}

int LaneLineDetection::mergeWithCluster()
{
    //// trans radium and theta to new image
    cv::Mat im_projection;


    return 0;
}

int LaneLineDetection::refine()
{
    //// use vanish pt to refine
    for (size_t i = 0; i < lane_lines_.size(); ++i)
    {
        double dist = fabs(ransac_.getDist(lane_lines_[i].vec_img, vanish_pt_));
        if (dist > max_dist_to_vanish_pt_)
        {
            lane_lines_.erase(lane_lines_.begin() + i);
            --i;
        }
    }

    return 0;
}

int LaneLineDetection::showImage(
        const cv::Mat& im,
        const int& mode, // 0-img, 1-ipm
        const std::string& win_name,
        const int& sleep_time
)
{
    // 0 = img, 1 = ipm
    cv::Mat im_show = im.clone();
    if (im.channels() == 1)
    {
        cv::cvtColor(im, im_show, cv::COLOR_GRAY2BGR);
    }
    if (mode == 0)
    {
        //// draw pts
        for (size_t i = 0; i < lane_lines_.size(); ++i)
        {
            cv::line(im_show, lane_lines_[i].pt_start_img, lane_lines_[i].pt_end_img,
                     cv::Scalar(0, 255, 0), 2);
            ROS_INFO("#%d lane line: (%f, %f) to (%f, %f)", lane_lines_[i].id,
                     lane_lines_[i].pt_start_img.x, lane_lines_[i].pt_start_img.y,
                     lane_lines_[i].pt_end_img.x, lane_lines_[i].pt_end_img.y);
        }


        cv::circle(im_show, vanish_pt_, max_dist_to_vanish_pt_, cv::Scalar(0, 0, 255), 2);
        cv::circle(im_show, vanish_pt_, 5, cv::Scalar(255, 255, 255), -1);

        cv::resize(im_show, im_show, cv::Size(im_show.cols / 2, im_show.rows / 2));

        //// draw line

    }
    else if (mode == 1)
    {
        //// draw pts
        for (size_t i = 0; i < lane_lines_.size(); ++i)
        {
            cv::line(im_show, lane_lines_[i].pt_start_ipm, lane_lines_[i].pt_end_ipm,
                     cv::Scalar(0, 255, 0), 2);
            ROS_INFO("#%d lane line: (%f, %f) to (%f, %f)\n"
                     "  theta: %f, r: %f"
            ,
                     lane_lines_[i].id,
                     lane_lines_[i].pt_start_ipm.x, lane_lines_[i].pt_start_ipm.y,
                     lane_lines_[i].pt_end_ipm.x, lane_lines_[i].pt_end_ipm.y,
                     lane_lines_[i].theta_ipm,
                     lane_lines_[i].radium_ipm
            );

            cv::String point1 = cv::format("start%d: (%f, %f)",lane_lines_[i].id, lane_lines_[i].pt_start_ipm.x, lane_lines_[i].pt_start_ipm.y);
            cv::String point2 = cv::format("end%d: (%f, %f)", lane_lines_[i].id,lane_lines_[i].pt_end_ipm.x, lane_lines_[i].pt_end_ipm.y);
            cv::putText(im_show, point1,
                        lane_lines_[i].pt_start_ipm,
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.6, cv::Scalar(255,255,255),
                        1
            );
            cv::putText(im_show, point2,
                        lane_lines_[i].pt_end_ipm,
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.6, cv::Scalar(255,255,255),
                        1
            );
        }

        //// draw line



    }

    cv::imshow(win_name, im_show);
    cv::waitKey(sleep_time);

    return 0;
}


int LaneLineDetection::doLaneLineDetection(
        const cv::Mat& img,
        const cv::Mat& img_cnn,
        const std::string& file_name
)
{
    lane_lines_.clear();
    img_ = img.clone();
    img_cnn_ = img_cnn.clone();
    ipm_cnn_ = img_cnn.clone();
    file_name_ = file_name;

    cv::warpPerspective(img, ipm_, ipm_tf_.getImgToIpm(), cv::Size(roi_out_.width, roi_out_.height));
    cv::imshow("ipm-original"+file_name, ipm_);

    img_mask_ = cv::Mat::ones(img.rows, img.cols, CV_8UC1);
    cv::warpPerspective(img_mask_, ipm_mask_, ipm_tf_.getImgToIpm(), cv::Size(roi_out_.width, roi_out_.height));
    cv::Mat element = getStructuringElement(0, cv::Size(11, 11));
    erode(ipm_mask_, ipm_mask_, element);

    //// coarse detection
    ROS_INFO("#### coarse detection");
#if 0
    initializeWithHough();      // use Hough transform to detect lines
#else
    initializeWithContours();   // use the contours and RANSAC to detect lines
#endif
    showImage(ipm_, 1, "coarse", 1);  // mode: 1-ipm,

    //// merge
    ROS_INFO("#### merge detection");
    //mergeWithCluster();
    mergeWithDist();
    showImage(ipm_, 1, "merge", 1);  // mode 1-ipm

    //// fine detection
    ROS_INFO("#### fine detection");
    refine();
    showImage(img_, 0, "fine", 1);  // mode 0-original image

    return 0;
}

std::string LaneLineDetection::paramAsString() const
{
    std::stringstream ss;

    ss << "\n";
    ss << camera_info_.asString();

    ss << "====================\n";
    ss << " ROI parameters\n";
    ss << "====================\n";
    ss << "in_x1: " << roi_in_.x << "\n";
    ss << "in_x2: " << roi_in_.width << "\n";
    ss << "in_y1: " << roi_in_.y << "\n";
    ss << "in_y2: " << roi_in_.height << "\n";
    ss << "out_w: " << roi_out_.width << "\n";
    ss << "out_h: " << roi_out_.height << "\n";

    ss << "--------------------\n";

    ss << ransac_.paramAsString();
    return ss.str();
}

#endif
