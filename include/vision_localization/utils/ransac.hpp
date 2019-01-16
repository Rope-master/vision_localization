#include <opencv2/opencv.hpp>
#include <time.h>

class Ransac
{
public:
    int doInit(
            const double& success_prob,
            const double& max_outliers,
            const double& thres);

    int fitLine(
            const std::vector<cv::Point2d>& pts,
            cv::Vec4d& vec,
            std::vector<cv::Point2d>& inliers,
            std::vector<cv::Point2d>& outliers);

    // std::vector<cv::Point2d> fitPolyLine();

    /**
     *
     * @param vec [in] (vec[0], vec[1]) is the unit normal vector of the line,
     *            (vec[2], vec[3]) is a point on the line
     * @param pt  [in] a point
     * @return Signed distance of the point to the line
     */
    double getDist(
            const cv::Vec4d& vec,
            const cv::Point2d& pt);

    int xyToRtheta(
            const cv::Point2d& start_pt,
            const cv::Point2d& end_pt,
            const cv::Point2d& refer_center_pt,
            cv::Point2d& rtheta);

    int xyToRtheta(
            const cv::Point2d& start_pt,
            const cv::Point2d& end_pt,
            const cv::Point2d& refer_center_pt,
            double& radium,
            double& theta);
public:
    std::string paramAsString() const;

private:
    bool isInlierInLine(
            const cv::Vec4d& vec,
            const cv::Point2d& pt);
    // bool isInlierInPoly(
    //     const cv::Vec4f& vec,
    //     const cv::Point2f& pt);

private:
    double success_prob_;
    double max_outliers_;
    double thres_;
};

int Ransac::xyToRtheta(
        const cv::Point2d& start_pt,
        const cv::Point2d& end_pt,
        const cv::Point2d& refer_center_pt,
        cv::Point2d& rtheta)
{
    cv::Point2d start_pt_new = start_pt - refer_center_pt;
    cv::Point2d end_pt_new = end_pt - refer_center_pt;

    // std::cout << refer_center_pt << std::endl;

    if (start_pt_new.x == end_pt_new.x)
    {
        rtheta.x = start_pt_new.x;//fabs(start_pt_new.y);
        rtheta.y = CV_PI / 2;//start_pt_new.y >= 0 ? CV_PI / 2 : -CV_PI / 2;
    }
    else if (start_pt_new.y == end_pt_new.y)
    {
        rtheta.x = start_pt_new.y;//fabs(start_pt_new.x);
        rtheta.y = 0;//start_pt_new.x >= 0 ? 0 : CV_PI;
    }
    else
    {
        rtheta.y = std::atan2(end_pt_new.y - start_pt_new.y,
                              end_pt_new.x - start_pt_new.x);
        double r1 = -start_pt_new.y * std::cos(rtheta.y) + start_pt_new.x * std::sin(rtheta.y);
        double r2 = -end_pt_new.y * std::cos(rtheta.y) + end_pt_new.x * std::sin(rtheta.y);

        // std::cout << r1 << " " << r2 << std::endl;

        if (r1 < 0 || r2 < 0)
        {
            rtheta.y += rtheta.y < 0 ? CV_PI : 0;
            // rtheta.y += CV_PI;
            // rtheta.y -= rtheta.y > CV_PI ? CV_PI : 0;
        }
        rtheta.x = r2;//fabs(r2);
    }

    return 0;
}

int Ransac::xyToRtheta(
        const cv::Point2d& start_pt,
        const cv::Point2d& end_pt,
        const cv::Point2d& refer_center_pt,
        double& radium,
        double& theta)
{
    cv::Point2d start_pt_new = start_pt - refer_center_pt;
    cv::Point2d end_pt_new = end_pt - refer_center_pt;

    // std::cout << refer_center_pt << std::endl;

    if (start_pt_new.x == end_pt_new.x)
    {
        radium = start_pt_new.x;//fabs(start_pt_new.y);
        theta = CV_PI / 2;//start_pt_new.y >= 0 ? CV_PI / 2 : -CV_PI / 2;
    }
    else if (start_pt_new.y == end_pt_new.y)
    {
        radium = start_pt_new.y;//fabs(start_pt_new.x);
        theta = 0;//start_pt_new.x >= 0 ? 0 : CV_PI;
    }
    else
    {
        theta = std::atan2(end_pt_new.y - start_pt_new.y,
                              end_pt_new.x - start_pt_new.x);
        double r1 = -start_pt_new.y * std::cos(theta) + start_pt_new.x * std::sin(theta);
        double r2 = -end_pt_new.y * std::cos(theta) + end_pt_new.x * std::sin(theta);
        std::cout << "start_pt: " << start_pt << std::endl;
        std::cout << "end_pt: " << end_pt << std::endl;
        std::cout << "start_pt_new: " << start_pt_new << std::endl;
        std::cout << "end_pt_new: " << end_pt_new << std::endl;
        std::cout << "r1: " << r1 << std::endl;
        std::cout << "r2: " << r2 << std::endl;
        std::cout << "theta: " << theta << std::endl;

        // std::cout << r1 << " " << r2 << std::endl;


        if (r1 < 0 || r2 < 0)
        {
            theta += theta < 0 ? CV_PI : 0;
            // rtheta.y += CV_PI;
            // rtheta.y -= rtheta.y > CV_PI ? CV_PI : 0;
        }
        radium = r2;//fabs(r2);
    }

    return 0;
}


int Ransac::fitLine(
        const std::vector<cv::Point2d>& pts,
        cv::Vec4d& vec,
        std::vector<cv::Point2d>& inliers,
        std::vector<cv::Point2d>& outliers)
{
    inliers.clear();
    outliers.clear();

    if (pts.size() < 2)
    {
        vec = cv::Vec4d(1, 1, 1, 1);

        return -1;
    }

    double numerator = log(1 - success_prob_);
    double denominator = log(1 - (1 - max_outliers_) * (1 - max_outliers_));
    int ransac_iter = ceil(numerator / denominator); // todo: (fangjun) recompute it after each iteration and exit the loop beforehand

    srand((unsigned)time(NULL));
    int len = pts.size();

    std::vector<cv::Point2d> candidates;
    int max_inlier = INT_MIN;

    for (int i = 0; i < ransac_iter; ++i)
    {
        candidates.clear();

        int idx1 = rand() % len;
        int idx2 = -1;
        while (1)
        {
            idx2 = rand() % len;
            if (idx1 != idx2)
            {
                break;
            }
        }

        candidates.push_back(pts[idx1]);
        candidates.push_back(pts[idx2]);

        cv::Vec4d vec_tmp;
        cv::fitLine(candidates, vec_tmp, CV_DIST_L2, 0, 0.01, 0.01);
        double tmp = vec_tmp.val[0];
        vec_tmp.val[0] = -vec_tmp.val[1];
        vec_tmp.val[1] = tmp; // (fangjun) now (vec_tmp[0], vec_tmp[1]) is the unit normal vector to the line

        int count_inlier = 0;
        for (int j = 0; j < len; ++j)
        {
            if (isInlierInLine(vec_tmp, pts[j]))
            {
                ++count_inlier;
            }
        }

        if (count_inlier > max_inlier)
        {
            max_inlier = count_inlier;
            vec = vec_tmp;
        }
    }

    // std::cout << vec << std::endl;
    // use the best pts to refine the vec
    for (int i = 0; i < len; ++i)
    {
        if (isInlierInLine(vec, pts[i]))
        {
            inliers.push_back(pts[i]);
        }
        else
        {
            outliers.push_back(pts[i]);
        }
    }
    if (inliers.size() < 2)
    {
        vec = cv::Vec4d(1, 1, 1, 1);

        return -2;
    }

    // std::cout << vec << std::endl;
    cv::fitLine(inliers, vec, CV_DIST_L2, 0, 0.01, 0.01);
    // std::cout << vec << std::endl;
    double tmp = vec.val[0];
    vec.val[0] = -vec.val[1];
    vec.val[1] = tmp;

    return 0;
}

double Ransac::getDist(
        const cv::Vec4d& vec,
        const cv::Point2d& pt)
{
    double dist = vec[0] * (pt.x - vec[2]) + vec[1] * (pt.y - vec[3]);
    return dist;
}

bool Ransac::isInlierInLine(
        const cv::Vec4d& vec,
        const cv::Point2d& pt)
{
    return fabs(getDist(vec, pt)) <= thres_;
}

// bool Ransac::isInlierInPoly(
//     const cv::Vec4f& vec,
//     const cv::Point2f& pt)
// {

// }

int Ransac::doInit(
        const double& success_prob,
        const double& max_outliers,
        const double& thres)
{
    success_prob_ = success_prob;
    max_outliers_ = max_outliers;
    thres_ = thres;
}

std::string Ransac::paramAsString() const
{
    std::stringstream ss;

    ss << "====================\n";
    ss << " RANSAC parameters  \n";
    ss << "====================\n";
    ss << "success probability: " << success_prob_ << "\n";
    ss << "max outlier ratio: " << max_outliers_ << "\n";
    ss << "threshold: " << thres_<< "\n";
    ss << "--------------------\n";

    return ss.str();
}
