/*****************************************************************************
 * Macros represent the configurable control variables; please refer to
 * ATN.pdf to understand what each of them means
 *****************************************************************************/
#define GRADIENT_OPERATOR					0

#define THRESHOLD_GRAYSCALE_DELTA_FACTOR	1
#define THRESHOLD_GRADIENT					2

#define LEFT_MOST							3
#define RIGHT_MOST							4

#define VANISHING_POINT_HEIGHT_FACTOR		5
#define SWEEP_LINE_LOWEST					6

#define FIRST_LANE_SEG_LENGTH_MIN			7
#define SECOND_LANE_SEG_LENGTH_MIN			8

#define FIRST_SECOND_DISTANCE				9
#define SECOND_AMPLITUDE					10

#define RECT_LEFT_FACTOR					11
#define RECT_TOP_FACTOR						12
#define RECT_HEIGHT_FACTOR					13

#define THRESHOLD_REG						14

/*****************************************************************************
 * Macros represent the gradient operators used to achieve the gradient image
 *****************************************************************************/
#define SOBEL	0	// Use Sobel operator
#define LAPLACE 1	// Use Laplace operator
#define CANNY	2	// Use Canny operator

/*****************************************************************************
 * Macros represent the kind of images
 *****************************************************************************/
#define ORIGINAL	0	// The original image
#define GRAYSCALE	1	// The grayscale image
#define GRADIENT	2	// The gradient image
#define BINARY		3	// The binary image

class Detector
{
public:
    int doInit(const int& im_w,
               const int& im_h)
    {
        m_size = cv::Size(im_w,im_h);
        m_nGradientOperator = CANNY;
        m_fThresholdGrayscaleDeltaFactor = 0.4f;
        m_nThresholdGradient = 25;
        m_nLeftMost = 0;
        m_nVerticalCenter = m_size.width / 2;
        m_nRightMost = m_size.width - 1;
        m_nVanishingPointHeight = (int) (m_size.height * 0.4);
        m_nSweepLineHighest = m_nVanishingPointHeight + (int) (m_size.height * 0.03);
        m_nSweepLineLowest = m_size.height - 30;
        m_nFirstLaneSegLengthMin = 10;
        m_nSecondLaneSegLengthMin = 10;
        m_nFirstSecondDistance = 20;
        m_nSecondAmplitude = 20;
        m_nRectLeft = (int) (m_size.width * 0.3);
        m_nRectTop = (int) (m_size.height * 0.85);
        m_nRectWidth = (int) (m_size.width * 0.4);
        m_nRectHeight = (int) (m_size.height * 0.1);
        m_nStripLeft = 0;
        m_nStripTop = (int) (m_size.height * 0.675);
        m_nStripWidth = m_size.width;
        m_nStripHeight = 15;
        m_nPrevLaneX[0] = m_nPrevLaneX[1] = m_nPrevLaneX[2] = -1;
        m_nPrevVerticalCenter = m_size.width / 2;

        m_fThresholdReg = 1.0f;

        m_bTrackingLeft = false;
        m_bTrackingRight = false;
    }

	bool Detect();

	bool Mark(int);
	bool ToFile(const std::string&) const;

public:
	bool SetControlValue(double, int);
	inline void EnableTracking(bool);
	inline bool IsActive() const;

private:
	void InitControlValues();
	void Grayscale(bool);
	void DetermineVerticalCenter();
	void GetThresholdGrayscale();
	void Gradient();
	void Binarize();

	bool NextPoint(bool, int&, int&, int = 0) const;
	bool FindNextWhitePoint(bool, int&, int&, int = 0) const;
	void DetectLeft1();
	void DetectRight1();
	void DetectLeft2();
	void DetectRight2();

	inline void Mark();

private:
	bool m_bActive;				// Indicate the Detector object is active

	cv::Size m_size;				// The size(width and height) of the image or each frame in the video

public:
	cv::Mat img_;
	cv::Mat img_cnn_;
	cv::Mat img_mask_;
	cv::Mat ipm_;
	cv::Mat ipm_gray_;
	cv::Mat ipm_cnn_;
	cv::Mat ipm_filter_;
	cv::Mat ipm_mask_;
	cv::Mat ipm_weight_;
	std::string file_name_;

private:
	// These are control values.
	// For further understanding, please refer to ATN.pdf
	int m_nGradientOperator;

	int m_nThresholdGrayscale;
	float m_fThresholdGrayscaleDeltaFactor;
	int m_nThresholdGradient;

	int m_nLeftMost;
	int m_nVerticalCenter;
	int m_nRightMost;

	int m_nVanishingPointHeight;
	int m_nSweepLineHighest;
	int m_nSweepLineLowest;

	int m_nFirstLaneSegLengthMin;
	int m_nSecondLaneSegLengthMin;

	int m_nFirstSecondDistance;
	int m_nSecondAmplitude;

	int m_nRectLeft;
	int m_nRectTop;
	int m_nRectWidth;
	int m_nRectHeight;

	int m_nStripLeft;
	int m_nStripTop;
	int m_nStripWidth;
	int m_nStripHeight;

	int m_nPrevLaneX[3];
	int m_nPrevVerticalCenter;

	float m_fThresholdReg;

private:
	LaneSeg m_left1;			// The first left lane segment
	LaneSeg m_right1;			// The first right lane segment
	LaneSeg m_left2;			// The second left lane segment
	LaneSeg m_right2;			// The second right lane segment

private:
	LaneSeg m_prevLeft1;
	LaneSeg m_prevRight1;
	bool m_bTrackingLeft;
	bool m_bTrackingRight;
};

Detector::Detector() :
m_bActive(false),
m_pOriginalImg(NULL),
m_pGrayscaleImg(NULL),
m_pGradientImg(NULL),
m_pBinaryImg(NULL),
m_pMarkImg(NULL)
{
	m_size.width = 0;
	m_size.height = 0;
}

/*****************************************************************************
 * Destructor
 *****************************************************************************/
Detector::~Detector()
{
	Destroy();
}

/*****************************************************************************
 * Function name:
 *     EnableTracking
 *
 * Remarks:
 *     Use this function to enable/disable the use of tracking
 *****************************************************************************/
void Detector::EnableTracking(bool bEnable)
{
	m_bTrackingLeft = bEnable;
	m_bTrackingRight = bEnable;
}

/*****************************************************************************
 * Function name:
 *     IsActive
 *
 * Return Value:
 *     true if the Detector object calling this method is active; otherwise
 *     false
 *
 * Remarks:
 *     This function returns whether the calling Detector object has been
 *     created with "Image" mode or "Video" mode
 *****************************************************************************/
bool Detector::IsActive() const
{
	return m_bActive;
}

/*****************************************************************************
 * Function name:
 *     Mark
 *
 * Remarks:
 *     This function marks the lane segments on m_pMarkImg
 *****************************************************************************/
void Detector::Mark()
{
	m_left1.Mark(m_MarkImg);
	m_right1.Mark(m_MarkImg);
	m_left2.Mark(m_MarkImg, cv::Scalar(255, 255, 0));
	m_right2.Mark(m_MarkImg, cv::Scalar(255, 255, 0));

	cv::Point p1, p2;
	// The first point
	p1.x = m_nVerticalCenter;
	p1.y = 0;
	// The second point
	p2.x = m_nVerticalCenter;
	p2.y = m_size.height;

	// Draw a line on pImg from p1 to p2
	cv::line(m_MarkImg, p1, p2, cv::Scalar(0, 0, 255), 1);
}


/*****************************************************************************
 * Function name:
 *     CreateImageMode
 *
 * Parameters:
 *     pOriginalImg: The original road image to be processed
 *
 * Return Value:
 *     true if successful; otherwise false
 *
 * Remarks:
 *     There are two modes one can choose from when creating a Detector
 *     object. If you are creating the Detector object with "Image" mode, the
 *     object is used to process exactly one image; the "Video" mode, as the
 *     name said, is used to process each frame in a video file. If you
 *     created a Detector object in its "Video" mode, you can then call
 *     SetFrameImage to dispatch one frame in the video file to the object.
 *     The main difference between "Image" mode and "Video" mode is something
 *     about performance. When you process a video file with "Video" mode,
 *     there are less creation and release, thus a performance enhancement
 *****************************************************************************/
bool Detector::CreateImageMode(cv::Mat&  OriginalImg)
{
	// It's bad if pOriginalImg is NULL

	if (!OriginalImg.empty())
	{
		Destroy();
		return false;
	}

	// If the calling Detector object is already active, then release resources allocated
	if (IsActive())
	{
		m_left1.Clear();
		m_right1.Clear();
		m_left2.Clear();
		m_right2.Clear();
	}

	m_bActive = true;
	m_size = OriginalImg.size();

	// m_pOriginalImg is a copy of pOriginalImg, but make sure m_pOriginalImg is a top-down image
	m_OriginalImg = OriginalImg.clone();

	// Produce m_GrayscaleImg from m_OriginalImg.
	// "true" means memory will be allcated in function Grayscale
	Grayscale(true);
	// Allocate memory for m_pGradientImg and m_pBinaryImg
	//
	//m_pGradientImg = cvCreateImage(m_size, IPL_DEPTH_8U, 1);
	//m_pBinaryImg = cvCreateImage(m_size, IPL_DEPTH_8U, 1);

	// Allocate memory for m_pMarkImg
	//m_pMarkImg = cvCreateImage(m_size, IPL_DEPTH_8U, 3);

	// Initialize all the control values
	InitControlValues();

	return true;
}

/*****************************************************************************
 * Function name:
 *     CreateVideoMode
 *
 * Parameters:
 *     size: The size of the video(it's width and height), thus this is also
 *         the size of each frame in the video.
 *
 * Remarks:
 *     See "Remarks" part of function CreateImageMode
 *****************************************************************************/
void Detector::CreateVideoMode(const CvSize& size)
{
	// If the calling Detector object is already active, then release resources allocated
	if (IsActive())
	{


		m_left1.Clear();
		m_right1.Clear();
		m_left2.Clear();
		m_right2.Clear();
	}

	m_bActive = true;
	m_size = size;

	// Allocate memroy for m_pGrayscaleImg, m_pGradientImg and m_pBinaryImg
	//m_pGrayscaleImg = cvCreateImage(m_size, IPL_DEPTH_8U, 1);
	//m_pGradientImg = cvCreateImage(m_size, IPL_DEPTH_8U, 1);
	//m_pBinaryImg = cvCreateImage(m_size, IPL_DEPTH_8U, 1);

	// Allocate memory for m_pMarkImg
	//m_pMarkImg = cvCreateImage(m_size, IPL_DEPTH_8U, 3);

	// Initialize all the control values
	InitControlValues();
}

/*****************************************************************************
 * Function name:
 *     SetFrameImage
 *
 * Parameters:
 *     pFrameImg: One frame in the video file
 *
 * Return Value:
 *     true if successful; otherwise false
 *
 * Remarks:
 *     This function dispatches one frame in the video file to the Detector
 *     object to be processed. Make sure the calling Detector object is active
 *****************************************************************************/
bool Detector::SetFrameImage(IplImage* pFrameImg)
{
	// It's bad if pFrameImg is NULL
	if (!pFrameImg)
	{
		g_nError = ERROR_BAD_ARGUMENT;
		return false;
	}

	// When calling this function, the calling Detector object should be active
	if (!IsActive())
	{
		g_nError = ERROR_DETECTOR_NOT_ACTIVE;
		return false;
	}

	// If there is already an m_pOriginalImg, release it and make it a copy of pFrameImg.
	// Also m_pOriginalImg should be a top-down image
	if (m_pOriginalImg)
		cvReleaseImage(&m_pOriginalImg);

	m_pOriginalImg = cvCloneImage(pFrameImg);
	if (pFrameImg->origin == IPL_ORIGIN_BL)
	{
		cvConvertImage(m_pOriginalImg, m_pOriginalImg, CV_CVTIMG_FLIP);
		m_pOriginalImg->origin = IPL_ORIGIN_TL;
	}

	// Produce m_pGrayscaleImg from m_pOriginalImg.
	// "false" means the memory for m_pGrayscaleImg has been already allocated
	Grayscale(false);

	return true;
}

/*****************************************************************************
 * Function name:
 *     Destroy
 *
 * Remarks:
 *     This function releases all the resources in the calling Detector object
 *****************************************************************************/
void Detector::Destroy()
{
	if (IsActive())
	{
		m_bActive = false;
		m_size.width = 0;
		m_size.height = 0;

		cvReleaseImage(&m_pOriginalImg);
		m_pOriginalImg = NULL;
		cvReleaseImage(&m_pGrayscaleImg);
		m_pGrayscaleImg = NULL;
		cvReleaseImage(&m_pGradientImg);
		m_pGradientImg = NULL;
		cvReleaseImage(&m_pBinaryImg);
		m_pBinaryImg = NULL;

		cvReleaseImage(&m_pMarkImg);
		m_pMarkImg = NULL;

		m_left1.Clear();
		m_right1.Clear();
		m_left2.Clear();
		m_right2.Clear();
	}
}

/*****************************************************************************
 * Function name:
 *     Detect
 *
 * Return Value:
 *     true if successful; otherwise false
 *
 * Remarks:
 *     This function processes the current image stored in m_pGrayscaleImg(
 *     which had already been produced from m_pOriginalImg), that means, finds
 *     out all the lane segments in the current image
 *****************************************************************************/
bool Detector::doLaneLineDetection(cv::Mat img)
{
	// When calling this function, the calling Detector object should be active

	// Determine the vertical center

	DetermineVerticalCenter();
	
	// If we are experencing a lane changing, tracking could not be used
	if (std::abs(m_nVerticalCenter - m_nPrevVerticalCenter) > 160)
	{
		m_bTrackingLeft = false;
		m_bTrackingRight = false;
	}

	// Generate the adaptive threshold for the graysacale image
	GetThresholdGrayscale();
	// Produce m_pGradientImg
	Gradient();
	// Produce m_pBinaryImg
	Binarize();

	// Detect the first left lane segment
	DetectLeft1();
	// If cannot find first left lane with tracking,
	// find it again without tracking
	if (m_left1.IsEmpty() && m_bTrackingLeft)
	{
		m_bTrackingLeft = false;
		DetectLeft1();
	}
	// Detect the first right lane segment
	DetectRight1();
	// If cannot find first right lane with tracking,
	// find it again without tracking
	if (m_right1.IsEmpty() && m_bTrackingRight)
	{
		m_bTrackingRight = false;
		DetectRight1();
	}

	// If first left lane cannot be found, tracking is off
	if (!m_left1.IsEmpty())
	{
		m_prevLeft1 = m_left1;
		m_bTrackingLeft = true;
	}
	else
	{
		m_prevLeft1.Clear();
		m_bTrackingLeft = false;
	}

	// If first right lane cannot be found, tracking is off
	if (!m_right1.IsEmpty())
	{
		m_prevRight1 = m_right1;
		m_bTrackingRight = true;
	}
	else
	{
		m_prevRight1.Clear();
		m_bTrackingRight = false;
	}

	// Detect the second left lane segment
	DetectLeft2();
	// Detect the second right lane segment
	DetectRight2();

	return true;
}

/*****************************************************************************
 * Function name:
 *     Mark
 *
 * Parameters:
 *     nWhichImage: The lane segment is marked on which image
 *
 * Return Value:
 *     true if successful; otherwise false
 *
 * Remarks:
 *     This function marks the result on m_pMarkImg
 *****************************************************************************/
bool Detector::Mark(int nWhichImage)
{
	// nWhichImage should be one of ORIGINAL, GRAYSCALE, GRADIENT and BINARY
	if (nWhichImage != ORIGINAL && nWhichImage != GRAYSCALE && nWhichImage != GRADIENT && nWhichImage != BINARY)
		return false;

	// All the lane segments are marked on m_pMarkImg.
	// nWhichImage is used to determine which image is used to be the background image that copied or converted to m_pMarkImg
	switch (nWhichImage)
	{
	case ORIGINAL:	// The original image is the background image.
		// If m_pOriginalImg is a grayscale image, m_pMarkImg is its 24-bit version;
		//     otherwise m_pMarkImg is just a copy of m_pOriginalImg
		if (m_pOriginalImg->depth == IPL_DEPTH_8U && m_pOriginalImg->nChannels == 1)
			cvCvtColor(m_pOriginalImg, m_pMarkImg, CV_GRAY2RGB);
		else
			cvCopy(m_pOriginalImg, m_pMarkImg);
		break;

	case GRAYSCALE:	// The grayscale image is the background image. Need conversion
		cvCvtColor(m_pGrayscaleImg, m_pMarkImg, CV_GRAY2RGB);
		break;

	case GRADIENT:	// The gradient image is the background image. Need conversion
		cvCvtColor(m_pGradientImg, m_pMarkImg, CV_GRAY2RGB);
		break;

	case BINARY:	// The binary image is the background image. Need conversion
		cvCvtColor(m_pBinaryImg, m_pMarkImg, CV_GRAY2RGB);
		break;
	}

	// Mark it
	Mark();

	return true;
}

/*****************************************************************************
 * Function name:
 *     ToFile
 *
 * Parameters:
 *     strFilename: Name of the output results file
 *
 * Return Value:
 *     true if successful; otherwise false
 *
 * Remarks:
 *     This function outputs the detection results of the current image to a
 *     file
 *****************************************************************************/
bool Detector::ToFile(const std::string& strFilename) const
{
	std::ofstream fout(strFilename.c_str());
	if (!fout)
	{
		g_nError = ERROR_CANNOT_OPEN_OR_CREATE_FILE;
		return false;
	}

	fout << "Left Lane 1: ";
	m_left1.ToFile(fout);

	fout << "Right Lane 1: ";
	m_right1.ToFile(fout);

	fout << "Left Lane 2: ";
	m_left2.ToFile(fout);

	fout << "Right Lane 2: ";
	m_right2.ToFile(fout);

	return true;
}

/*****************************************************************************
 * Function name:
 *     SetControlValue
 *
 * Parameters:
 *     dValue: The new control value
 *     nWhat: Which control value to set
 *
 * Return Value:
 *     true if successful; otherwise false
 *
 * Remarks:
 *     This function is used to set all the configurable control values. Use
 *     nWhat and dValue to indicate which control value to set and set to
 *     what, respectively. There is always a range for each control value,
 *     specifies dValue out of this range will cause a failure. For further
 *     understanding of the meaning of each control value, please refer to
 *     ATN.pdf
 *****************************************************************************/
bool Detector::SetControlValue(double dValue, int nWhat)
{
	int nValue = (int) dValue;

	switch (nWhat)
	{
	case GRADIENT_OPERATOR:
		if (nValue != SOBEL && nValue != LAPLACE && nValue != CANNY)
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_nGradientOperator = nValue;
		return true;

	case THRESHOLD_GRAYSCALE_DELTA_FACTOR:
		if (dValue < 0.1 || dValue > 1.0)
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_fThresholdGrayscaleDeltaFactor = (float) dValue;
		return true;

	case THRESHOLD_GRADIENT:
		if (nValue < 0 || nValue > 255)
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_nThresholdGradient = nValue;
		return true;

	case LEFT_MOST:
		if (nValue < 0 || (IsActive() && nValue >= m_nVerticalCenter))
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_nLeftMost = nValue;
		return true;

	case RIGHT_MOST:
		if (nValue < 0 || (IsActive() && (nValue <= m_nVerticalCenter || nValue >= m_size.width)))
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_nRightMost = nValue;
		return true;

	case VANISHING_POINT_HEIGHT_FACTOR:
		if (dValue < 0 || dValue > 0.5)
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_nVanishingPointHeight = (int) (m_size.height * dValue);
		m_nSweepLineHighest = m_nVanishingPointHeight + (int) (m_size.height * 0.03);
		return true;

	case SWEEP_LINE_LOWEST:
		if (nValue < 0 || (IsActive() && nValue > m_nSweepLineHighest - m_nFirstLaneSegLengthMin + 1))
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_nSweepLineLowest = nValue;
		return true;

	case FIRST_LANE_SEG_LENGTH_MIN:
		if (nValue < 10)
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_nFirstLaneSegLengthMin = nValue;
		return true;

	case SECOND_LANE_SEG_LENGTH_MIN:
		if (nValue < 10)
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_nSecondLaneSegLengthMin = nValue;
		return true;

	case FIRST_SECOND_DISTANCE:
		if (nValue < 10 || nValue > 25)
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_nFirstSecondDistance = nValue;
		return true;

	case SECOND_AMPLITUDE:
		if (nValue < 10 || nValue > 25)
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_nSecondAmplitude = nValue;
		return true;

	case RECT_LEFT_FACTOR:
		if (dValue <0.0 || dValue > 0.5)
		{
			g_nError =	ERROR_BAD_ARGUMENT;
			return false;
		}
		m_nRectLeft = (int) (m_size.width * dValue);
		m_nRectWidth = (int) (m_size.width * (1 - 2 * dValue));
		return true;

	case RECT_TOP_FACTOR:
		if (dValue < 0.5 || dValue > 1.0)
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_nRectTop = (int) (m_size.height * dValue);
		return true;

	case RECT_HEIGHT_FACTOR:
		if (dValue < 0.0 || dValue > 0.2)
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_nRectHeight = (int) (m_size.height * dValue);
		return true;

	case THRESHOLD_REG:
		if (dValue < 0.5 || dValue > 2.0)
		{
			g_nError = ERROR_BAD_ARGUMENT;
			return false;
		}
		m_fThresholdReg = (float) dValue;
		return true;

	default:
		return false;
	}
}

/*****************************************************************************
 * Function name:
 *     InitControlValues
 *
 * Remarks:
 *     Each control values has its default value. This function initializes
 *     them use their default values
 *****************************************************************************/
void Detector::InitControlValues()
{
	m_nGradientOperator = CANNY;
	m_fThresholdGrayscaleDeltaFactor = 0.4f;
	m_nThresholdGradient = 25;
	m_nLeftMost = 0;
	m_nVerticalCenter = m_size.width / 2;
	m_nRightMost = m_size.width - 1;
	m_nVanishingPointHeight = (int) (m_size.height * 0.4);
	m_nSweepLineHighest = m_nVanishingPointHeight + (int) (m_size.height * 0.03);
	m_nSweepLineLowest = m_size.height - 30;
	m_nFirstLaneSegLengthMin = 10;
	m_nSecondLaneSegLengthMin = 10;
	m_nFirstSecondDistance = 20;
	m_nSecondAmplitude = 20;
	m_nRectLeft = (int) (m_size.width * 0.3);
	m_nRectTop = (int) (m_size.height * 0.85);
	m_nRectWidth = (int) (m_size.width * 0.4);
	m_nRectHeight = (int) (m_size.height * 0.1);
	m_nStripLeft = 0;
	m_nStripTop = (int) (m_size.height * 0.675);
	m_nStripWidth = m_size.width;
	m_nStripHeight = 15;
	m_nPrevLaneX[0] = m_nPrevLaneX[1] = m_nPrevLaneX[2] = -1;
	m_nPrevVerticalCenter = m_size.width / 2;

	m_fThresholdReg = 1.0f;

	m_bTrackingLeft = false;
	m_bTrackingRight = false;
}

/*****************************************************************************
 * Function name:
 *     Garyscale
 *
 * Parameters:
 *     bNeedAlloc: true means memory should be allocated in this function
 *         call; false means memory has already been allocated
 *
 * Remarks:
 *     This function produces m_pGrayscaleImg from m_pOriginalImg
 *****************************************************************************/
void Detector::Grayscale(bool bNeedAlloc)
{
	if (bNeedAlloc)
	{
		if (m_pOriginalImg->depth == IPL_DEPTH_8U && m_pOriginalImg->nChannels == 1)
			m_pGrayscaleImg = cvCloneImage(m_pOriginalImg);
		else
		{
			m_pGrayscaleImg = cvCreateImage(m_size, IPL_DEPTH_8U, 1);
			cvCvtColor(m_pOriginalImg, m_pGrayscaleImg, CV_RGB2GRAY);
		}
	}
	else
	{
		if (m_pOriginalImg->depth == IPL_DEPTH_8U && m_pOriginalImg->nChannels == 1)
			cvCopy(m_pOriginalImg, m_pGrayscaleImg);
		else
			cvCvtColor(m_pOriginalImg, m_pGrayscaleImg, CV_RGB2GRAY);
	}
}

/*****************************************************************************
 * Function name:
 *     DetermineVerticalCenter
 *
 * Remarks:
 *     This function determines the vertical center from which each scan line
 *     begins. It's mainly used to solve the lane changing condition.
 *****************************************************************************/
void Detector::DetermineVerticalCenter()
{
	m_nPrevVerticalCenter = m_nVerticalCenter;
	// The first int in the pair is used to store the sum of grayscale of
	// each vertical line segment; and the second int stores the index
	// of the vertical line segment
	std::pair<int, int>* strip = new std::pair<int, int>[m_nStripWidth];
	if (!strip)
		return;

	int i;

	// Initially all the sums are set to 0 and indexes are set to proper value
	for (i = 0; i < m_nStripWidth; ++i)
	{
		strip[i].first = 0;
		strip[i].second = i + m_nStripLeft;
	}

	// Calculate each sum
	unsigned char* pGrayscale = (unsigned char*) (m_pGrayscaleImg->imageData + m_pGrayscaleImg->widthStep * m_nStripTop + m_nStripLeft);
	for (int y = 0; y < m_nStripHeight; ++y)
	{
		for (int x = 0; x < m_nStripWidth; ++x)
			strip[x].first += pGrayscale[x];

		pGrayscale += m_pGrayscaleImg->widthStep;
	}

	// This threshold is used to distinguish the peaks
	int nThres = 0;
	for (i = 0; i < m_nStripWidth; ++i)
		nThres += strip[i].first;
	nThres /= m_nStripWidth;
	nThres += nThres / 10;

	// There are several peaks in the strip, and each of them is considered to be a
	// lane. We use the threshold value calculated above to determine whether this
	// vertical line belongs to a peak
	i = 0;
	int nCurX = 0;
	int nCount = 0;
	std::vector<int> vecLaneX;
	while (i < m_nStripWidth)
	{
		// Belongs to one peak
		if (strip[i].first > nThres)
		{
			nCurX += i;
			++nCount;
		}
		else
		{
			// The peak ends, thus we calculate the average index (the center of
			// the lane)
			if (nCount >= 15 && nCount <= 75)
				vecLaneX.push_back(nCurX / nCount);

			nCurX = 0;
			nCount = 0;
		}

		++i;
	}

	if (nCount >= 15 && nCount <= 75)
		vecLaneX.push_back(nCurX / nCount);

	// Now with the values achieved from the previous frame, we could
	// calculate the m_nVerticalCenter based on several conditions

	// No lane found in this frame
	if (vecLaneX.empty())
	{
		// No lane found in the previous frame
		if (m_nPrevLaneX[0] == -1)
			// Set m_nVerticalCenter to its initial value
			m_nVerticalCenter = m_size.width / 2;
		// If lanes are found in the previous frame, the m_nVerticalCenter
		// will remain unchanged in this frame
		goto End;
	}
	// One lane found in this frame
	else if (vecLaneX.size() == 1)
	{
		// No lane found in the previous frame
		if (m_nPrevLaneX[0] == -1)
		{
			m_nPrevLaneX[0] = vecLaneX[0];

			// Calculate the m_nVerticalCenter based on whether this lane
			// is in the left part of the road image
			if (m_nPrevLaneX[0] < m_size.width / 2)
				m_nVerticalCenter = (m_nPrevLaneX[0] + m_size.width) / 2;
			else
				m_nVerticalCenter = m_nPrevLaneX[0] / 2;

			goto End;
		}
		// One lane found in the previous frame
		else if (m_nPrevLaneX[1] == -1)
		{
			// Is it the same lane we found in the previous frame
			if (std::abs(vecLaneX[0] - m_nPrevLaneX[0]) < 50)
			{
				m_nPrevLaneX[0] = vecLaneX[0];

				if (m_nPrevLaneX[0] < m_size.width / 2)
					m_nVerticalCenter = (m_nPrevLaneX[0] + m_size.width) / 2;
				else
					m_nVerticalCenter = m_nPrevLaneX[0] / 2;

				goto End;
			}
			// Is it a new lane
			else if (std::abs(m_nPrevLaneX[0] - vecLaneX[0]) > 200)
			{
				m_nPrevLaneX[1] = vecLaneX[0];
				goto CalcBy2Lanes;
			}
		}
		// Two lane found in the previous frame
		else if (m_nPrevLaneX[2] == -1)
		{
			// Is it one of the lanes we found in the previous frame
			if (std::abs( m_nPrevLaneX[0] - vecLaneX[0]) < 50)
			{
				m_nPrevLaneX[0] = vecLaneX[0];
				goto CalcBy2Lanes;
			}
			else if (std::abs(m_nPrevLaneX[1] - vecLaneX[0]) < 50)
			{
				m_nPrevLaneX[1] = vecLaneX[0];
				goto CalcBy2Lanes;
			}
			// Is it a new lane
			else if ((m_nPrevLaneX[0] > vecLaneX[0] && m_nPrevLaneX[0] - vecLaneX[0] > 200)
				|| (m_nPrevLaneX[1] < vecLaneX[0]  && vecLaneX[0] - m_nPrevLaneX[1] > 200)
				|| ((m_nPrevLaneX[0] < vecLaneX[0] && vecLaneX[0] - m_nPrevLaneX[0] > 200) && (m_nPrevLaneX[1] > vecLaneX[0] && m_nPrevLaneX[1] - vecLaneX[0] > 200)))
			{
				m_nPrevLaneX[2] = vecLaneX[0];
				goto CalcBy3Lanes;
			}

			goto End;
		}
		// Already three lanes found in the previous frame
		else
		{
			for (int i = 0; i < 3; ++i)
			{
				if (std::abs(m_nPrevLaneX[i] - vecLaneX[0]) < 50)
				{
					m_nPrevLaneX[i] = vecLaneX[0];
					goto CalcBy3Lanes;
				}
			}

			goto End;
		}
	}
	// Two lanes found in this frame
	if (vecLaneX.size() == 2)
	{
		// No lane found in the previous frame
		if (m_nPrevLaneX[0] == -1)
		{
			m_nPrevLaneX[0] = vecLaneX[0];
			m_nPrevLaneX[1] = vecLaneX[1];
			goto CalcBy2Lanes;
		}
		// One lane found in the previous frame
		else if (m_nPrevLaneX[1] == -1)
		{
			// Is one of the lane the same as the lane we found in the previous frame
			if (std::abs(m_nPrevLaneX[0] - vecLaneX[0]) < 50 || std::abs(m_nPrevLaneX[0] - vecLaneX[1]) < 50)
			{
				m_nPrevLaneX[0] = vecLaneX[0];
				m_nPrevLaneX[1] = vecLaneX[1];
				goto CalcBy2Lanes;
			}
			// Then there are two new lanes found
			else if ((m_nPrevLaneX[0] < vecLaneX[0] && vecLaneX[0] - m_nPrevLaneX[0] > 200)
				|| (m_nPrevLaneX[0] > vecLaneX[1] && m_nPrevLaneX[0] - vecLaneX[1] > 200)
				|| ((m_nPrevLaneX[0] > vecLaneX[0] && m_nPrevLaneX[0] - vecLaneX[0] > 200) && (m_nPrevLaneX[0] < vecLaneX[1] && vecLaneX[1] - m_nPrevLaneX[0] > m_size.width /  6)))
			{
				m_nPrevLaneX[1] = vecLaneX[0];
				m_nPrevLaneX[2] = vecLaneX[1];
				goto CalcBy3Lanes;
			}
		}
		// Two lanes found in the previous frame
		else if (m_nPrevLaneX[2] == -1)
		{
			// There must be at least one lane the same as we found in the previous frame
			if (std::abs(m_nPrevLaneX[0] - vecLaneX[0]) < 50)
			{
				m_nPrevLaneX[0] = vecLaneX[0];
				if (std::abs(m_nPrevLaneX[1] - vecLaneX[1]) < 50)
				{
					m_nPrevLaneX[1] = vecLaneX[1];
					goto CalcBy2Lanes;
				}
				else if (std::abs(m_nPrevLaneX[1] - vecLaneX[1]) > 200)
				{
					m_nPrevLaneX[2] = vecLaneX[1];
					goto CalcBy3Lanes;
				}
			}
			else if (std::abs(m_nPrevLaneX[0] - vecLaneX[1]) < 50)
			{
				m_nPrevLaneX[0] = vecLaneX[1];
				m_nPrevLaneX[2] = vecLaneX[0];
				goto CalcBy3Lanes;
			}
			else if (std::abs(m_nPrevLaneX[1] - vecLaneX[0]) < 50)
			{
				m_nPrevLaneX[1] = vecLaneX[0];
				m_nPrevLaneX[2] = vecLaneX[1];
				goto CalcBy3Lanes;
			}
			else if ((std::abs(m_nPrevLaneX[1] - vecLaneX[1]) < 50) && (std::abs(m_nPrevLaneX[0] - vecLaneX[0]) > 200))
			{
				m_nPrevLaneX[2] = vecLaneX[0];
				goto CalcBy3Lanes;
			}
		}
		// Already three lanes found in the previous frame
		else
		{
			if (std::abs(m_nPrevLaneX[0] - vecLaneX[0]) < 50)
			{
				if (std::abs(m_nPrevLaneX[1] - vecLaneX[1]) < 50)
				{
					m_nPrevLaneX[0] = vecLaneX[0];
					m_nPrevLaneX[1] = vecLaneX[1];
					goto CalcBy3Lanes;
				}
				else if (std::abs(m_nPrevLaneX[2] - vecLaneX[1]) < 50)
				{
					m_nPrevLaneX[0] = vecLaneX[0];
					m_nPrevLaneX[2] = vecLaneX[1];
					goto CalcBy3Lanes;
				}
			}
			else if (std::abs(m_nPrevLaneX[1] - vecLaneX[0]) < 50 && std::abs(m_nPrevLaneX[2] - vecLaneX[1]) < 50)
			{
				m_nPrevLaneX[1] = vecLaneX[0];
				m_nPrevLaneX[2] = vecLaneX[1];
				goto CalcBy3Lanes;
			}
			else
				goto End;
		}
	}
	// Three lanes found in this frame
	else if (vecLaneX.size() == 3)
	{
		// No lane found in the previous frame
		if (m_nPrevLaneX[0] == -1)
		{
			m_nPrevLaneX[0] = vecLaneX[0];
			m_nPrevLaneX[1] = vecLaneX[1];
			m_nPrevLaneX[2] = vecLaneX[2];
			goto CalcBy3Lanes;
		}
		// One lane found in the previous frame, should be overlayed
		else if (m_nPrevLaneX[1] == -1)
		{
			if (std::abs(m_nPrevLaneX[0] - vecLaneX[0]) < 50 || std::abs(m_nPrevLaneX[0] - vecLaneX[1]) < 50 || std::abs(m_nPrevLaneX[0] - vecLaneX[2]) < 50)
			{
				m_nPrevLaneX[0] = vecLaneX[0];
				m_nPrevLaneX[1] = vecLaneX[1];
				m_nPrevLaneX[2] = vecLaneX[2];
				goto CalcBy3Lanes;
			}
		}
		// Two lanes found in the previous frame, should be overlayed
		else if (m_nPrevLaneX[2] == -1)
		{
			if (std::abs(m_nPrevLaneX[0] - vecLaneX[0]) < 50 && (std::abs(m_nPrevLaneX[1] - vecLaneX[1]) < 50 || std::abs(m_nPrevLaneX[1] - vecLaneX[2]) < 50)
				|| (std::abs(m_nPrevLaneX[0] - vecLaneX[1]) < 50 && std::abs(m_nPrevLaneX[1] - vecLaneX[2]) < 50))
			{
				m_nPrevLaneX[0] = vecLaneX[0];
				m_nPrevLaneX[1] = vecLaneX[1];
				m_nPrevLaneX[2] = vecLaneX[2];
				goto CalcBy3Lanes;
			}
		}
		// Three lanes found in the previous frame, all overlayed
		else
		{
			if (std::abs(m_nPrevLaneX[0] - vecLaneX[0]) < 50 && std::abs(m_nPrevLaneX[1] - vecLaneX[1]) < 50 && std::abs(m_nPrevLaneX[2] - vecLaneX[2]) < 50)
			{
				m_nPrevLaneX[0] = vecLaneX[0];
				m_nPrevLaneX[1] = vecLaneX[1];
				m_nPrevLaneX[2] = vecLaneX[2];
				goto CalcBy3Lanes;
			}
			else
				 goto End;
		}
	}
	// Over three lanes found in this frame
	else
		goto End;

CalcBy2Lanes:
	std::sort(m_nPrevLaneX, m_nPrevLaneX + 2);
	m_nVerticalCenter = (m_nPrevLaneX[0] + m_nPrevLaneX[1]) / 2;
	goto End;

CalcBy3Lanes:
	std::sort(m_nPrevLaneX, m_nPrevLaneX + 3);
	if (m_nPrevLaneX[1] < m_size.width / 2)
		m_nVerticalCenter = (m_nPrevLaneX[1] + m_nPrevLaneX[2]) / 2;
	else
		m_nVerticalCenter = (m_nPrevLaneX[0] + m_nPrevLaneX[1]) / 2;
	goto End;

End:
	delete[] strip;
}

/*****************************************************************************
 * Function name:
 *     GetThresholdGrayscale
 *
 * Remarks:
 *     Taking out a rectangle from the grayscale image to represent the road
 *     surface. Taking the average grayscale in the rectangle as the average
 *     road surface grayscale, the threshold of the grayscale image is
 *     achieved by multiplying it by certain value(namely, 1 +
 *     m_fThresholdGrayscaleDeltaFactor)
 *****************************************************************************/
void Detector::GetThresholdGrayscale()
{
	unsigned char* pGrayscale = (unsigned char*) m_pGrayscaleImg->imageData + m_pGrayscaleImg->widthStep * m_nRectTop;
	int nSum = 0;

	for (int y = m_nRectTop; y < m_nRectTop + m_nRectHeight; ++y)
	{
		for (int x = m_nRectLeft; x < m_nRectLeft + m_nRectWidth; ++x)
			nSum += pGrayscale[x];
		pGrayscale += m_pGrayscaleImg->widthStep;
	}

	m_nThresholdGrayscale = (int) ((1 + m_fThresholdGrayscaleDeltaFactor) * nSum / (m_nRectWidth * m_nRectHeight));
}

/*****************************************************************************
 * Function name:
 *     Gradient
 *
 * Remarks:
 *     This function produces m_pGradientImg according to the
 *     m_nGradientOperator member. There are three kinds of gradient operators
 *     you could choose from: Sobel, Laplace and Canny
 *****************************************************************************/
void Detector::Gradient()
{
	switch (m_nGradientOperator)
	{
	case SOBEL:
		{
			IplImage* pImg1 = cvCreateImage(m_size, IPL_DEPTH_16S, 1);
			IplImage* pImg2 = cvCreateImage(m_size, IPL_DEPTH_16S, 1);
			IplImage* pImg = cvCreateImage(m_size, IPL_DEPTH_16S, 1);

			cvSobel(m_pGrayscaleImg, pImg1, 1, 0);
			cvSobel(m_pGrayscaleImg, pImg2, 0, 1);
			cvAdd(pImg1, pImg2, pImg);
			cvConvertScale(pImg, m_pGradientImg);

			cvReleaseImage(&pImg1);
			cvReleaseImage(&pImg2);
			cvReleaseImage(&pImg);
		}
		break;

	case LAPLACE:
		{
			IplImage* pImg = cvCreateImage(m_size, IPL_DEPTH_16S, 1);

			cvLaplace(m_pGrayscaleImg, pImg);
			cvConvertScale(pImg, m_pGradientImg);

			cvReleaseImage(&pImg);
		}
		break;

	case CANNY:
		cvCanny(m_pGrayscaleImg, m_pGradientImg, 40, 100);
		break;
	}
}

/*****************************************************************************
 * Function name:
 *     Binarize
 *
 * Remarks:
 *     This function produces m_pBinaryImg using one threshold for
 *     m_pGrayscaleImg and one threshold for m_pGradientImg
 *****************************************************************************/
void Detector::Binarize()
{
	unsigned char* pGrayscale = (unsigned char*) m_pGrayscaleImg->imageData;
	unsigned char* pGradient = (unsigned char*) m_pGradientImg->imageData;
	unsigned char* pBinary = (unsigned char*) m_pBinaryImg->imageData;

	for (int y = 0; y < m_size.height; ++y)
	{
		for (int x = 0; x < m_size.width; ++x)
		{
			// This is how bi-threshold is used
			if (pGrayscale[x] > m_nThresholdGrayscale && pGradient[x] > m_nThresholdGradient)
				pBinary[x] = 255;
			else
				pBinary[x] = 0;
		}

		pGrayscale += m_pGrayscaleImg->widthStep;
		pGradient += m_pGradientImg->widthStep;
		pBinary += m_pBinaryImg->widthStep;
	}
}

/*****************************************************************************
 * Function name:
 *     NextPoint
 *
 * Parameters:
 *     bIsLeft: true if the current scan direction is from the vertical center
 *         of the image to left; otherwise false
 *     nX: A reference, used to accept the x-coordinate of the next point
 *     nY: A reference, used to accept the y-coordinate of the next point
 *     nLimit: The scan line will not go over this limit to scan any further
 *
 * Return Value:
 *     true if there exist the next point; othwise false
 *
 * Remarks:
 *     This is an assistant function used to achieve the next point on the
 *     scan line
 *****************************************************************************/
bool Detector::NextPoint(bool bIsLeft, int& nX, int& nY, int nLimit) const
{
	if (nLimit == m_nVerticalCenter)
		return false;

	if (bIsLeft)
	{
		if (nLimit == 0)
			nLimit = m_nLeftMost;

		--nX;
		if (nX < nLimit)
		{
			--nY;
			if (nY < m_nSweepLineHighest)
				return false;
			else
			{
				nX = m_bTrackingLeft ? (int) (m_prevLeft1.m_fK * nY + m_prevLeft1.m_fB + 10) : m_nVerticalCenter;
				return true;
			}
		}
		else
			return true;
	}
	else
	{
		if (nLimit == 0)
			nLimit = m_nRightMost;

		++nX;
		if (nX > nLimit)
		{
			--nY;
			if (nY < m_nSweepLineHighest)
				return false;
			else
			{
				nX = m_bTrackingRight ? (int) (m_prevRight1.m_fK * nY + m_prevRight1.m_fB - 10) : m_nVerticalCenter;
				return true;
			}
		}
		else
			return true;
	}
}

/*****************************************************************************
 * Function name:
 *     FindNextWhitePoint
 *
 * Parameters:
 *     bIsLeft: true if the current scan direction is from the vertical center
 *         of the image to left; otherwise false
 *     nX: A reference, used to accept the x-coordinate of the next white
 *         point
 *     nY: A reference, used to accept the y-coordinate of the next white
 *         point
 *     nLimit: The scan line will not go over this limit to scan any further
 *
 * Return Value:
 *     true if there exist the next white point; othwise false
 *
 * Remarks:
 *     This is an assistant function used to achieve the next white point on
 *     the scan line
 *****************************************************************************/
bool Detector::FindNextWhitePoint(bool bIsLeft, int& nX, int& nY, int nLimit) const
{
	if (nLimit == m_nVerticalCenter)
		return false;

	unsigned char* pBinary = (unsigned char*) (m_pBinaryImg->imageData + m_pBinaryImg->widthStep * nY);

	if (pBinary[nX] == 255)
		return true;

	while (NextPoint(bIsLeft, nX, nY, nLimit))
	{
		pBinary = (unsigned char*) (m_pBinaryImg->imageData + m_pBinaryImg->widthStep * nY);
		if (pBinary[nX] == 255)
			return true;
	}

	return false;
}

/*****************************************************************************
 * Function name:
 *     DetectLeft1
 *
 * Remarks:
 *     This function detects the first left lane segment. Please refer to
 *     ATN.pdf to get further understanding
 *****************************************************************************/
void Detector::DetectLeft1()
{
	m_left1.Clear();

	// Scan from this point (nX, nY)
	int nY = m_nSweepLineLowest;
	int nX = m_bTrackingLeft ? (int) (m_prevLeft1.m_fK * nY + m_prevLeft1.m_fB + 10) : m_nVerticalCenter;
	if (nX < m_nLeftMost)
	{
		nY = (int) ((m_nLeftMost - m_prevLeft1.m_fB) / m_prevLeft1.m_fK - 1);
		nX = (int) (m_nLeftMost - m_prevLeft1.m_fK + 10);
	}
	// Variables used to store the prefetch point
	int nX2 = 0, nY2 = 0;
	// Used to store the last point in m_left1
	int nLastX = 0, nLastY = 0;
	// The x and y difference between the current found point and the last point in m_left1
	int nDx = 0, nDy = 0;
	// bool variable indicate whether a lane pixel is found
	bool bFound = false;

	if (m_bTrackingLeft)
		bFound = FindNextWhitePoint(true, nX, nY, std::max<int>(nX - 15, m_nLeftMost));
	else
		bFound = FindNextWhitePoint(true, nX, nY);
	if (!bFound)
		return;

	if (nY <= m_nSweepLineHighest - m_nFirstLaneSegLengthMin)
		return;

AddPoint:
	m_left1.Add(nX, nY);

Next:
	--nY;
	nX = m_bTrackingLeft ? (int) (m_prevLeft1.m_fK * nY + m_prevLeft1.m_fB + 10) : m_nVerticalCenter;
	if (nY < m_nSweepLineHighest)
		goto LengthCheck;

	m_left1.GetLastCoordinate(nLastX, nLastY);

	if (m_bTrackingLeft)
		bFound = FindNextWhitePoint(true, nX, nY, nX - 15);
	else
		bFound = FindNextWhitePoint(true, nX, nY, nLastX);
	if (!bFound)
		goto LengthCheck;

	nDx = nX - nLastX;
	nDy = nLastY - nY;

	// Cannot find any white point for 15 scan lines, m_left1 ends
	if ((nY >= 280 && nDy >= 15) || (nY < 280 && nY >= 240 && nDy >= 10) || (nY < 240 && nDy >= 5))
	{
		// Examine m_left1's length
		if (m_left1.Length() < m_nFirstLaneSegLengthMin)
		{
			m_left1.Clear();

			nY = nLastY - 1;
			nX = m_bTrackingLeft ? (int) (m_prevLeft1.m_fK * nY + m_prevLeft1.m_fB + 10) : m_nVerticalCenter;

			if (m_bTrackingLeft)
				bFound = FindNextWhitePoint(true, nX, nY, std::max<int>(nX - 15, m_nLeftMost));
			else
				bFound = FindNextWhitePoint(true, nX, nY);
			if (!bFound)
				return;

			goto AddPoint;
		}

		goto RegCheck;
	}

	// (nX, nY) is too far away from (nLastX, nLastY)
	if (nDx >= 20 * nDy)
	{
		// Is it because the right lanes passes over the vertical center of the image
		if (std::abs(nX - m_nVerticalCenter) < 15 && m_left1.Length() >= m_nFirstLaneSegLengthMin)
		{
			// Move the vertical center a little bit left
			m_nVerticalCenter = (nLastX + m_nVerticalCenter) / 2;
			goto Next;
		}
		else
		{
			// Prefetch a point
			nY2 = nY - 1;
			nX2 = m_nVerticalCenter;

			if (!FindNextWhitePoint(true, nX2, nY2, nX) || nY - nY2 >= 15 || (nX2 - nX) >= 5 * (nY - nY2))
				goto Next;
			else
			{
				m_left1.Clear();
				goto AddPoint;
			}
		}
	}
	// (nX, nY) is a noise
	else if (nDx >= 5 * nDy)
		goto Next;
	else
		goto AddPoint;

LengthCheck:
	if (m_left1.Length() < m_nFirstLaneSegLengthMin)
	{
		m_left1.Clear();
		return;
	}

RegCheck:
	if (!m_left1.RegCheck(m_fThresholdReg))
	{
		m_left1.Adjust();
		if (m_left1.Length() < m_nFirstLaneSegLengthMin)
			m_left1.Clear();
	}
}

/*****************************************************************************
 * Function name:
 *     DetectRight1
 *
 * Remarks:
 *     This function detects the first right lane segment. Please refer to
 *     ATN.pdf to get further understanding
 *****************************************************************************/
void Detector::DetectRight1()
{
	m_right1.Clear();

	// Scan from this point (nX, nY)
	int nY = m_nSweepLineLowest;
	int nX = m_bTrackingRight ? (int) (m_prevRight1.m_fK * nY + m_prevRight1.m_fB - 10) : m_nVerticalCenter;
	if (nX > m_nRightMost)
	{
		nY = (int) ((m_nRightMost - m_prevRight1.m_fB) / m_prevRight1.m_fK - 1);
		nX = (int) (m_nRightMost - m_prevRight1.m_fK - 10);
	}
	// Variables used to store the prefetch point
	int nX2 = 0, nY2 = 0;
	// Used to store the last point in m_right1
	int nLastX = 0, nLastY = 0;
	// The x and y difference between the current found point and the last point in m_right1
	int nDx = 0, nDy = 0;
	// bool variable indicate whether a lane pixel is found
	bool bFound = false;

	if (m_bTrackingRight)
		bFound = FindNextWhitePoint(false, nX, nY, std::min<int>(nX + 15, m_nRightMost));
	else
		bFound = FindNextWhitePoint(false, nX, nY);
	if (!bFound)
		return;

	if (nY <= m_nSweepLineHighest - m_nFirstLaneSegLengthMin)
		return;

AddPoint:
	m_right1.Add(nX, nY);
	
Next:
	--nY;
	nX = m_bTrackingRight ? (int) (m_prevRight1.m_fK * nY + m_prevRight1.m_fB - 10) : m_nVerticalCenter;
	if (nY < m_nSweepLineHighest)
		goto LengthCheck;

	m_right1.GetLastCoordinate(nLastX, nLastY);

	if (m_bTrackingRight)
		bFound = FindNextWhitePoint(false, nX, nY, nX + 15);
	else
		bFound = FindNextWhitePoint(false, nX, nY, nLastX);
	if (!bFound)
		goto LengthCheck;

	nDx = nLastX - nX;
	nDy = nLastY - nY;

	// Cannot find any white point for 15 scan lines, m_right1 ends
	if ((nY >= 280 && nDy >= 15) || (nY < 280 && nY >= 240 && nDy >= 10) || (nY < 240 && nDy >= 5))
	{
		// Examine m_right1's length
		if (m_right1.Length() < m_nFirstLaneSegLengthMin)
		{
			m_right1.Clear();

			nY = nLastY - 1;
			nX = m_bTrackingRight ? (int) (m_prevRight1.m_fK * nY + m_prevRight1.m_fB - 10) : m_nVerticalCenter;

			if (m_bTrackingRight)
				bFound = FindNextWhitePoint(false, nX, nY, std::min<int>(nX + 15, m_nRightMost));
			else
				bFound = FindNextWhitePoint(false, nX, nY);
			if (!bFound)
				return;

			goto AddPoint;
		}

		goto RegCheck;
	}

	// (nX, nY) is too far away from (nLastX, nLastY)
	if (nDx >= 20 * nDy)
	{
		// Is it because the left lanes passes over the vertical center of the image
		if (std::abs(nX - m_nVerticalCenter) < 15 && m_right1.Length() >= m_nFirstLaneSegLengthMin)
		{
			// Move the vertical center a little bit right
			m_nVerticalCenter = (nLastX + m_nVerticalCenter) / 2;
			goto Next;
		}
		else
		{
			// Prefetch a point
			nY2 = nY - 1;
			nX2 = m_nVerticalCenter;

			if (!FindNextWhitePoint(false, nX2, nY2, nX) || nY - nY2 >= 15 || (nX - nX2) >= 5 * (nY - nY2))
				goto Next;
			else
			{
				m_right1.Clear();
				goto AddPoint;
			}
		}
	}
	// (nX, nY) is a noise
	else if (nDx >= 5 * nDy)
		goto Next;
	else
		goto AddPoint;

LengthCheck:
	if (m_right1.Length() < m_nFirstLaneSegLengthMin)
	{
		m_right1.Clear();
		return;
	}

RegCheck:
	if (!m_right1.RegCheck(m_fThresholdReg))
	{
		m_right1.Adjust();
		if (m_right1.Length() < m_nFirstLaneSegLengthMin)
			m_right1.Clear();
	}
}

/*****************************************************************************
 * Function name:
 *     DetectLeft2
 *
 * Remarks:
 *     This function detects the second left lane segment. Please refer to
 *     ATN.pdf to get further understanding
 *****************************************************************************/
void Detector::DetectLeft2()
{
	m_left2.Clear();

	// Didn't find m_left1, need not to find m_left2
	if (m_left1.IsEmpty())
		return;
	
	int nLastX, nLastY;
	m_left1.GetLastCoordinate(nLastX, nLastY);
	
	int nCenterX = 0;
	unsigned char* pBinary = NULL;
	
	bool bUsefulLine = false;
	int nSpareLine = 0;

	for (int y = nLastY - m_nFirstSecondDistance; y >= m_nSweepLineHighest; --y)
	{
		// For each y, calculate the x value according to the lane equation of m_left1
		nCenterX = (int) (m_left1.m_fK * y + m_left1.m_fB);
		pBinary = (unsigned char*) (m_pBinaryImg->imageData + m_pBinaryImg->widthStep * y);
		
		bUsefulLine = false;

		for (int x = nCenterX + m_nSecondAmplitude; x > nCenterX - m_nSecondAmplitude; --x)
		{
			if (pBinary[x] == 255)
			{
				m_left2.Add(x, y);
				bUsefulLine = true;
				break;
			}
		}

		if (!bUsefulLine)
		{
			++nSpareLine;

			if (nSpareLine >= 5)
			{
				if (m_left2.Length() < m_nSecondLaneSegLengthMin)
				{
					m_left2.Clear();
					nSpareLine = 0;
				}
				else
					goto RegCheck;
			}
		}
		else
			nSpareLine = 0;
	}

RegCheck:
	if (!m_left2.RegCheck(m_fThresholdReg * 2))
		m_left2.Clear();
}

/*****************************************************************************
 * Function name:
 *     DetectRight2
 *
 * Remarks:
 *     This function detects the second right lane segment. Please refer to
 *     ATN.pdf to get further understanding
 *****************************************************************************/
void Detector::DetectRight2()
{
	m_right2.Clear();

	// Didn't find m_right1, need not to find m_right2
	if (m_right1.IsEmpty())
		return;
	
	int nLastX, nLastY;
	m_right1.GetLastCoordinate(nLastX, nLastY);
	
	int nCenterX = 0;
	unsigned char* pBinary = NULL;

	bool bUsefulLine = false;
	int nSpareLine = 0;
	
	for (int y = nLastY - m_nFirstSecondDistance; y >= m_nSweepLineHighest; --y)
	{
		// For each y, calculate the x value according to the lane equation of m_right1
		nCenterX = (int) (m_right1.m_fK * y + m_right1.m_fB);
		pBinary = (unsigned char*) (m_pBinaryImg->imageData + m_pBinaryImg->widthStep * y);

		bUsefulLine = false;
		
		for (int x = nCenterX - m_nSecondAmplitude; x < nCenterX + m_nSecondAmplitude; ++x)
		{
			if (pBinary[x] == 255)
			{
				m_right2.Add(x, y);
				bUsefulLine = true;
				break;
			}
		}

		if (!bUsefulLine)
		{
			++nSpareLine;

			if (nSpareLine >= 5)
			{
				if (m_right2.Length() < m_nSecondLaneSegLengthMin)
				{
					m_right2.Clear();
					nSpareLine = 0;
				}
				else
					goto RegCheck;
			}
		}
		else
			nSpareLine = 0;
	}

RegCheck:
	if (!m_right2.RegCheck(m_fThresholdReg * 2))
		m_right2.Clear();
}


typedef std::pair<int, int> Coordinate;

class LaneSeg
{
public:
	inline LaneSeg();
	inline LaneSeg& operator=(const LaneSeg&);

	inline void Add(int, int);
	inline void Clear();

	bool RegCheck(float);

	bool Adjust();

public:
	void ToFile(std::ofstream&) const;
	bool Mark(IplImage*, CvScalar = CV_RGB(255, 0, 0)) const;

public:
	inline bool IsEmpty() const;
	inline int Length() const;
	bool GetLastCoordinate(int&, int&) const;

private:
	void CalcCenter(float&, float&);
	void CalcKB();

public:
	std::vector<Coordinate> m_vecCoordinates;	// Used to store all the points in the lane segment

	float m_fK;									// Slope of the resulting line
	float m_fB;									// Intercept of the resulting line

private:
	bool m_bRegChecked;
};

LaneSeg::LaneSeg() :
m_bRegChecked(false)
{
}

/*****************************************************************************
 * operator =
 *****************************************************************************/
LaneSeg& LaneSeg::operator=(const LaneSeg& rhs)
{
	if (this != &rhs)
	{
		m_fK = rhs.m_fK;
		m_fB = rhs.m_fB;
	}

	return *this;
}

/*****************************************************************************
 * Function name:
 *     Add
 *
 * Parameters:
 *     nX: The x-coordinate of the point to be added to the lane segment
 *     nY: The y-coordinate of the point to be added to the lane segment
 *
 * Remarks:
 *     This function adds one new pixel to the lane segment
 *****************************************************************************/
void LaneSeg::Add(int nX, int nY)
{
	m_vecCoordinates.push_back(std::make_pair(nX, nY));
}

/*****************************************************************************
 * Function name:
 *     Clear
 *
 * Remarks:
 *     This function makes the lane segment empty
 *****************************************************************************/
void LaneSeg::Clear()
{
	m_vecCoordinates.clear();
	m_bRegChecked = false;
}

/*****************************************************************************
 * Function name:
 *     IsEmpty
 *
 * Return Value:
 *     true if the lane segment has no point; otherwise false
 *
 * Remarks:
 *     This function is used to detect whether the lane segment is empty
 *****************************************************************************/
inline bool LaneSeg::IsEmpty() const
{
	return m_vecCoordinates.empty();
}

/*****************************************************************************
 * Function name:
 *     ToFile
 *
 * Return Value:
 *     Number of points in the lane segment
 *
 * Remarks:
 *     This function retrieve the number of points in the lane segment
 *****************************************************************************/
int LaneSeg::Length() const
{
	return (int) m_vecCoordinates.size();
}
/*****************************************************************************
 * Function name:
 *     RegCheck
 *
 * Parameters:
 *     fThreshold: The threshold of the regression checking
 *
 * Return Value:
 *     true if the lane segment can pass the regression checking; otherwise
 *     false
 *
 * Remarks:
 *     This function performs line fitting using least square method. It also
 *     performs regression checking to see whether those points in the lane
 *     segment really formed a straight line
 *****************************************************************************/
bool LaneSeg::RegCheck(float fThreshold)
{
	if (m_vecCoordinates.size() < 3)
		return true;

	if (fThreshold < 0.5f)
		fThreshold = 0.5f;
	else if (fThreshold > 5.0f)
		fThreshold = 5.0f;

	// Calculate slope(m_fK) and intercept(m_fB) using least square method
	CalcKB();
	// Performing regression checking
	float fError = 0.0f;
	for (int i = 0; i < (int) m_vecCoordinates.size(); ++i)
		fError += std::abs(m_vecCoordinates[i].first - m_fK * m_vecCoordinates[i].second - m_fB);
	fError /= m_vecCoordinates.size();

	// Set m_bRegChecked to indicate this lane segment has been checked
	m_bRegChecked = true;

	return fError < fThreshold;
}

/*****************************************************************************
 * Function name:
 *     Adjust
 *
 * Return Value:
 *     true if successful; otherwise false
 *
 * Remarks:
 *     Sometimes the lane segment cannot pass the regression checking only
 *     because a few points are too far away from the fitted line. So we can
 *     "Adjust" the lane segment by deleting these points to make the lane
 *     segment pass the regression checking easier. Make sure to call this
 *     function after the regression checking
 *****************************************************************************/
bool LaneSeg::Adjust()
{
	if (!m_bRegChecked)
		return false;

	std::vector<Coordinate>::iterator iter = m_vecCoordinates.begin();
	float fError = 0.0f;

	while (iter != m_vecCoordinates.end())
	{
		fError = std::abs(iter->first - m_fK * iter->second - m_fB);
		if (fError > 1.5f)
			// This point is too far away from the fitted line, so delete it
			m_vecCoordinates.erase(iter);
		else
			++iter;
	}

	return true;
}

/*****************************************************************************
 * Function name:
 *     ToFile
 *
 * Parameters:
 *     fout: An std::ofstream object indicating an opened file
 *
 * Remarks:
 *     This function serializes all the points in the lane segment to a file
 *****************************************************************************/
void LaneSeg::ToFile(std::ofstream& fout) const
{
	if (m_bRegChecked)
		fout << "x=" << m_fK << "y+" << m_fB;
	fout << "\n";
	for (int i = 0; i < (int) m_vecCoordinates.size(); ++i)
		fout << "(" << m_vecCoordinates[i].first << ", " << m_vecCoordinates[i].second << ")\n";
	fout << "\n";
}

/*****************************************************************************
 * Function name:
 *     Mark
 *
 * Parameters:
 *     pImg: The image on which to mark
 *     color: The color used to mark
 *
 * Return Value:
 *     true if successful; otherwise false
 *
 * Remarks:
 *     This function marks the lane segment on specified image with a straight
 *     line in specified color. Make sure to call this function after the
 *     regression checking
 *****************************************************************************/
bool LaneSeg::Mark(IplImage* pImg, CvScalar color) const
{
	if (!m_bRegChecked)
		return false;

	CvPoint p1, p2;
	// The first point
	p1.y = m_vecCoordinates[0].second;
	p1.x = (int) (m_fK * p1.y + m_fB);
	// The second point
	p2.y = m_vecCoordinates[m_vecCoordinates.size() - 1].second;
	p2.x = (int) (m_fK * p2.y + m_fB);

	// Draw a line on pImg from p1 to p2
	cvLine(pImg, p1, p2, color, 3);

	return true;
}

/*****************************************************************************
 * Function name:
 *     GetLastCoordinate
 *
 * Parameters:
 *     nX: Used to return the x-coordinate of the last point in the lane
 *         segment
 *     nY: Used to return the y-coordinate of the last point in the lane
 *         segment
 * Return Value:
 *     true if successful; otherwise false
 *
 * Remarks:
 *     This function retrieves the coordinates of the last point in the lane
 *     segment and returns them by reference variables if the lane segment is
 *     not empty, otherwise returns false
 *****************************************************************************/
bool LaneSeg::GetLastCoordinate(int& nX, int& nY) const
{
	if (m_vecCoordinates.empty())
		return false;

	nX = m_vecCoordinates[m_vecCoordinates.size() - 1].first;
	nY = m_vecCoordinates[m_vecCoordinates.size() - 1].second;

	return true;
}

/*****************************************************************************
 * Function name:
 *     CalcCenter
 *
 * Parameters:
 *     fCenterX: Used to return the average x value of points in the lane
 *         segment
 *     fCenterY: Used to return the average y value of points in the lane
 *         segment
 *
 * Remarks:
 *     This function calculates the average x and y value of all the points
 *     in the lane segment and returns them by reference variables. These two
 *     values are used in the further least square method
 *****************************************************************************/
void LaneSeg::CalcCenter(float& fCenterX, float& fCenterY)
{
	int nSumX = 0;
	int nSumY = 0;

	for (int i = 0; i < (int) m_vecCoordinates.size(); ++i)
	{
		nSumX += m_vecCoordinates[i].first;
		nSumY += m_vecCoordinates[i].second;
	}

	fCenterX = (float) nSumX / m_vecCoordinates.size();
	fCenterY = (float) nSumY / m_vecCoordinates.size();
}

/*****************************************************************************
 * Function name:
 *     CalcKB
 *
 * Remarks:
 *     This function performs line fitting using the least square method
 *****************************************************************************/
void LaneSeg::CalcKB()
{
	// Calculate the average x and y value of points in the lane segment
	float fCenterX, fCenterY;
	CalcCenter(fCenterX, fCenterY);

	float fDx = 0.0f;
	float fDy = 0.0f;

	float fNumerator = 0.0f;
	float fDenominator = 0.0f;

	// Performs the least square method
	for (int i = 0; i < (int) m_vecCoordinates.size(); ++i)
	{
		fDx = m_vecCoordinates[i].first - fCenterX;
		fDy = m_vecCoordinates[i].second - fCenterY;

		fNumerator += fDx * fDy;
		fDenominator += fDy * fDy;
	}

	m_fK = fNumerator / fDenominator;
	m_fB = fCenterX - m_fK * fCenterY;
}

