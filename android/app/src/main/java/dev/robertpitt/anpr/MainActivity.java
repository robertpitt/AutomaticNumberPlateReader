package dev.robertpitt.anpr;


import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    /**
     * Tag used for debugging
     */
    private static final String TAG = "MainActivity";

    /**
     * Permission Request Code used for async permission dialog communication.
     */
    private static final int CAMERA_PERMISSION_REQUEST = 0x00001;

    /**
     * Camera Bridge View
     */
    private CameraBridgeViewBase mOpenCvCameraView;

    /**
     * List of permissions required for the camera to be initialised
     */
    private String[] requiredPermissions = new String[] {
        Manifest.permission.CAMERA
    };

    /**
     * Callback for the manager
     */
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                mOpenCvCameraView.enableView();
            } else {
                super.onManagerConnected(status);
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        // Lock Screen orientation
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // Permissions for Android 6+
        ActivityCompat.requestPermissions(this, requiredPermissions, CAMERA_PERMISSION_REQUEST);

        // Set the view
        setContentView(R.layout.activity_main);

        // Initialise the preview window
        mOpenCvCameraView = (JavaCamera2View)findViewById(R.id.main_surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    /**
     *
     * @param requestCode
     * @param permissions
     * @param grantResults
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                mOpenCvCameraView.setCameraPermissionGranted();
            } else {
                String message = "Camera permission was not granted";
                Log.e(TAG, message);
                Toast.makeText(this, message, Toast.LENGTH_LONG).show();
            }
        } else {
            Log.e(TAG, "Unexpected permission request");
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame frame) {
        // get current camera frame as OpenCV Mat object
        Mat gray = frame.gray();
        Mat rgba = frame.rgba();

        // Flip the rgba and gray so they are upright
        Core.flip(rgba.t(), rgba, 1);
        Core.flip(gray.t(), gray, 1);

        // Perform a canny edge detection process in the gray image.
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, 60, 175);

        // Find Contours
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(edges, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        // Find rectangles
        List<MatOfPoint> licencePlateContour = detectLicencePlateFromContours(contours);

//        // Sort the contours by area size
//        Collections.sort(contours, new Comparator<MatOfPoint>() {
//            @Override
//            public int compare(MatOfPoint a, MatOfPoint b) {
//                // Not sure how expensive this will be computationally as the
//                // area is computed for each comparison
//                double areaA = Imgproc.contourArea(a);
//                double areaB = Imgproc.contourArea(b);
//
//                // Change sign depending on whether your want sorted small to big
//                // or big to small
//                if (areaA > areaB) {
//                    return -1;
//                } else if (areaA < areaB) {
//                    return 1;
//                }
//
//                return 0;
//            }
//        });

//        List<MatOfPoint> topTenContours = contours.subList(0, contours.size() > 10 ? 10 : contours.size());

        // Draw contours on the rgba for inspection
        Imgproc.drawContours(rgba, licencePlateContour, -1, new Scalar(255, 0, 0), 3);

        // return processed frame for live preview
        return rgba;
    }

    /// https://github.com/opencv/opencv/blob/master/samples/cpp/squares.cpp
    private double angle(Point pt1, Point pt2, Point pt0 )
    {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return ( dx1*dx2 + dy1*dy2 ) / Math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
    }

    private List<MatOfPoint> detectLicencePlateFromContours(List<MatOfPoint> contours) {

        List<MatOfPoint> results = new ArrayList<MatOfPoint>();
        MatOfPoint2f approxCurve = new MatOfPoint2f();

        // Itterate over
        for(MatOfPoint contour : contours) {
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            // Common values for the second parameter to cv2.approxPolyDP  are normally in the
            // range of 1-5% of the original contour perimeter.
            Imgproc.approxPolyDP(contour2f, approxCurve, Imgproc.arcLength(contour2f, true) * 0.018, true);
            Point[] approxCurveArr = approxCurve.toArray();
            MatOfPoint approxCurveMat = new MatOfPoint(approxCurveArr);

            // If the rectangle doesn't meet the minimum area
            // The higher this value the large the plate has to be within the vamera view.
            if(approxCurve.total() == 4 && Math.abs(Imgproc.contourArea(approxCurve)) > 500 && Imgproc.isContourConvex(approxCurveMat)) {
                double maxCosine = 0;
                for( int j = 2; j < 5; j++ )
                {
                    // Determine Angle
                    // find the maximum cosine of the angle between joint edges
                    double cosine = Math.abs(angle(approxCurveArr[j % 4], approxCurveArr[j - 2], approxCurveArr[j - 1]));
                    maxCosine = Math.max(maxCosine, cosine);
                }

                // if cosines of all angles are small
                // (all angles are ~90 degree) then write quandrange
                // vertices to resultant sequence
                if( maxCosine < 0.3 )
                    results.add(approxCurveMat);
            }

//            // Calculate the bounding rect
//            Rect boundingRect = Imgproc.boundingRect(approxCurve);
//            double ratio = boundingRect.width / boundingRect.height;
//
//            if(ratio != 3 && ratio != 4) {
//                continue;
//            }
        }

        return results;
    }
}