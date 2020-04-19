package dev.robertpitt.anpr;


import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;

import com.googlecode.tesseract.android.TessBaseAPI;

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
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

import static dev.robertpitt.anpr.Utils.rotateBasedOnRect;

public class MainActivity extends Activity implements CvCameraViewListener2, View.OnClickListener {
    /**
     * Tag used for debugging
     */
    private static final String TAG = "MainActivity";

    /**
     * Permission Request Code used for async permission dialog communication.
     */
    private static final int CAMERA_PERMISSION_REQUEST = 0x00001;

    /**
     * Default Lower threshold for the canny edge detection process.
     */
    private static final int CANNY_LOWER_THRESHOLD = 100;

    /**
     * Default Upper threshold for the canny edge detection process.
     */
    private static final int CANNY_UPPER_THRESHOLD = 800;

    /**
     * Base path for tesseract storage
     */
    private static final String TESS_BASE_BATH = Environment.getExternalStorageDirectory().getAbsolutePath() + "/tesseract/";

    /**
     * Base path for tesseract models
     */
    private static final String TESS_DATA_PATH = TESS_BASE_BATH + "tessdata/";

    /**
     * The color red in scalar format.
     */
    private static final Scalar COLOR_RED = new Scalar(255, 0, 0);

    /**
     * Tesseract API
     */
    TessBaseAPI tessBaseAPI = new TessBaseAPI();

    /**
     * Deter Engine
     */
    PlateDetector detector = new PlateDetector();

    /**
     * Camera Bridge View
     */
    private CameraBridgeViewBase mOpenCvCameraView;

    /**
     * List of permissions required for the camera to be initialised
     */
    private String[] requiredPermissions = new String[] {
        Manifest.permission.CAMERA,
        Manifest.permission.WRITE_EXTERNAL_STORAGE,
        Manifest.permission.READ_EXTERNAL_STORAGE
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

    /**
     * Lower SeekBar reference
     */
    private SeekBar mLowerThreshSeekBar;

    /**
     * Upper SeekBar reference
     */
    private SeekBar mUpperThreshSeekBar;

    /**
     *
     */
    private Button mSettinsButton;

    /**
     * Image frame references
     */
    private Mat rgba;
    private Mat gray;

    /**
     * Executed when the activity is created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
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

        // Initialize seek options
        mLowerThreshSeekBar = findViewById(R.id.lowerThreshold);
        mUpperThreshSeekBar = findViewById(R.id.upperThreshold);

        // Set the default canny values
        mLowerThreshSeekBar.setProgress(CANNY_LOWER_THRESHOLD);
        mUpperThreshSeekBar.setProgress(CANNY_UPPER_THRESHOLD);

        // Settings Button
        mSettinsButton = findViewById(R.id.settingsButton);
        mSettinsButton.setOnClickListener(this);

        // Initialise tessBaseAPI
        initialiseTesseract();
    }

    /**
     *
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

    /**
     *
     */
    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    /**
     *
     */
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

    /**
     *
     */
    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    /**
     *
     */
    private void initialiseTesseract() {

        // Create teh folders if they don't exists
        mkdir(TESS_BASE_BATH);
        mkdir(TESS_DATA_PATH);

        // copy the eng lang file from assets folder if not exists.
        File f2 = new File(TESS_DATA_PATH + "eng.traineddata");
        if(!f2.exists()){
            InputStream in = null;
            try {
                in = getAssets().open( "tessdata/eng.traineddata");
                FileOutputStream fout = new FileOutputStream(f2);
                byte[] buf = new byte[1024];
                int len;
                while ((len = in.read(buf)) > 0) {
                    fout.write(buf, 0, len);
                }
                in.close();
                fout.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        tessBaseAPI.init(TESS_BASE_BATH, "eng", TessBaseAPI.OEM_LSTM_ONLY);
        tessBaseAPI.setDebug(false);
        tessBaseAPI.setVariable("tessedit_char_whitelist", " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");

        // Disable dictionary lookups as we are not looking
        // https://tesseract-ocr.github.io/tessdoc/ImproveQuality#dictionaries-word-lists-and-patterns
        tessBaseAPI.setVariable("load_system_dawg", "false");
        tessBaseAPI.setVariable("load_freq_dawg", "false");
    }

    /**
     *
     * @param width -  the width of the frames that will be delivered
     * @param height - the height of the frames that will be delivered
     */
    @Override
    public void onCameraViewStarted(int width, int height) {
        detector.onCameraViewStarted(width, height);
        rgba = new Mat();
        gray = new Mat();
    }

    /**
     *
     */
    @Override
    public void onCameraViewStopped() {
    }

    /**
     *
     * @param frame
     * @return
     */
    @Override
    public Mat onCameraFrame(CvCameraViewFrame frame) {
        rgba = frame.rgba();
        gray = frame.gray();

        // This fixes image rotation, shape really that we have to do this computation on every frame.
        Core.flip(rgba, rgba, 0);
        Core.flip(gray, gray, 0);
        Core.transpose(rgba, rgba);
        Core.transpose(gray, gray);

        // List of detected plates
        List<RotatedRect> plates = detector.detect(gray, mLowerThreshSeekBar.getProgress(), mUpperThreshSeekBar.getProgress());

        // Fetch the largest registration
        RotatedRect detection = Utils.getLargestContourFromList(plates);

        /**
         * Extract the
         */
        if(detection != null) {
            Mat cropped = new Mat(gray, detection.boundingRect());
//            Mat cropped = Utils.rotateAndDeskew(gray, detection);

            // Threshold the plate
            Imgproc.threshold(cropped, cropped, 50, 255, Imgproc.THRESH_BINARY);

            // Output bitmap for OCR
            Bitmap bitmap = Bitmap.createBitmap(cropped.width(), cropped.height(), Bitmap.Config.ARGB_8888);
            org.opencv.android.Utils.matToBitmap(cropped, bitmap);

            // Perform OCR
            tessBaseAPI.setImage(bitmap);
            String plate = tessBaseAPI.getUTF8Text();

            Point[] vertices = new Point[4];
            detection.points(vertices);

            // Draw Bounding Box
            Imgproc.drawContours(rgba, Arrays.asList(new MatOfPoint(vertices)), -1, COLOR_RED, 3);

            // Write plate and confidence
            Imgproc.putText(rgba,  detection.angle + " deg - " + plate, detection.boundingRect().tl(), Imgproc.FONT_HERSHEY_PLAIN, 2, COLOR_RED, 2);
        }

//        // If we are debugging then show the plate overlay
//        if(largestPlate != null) {
//            Mat cropped = new Mat(gray, largestPlate.boundingRect());
//
//            // Fix angle of rotation
//            double angle = largestPlate.angle < -45. ? largestPlate.angle + 90. : largestPlate.angle;
//
//            Point[] vertices = new Point[4];
//            largestPlate.points(vertices);
//
//            // While tesseract version 3.05 (and older) handle inverted image
//            // (dark background and light text) without problem, for 4.x version use dark
//            // text on light background.
//            // THRESH_BINARY == White background + Black Text
//            Imgproc.threshold(cropped, cropped, 50, 255, Imgproc.THRESH_BINARY);
//            rotateBasedOnRect(cropped, cropped, largestPlate);
//
//            Bitmap bitmap = Bitmap.createBitmap(cropped.width(), cropped.height(), Bitmap.Config.ARGB_8888);
//            org.opencv.android.Utils.matToBitmap(cropped, bitmap);
//
//            // Perform OCR
//            tessBaseAPI.setImage(bitmap);
//            String plate = tessBaseAPI.getUTF8Text();
//
//            Imgproc.putText(rgba, tessBaseAPI.wordConfidences() + "", largestPlate.boundingRect().tl(), Imgproc.FONT_HERSHEY_PLAIN, 3, COLOR_RED, 3);
//
//            MatOfPoint contour = new MatOfPoint();
//            Imgproc.drawContours(rgba, Arrays.asList(new MatOfPoint(vertices)), -1, COLOR_RED, 3);
//            cropped.release();
//        }

        gray.release();
        return rgba;
    }

    @Override
    public void onClick(View view) {
        if(view.getId() == R.id.settingsButton) {
            Intent myIntent = new Intent(this, SettingsActivity.class);
            this.startActivity(myIntent);
        }
    }
}