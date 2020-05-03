package dev.robertpitt.anprX;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;

import com.google.common.util.concurrent.ListenableFuture;
import com.googlecode.tesseract.android.TessBaseAPI;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.FocusMeteringAction;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.MeteringPoint;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.view.MotionEvent;
import android.view.Surface;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/**
 * Main Camera Activity
 */
public class MainActivity extends AppCompatActivity {
  /**
   * Log Tag, used to tag logs so we can easily file them using logcat.
   */
  private final static String TAG = "ANPRX::MainActivity";

  /**
   * Camera Permission Message ID, this is used for async communication with
   * the permission requester process, which is spawned if we have missing permissions.
   */
  private final static int CAMERA_PERMISSION_REQUEST = 0x01;

  /**
   * Static list of permissions this activity requires, this list will
   * be compared to the actual authorized permissions and if we are missing a
   * invocation to the android permission layer to prompt the user for those permissions.
   */
  private final static String[] permissions = {
      Manifest.permission.CAMERA,
      Manifest.permission.WRITE_EXTERNAL_STORAGE
  };

  /**
   * Tesseract Base API instance, used for processing cropped images
   * for textual representation
   */
  private TessBaseAPI tessBaseAPI = new TessBaseAPI();

  /**
   * Plate Detector Logic
   */
  private PlateDetector detector = new PlateDetector();

  /**
   * Camera Instance
   */
  private Camera camera;

  /**
   * Camera Provider Instance
   */
  private ProcessCameraProvider cameraProvider;

  /**
   * Camera Provider Future
   */
  private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

  /**
   * Camera Selector Config
   */
  private CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();

  /**
   * Preview Config
   * This needs to be initialised after the camera is ready
   */
  private Preview preview;

  /**
   * Preview View Container
   */
  private PreviewView previewView;

  /**
   * Executor thread, used for background image processing
   */
  private Executor analysisExecutor = Executors.newSingleThreadExecutor();

  /**
   * UI Component for the Toolbar
   */
  private Toolbar toolbar;

  /**
   * UI Component for drawing detector results for debugging purposes.
   */
  private ImageView imageOverlayView;
  private TextView registrationTextView;

  /**
   * Activity Creation Handler
   * @param savedInstanceState
   */
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    // Execute the parent onCreate command to initialise the activity.
    super.onCreate(savedInstanceState);

    // Configure the primary view to display
    setContentView(R.layout.activity_main);
    
    // Get references to view components
    toolbar = findViewById(R.id.toolbar);
    previewView = findViewById(R.id.preview_view);
    imageOverlayView = findViewById(R.id.imageOverlayView);
    registrationTextView = findViewById(R.id.registrationTextView);
    registrationTextView.setVisibility(View.VISIBLE);

    // Configure Toolbar
    setSupportActionBar(toolbar);

    // Request Permissions (Once granted notification is received we bind the camera)
    ActivityCompat.requestPermissions(this, permissions, CAMERA_PERMISSION_REQUEST);

    // Initialise the Tesseract library
    tessBaseAPI.init(ANPRXApplication.TESS_BASE_PATH, "eng", TessBaseAPI.OEM_TESSERACT_LSTM_COMBINED);
    tessBaseAPI.setVariable("tessedit_char_whitelist", " 0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ");
    tessBaseAPI.setDebug(false);

    // Disable dictionary lookups as we are not looking
    // https://tesseract-ocr.github.io/tessdoc/ImproveQuality#dictionaries-word-lists-and-patterns
    tessBaseAPI.setVariable("load_system_dawg", "false");
    tessBaseAPI.setVariable("load_freq_dawg", "false");
  }

  /**
   * Handle the result of a permission request, this is the response of the users
   * interaction with the Allow/Deny dialog.
   */
  @Override
  public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == CAMERA_PERMISSION_REQUEST) {
      if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        initialiseCamera();
      } else {
        Toast.makeText(this,  "Camera permission was not granted", Toast.LENGTH_LONG).show();
      }
    }
  }

  /**
   * Bind the camera
   */
  @SuppressLint("ClickableViewAccessibility")
  private void initialiseCamera() {
    // Initialize the provider instance
    cameraProviderFuture = ProcessCameraProvider.getInstance(this);

    // Listen for provider initialisation success event
    cameraProviderFuture.addListener(() -> {
      try {
        /**
         * Initialize the preview UseCase
         */
        preview = buildPreviewUseCase();

        /**
         * Configure our Image Analysis pipeline for detecting numbers
         */
        ImageAnalysis imageAnalysis = buildImageAnalysisUseCase();

        /**
         * Create the camera instance with the configured use cases.
         */
        cameraProvider = cameraProviderFuture.get();

        /**
         * Attach use cases to the camera with the same lifecycle owner
         */
        camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);

        /**
         * Connect the preview use case to the previewView
         */
        preview.setSurfaceProvider(previewView.createSurfaceProvider(camera.getCameraInfo()));

        /**
         * Attach event listener to the view finder to allow touch to focus
         */
        previewView.setOnTouchListener((view, motionEvent) -> {
          if (motionEvent.getAction() != MotionEvent.ACTION_UP) {
            MeteringPoint meteringPoint = previewView.createMeteringPointFactory(cameraSelector).createPoint(motionEvent.getX(), motionEvent.getY());
            camera.getCameraControl().startFocusAndMetering(new FocusMeteringAction.Builder(meteringPoint).build());
            return true;
          }
          return false;
        });
      } catch (ExecutionException | InterruptedException e) {
        // No errors need to be handled for this Future.
        // This should never be reached.
      }
    }, ContextCompat.getMainExecutor(this));
  }

  /**
   * Build the preview user case from the camerax library, this will be used to
   * display a video feed of the camera on the screen, this preview will be running on
   * a seperate thread to the numberplate extraction process.
   */
  private Preview buildPreviewUseCase() {
    return new Preview.Builder()
        .setTargetRotation(Surface.ROTATION_0)
        .build();
  }

  /**
   * Create the Image Analysis use case to scan for number plates in the background.
   */
  private ImageAnalysis buildImageAnalysisUseCase() {
    /**
     * Create the base configuration
     */
    ImageAnalysis imageAnalysisUseCase = new ImageAnalysis.Builder()
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setTargetRotation(Surface.ROTATION_0)
        .build();

    /**
     * Connect the analyzer handler to the analysis pipeline.
     */
    imageAnalysisUseCase.setAnalyzer(analysisExecutor, image -> this.analyzeFrame(image));

    /**
     * Return the use case
     */
    return imageAnalysisUseCase;
  }

  /**
   *
   * @param image
   */
  private void analyzeFrame(ImageProxy image) {
    /**
     * 1. Convert image to mat
     */
    Mat singleChannel8BitImage = Utils.imageToMat(image);

    /**
     * Rotate the view to match the preview window
     */
    Core.rotate(singleChannel8BitImage, singleChannel8BitImage, image.getImageInfo().getRotationDegrees() - 90);

    /**
     * 2 Scan image for rectangle shapes
     */
    final List<RotatedRect> results = detector.detect(singleChannel8BitImage, 100, 400);

    /**
     * 3. Extract the largest shape.
     */
    RotatedRect detection = Utils.getLargestContourFromList(results);
    if(detection == null) {
      singleChannel8BitImage.release();
      image.close();
      return;
    }

    /**
     * 4. Crop the detection from the greyspace and apply a threshold
     */
    Mat cropped = singleChannel8BitImage.submat(detection.boundingRect());
//    Mat cropped = Utils.rotateAndDeskew(singleChannel8BitImage, detection);
    Imgproc.threshold(cropped, cropped, 120, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

    /**
     * 5. Convert the cropped mat into a bitmap
     */
    final Bitmap bitmap = Bitmap.createBitmap(cropped.width(), cropped.height(), Bitmap.Config.ARGB_8888);
    org.opencv.android.Utils.matToBitmap(cropped, bitmap);

    // Perform OCR
    tessBaseAPI.setImage(bitmap);
    final String possibleRegistration = tessBaseAPI.getUTF8Text();
    tessBaseAPI.clear();

    /**
     * Update the UI
     */
    runOnUiThread(() -> {
      registrationTextView.setText(possibleRegistration);
      imageOverlayView.setImageBitmap(bitmap);
      imageOverlayView.setVisibility(View.VISIBLE);
    });

    singleChannel8BitImage.release();
    image.close();
  }

  private void handlePlateDetections(List<MatOfPoint2f> results) {
//    detectorResultsTextBox.scrollTo(detectorResultsTextBox.getBottom(), 0);
    for(MatOfPoint2f result : results) {

    }
  }
}
