package dev.robertpitt.anpr;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import dev.robertpitt.anpr.Utils;

/**
 * Algorithm Version 1.
 *
 * 1. Start Video Capture
 * 2. Read Video_Frame
 * 3. Resize Video_Frame
 * 4. Gray_Frame = Transform to grayscale (Video_Frame)
 * 5. Apply filter to remove the noise (Gray_Frame)
 * 6. Frame_Edges = Detect edges (Gray_Frame)
 * 7. Frame_Contours = Find contours (Frame_Edges)
 * 8. Sort (Frame_Contours)
 * 9. Declare Number_Plate_Contour
 * 10. Declare Largest_Rectangle
 * 11. for Contour in Frame_Contours do
 * 12.   Perimeter_Rectangle = Calculate perimeter (Contour)
 * 13.   Approx_Rectangle = Find the approximate rectangle (Perimeter_Rectangle)
 * 14.     if (length (Approx_Rectangle) == 4) then
 * 15.       Number_Plate_Contour = Approx_Rectangle
 * 16.       Largest_Rectangle = Find area (Approx_Rectangle)
 * 17.       break
 * 18.    x,y,w,h = Calculate up-right bounding rectangle (Largest_Rectangle)
 * 19.   Cropped_Frame = Crop using x,y,w,h (Video_Frame)
 * 20.   Draw Largest_Rectangle contours on Video_Frame
 * 21.   Transform to grayscale (Cropped_Frame)
 * 22.   Frame_Threshold = Binarize (Cropped_Frame)
 * 23.   Kernel = New square object of size 1x1
 * 24.   Image_Dilation = Dilates using Kernel (Frame_Threshold)
 * 25.   Dilated_Image_Contours = Find contours (Image_Dilation)
 * 26.   Sorted_Dilated_Contours = Sort (Dilated_Image_Contours)
 * 27.   for Dilated_Contour in Sorted_Dilated_Contours
 * 28.     x,y,w,h = Calculate up-right bounding rectangle (Dilated_Contour)
 * 29.     Draw a rectangle of dimensions x,y,w,h on Video_Frame
 * 30.   end for
 * 31.   Transform to binary (Gray_Frame)
 * 32.   License_Plate_Characters = Transform to string (Gray_Frame)
 * 33.   if length(License_Plate_Characters) > 0 then
 * 34.     Get License_Plate_Characters
 * 35.   end if
 * 36. return Video_Frame
 */
public class PlateDetector {
  /**
   * Log Tag
   */
  private static String TAG = "Detector";

  /**
   * Frame size we are going to
   */
  Size frameDownsizeDimentions = new Size(640, 480);

  /**
   * Container that will hold the resized frame.
   */
  private Mat resized;

  /**
   * mat container to hold the 640x480 edge detection result
   */
  private Mat edges;

  /**
   * Hierarchy Mat from the contour extraction process.
   */
  private Mat hierarchy;

  /**
   * Array list to hold on to the contours
   */
  private List<MatOfPoint> contours = new ArrayList<>();

  /**
   *
   */
  private MatOfPoint contour;

  /**
   *
   */
  private MatOfPoint2f approxCurve;

  /**
   *
   */
  private double minAreaSize = 1000.0;
  private double maxAreaSize = 100000.0;

  /**
   *
   * @param width
   * @param hieght
   */
  public void onCameraViewStarted(int width, int hieght) {
    /**
     * Reset variable instances to clear state between frames.
     */
    contour = new MatOfPoint();
    resized = new Mat();
    hierarchy = new Mat();
    approxCurve = new MatOfPoint2f();
    edges = new Mat();
    contour = new MatOfPoint();
  }

  /**
   * Perform detection on the greyscale version of the frame.
   * @param grayscale
   * @return
   *
   * @// TODO: 2020-04-12 assert the structure if the input mat is 16 bit.
   */
  public List<RotatedRect> detect(Mat grayscale, int lowerThreshold, int upperThreshold) {
    /**
     * 1. Reduce the size of the input frame in order to improve performance.
     * 2. Convert the input to gray scale and known size.
     */
//    Imgproc.resize(grayscale, resized, frameDownsizeDimentions, Imgproc.INTER_AREA);

    /**
     * 3. Perform edge detections
     */
    Imgproc.Canny(grayscale, edges, lowerThreshold, upperThreshold);
//    resized.release();

    /**
     * 4. Extract the contours from the view
     */
    contours = new ArrayList<>();
    Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
    edges.release();
    hierarchy.release();

    /**
     * 5. Extract contours
     */
    List<RotatedRect> plates = scanForLicensePlate(contours);
    contours.clear();

    return plates;
  }

  /**
   * Attempt to locate the license plate within the array of detected contours
   */
  private List<RotatedRect> scanForLicensePlate(List<MatOfPoint> contours) {
    /**
     * Return null immediately if the input size os 0, this saves setting up the
     * itterator every frame.
     */
    if(contours.size() == 0) {
      return null;
    }

    /**
     * Create a new container for the results
     */
    List<MatOfPoint> plates = new ArrayList<>();
    List<RotatedRect> rectangles = new ArrayList<>();

    /**
     * Itterate over the contours, skipping contours that we are not interested in.
     */
    for(int i = 0; i < contours.size(); i++) {
      /**
       * Extract the points of the contour.
       */
      MatOfPoint2f contour2f = new MatOfPoint2f(contours.get(i).toArray());

      /**
       * Approximate the polygon from the contour
       */
      Imgproc.approxPolyDP(contour2f, approxCurve, Imgproc.arcLength(contour2f, true) * 0.018, true);
      contour2f.release();

      /**
       * Remove those where the total sides of the approximated curve is not rectangle
       */
      if(approxCurve.total() != 4){
        approxCurve.release();
        continue;
      }

      /**
       * Calculate the total area size for the shape so we can filter
       * the selections that are too small or to0 big.
       */
      double areaSize = Math.abs(Imgproc.contourArea(approxCurve));
      if(areaSize < minAreaSize || areaSize > maxAreaSize) {
        approxCurve.release();
        continue;
      }

      /**
       * Exclude the contour of the approximation is not convex
       *
       * @see https://en.wikipedia.org/wiki/Convex_polygon#Properties
       */
      // if(!Imgproc.isContourConvex(contours.get(i))) {
      //  approxCurve.release();
      //  continue;
      // }

      /**
       * Determine if the shape is rectangular
       */
      if(!Utils.isRectangleInShape(approxCurve)) {
        approxCurve.release();
        continue;
      }

//      RotatedRect rotatedRect = Imgproc.minAreaRect(approxCurve);

      plates.add(new MatOfPoint(approxCurve.toArray()));
      rectangles.add(Imgproc.minAreaRect(approxCurve));

      // Release the approxCurve memory allocation
      approxCurve.release();
    }

    return rectangles;
//    return plates;
  }
}
