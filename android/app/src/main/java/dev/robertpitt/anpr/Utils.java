package dev.robertpitt.anpr;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Iterator;
import java.util.List;

import static org.opencv.imgproc.Imgproc.INTER_CUBIC;
import static org.opencv.imgproc.Imgproc.warpAffine;

public class Utils {
 public static Mat rotateBasedOnRect(Mat source, Mat dst, RotatedRect rect) {
    Mat rot_mat = Imgproc.getRotationMatrix2D(rect.center, rect.angle, 1.0);
    warpAffine(source, dst, rot_mat, source.size(), Imgproc.INTER_CUBIC);
    return dst;
  }

  /**
   * Determine the angle
   * @param pt1
   * @param pt2
   * @param pt0
   * @return
   */
  public static double determineAngle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return ( dx1*dx2 + dy1*dy2 ) / Math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
  }

  public static MatOfPoint2f convertMatOfPointToMatOfPoint2f(MatOfPoint in) {
    MatOfPoint2f out = new MatOfPoint2f();
    in.convertTo(out, CvType.CV_32F);
    return out;
  }

  /**
   * Determine if the MatOfPoint is a rectangle, it does this by checking each point and its
   * relation to the opposing node, if the angle between the two is ~90 for all points we
   * return true.
   * @param approxCurve
   * @return
   */
  public static boolean isRectangleInShape(MatOfPoint2f approxCurve) {
    Point[] points = approxCurve.toArray();

    double maxCosine = 0;
    for( int j = 2; j < 5; j++ ) {
      double cosine = Math.abs(Utils.determineAngle(points[j % 4], points[j - 2], points[j - 1]));
      maxCosine = Math.max(maxCosine, cosine);
    }

    // if cosines of all angles are small
    // (all angles are ~90 degree) then write quadrangle
    // vertices to resultant sequence
    return maxCosine < 0.3;
  }

  public static RotatedRect getLargestContourFromList(List<RotatedRect> plates) {
    if(plates == null || plates.size() == 0)
      return null;

    double currentLargestSize = 0;
    RotatedRect largest = null;
    for(int i = 0; i < plates.size(); i++) {
      double area = plates.get(i).size.area();
      if(area > currentLargestSize) {
        currentLargestSize = area;
        largest = plates.get(i);
      }
    }

    return largest;
  }

  public static Mat rotateAndDeskew(Mat scene, RotatedRect rect) {
    // Crop the bounding box from the scene.
    Mat cropped = new Mat(scene, rect.boundingRect());

    // Rotate the cropped scene by the angle.
    double angle = rect.angle;
    Size size = rect.size;

    if (angle < -45.) {
      angle += 90.0;
    }
    Log.d("Utils:Angle", rect.angle + " > " + angle);

    // We compute the rotation matrix using the corresponding OpenCV function, we specify the center
    // of the rotation (the center of our bounding box), the rotation angle (the skew angle) and the
    // scale factor (none here).
    Mat rotationMat = Imgproc.getRotationMatrix2D(rect.center, angle, 1);

    // Now that we have the rotation matrix, we can apply the geometric transformation using the function warpAffine
    Imgproc.warpAffine(cropped, cropped, rotationMat, rect.size, INTER_CUBIC);

    Mat out = new Mat();
    Imgproc.getRectSubPix(cropped, rect.size, rect.center, cropped);

    cropped.release();
    return out;
  }
}
