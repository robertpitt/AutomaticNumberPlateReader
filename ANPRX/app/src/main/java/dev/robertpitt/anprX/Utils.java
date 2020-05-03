package dev.robertpitt.anprX;

import android.graphics.ImageFormat;
import android.util.Log;

import androidx.camera.core.ImageProxy;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.List;

import static org.opencv.imgproc.Imgproc.INTER_AREA;
import static org.opencv.imgproc.Imgproc.INTER_CUBIC;
import static org.opencv.imgproc.Imgproc.warpAffine;

public class Utils {

  /**
   *
   * @param path
   * @return
   */
  public static boolean mkdir(String path) {
    File file = new File(path);
    if(!file.exists()){
      return file.mkdirs();
    }

    return true;
  }

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

  /**
   * https://gist.github.com/FWStelian/4c3dcd35960d6eabbe661c3448dd5539
   * @param image
   * @return
   */
  public static Mat imageToMat(ImageProxy image) {
    int width = image.getWidth();
    int height = image.getHeight();

    ImageProxy.PlaneProxy yPlane = image.getPlanes()[0];
    int ySize = yPlane.getBuffer().remaining();

    byte[] data = new byte[ySize];
    yPlane.getBuffer().get(data, 0, ySize);

    Mat greyMat = new Mat(height, width, CvType.CV_8UC1);
    greyMat.put(0, 0, data);

    return greyMat;
  }

  public static Mat yuvToMat(ImageProxy image) {
    ByteBuffer buffer;
    int rowStride;
    int pixelStride;
    int width = image.getWidth();
    int height = image.getHeight();
    int offset = 0;

    ImageProxy.PlaneProxy[] planes = image.getPlanes();
    byte[] data = new byte[image.getWidth() * image.getHeight() * ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8];
    byte[] rowData = new byte[planes[0].getRowStride()];

    for (int i = 0; i < 2; i++) {
      buffer = planes[i].getBuffer();
      rowStride = planes[i].getRowStride();
      pixelStride = planes[i].getPixelStride();
      int w = (i == 0) ? width : width / 2;
      int h = (i == 0) ? height : height / 2;
      for (int row = 0; row < h; row++) {
        int bytesPerPixel = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8;
        if (pixelStride == bytesPerPixel) {
          int length = w * bytesPerPixel;
          buffer.get(data, offset, length);

          if (h - row != 1) {
            buffer.position(buffer.position() + rowStride - length);
          }
          offset += length;
        } else {
          if (h - row == 1) {
            buffer.get(rowData, 0, width - pixelStride + 1);
          } else {
            buffer.get(rowData, 0, rowStride);
          }

          for (int col = 0; col < w; col++) {
            data[offset++] = rowData[col * pixelStride];
          }
        }
      }
    }

    Mat mat = new Mat(height + height / 2, width, CvType.CV_8UC1);
    mat.put(0, 0, data);

    return mat;
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
    // We compute the rotation matrix using the corresponding OpenCV function, we specify the center
    // of the rotation (the center of our bounding box), the rotation angle (the skew angle) and the
    // scale factor (none here).
//    double angle = rect.angle;
/// This will work for slight rotations, once going over 45 degrees this might result in wrong transformations
    double angle = 0.0;
    if((int)rect.angle < -45){
      angle = (rect.angle + 90) % angle;
    } else {
      angle = (int) rect.angle;
    }

    if(rect.size.height > rect.size.width) {
      angle += 90;
    }
    // the `cv2.minAreaRect` function returns values in the
    // range [-90, 0); as the rectangle rotates clockwise the
    // returned angle trends to 0 -- in this special case we
    // need to add 90 degrees to the angle
//    if (angle < -45) {
//      angle = -(90 + angle);
//    }
//    Log.d("ANPRX::Utils", rect.angle + " > " + angle);

    Mat rotationMat = Imgproc.getRotationMatrix2D(rect.center, angle, 1);

    // Now that we have the rotation matrix, we can apply the geometric transformation using the function warpAffine
    Mat sceneRotated = new Mat();
    Imgproc.warpAffine(scene, sceneRotated, rotationMat, scene.size(), INTER_AREA);
    Mat patch = new Mat();
    Imgproc.getRectSubPix(sceneRotated, rect.size, rect.center, patch);
    sceneRotated.release();
    return patch;
  }
}
