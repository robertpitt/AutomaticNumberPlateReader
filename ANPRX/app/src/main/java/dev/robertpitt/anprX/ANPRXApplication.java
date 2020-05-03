package dev.robertpitt.anprX;

import android.app.Application;
import android.os.Environment;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.camera.camera2.Camera2Config;
import androidx.camera.core.CameraXConfig;

import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import io.sentry.android.core.SentryAndroid;
import io.sentry.core.SentryLevel;

public class ANPRXApplication extends Application implements CameraXConfig.Provider {
  /**
   * Log Tag
   */
  private static final String TAG = "ANPRX::ANPRXApplication";

  /**
   * Base path for tesseract storage
   */
  public static final String TESS_BASE_PATH = Environment.getExternalStorageDirectory().getAbsolutePath() + "/tesseract";

  /**
   * Override CameraX Config
   * @return
   */
  @NonNull
  @Override
  public CameraXConfig getCameraXConfig() {
    return Camera2Config.defaultConfig();
  }

  /**
   * Initialise the Application
   */
  @Override
  public void onCreate() {
    initialiseSentry();
    initialiseTesseract();
    OpenCVLoader.initDebug();
    super.onCreate();
  }

  /**
   * Configure Sentry
   */
  private void initialiseSentry() {
    /**
     * Manual Initialization of the Sentry Android SDK
     * @Context - Instance of the Android Context
     * @Options - Call back function that you need to provide to be able to modify the options.
     * The call back function is provided with the options loaded from the manifest.
     */
    SentryAndroid.init(this, options -> {
      // Add a callback that will be used before the event is sent to Sentry.
      // With this callback, you can modify the event or, when returning null, also discard the event.
      options.setBeforeSend((event, hint) -> {
        if (SentryLevel.DEBUG.equals(event.getLevel()))
          return null;
        else
          return event;
      });
    });
  }

  /**
   * Initialise tesseract models and copy them to the SD-Card.
   */
  private void initialiseTesseract() {

    // Create the folders if they don't exists
    Utils.mkdir(TESS_BASE_PATH);
    Utils.mkdir(TESS_BASE_PATH + "/tessdata");

    // copy the eng lang file from assets folder if not exists.
    File f2 = new File(TESS_BASE_PATH + "/tessdata/eng.traineddata");
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
        Log.e(TAG, e.toString());
        e.printStackTrace();
      }
    }
  }
}
