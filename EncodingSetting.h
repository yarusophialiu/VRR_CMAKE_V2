#pragma once
#include <string>

class EncodingSetting
{
public:
    EncodingSetting(int speed, int fps, int width, int height, int bitrate)
        : mSpeed(speed), mFps(fps), mWidth(width), mHeight(height), mBitrate(bitrate) {}

    // int getFps() const { return mFps; }
    // int getWidth() const { return mWidth; }
    // int getHeight() const { return mHeight; }
    // int getBitrate() const { return mBitrate; }

    std::string toFilename() const
    {
        return "output_" + std::to_string(mFps) + "fps_" +
               std::to_string(mWidth) + "x" + std::to_string(mHeight) + "_" +
               std::to_string(mBitrate) + "kbps.mp4";
    }

public:
    int mSpeed;
    int mFps;
    int mWidth;
    int mHeight;
    int mBitrate;
};
