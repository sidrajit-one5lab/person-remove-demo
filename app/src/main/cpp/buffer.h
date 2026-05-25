#pragma once

#include <opencv2/core.hpp>
#include <mutex>
#include <vector>
#include <cstdint>

/**
 * A single frame held in the ring buffer.
 *
 * `image` is the camera frame in RGB (CV_8UC3).
 * `personMasks` are binary masks (CV_8UC1, 0/255) — one per detected person, all sharing
 * the frame's dimensions. `anyPersonMask` is the union of those masks, precomputed once
 * so the stitching pass doesn't have to recompute it per pixel.
 */
struct BufferedFrame {
    cv::Mat image;
    std::vector<cv::Mat> personMasks;
    cv::Mat anyPersonMask;
    cv::Mat thumbnail;       // 32x24 grayscale, for diversity-based eviction
    int64_t timestampMs = 0;
};

/**
 * Bounded FIFO of [BufferedFrame]s. When full, push() overwrites the oldest entry.
 *
 * Thread-safety: push, snapshot, clear, size are all internally synchronized so the
 * camera thread can push while the capture thread reads.
 */
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity);

    void push(BufferedFrame&& f);

    /** Returns a copy of the current contents in chronological order (oldest → newest). */
    std::vector<BufferedFrame> snapshot() const;

    void clear();

    size_t size() const;
    size_t capacity() const { return capacity_; }

private:
    mutable std::mutex mu_;
    const size_t capacity_;
    std::vector<BufferedFrame> buf_;   // size <= capacity_
    size_t head_ = 0;                  // index of the OLDEST element when full
};
