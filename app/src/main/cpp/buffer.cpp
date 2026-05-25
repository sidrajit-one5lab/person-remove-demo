#include "buffer.h"
#include <opencv2/imgproc.hpp>

RingBuffer::RingBuffer(size_t capacity) : capacity_(capacity) {
    buf_.reserve(capacity);
}

void RingBuffer::push(BufferedFrame&& f) {
    // Compute thumbnail if not already set (jni_bridge may have set it)
    if (f.thumbnail.empty() && !f.image.empty()) {
        cv::Mat gray;
        cv::cvtColor(f.image, gray, cv::COLOR_RGB2GRAY);
        cv::resize(gray, f.thumbnail, cv::Size(32, 24), 0, 0, cv::INTER_AREA);
    }

    std::lock_guard<std::mutex> g(mu_);
    if (buf_.size() < capacity_) {
        buf_.push_back(std::move(f));
    } else {
        // Evict the frame most similar to its predecessor (lowest diversity
        // contribution). Scan is O(capacity) with 32x24 diffs — microseconds.
        size_t evictIdx = head_;  // default: oldest (FIFO fallback)
        double minDiff = 1e9;
        for (size_t i = 0; i < capacity_; ++i) {
            size_t cur = (head_ + i) % capacity_;
            size_t prev = (head_ + i + capacity_ - 1) % capacity_;
            if (buf_[cur].thumbnail.empty() || buf_[prev].thumbnail.empty()) continue;
            cv::Mat diff;
            cv::absdiff(buf_[cur].thumbnail, buf_[prev].thumbnail, diff);
            double d = cv::mean(diff)[0];
            if (d < minDiff) {
                minDiff = d;
                evictIdx = cur;
            }
        }
        buf_[evictIdx] = std::move(f);
        // head_ stays at the oldest logical position; eviction is content-based
        // so chronological ordering may have a gap, but snapshot still returns
        // all frames and the stitcher doesn't depend on ordering.
    }
}

std::vector<BufferedFrame> RingBuffer::snapshot() const {
    std::lock_guard<std::mutex> g(mu_);
    std::vector<BufferedFrame> out;
    out.reserve(buf_.size());
    if (buf_.size() < capacity_) {
        // Not yet full → contents are simply 0..size-1.
        for (const auto& f : buf_) out.push_back(f);
    } else {
        // Full → start at head (oldest) and wrap.
        for (size_t i = 0; i < capacity_; ++i) {
            out.push_back(buf_[(head_ + i) % capacity_]);
        }
    }
    return out;
}

void RingBuffer::clear() {
    std::lock_guard<std::mutex> g(mu_);
    buf_.clear();
    head_ = 0;
}

size_t RingBuffer::size() const {
    std::lock_guard<std::mutex> g(mu_);
    return buf_.size();
}
