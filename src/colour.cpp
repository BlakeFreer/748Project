#include "colour.hpp"

#include <stdexcept>


Colour Colour::Interpolate(const Colour& c1, const Colour& c2, float t) {
    Colour c{};
    c.red = static_cast<uint8_t>((1 - t) * (float)c1.red + t * (float)c2.red);
    c.green =
        static_cast<uint8_t>((1 - t) * (float)c1.green + t * (float)c2.green),
    c.blue =
        static_cast<uint8_t>((1 - t) * (float)c1.blue + t * (float)c2.blue);
    return c;
}

bool Colour::operator==(const Colour& other) const {
    return this->red == other.red && this->green == other.green &&
           this->blue == other.blue;
}
bool Colour::operator!=(const Colour& other) const {
    return !(*this == other);
}

ContinuousColourMap::ContinuousColourMap(std::initializer_list<Colour> colours)
    : colours_(colours), mini_(0.f), maxi_(1.f) {}

Colour ContinuousColourMap::Get(float t) const {
    if (t <= mini_) return colours_.front();
    if (t >= maxi_) return colours_.back();
    if (std::isnan(t)) return colours_.front();

    t = (t - mini_) / (maxi_ - mini_);  // rescale
    t *= colours_.size() - 1;           // convert to index
    int idx = std::floor(t);

    Colour lower = colours_[idx];
    Colour upper = colours_[idx + 1];
    return Colour::Interpolate(lower, upper, t - idx);
}

void ContinuousColourMap::Rescale(float minimum, float maximum) {
    if (minimum >= maximum) {
        throw std::runtime_error("minimum must be less than maximum");
    }

    mini_ = minimum;
    maxi_ = maximum;
}