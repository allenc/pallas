#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace pallas {

struct Point {
    int x{0};
    int y{0};
};

struct Segment {
    Point start;
    Point end;
};

struct Polygon {
    std::vector<Point> vertices;
    std::vector<Segment> edges;
};

// if a point is inside a polygon
bool inside(const Point& point, const Polygon& polygon);

// if two points intersects a segment
bool intersects(const Point& lhs_point, const Point& rhs_point,
                const Segment& segment);

}  // namespace pallas
