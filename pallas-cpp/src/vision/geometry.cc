#include "geometry.h"

namespace pallas {

// Check if two line segments intersect
bool intersects(const Point& lhs_point, const Point& rhs_point,
                const Segment& segment) {
    // Convert to line segment
    Segment line{lhs_point, rhs_point};

    // Helper function to determine direction of three points
    auto direction = [](const Point& p1, const Point& p2,
                        const Point& p3) -> int {
        return (p3.y - p1.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p3.x - p1.x);
    };

    // Helper function to check if a point lies on a segment
    auto onSegment = [](const Point& p, const Point& q,
                        const Point& r) -> bool {
        return (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) &&
                q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y));
    };

    // Find the four directions needed for general case
    int d1 = direction(segment.start, segment.end, line.start);
    int d2 = direction(segment.start, segment.end, line.end);
    int d3 = direction(line.start, line.end, segment.start);
    int d4 = direction(line.start, line.end, segment.end);

    // General case - if the directions are different, the lines intersect
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)))
        return true;

    // Special cases - check if a point from one segment lies on the other
    // segment
    if (d1 == 0 && onSegment(segment.start, line.start, segment.end))
        return true;
    if (d2 == 0 && onSegment(segment.start, line.end, segment.end)) return true;
    if (d3 == 0 && onSegment(line.start, segment.start, line.end)) return true;
    if (d4 == 0 && onSegment(line.start, segment.end, line.end)) return true;

    return false;
}

// Check if a point is inside a polygon using ray casting algorithm
bool inside(const Point& point, const Polygon& polygon) {
    if (polygon.vertices.size() < 3) return false;  // Not a valid polygon

    // Create a point definitely outside the polygon
    Point outside{-1000, point.y};

    // Count intersections of the ray from our point to the outside point
    int count = 0;
    Segment ray{point, outside};

    for (const auto& edge : polygon.edges) {
        if (intersects(ray.start, ray.end, edge)) {
            // Special case: if ray passes through a vertex
            if ((edge.start.y == point.y && edge.start.x >= point.x) ||
                (edge.end.y == point.y && edge.end.x >= point.x)) {
                // Handle the vertex case carefully to avoid double counting
                const Point& next =
                    (edge.end.y == point.y) ? edge.end : edge.start;
                const Point& prev =
                    (edge.end.y == point.y) ? edge.start : edge.end;

                // Check if the vertex is a local maximum/minimum
                if ((prev.y < point.y && next.y > point.y) ||
                    (prev.y > point.y && next.y < point.y)) {
                    count++;
                }
            } else {
                count++;
            }
        }
    }

    // If the count is odd, the point is inside the polygon
    return count % 2 == 1;
}

}  // namespace pallas
