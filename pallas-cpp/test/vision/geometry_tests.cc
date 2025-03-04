#include <gtest/gtest.h>
#include <vision/geometry.h>

namespace pallas {

Polygon square(int size, int x = 0, int y = 0) {
    Polygon polygon;
    polygon.vertices = {
        {x, y}, {x + size, y}, {x + size, y + size}, {x, y + size}};
    polygon.edges = {{{x, y}, {x + size, y}},
                     {{x + size, y}, {x + size, y + size}},
                     {{x + size, y + size}, {x, y + size}},
                     {{x, y + size}, {x, y}}};

    return polygon;
}

class GeometryTests : public testing::Test {
   protected:
};
TEST_F(GeometryTests, Inside_PointInsidePolygon) {
    // Precondition: A square polygon from (0,0) to (10,10)
    const Point inside_point{5, 5};

    // Postcondition: The point should be inside
    EXPECT_TRUE(inside(inside_point, square(10)));
}

TEST_F(GeometryTests, Inside_PointOutsidePolygon) {
    // Precondition: A square polygon
    const Point outside_point{15, 15};

    // Postcondition: The point should be outside
    EXPECT_FALSE(inside(outside_point, square(10)));
}

TEST_F(GeometryTests, Inside_PointOnEdge) {
    // Precondition: A square polygon
    const Point edge_point{0, 5};

    // Postcondition: Depending on definition, it might be inside or on boundary
    EXPECT_TRUE(inside(edge_point, square(10)));  // Adjust if necessary
}

TEST_F(GeometryTests, Intersects_SegmentIntersectingPoints) {
    // Precondition: A segment from (0,0) to (10,10)
    Segment segment{{0, 0}, {10, 10}};

    // Under test: Two points that should intersect the segment
    Point p1{5, 5};
    Point p2{7, 7};

    // Postcondition: The function should return true
    EXPECT_TRUE(intersects(p1, p2, segment));
}

TEST_F(GeometryTests, Intersects_SegmentNotIntersectingPoints) {
    // Precondition: A segment from (0,0) to (10,10)
    Segment segment{{0, 0}, {10, 10}};

    // Under test: Two points that do not intersect the segment
    Point p1{15, 15};
    Point p2{20, 20};

    // Postcondition: The function should return false
    EXPECT_FALSE(intersects(p1, p2, segment));
}

TEST_F(GeometryTests, Intersects_PointCoincidesWithSegmentEndpoint) {
    // Precondition: A segment from (0,0) to (10,10)
    Segment segment{{0, 0}, {10, 10}};

    // Under test: One point at an endpoint, one inside
    Point p1{0, 0};
    Point p2{5, 5};

    // Postcondition: The function should return true
    EXPECT_TRUE(intersects(p1, p2, segment));
}

}  // namespace pallas
