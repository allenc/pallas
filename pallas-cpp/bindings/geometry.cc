#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include "../src/vision/geometry.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(pallas_py, m) {
    // Define Python bindings for Point class
    nb::class_<pallas::Point>(m, "Point")
        .def(nb::init<>())
        .def(nb::init<int, int>(), "x"_a, "y"_a)
        .def_rw("x", &pallas::Point::x)
        .def_rw("y", &pallas::Point::y)
        .def("__repr__", [](const pallas::Point &p) {
            return "Point(" + std::to_string(p.x) + ", " + std::to_string(p.y) + ")";
        });

    // Define Python bindings for Segment class
    nb::class_<pallas::Segment>(m, "Segment")
        .def(nb::init<>())
        .def(nb::init<pallas::Point, pallas::Point>(), "start"_a, "end"_a)
        .def_rw("start", &pallas::Segment::start)
        .def_rw("end", &pallas::Segment::end)
        .def("__repr__", [](const pallas::Segment &s) {
            return "Segment(start=Point(" + std::to_string(s.start.x) + ", " + 
                   std::to_string(s.start.y) + "), end=Point(" + 
                   std::to_string(s.end.x) + ", " + std::to_string(s.end.y) + "))";
        });

    // Define Python bindings for Polygon class
    nb::class_<pallas::Polygon>(m, "Polygon")
        .def(nb::init<>())
        .def_rw("vertices", &pallas::Polygon::vertices)
        .def_rw("edges", &pallas::Polygon::edges)
        .def("__repr__", [](const pallas::Polygon &poly) {
            std::string repr = "Polygon(vertices=[";
            for (size_t i = 0; i < poly.vertices.size(); ++i) {
                if (i > 0) repr += ", ";
                repr += "Point(" + std::to_string(poly.vertices[i].x) + ", " + 
                       std::to_string(poly.vertices[i].y) + ")";
            }
            repr += "], edges=[";
            for (size_t i = 0; i < poly.edges.size(); ++i) {
                if (i > 0) repr += ", ";
                repr += "Segment(...)";  // Simplified representation for edges
            }
            repr += "])";
            return repr;
        });

    // Bind free functions
    m.def("inside", &pallas::inside, "Check if a point is inside a polygon", 
          "point"_a, "polygon"_a);
    m.def("intersects", &pallas::intersects, 
          "Check if a line between two points intersects a segment", 
          "lhs_point"_a, "rhs_point"_a, "segment"_a);
}
