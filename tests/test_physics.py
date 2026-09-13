import math
import unittest

import numpy as np

from optics2d.model import FrequencyChannel, OpticalElement, OpticalSystem
from optics2d.physics import (
    BeamSegment,
    C_MM_GHZ,
    apply_focusing_power,
    ellipse_effective_focal_length,
    ellipse_geometry,
    ellipse_vertices,
    reflected,
    segment_waist,
    trace_all_beams,
    trace_system,
)
from optics2d.model import default_band6_system
from optics2d.constraints import (
    attach_assembly_to_block,
    block_endpoints,
    enforce_links,
    move_with_constraints,
    ruler_attachments,
)
from optics2d.visualization import cryostat_window_corners


class PhysicsTests(unittest.TestCase):
    def test_frequency_to_wavelength(self):
        system = OpticalSystem([OpticalElement("horn", "H", 0, 0, frequency_ghz=243.0)])
        result = trace_system(system)
        self.assertAlmostEqual(result.wavelength_mm, C_MM_GHZ / 243.0, places=10)

    def test_plane_mirror_reflection(self):
        mirror = OpticalElement("plane_mirror", "M", 0, 0, angle_deg=90)
        output = reflected(np.array([1.0, 0.0]), mirror)
        np.testing.assert_allclose(output, [-1.0, 0.0], atol=1e-12)

    def test_unlinked_mirror_after_linked_mirror_still_reflects(self):
        horn = OpticalElement("horn", "H", 0.0, 0.0, angle_deg=0.0)
        first = OpticalElement("plane_mirror", "M1", 50.0, 0.0, angle_deg=45.0, aperture=40.0)
        second = OpticalElement("plane_mirror", "M2", 50.0, 50.0, angle_deg=0.0, aperture=40.0)
        system = OpticalSystem([horn, first, second], max_path_mm=180.0)
        system.connect(horn.uid, first.uid)
        result = trace_system(system)
        self.assertEqual(result.segments[0].hit_uid, first.uid)
        self.assertEqual(result.segments[1].hit_uid, second.uid)

    def test_segment_waist_location(self):
        segment = BeamSegment(
            np.array([2.0, 3.0]), np.array([1.0, 0.0]), 20.0,
            complex(-7.0, 12.0), 1.0,
        )
        waist = segment_waist(segment)
        self.assertIsNotNone(waist)
        self.assertAlmostEqual(waist.distance, 7.0)
        np.testing.assert_allclose(waist.position, [9.0, 3.0])

    def test_lens_abcd_law(self):
        q_in = complex(100.0, 50.0)
        f = 80.0
        expected = 1.0 / (1.0 / q_in - 1.0 / f)
        self.assertAlmostEqual(apply_focusing_power(q_in, f), expected)

    def test_beam_radius_at_initial_waist(self):
        horn = OpticalElement("horn", "H", 0, 0, waist_radius=2.0, frequency_ghz=243.0)
        result = trace_system(OpticalSystem([horn], max_path_mm=10))
        self.assertAlmostEqual(float(result.segments[0].radius(0)), 2.0, places=10)

    def test_trace_hits_and_reflects(self):
        horn = OpticalElement("horn", "H", 0, 0, angle_deg=0)
        mirror = OpticalElement("plane_mirror", "M", 10, 0, angle_deg=90, aperture=20)
        result = trace_system(OpticalSystem([horn, mirror], max_path_mm=30))
        self.assertEqual(len(result.segments), 2)
        self.assertAlmostEqual(result.segments[0].length, 10.0, places=4)
        self.assertLess(result.segments[1].direction[0], -0.999)

    def test_band6_ellipse_matches_original_notebook(self):
        system = default_band6_system()
        mirror = next(item for item in system.elements if item.kind == "elliptical_mirror")
        _, semi_major, semi_minor, _ = ellipse_geometry(mirror)
        self.assertAlmostEqual(semi_major, 135.379, places=3)
        self.assertAlmostEqual(semi_minor, 97.1447, places=3)
        self.assertAlmostEqual(ellipse_effective_focal_length(mirror), 41.3935, places=3)

    def test_ellipse_four_vertices_in_global_coordinates(self):
        mirror = OpticalElement(
            "elliptical_mirror", "E", 0.0, 4.0,
            focus1_x=-3.0, focus1_y=0.0,
            focus2_x=3.0, focus2_y=0.0,
        )
        vertices = ellipse_vertices(mirror)
        self.assertIsNotNone(vertices)
        expected = ([5.0, 0.0], [-5.0, 0.0], [0.0, 4.0], [0.0, -4.0])
        for actual, target in zip(vertices, expected):
            self.assertTrue(np.allclose(actual, target))

    def test_moving_focus_recomputes_ellipse(self):
        system = default_band6_system()
        mirror = next(item for item in system.elements if item.kind == "elliptical_mirror")
        old_a = ellipse_geometry(mirror)[1]
        mirror.focus2_x += 20.0
        new_a = ellipse_geometry(mirror)[1]
        self.assertNotAlmostEqual(old_a, new_a)

    def test_gaussian_boundaries_are_curved_not_straight_rays(self):
        result = trace_system(default_band6_system())
        segment = result.segments[-1]
        w0 = float(segment.radius(0.0))
        wm = float(segment.radius(segment.length / 2.0))
        w1 = float(segment.radius(segment.length))
        self.assertNotAlmostEqual(wm, (w0 + w1) / 2.0, places=5)

    def test_horn_link_projects_motion_onto_m1_f1_line(self):
        system = default_band6_system()
        horn = system.horn
        ellipse = next(item for item in system.elements if item.kind == "elliptical_mirror")
        move_with_constraints(system, horn, 80.0, 40.0)
        line = np.array([ellipse.focus1_x - ellipse.x, ellipse.focus1_y - ellipse.y])
        horn_vector = np.array([horn.x - ellipse.x, horn.y - ellipse.y])
        cross = line[0] * horn_vector[1] - line[1] * horn_vector[0]
        self.assertAlmostEqual(float(cross), 0.0, places=8)
        self.assertAlmostEqual(horn.x, ellipse.focus1_x)
        self.assertAlmostEqual(horn.y, ellipse.focus1_y)

    def test_plane_link_projects_motion_onto_m1_f2_line(self):
        system = default_band6_system()
        plane = next(item for item in system.elements if item.kind == "plane_mirror")
        ellipse = next(item for item in system.elements if item.kind == "elliptical_mirror")
        move_with_constraints(system, plane, 10.0, 80.0)
        line = np.array([ellipse.focus2_x - ellipse.x, ellipse.focus2_y - ellipse.y])
        plane_vector = np.array([plane.x - ellipse.x, plane.y - ellipse.y])
        cross = line[0] * plane_vector[1] - line[1] * plane_vector[0]
        self.assertAlmostEqual(float(cross), 0.0, places=8)

    def test_motion_constraint_can_be_disabled(self):
        system = default_band6_system()
        horn = system.horn
        system.link_horn_to_ellipse = False
        move_with_constraints(system, horn, 81.0, 47.0)
        self.assertAlmostEqual(horn.x, 81.0)
        self.assertAlmostEqual(horn.y, 47.0)

    def test_multiple_horns_and_frequencies_are_independent(self):
        system = default_band6_system()
        first_horn = system.horn
        first_horn.frequency_channels = [FrequencyChannel(211.0, True), FrequencyChannel(243.0, True)]
        second_horn = OpticalElement(
            "horn", "H2", -40.0, 70.0, angle_deg=-20.0,
            frequency_channels=[FrequencyChannel(275.0, True), FrequencyChannel(300.0, False)],
        )
        system.elements.append(second_horn)
        traces = trace_all_beams(system)
        self.assertEqual(len(traces), 3)
        self.assertEqual(sum(trace.horn_uid == first_horn.uid for trace in traces), 2)
        self.assertEqual(sum(trace.horn_uid == second_horn.uid for trace in traces), 1)

    def test_link_can_target_any_ellipse(self):
        system = default_band6_system()
        horn = system.horn
        second = OpticalElement(
            "elliptical_mirror", "E2", 200.0, 50.0,
            focus1_x=horn.x, focus1_y=horn.y, focus2_x=300.0, focus2_y=50.0,
        )
        system.elements.append(second)
        system.connect(horn.uid, second.uid)
        self.assertIn(second.uid, {item.uid for item in system.outgoing(horn.uid)})

    def test_multiple_and_parallel_links_are_preserved(self):
        first = OpticalElement("horn", "H", 0.0, 0.0)
        second = OpticalElement("plane_mirror", "M", 10.0, 0.0)
        third = OpticalElement("lens", "L", 20.0, 0.0)
        system = OpticalSystem([first, second, third])
        system.connect(first.uid, second.uid)
        system.connect(first.uid, second.uid)
        system.connect(first.uid, third.uid)
        self.assertEqual(len(system.links), 3)
        self.assertEqual(len(system.outgoing(first.uid)), 3)

    def test_system_snapshot_round_trip_preserves_parallel_links(self):
        system = default_band6_system()
        next(item for item in system.elements if item.kind == "cryostat").window_count = 4
        source, target = system.elements[1], system.elements[2]
        system.connect(source.uid, target.uid)
        restored = OpticalSystem.from_dict(system.to_dict())
        self.assertEqual(len(restored.elements), len(system.elements))
        self.assertEqual(len(restored.links), len(system.links))
        pairs = [(link.source_uid, link.target_uid) for link in restored.links]
        self.assertGreaterEqual(pairs.count((source.uid, target.uid)), 2)
        self.assertEqual(next(item for item in restored.elements if item.kind == "cryostat").window_count, 4)

    def test_four_cryostat_windows_are_spaced_by_90_degrees(self):
        cryostat = OpticalElement("cryostat", "C", 5.0, -3.0, angle_deg=17.0, radius=100.0, window_count=4)
        centers = [np.mean(cryostat_window_corners(cryostat, index), axis=0) for index in range(4)]
        vectors = [center - np.array([cryostat.x, cryostat.y]) for center in centers]
        for vector in vectors:
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 120.0, places=8)
        for index in range(4):
            self.assertAlmostEqual(float(np.dot(vectors[index], vectors[(index + 1) % 4])), 0.0, places=8)

    def test_ruler_follows_two_unique_linked_objects(self):
        first = OpticalElement("lens", "A", 10.0, 20.0)
        second = OpticalElement("plane_mirror", "B", 80.0, -15.0)
        ruler = OpticalElement("ruler", "R", 0.0, 0.0, target_x=1.0, target_y=1.0)
        system = OpticalSystem([first, second, ruler])
        system.connect(first.uid, ruler.uid)
        system.connect(first.uid, ruler.uid)  # parallel duplicate is not a second attachment
        system.connect(ruler.uid, second.uid)
        enforce_links(system)
        self.assertEqual(len(ruler_attachments(system, ruler)), 2)
        self.assertEqual((ruler.x, ruler.y), (first.x, first.y))
        self.assertEqual((ruler.target_x, ruler.target_y), (second.x, second.y))
        self.assertAlmostEqual(math.hypot(ruler.target_x - ruler.x, ruler.target_y - ruler.y), math.hypot(70.0, -35.0))

    def test_ruler_does_not_intercept_beam(self):
        horn = OpticalElement("horn", "H", 0.0, 0.0, angle_deg=0.0)
        ruler = OpticalElement("ruler", "R", 20.0, -10.0, target_x=20.0, target_y=10.0)
        mirror = OpticalElement("plane_mirror", "M", 50.0, 0.0, angle_deg=90.0)
        result = trace_system(OpticalSystem([horn, ruler, mirror], max_path_mm=100.0))
        self.assertEqual(result.segments[0].hit_uid, mirror.uid)

    def test_relative_target_follows_predecessor_and_updates_mirror(self):
        mirror = OpticalElement("plane_mirror", "M", 10.0, 20.0)
        target = OpticalElement(
            "target", "T", 0.0, 0.0,
            position_mode="relative", relative_distance=100.0, relative_angle_deg=30.0,
        )
        system = OpticalSystem([mirror, target])
        system.connect(mirror.uid, target.uid)
        enforce_links(system)
        self.assertAlmostEqual(target.x, 10.0 + 100.0 * math.cos(math.radians(30.0)))
        self.assertAlmostEqual(target.y, 20.0 + 100.0 * math.sin(math.radians(30.0)))
        self.assertAlmostEqual(mirror.target_x, target.x)
        self.assertAlmostEqual(mirror.target_y, target.y)

    def test_coordinate_target_calculates_polar_values(self):
        mirror = OpticalElement("plane_mirror", "M", 10.0, 10.0)
        target = OpticalElement("target", "T", 40.0, 50.0, position_mode="coordinates")
        system = OpticalSystem([mirror, target])
        system.connect(mirror.uid, target.uid)
        enforce_links(system)
        self.assertAlmostEqual(target.relative_distance, 50.0)
        self.assertAlmostEqual(target.relative_angle_deg, math.degrees(math.atan2(40.0, 30.0)))

    def test_target_does_not_intercept_beam(self):
        horn = OpticalElement("horn", "H", 0.0, 0.0, angle_deg=0.0)
        target = OpticalElement("target", "T", 20.0, 0.0)
        mirror = OpticalElement("plane_mirror", "M", 50.0, 0.0, angle_deg=90.0)
        result = trace_system(OpticalSystem([horn, target, mirror], max_path_mm=100.0))
        self.assertEqual(result.segments[0].hit_uid, mirror.uid)

    def test_block_attaches_two_horns_at_waveguides(self):
        first = OpticalElement(
            "horn", "H1", 10.0, 20.0, angle_deg=30.0,
            horn_length=30.0, pcl_depth=5.0,
        )
        block = OpticalElement("block", "B", 0.0, 0.0, block_length=60.0)
        second = OpticalElement(
            "horn", "H2", 0.0, 0.0,
            horn_length=20.0, pcl_depth=2.0,
        )
        system = OpticalSystem([first, block, second])
        system.connect(first.uid, block.uid)
        system.connect(block.uid, second.uid)
        enforce_links(system)
        endpoint_a, endpoint_b = block_endpoints(block)
        self.assertTrue(np.allclose(endpoint_a, [first.waveguide_x, first.waveguide_y]))
        self.assertAlmostEqual(block.angle_deg, first.angle_deg)
        self.assertAlmostEqual(second.angle_deg, first.angle_deg + 180.0)
        self.assertTrue(np.allclose(endpoint_b, [second.waveguide_x, second.waveguide_y]))

    def test_block_rejects_non_horn_and_extra_end_connection(self):
        horn = OpticalElement("horn", "H1", 0.0, 0.0)
        extra = OpticalElement("horn", "H2", 0.0, 0.0)
        block = OpticalElement("block", "B", 0.0, 0.0)
        mirror = OpticalElement("plane_mirror", "M", 0.0, 0.0)
        system = OpticalSystem([horn, extra, block, mirror])
        with self.assertRaises(ValueError):
            system.connect(block.uid, mirror.uid)
        system.connect(horn.uid, block.uid)
        with self.assertRaises(ValueError):
            system.connect(extra.uid, block.uid)

    def test_block_does_not_intercept_beam(self):
        horn = OpticalElement("horn", "H", 0.0, 0.0, angle_deg=0.0)
        block = OpticalElement("block", "B", 20.0, 0.0, angle_deg=90.0, block_length=40.0)
        mirror = OpticalElement("plane_mirror", "M", 50.0, 0.0, angle_deg=90.0)
        result = trace_system(OpticalSystem([horn, block, mirror], max_path_mm=100.0))
        self.assertEqual(result.segments[0].hit_uid, mirror.uid)

    def test_selected_horn_assembly_attaches_as_rigid_body(self):
        block = OpticalElement("block", "B", 100.0, 100.0, angle_deg=90.0, block_length=40.0)
        horn = OpticalElement("horn", "H", 10.0, 0.0, angle_deg=15.0)
        mirror = OpticalElement(
            "elliptical_mirror", "M", 55.0, 25.0,
            focus1_x=10.0, focus1_y=0.0,
            focus2_x=90.0, focus2_y=40.0,
            target_x=90.0, target_y=40.0,
        )
        system = OpticalSystem([block, horn, mirror])
        distance_before = math.hypot(mirror.x - horn.x, mirror.y - horn.y)
        focus_distance_before = math.hypot(mirror.focus2_x - horn.x, mirror.focus2_y - horn.y)
        attach_assembly_to_block(system, block, horn, {horn.uid, mirror.uid})
        _first, endpoint_b = block_endpoints(block)
        self.assertTrue(np.allclose([horn.waveguide_x, horn.waveguide_y], endpoint_b))
        self.assertAlmostEqual(horn.angle_deg, 270.0)
        self.assertAlmostEqual(mirror.angle_deg, 255.0)
        self.assertAlmostEqual(math.hypot(mirror.x - horn.x, mirror.y - horn.y), distance_before)
        self.assertAlmostEqual(
            math.hypot(mirror.focus2_x - horn.x, mirror.focus2_y - horn.y),
            focus_distance_before,
        )


if __name__ == "__main__":
    unittest.main()
