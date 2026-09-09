#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from graderUtil import graded, CourseTestRunner, GradedTestCase

import json
from typing import List, Optional

import util

from mapUtil import (
    CityMap,
    checkValid,
    createGridMap,
    createGridMapWithCustomTags,
    createStanfordMap,
    getTotalCost,
    locationFromTag,
    makeGridLabel,
    makeTag,
)

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

def extractPath(startLocation: str, search: util.SearchAlgorithm) -> List[str]:
    """
    Assumes that `solve()` has already been called on the `searchAlgorithm`.

    We extract a sequence of locations from `search.path` (see util.py to better
    understand exactly how this list gets populated).
    """
    return [startLocation] + search.actions


def printPath(
    path: List[str],
    waypointTags: List[str],
    cityMap: CityMap,
    outPath: Optional[str] = "path.json",
):
    doneWaypointTags = set()
    for location in path:
        for tag in cityMap.tags[location]:
            if tag in waypointTags:
                doneWaypointTags.add(tag)
        tagsStr = " ".join(cityMap.tags[location])
        doneTagsStr = " ".join(sorted(doneWaypointTags))
        print(f"Location {location} tags:[{tagsStr}]; done:[{doneTagsStr}]")
    print(f"Total distance: {getTotalCost(path, cityMap)}")

    # (Optional) Write path to file, for use with `visualize.py`
    if outPath is not None:
        with open(outPath, "w") as f:
            data = {"waypointTags": waypointTags, "path": path}
            json.dump(data, f, indent=2)


# Instantiate the Stanford Map as a constant --> just load once!
stanfordMap = createStanfordMap()

# To test your reduction, we'll define an admissible (but fairly unhelpful) heuristic
class ZeroHeuristic(util.Heuristic):
    """Estimates the cost between locations as 0 distance."""
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

    def evaluate(self, state: util.State) -> float:
        return 0.0


#########
# TESTS #
#########


class Test_2a(GradedTestCase):

    def t_2a(
        self,
        cityMap: CityMap,
        startLocation: str,
        endTag: str,
        expectedCost: Optional[float] = None,
    ):
        """
        Run UCS on a ShortestPathProblem, specified by
            (startLocation, endTag).
        Check that the cost of the minimum cost path is `expectedCost`.
        """
        ucs = util.UniformCostSearch(verbose=0)
        ucs.solve(submission.ShortestPathProblem(startLocation, endTag, cityMap))
        path = extractPath(startLocation, ucs)
        self.assertTrue(checkValid(path, cityMap, startLocation, endTag, []))
        if expectedCost is not None:
            self.assertAlmostEqual(expectedCost, getTotalCost(path, cityMap), places=4)

        # BEGIN_HIDE
        # END_HIDE

    @graded(timeout=1)
    def test_0(self):
        """2a-0-basic: shortest path on small grid."""

        self.t_2a(
            cityMap=createGridMap(3, 5),
            startLocation=makeGridLabel(0, 0),
            endTag=makeTag("label", makeGridLabel(2, 2)),
            expectedCost=4,
        )

    @graded(timeout=1)
    def test_1(self):
        """2a-1-basic: shortest path with multiple end locations"""

        self.t_2a(
            cityMap=createGridMap(30, 30),
            startLocation=makeGridLabel(20, 10),
            endTag=makeTag("x", "5"),
            expectedCost=15,
        )

    @graded(timeout=1, is_hidden=True)
    def test_2(self):
        """2a-2-hidden: shortest path with larger grid"""
        self.t_2a(
            cityMap=createGridMap(100, 100),
            startLocation=makeGridLabel(0, 0),
            endTag=makeTag("label", makeGridLabel(99, 99)),
        )

    @graded(timeout=1)
    def test_3(self):
        """2a-3-basic: basic shortest path test case (2a-3)"""

        self.t_2a(
            cityMap=stanfordMap,
            startLocation=locationFromTag(makeTag("landmark", "gates"), stanfordMap),
            endTag=makeTag("landmark", "oval"),
            expectedCost=446.99724421432353,
        )

    @graded(timeout=1)
    def test_4(self):
        """2a-4-basic: basic shortest path test case (2a-4)"""

        self.t_2a(
            cityMap=stanfordMap,
            startLocation=locationFromTag(makeTag("landmark", "rains"), stanfordMap),
            endTag=makeTag("landmark", "evgr_a"),
            expectedCost=660.9598696201658,
        )

    @graded(timeout=1)
    def test_5(self):
        """2a-5-basic: basic shortest path test case (2a-5)"""

        self.t_2a(
            cityMap=stanfordMap,
            startLocation=locationFromTag(makeTag("landmark", "rains"), stanfordMap),
            endTag=makeTag("landmark", "bookstore"),
            expectedCost=1109.3271626156256,
        )

    @graded(timeout=1, is_hidden=True)
    def test_6(self):
        """2a-6-hidden: hidden shortest path test case (2a-6)"""

        self.t_2a(
            cityMap=stanfordMap,
            startLocation=locationFromTag(makeTag("landmark", "hoover_tower"), stanfordMap),
            endTag=makeTag("landmark", "cantor_arts_center"),
        )

    @graded(timeout=1, is_hidden=True)
    def test_7(self):
        """2a-7-hidden: hidden shortest path test case (2a-7)"""

        self.t_2a(
            cityMap=stanfordMap,
            startLocation=locationFromTag(makeTag("landmark", "hoover_tower"), stanfordMap),
            endTag=makeTag("landmark", "cantor_arts_center"),
        )

class Test_2b(GradedTestCase):

    @graded(timeout=10)
    def test_0(self):
        """2b-0-helper: Helper function to get customized shortest path through Stanford for question 1b."""

        """Given custom ShortestPathProblem, output path for visualization."""
        problem = submission.getStanfordShortestPathProblem()
        ucs = util.UniformCostSearch(verbose=0)
        ucs.solve(problem)
        path = extractPath(problem.startLocation, ucs)
        printPath(path=path, waypointTags=[], cityMap=stanfordMap)

        self.assertTrue(checkValid(path, stanfordMap, problem.startLocation, problem.endTag, []))

        self.skipTest("This test case is a helper function for students.")

class Test_3a(GradedTestCase):

    def t_3ab(
        self,
        cityMap: CityMap,
        startLocation: str,
        endTag: str,
        waypointTags: List[str],
        expectedCost: Optional[float] = None,
    ):
        """
        Run UCS on a WaypointsShortestPathProblem, specified by
            (startLocation, waypointTags, endTag).
        """
        ucs = util.UniformCostSearch(verbose=0)
        problem = submission.WaypointsShortestPathProblem(
            startLocation,
            waypointTags,
            endTag,
            cityMap,
        )
        ucs.solve(problem)
        self.assertTrue(ucs.pathCost is not None)
        path = extractPath(startLocation, ucs)
        self.assertTrue(checkValid(path, cityMap, startLocation, endTag, waypointTags))
        if expectedCost is not None:
            self.assertAlmostEqual(expectedCost, getTotalCost(path, cityMap), places=4)

        # BEGIN_HIDE
        # END_HIDE

    @graded(timeout=3)
    def test_0(self):
        """3a-0-basic: shortest path on small grid with 1 waypoint."""

        self.t_3ab(
            cityMap=createGridMap(3, 5),
            startLocation=makeGridLabel(0, 0),
            waypointTags=[makeTag("y", 4)],
            endTag=makeTag("label", makeGridLabel(2, 2)),
            expectedCost=8,
        )

    @graded(timeout=3)
    def test_1(self):
        """3a-1-basic: shortest path on medium grid with 2 waypoints."""

        self.t_3ab(
            cityMap=createGridMap(30, 30),
            startLocation=makeGridLabel(20, 10),
            waypointTags=[makeTag("x", 5), makeTag("x", 7)],
            endTag=makeTag("label", makeGridLabel(3, 3)),
            expectedCost=24.0,
        )

    @graded(timeout=3)
    def test_2(self):
        """3a-2-basic: shortest path with 3 waypoints and some locations covering multiple waypoints."""

        self.t_3ab(
            cityMap=createGridMapWithCustomTags(2, 2, {(0,0): [], (0,1): ["food", "fuel", "books"], (1,0): ["food"], (1,1): ["fuel"]}),
            startLocation=makeGridLabel(0, 0),
            waypointTags=[
                "food", "fuel", "books"
            ],
            endTag=makeTag("label", makeGridLabel(0, 1)),
            expectedCost=1.0,
        )

    @graded(timeout=3)
    def test_3(self):
        """3a-3-basic: shortest path with 3 waypoints and start location covering some waypoints."""

        self.t_3ab(
            cityMap=createGridMapWithCustomTags(2, 2, {(0,0): ["food"], (0,1): ["fuel"], (1,0): ["food"], (1,1): ["food", "fuel"]}),
            startLocation=makeGridLabel(0, 0),
            waypointTags=[
                "food", "fuel"
            ],
            endTag=makeTag("label", makeGridLabel(0, 1)),
            expectedCost=1.0,
        )

    @graded(timeout=3, is_hidden=True)
    def test_4(self):
        """3a-4-hidden: shortest path with 3 waypoints and start location covering some waypoints."""

        self.t_3ab(
            cityMap=stanfordMap,
            startLocation=locationFromTag(makeTag("landmark", "gates"), stanfordMap),
            waypointTags=[makeTag("landmark", "hoover_tower")],
            endTag=makeTag("landmark", "oval"),
            expectedCost=1108.3623108845995,
        )

    @graded(timeout=3)
    def test_5(self):
        """3a-5-basic: basic waypoints test case (3a-5)."""

        self.t_3ab(
            cityMap=createGridMapWithCustomTags(2, 2, {(0,0): ["food"], (0,1): ["fuel"], (1,0): ["food"], (1,1): ["food", "fuel"]}),
            startLocation=makeGridLabel(0, 0),
            waypointTags=[
                "food", "fuel"
            ],
            endTag=makeTag("label", makeGridLabel(0, 1)),
            expectedCost=1.0,
        )


    @graded(timeout=3)
    def test_6(self):
        """3a-6-basic: basic waypoints test case (3a-6)."""

        self.t_3ab(
            cityMap=stanfordMap,
            startLocation=locationFromTag(makeTag("landmark", "evgr_a"), stanfordMap),
            waypointTags=[
                makeTag("landmark", "memorial_church"),
                makeTag("landmark", "tressider"),
                makeTag("landmark", "gates"),
            ],
            endTag=makeTag("landmark", "evgr_a"),
            expectedCost=3381.952714299139,
        )

    @graded(timeout=3)
    def test_7(self):
        """3a-7-basic: basic waypoints test case (3a-7)."""

        self.t_3ab(
            cityMap=stanfordMap,
            startLocation=locationFromTag(makeTag("landmark", "rains"), stanfordMap),
            waypointTags=[
                makeTag("landmark", "gates"),
                makeTag("landmark", "AOERC"),
                makeTag("landmark", "bookstore"),
                makeTag("landmark", "hoover_tower"),
            ],
            endTag=makeTag("landmark", "green_library"),
            expectedCost=3946.478546309725,
        )


    @graded(timeout=3, is_hidden=True)
    def test_8(self):
        """3a-8-hidden: hidden waypoints test case (3a-8)."""

        self.t_3ab(
            cityMap=stanfordMap,
            startLocation=locationFromTag(makeTag("landmark", "oval"), stanfordMap),
            waypointTags=[
                makeTag("landmark", "memorial_church"),
                makeTag("landmark", "hoover_tower"),
                makeTag("landmark", "bookstore"),
            ],
            endTag=makeTag("landmark", "AOERC"),
        )

    @graded(timeout=3, is_hidden=True)
    def test_9(self):
        """3a-9-hidden: hidden waypoints test case (3a-9)."""

        self.t_3ab(
            cityMap=stanfordMap,
            startLocation=locationFromTag(makeTag("landmark", "oval"), stanfordMap),
            waypointTags=[
                makeTag("landmark", "memorial_church"),
                makeTag("landmark", "stanford_stadium"),
                makeTag("landmark", "rains"),
            ],
            endTag=makeTag("landmark", "oval"),
        )

    @graded(timeout=5, is_hidden=True)
    def test_10(self):
        """3a-10-hidden: hidden waypoints test case (3a-10)."""

        self.t_3ab(
            cityMap=stanfordMap,
            startLocation=locationFromTag(makeTag("landmark", "gates"), stanfordMap),
            waypointTags=[
                makeTag("landmark", "lathrop_library"),
                makeTag("landmark", "green_library"),
                makeTag("landmark", "tressider"),
                makeTag("landmark", "AOERC"),
            ],
            endTag=makeTag("landmark", "evgr_a"),
        )


class Test_3c(GradedTestCase):

    @graded(timeout=10)
    def test_0(self):
        """3c-0-helper: Helper function to get customized shortest path with waypoints through Stanford for question 2c."""

        """Given custom WaypointsShortestPathProblem, output path for visualization."""
        problem = submission.getStanfordWaypointsShortestPathProblem()
        ucs = util.UniformCostSearch(verbose=0)
        ucs.solve(problem)
        path = extractPath(problem.startLocation, ucs)
        printPath(path=path, waypointTags=problem.waypointTags, cityMap=stanfordMap)
        self.assertTrue(
            checkValid(
                path,
                stanfordMap,
                problem.startLocation,
                problem.endTag,
                problem.waypointTags,
            )
        )

        self.skipTest("This test case is a helper function for students.")

class Test_4a(GradedTestCase):

    def t_4a(
        self,
        cityMap: CityMap,
        startLocation: str,
        endTag: str,
        expectedCost: Optional[float] = None,
    ):
        """
        Run UCS on the A* Reduction of a ShortestPathProblem, specified by
            (startLocation, endTag).
        """
        # We'll use the ZeroHeuristic to verify that the reduction works as expected
        zeroHeuristic = ZeroHeuristic(endTag, cityMap)

        # Define the baseProblem and corresponding reduction (using `zeroHeuristic`)
        baseProblem = submission.ShortestPathProblem(startLocation, endTag, cityMap)
        aStarProblem = submission.aStarReduction(baseProblem, zeroHeuristic)

        # Solve the reduction via a call to `ucs.solve` (similar to prior tests)
        ucs = util.UniformCostSearch(verbose=0)
        ucs.solve(aStarProblem)
        path = extractPath(startLocation, ucs)
        self.assertTrue(checkValid(path, cityMap, startLocation, endTag, []))
        if expectedCost is not None: 
            self.assertAlmostEqual(expectedCost, getTotalCost(path, cityMap), places=4)

        # BEGIN_HIDE
        # END_HIDE

    @graded(timeout=1)
    def test_0(self):
        """4a-0-basic: A* shortest path on small grid."""

        self.t_4a(
            cityMap=createGridMap(3, 5),
            startLocation=makeGridLabel(0, 0),
            endTag=makeTag("label", makeGridLabel(2, 2)),
            expectedCost=4,
        )

    @graded(timeout=1)
    def test_1(self):
        """4a-1-basic: A* shortest path with multiple end locations."""

        self.t_4a(
            cityMap=createGridMap(30, 30),
            startLocation=makeGridLabel(20, 10),
            endTag=makeTag("x", "5"),
            expectedCost=15,
        )

    @graded(timeout=2, is_hidden=True)
    def test_2(self):
        """4a-2-hidden: A* shortest path with larger grid."""

        self.t_4a(
            cityMap=createGridMap(100, 100),
            startLocation=makeGridLabel(0, 0),
            endTag=makeTag("label", makeGridLabel(99, 99)),
        )

class Test_4b(GradedTestCase):

    def setUp(self):

        # Initialize a `StraightLineHeuristic` using `endTag3b` and the `stanfordMap`
        self.endTag3b = makeTag("landmark", "gates")

        try:
            self.stanfordStraightLineHeuristic = submission.StraightLineHeuristic(
                self.endTag3b, stanfordMap
            )
        except:
            self.stanfordStraightLineHeuristic = None

    def t_4b_heuristic(
        self,
        cityMap: CityMap,
        startLocation: str,
        endTag: str,
        expectedCost: Optional[float] = None,
    ):
        """Targeted test for `StraightLineHeuristic` to ensure correctness."""
        heuristic = submission.StraightLineHeuristic(endTag, cityMap)
        heuristicCost = heuristic.evaluate(util.State(startLocation))
        if expectedCost is not None:
            self.assertAlmostEqual(expectedCost, heuristicCost, places=4)

        # BEGIN_HIDE
        # END_HIDE

    @graded(timeout=1)
    def test_0(self):
        """4b-0-basic: basic straight line heuristic unit test."""

        self.t_4b_heuristic(
            cityMap=createGridMap(3, 5),
            startLocation=makeGridLabel(0, 0),
            endTag=makeTag("label", makeGridLabel(2, 2)),
            expectedCost=3.145067466556296,
        )

    @graded(timeout=1, is_hidden=True)
    def test_1(self):
        """4b-1-hidden: hidden straight line heuristic unit test."""

        self.t_4b_heuristic(
            cityMap=createGridMap(100, 100),
            startLocation=makeGridLabel(0, 0),
            endTag=makeTag("label", makeGridLabel(99, 99)),
        )

    def t_4b_aStar(
        self,
        startLocation: str, 
        heuristic: util.Heuristic, 
        expectedCost: Optional[float] = None
    ):
        """Run UCS on the A* Reduction of a ShortestPathProblem, w/ `heuristic`"""
        baseProblem = submission.ShortestPathProblem(startLocation, self.endTag3b, stanfordMap)
        aStarProblem = submission.aStarReduction(baseProblem, heuristic)

        # Solve the reduction via a call to `ucs.solve` (similar to prior tests)
        ucs = util.UniformCostSearch(verbose=0)
        ucs.solve(aStarProblem)
        path = extractPath(startLocation, ucs)
        self.assertTrue(checkValid(path, stanfordMap, startLocation, self.endTag3b, []))
        if expectedCost is not None:
            self.assertAlmostEqual(expectedCost, getTotalCost(path, stanfordMap), places=4)

        # BEGIN_HIDE
        # END_HIDE

    @graded(timeout=2)
    def test_2(self):
        """4b-2-basic: basic straight line heuristic A* on Stanford map (4b-astar-1)."""

        self.t_4b_aStar(
            startLocation=locationFromTag(makeTag("landmark", "oval"), stanfordMap),
            heuristic=self.stanfordStraightLineHeuristic,
            expectedCost=446.9972442143235,
        )

    @graded(timeout=2)
    def test_3(self):
        """4b-3-basic: basic straight line heuristic A* on Stanford map (4b-astar-2)."""

        self.t_4b_aStar(
            startLocation=locationFromTag(makeTag("landmark", "rains"), stanfordMap),
            heuristic=self.stanfordStraightLineHeuristic,
            expectedCost=2005.4388573303765,
        )

    @graded(timeout=2, is_hidden=True)
    def test_4(self):
        """4b-4-hidden: hidden straight line heuristic A* on Stanford map (4b-astar-3)."""

        self.t_4b_aStar(
            startLocation=locationFromTag(makeTag("landmark", "bookstore"), stanfordMap),
            heuristic=self.stanfordStraightLineHeuristic,
        )

    @graded(timeout=2, is_hidden=True)
    def test_5(self):
        """4b-5-hidden: hidden straight line heuristic A* on Stanford map (4b-astar-4)."""

        self.t_4b_aStar(
            startLocation=locationFromTag(makeTag("landmark", "evgr_a"), stanfordMap),
            heuristic=self.stanfordStraightLineHeuristic,
        )

class Test_4c(GradedTestCase):

    def setUp(self):

        self.endTag3c = makeTag("wheelchair", "yes")

        try:
            self.stanfordNoWaypointsHeuristic = submission.NoWaypointsHeuristic(
                self.endTag3c, stanfordMap
            )
        except:
            self.stanfordNoWaypointsHeuristic = None

    def t_4c_heuristic(
        self,
        startLocation: str, 
        endTag: str, 
        expectedCost: Optional[float] = None
    ):
        """Targeted test for `NoWaypointsHeuristic` -- uses the full Stanford map."""
        heuristic = submission.NoWaypointsHeuristic(endTag, stanfordMap)
        heuristicCost = heuristic.evaluate(util.State(startLocation))
        if expectedCost is not None:
            self.assertAlmostEqual(expectedCost, heuristicCost, places=4)

        # BEGIN_HIDE
        # END_HIDE

    @graded(timeout=2)
    def test_0(self):
        """4c-0-basic: basic no waypoints heuristic unit test."""

        self.t_4c_heuristic(
            startLocation=locationFromTag(makeTag("landmark", "oval"), stanfordMap),
            endTag=makeTag("landmark", "gates"),
            expectedCost=446.99724421432353,
        )

    @graded(timeout=2, is_hidden=True)
    def test_1(self):
        """4c-1-hidden: hidden no waypoints heuristic unit test w/ multiple end locations."""

        self.t_4c_heuristic(
            startLocation=locationFromTag(makeTag("landmark", "bookstore"), stanfordMap),
            endTag=makeTag("amenity", "food"),
        )

    def t_4c_aStar(
        self,
        startLocation: str,
        waypointTags: List[str],
        heuristic: util.Heuristic,
        expectedCost: Optional[float] = None,
    ):
        """Run UCS on the A* Reduction of a WaypointsShortestPathProblem, w/ `heuristic`"""
        baseProblem = submission.WaypointsShortestPathProblem(
            startLocation, waypointTags, self.endTag3c, stanfordMap
        )
        aStarProblem = submission.aStarReduction(baseProblem, heuristic)

        # Solve the reduction via a call to `ucs.solve` (similar to prior tests)
        ucs = util.UniformCostSearch(verbose=0)
        ucs.solve(aStarProblem)
        path = extractPath(startLocation, ucs)
        self.assertTrue(
            checkValid(path, stanfordMap, startLocation, self.endTag3c, waypointTags)
        )
        if expectedCost is not None:
            self.assertAlmostEqual(expectedCost, getTotalCost(path, stanfordMap), places=4)

        # BEGIN_HIDE
        # END_HIDE

    @graded(timeout=2)
    def test_2(self):
        """4c-2-basic: basic no waypoints heuristic A* on Stanford map (4c-astar-1)."""

        self.t_4c_aStar(
            startLocation=locationFromTag(makeTag("landmark", "oval"), stanfordMap),
            waypointTags=[
                makeTag("landmark", "gates"),
                makeTag("landmark", "AOERC"),
                makeTag("landmark", "bookstore"),
                makeTag("landmark", "hoover_tower"),
            ],
            heuristic=self.stanfordNoWaypointsHeuristic,
            expectedCost=2943.242598551967,
        )

    @graded(timeout=2)
    def test_3(self):
        """4c-3-basic: basic no waypoints heuristic A* on Stanford map (4c-astar-1)."""

        self.t_4c_aStar(
            startLocation=locationFromTag(makeTag("landmark", "AOERC"), stanfordMap),
            waypointTags=[
                makeTag("landmark", "tressider"),
                makeTag("landmark", "hoover_tower"),
                makeTag("amenity", "food"),
            ],
            heuristic=self.stanfordNoWaypointsHeuristic,
            expectedCost=1677.3814380413373,
        )

    @graded(timeout=10, is_hidden=True)
    def test_4(self):
        """4c-4-hidden: hidden no waypoints heuristic A* on Stanford map (4c-astar-3)."""

        self.t_4c_aStar(
            startLocation=locationFromTag(makeTag("landmark", "tressider"), stanfordMap),
            waypointTags=[
                makeTag("landmark", "gates"),
                makeTag("amenity", "food"),
                makeTag("landmark", "rains"),
                makeTag("landmark", "stanford_stadium"),
                makeTag("bicycle", "yes"),
            ],
            heuristic=self.stanfordNoWaypointsHeuristic,
        )

def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)


if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
