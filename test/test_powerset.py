import unittest

from muller.monad.powerset import (
    Powerset,
    choice,
    empty,
    from_list,
    singleton,
)


class TestPowersetMonad(unittest.TestCase):
    """Test cases for the Powerset monad implementation."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def assertPowersetEqual(self, ps1: Powerset, ps2: Powerset, msg=None):
        """Helper method for comparing powersets."""
        self.assertEqual(ps1._inner_value, ps2._inner_value, msg=msg)

    def test_initialization(self):
        """Test Powerset monad initialization."""
        # Test with list
        ps = Powerset([1, 2, 3])
        self.assertEqual(ps._inner_value, frozenset({1, 2, 3}))

        # Test with set
        ps_set = Powerset({4, 5, 6})
        self.assertEqual(ps_set._inner_value, frozenset({4, 5, 6}))

        # Test with duplicates (should be removed)
        ps_dup = Powerset([1, 1, 2, 2, 3])
        self.assertEqual(ps_dup._inner_value, frozenset({1, 2, 3}))

    def test_insert_operation(self):
        """Test the insert (unit/return) operation."""
        value = "single"
        ps = Powerset.from_value(value)

        # Should contain only the single value
        self.assertEqual(ps._inner_value, frozenset({"single"}))
        self.assertEqual(len(ps._inner_value), 1)

    def test_map_operation(self):
        """Test the map (functor) operation."""
        # Create powerset with numbers
        ps = Powerset([1, 2, 3])

        # Map with a function that doubles values
        mapped = ps.map(lambda x: x * 2)

        # Should contain doubled values
        expected = Powerset([2, 4, 6])
        self.assertPowersetEqual(mapped, expected)

    def test_map_with_duplicates(self):
        """Test map operation when function produces duplicate values."""
        # Create powerset where map will create duplicates
        ps = Powerset([1, 2, 3, 4])

        # Map with a function that makes some values the same
        mapped = ps.map(lambda x: x // 2)  # 1->0, 2->1, 3->1, 4->2

        # Should automatically deduplicate
        expected = Powerset([0, 1, 2])
        self.assertPowersetEqual(mapped, expected)

    def test_bind_operation(self):
        """Test the bind (monadic composition) operation."""
        # Create initial powerset
        ps = Powerset([1, 2])

        # Bind with a function that creates new powersets
        def expand(x):
            return Powerset([x, x + 10])

        result = ps.bind(expand)

        # Should contain union of all results
        # 1 -> {1, 11}, 2 -> {2, 12}
        # Union: {1, 11, 2, 12}
        expected = Powerset([1, 11, 2, 12])
        self.assertPowersetEqual(result, expected)

    def test_bind_with_overlapping_results(self):
        """Test bind operation when results overlap."""
        ps = Powerset([1, 2, 3])

        # Bind with function that can produce overlapping results
        def neighbors(x):
            return Powerset([x - 1, x, x + 1])

        result = ps.bind(neighbors)

        # 1 -> {0, 1, 2}, 2 -> {1, 2, 3}, 3 -> {2, 3, 4}
        # Union: {0, 1, 2, 3, 4}
        expected = Powerset([0, 1, 2, 3, 4])
        self.assertPowersetEqual(result, expected)

    def test_monad_laws(self):
        """Test that the monad laws hold."""
        # Left identity: unit(a).bind(f) == f(a)
        a = 5

        def f(x):
            return Powerset([x, x + 1])

        left = Powerset.from_value(a).bind(f)
        right = f(a)
        self.assertPowersetEqual(left, right)

        # Right identity: m.bind(unit) == m
        m = Powerset([1, 2, 3])
        bound = m.bind(Powerset.from_value)
        self.assertPowersetEqual(m, bound)

        # Associativity: (m.bind(f)).bind(g) == m.bind(lambda x: f(x).bind(g))
        def g(x):
            return Powerset([x * 2, x * 3])

        left_assoc = m.bind(f).bind(g)
        right_assoc = m.bind(lambda x: f(x).bind(g))
        self.assertPowersetEqual(left_assoc, right_assoc)

    def test_functor_laws(self):
        """Test that the functor laws hold."""
        # Identity law: map(id) == id
        ps = Powerset([1, 2, 3])
        mapped_id = ps.map(lambda x: x)
        self.assertPowersetEqual(ps, mapped_id)

        # Composition law: map(f . g) == map(f) . map(g)
        def f(x):
            return x * 2

        def g(x):
            return x + 1

        # map(f . g)
        def composed_func(x):
            return f(g(x))

        left_side = ps.map(composed_func)

        # map(f) . map(g)
        right_side = ps.map(g).map(f)

        self.assertPowersetEqual(left_side, right_side)

    def test_empty_powerset(self):
        """Test empty powerset behavior."""
        empty_ps = empty()

        # Should be empty
        self.assertEqual(len(empty_ps._inner_value), 0)
        self.assertEqual(empty_ps._inner_value, frozenset())

        # Map on empty should return empty
        mapped = empty_ps.map(lambda x: x * 2)
        self.assertPowersetEqual(mapped, empty_ps)

        # Bind on empty should return empty
        bound = empty_ps.bind(lambda x: Powerset([x, x + 1]))
        self.assertPowersetEqual(bound, empty_ps)

    def test_singleton_function(self):
        """Test singleton convenience function."""
        ps = singleton("test")
        expected = Powerset(["test"])
        self.assertPowersetEqual(ps, expected)

    def test_from_list_function(self):
        """Test from_list convenience function."""
        ps = from_list([1, 2, 3, 2, 1])  # With duplicates
        expected = Powerset([1, 2, 3])
        self.assertPowersetEqual(ps, expected)

    def test_choice_function(self):
        """Test choice convenience function."""
        ps = choice("A", "B", "C")
        expected = Powerset(["A", "B", "C"])
        self.assertPowersetEqual(ps, expected)

        # Test with single choice
        ps_single = choice("only")
        expected_single = Powerset(["only"])
        self.assertPowersetEqual(ps_single, expected_single)

    def test_equality_and_hashing(self):
        """Test equality and hashing of powersets."""
        ps1 = Powerset([1, 2, 3])
        ps2 = Powerset([3, 2, 1])  # Same elements, different order
        ps3 = Powerset([1, 2, 4])  # Different elements

        # Should be equal (same elements)
        self.assertEqual(ps1, ps2)
        self.assertNotEqual(ps1, ps3)

        # Should have same hash if equal
        self.assertEqual(hash(ps1), hash(ps2))

    def test_string_representation(self):
        """Test string representation of powersets."""
        ps = Powerset([3, 1, 2])
        repr_str = repr(ps)

        # Should contain "Powerset" and be sorted
        self.assertTrue("Powerset" in repr_str)
        # Should be sorted in the representation
        self.assertTrue(repr_str.index("1") < repr_str.index("2"))
        self.assertTrue(repr_str.index("2") < repr_str.index("3"))

    def test_complex_composition(self):
        """Test complex monadic compositions."""
        # Start with initial choices
        initial = Powerset([1, 2])

        # Chain multiple operations
        result = (
            initial.map(lambda x: x + 1)  # {2, 3}
            .bind(lambda x: Powerset([x, x * 2]))  # {2, 4, 3, 6}
            .map(lambda x: x - 1)  # {1, 3, 2, 5}
            .bind(lambda x: Powerset([x]) if x % 2 == 1 else Powerset([]))
        )  # {1, 3, 5}

        expected = Powerset([1, 3, 5])
        self.assertPowersetEqual(result, expected)

    def test_non_deterministic_computation(self):
        """Test non-deterministic computation patterns."""
        # Simulate guessing a number
        guesses = Powerset([1, 2, 3, 4, 5])

        # Each guess leads to multiple possible outcomes
        def guess_outcome(guess):
            if guess <= 2:
                return Powerset(["too_low", "correct"])
            elif guess >= 4:
                return Powerset(["too_high", "correct"])
            else:
                return Powerset(["correct"])

        outcomes = guesses.bind(guess_outcome)
        expected = Powerset(["too_low", "correct", "too_high"])
        self.assertPowersetEqual(outcomes, expected)

    def test_filtering_with_bind(self):
        """Test filtering patterns using bind."""
        ps = Powerset([1, 2, 3, 4, 5, 6])

        # Filter to keep only even numbers using bind
        evens = ps.bind(lambda x: Powerset([x]) if x % 2 == 0 else Powerset([]))
        expected = Powerset([2, 4, 6])
        self.assertPowersetEqual(evens, expected)

    def test_cartesian_product_simulation(self):
        """Test simulating cartesian product with monadic operations."""
        # Two independent choices
        first_choice = Powerset(["A", "B"])
        second_choice = Powerset([1, 2])

        # Combine them using bind
        combinations = first_choice.bind(
            lambda x: second_choice.bind(lambda y: Powerset([(x, y)]))
        )

        expected = Powerset([("A", 1), ("A", 2), ("B", 1), ("B", 2)])
        self.assertPowersetEqual(combinations, expected)

    def test_search_space_exploration(self):
        """Test search space exploration patterns."""
        # Start with possible moves
        moves = Powerset(["up", "down", "left", "right"])

        # Each move leads to new positions
        def next_positions(move):
            if move == "up":
                return Powerset(["position_1", "position_2"])
            elif move == "down":
                return Powerset(["position_3"])
            elif move == "left":
                return Powerset(["position_4", "position_5"])
            else:  # right
                return Powerset(["position_6"])

        reachable = moves.bind(next_positions)
        expected = Powerset(
            [
                "position_1",
                "position_2",
                "position_3",
                "position_4",
                "position_5",
                "position_6",
            ]
        )
        self.assertPowersetEqual(reachable, expected)

    def test_flatten_behavior(self):
        """Test flattening behavior of bind."""
        # Create nested structure manually
        ps = Powerset([1, 2])

        # Bind with function that returns powersets
        def duplicate_options(x):
            return Powerset([x, x, x + 10])  # Duplicates should be removed

        result = ps.bind(duplicate_options)

        # Should be flattened and deduplicated
        expected = Powerset([1, 11, 2, 12])
        self.assertPowersetEqual(result, expected)

    def test_associativity_with_complex_functions(self):
        """Test associativity with more complex binding functions."""
        m = Powerset([1, 2])

        def f(x):
            return Powerset([x, x + 1])

        def g(y):
            return Powerset([y * 2, y * 3])

        # (m.bind(f)).bind(g)
        left = m.bind(f).bind(g)

        # m.bind(lambda x: f(x).bind(g))
        right = m.bind(lambda x: f(x).bind(g))

        self.assertPowersetEqual(left, right)

    def test_empty_propagation(self):
        """Test how empty sets propagate through operations."""
        ps = Powerset([1, 2, 3])

        # Bind with function that sometimes returns empty
        def conditional_empty(x):
            if x == 2:
                return empty()
            else:
                return Powerset([x * 2])

        result = ps.bind(conditional_empty)

        # Should only contain results from non-empty cases
        expected = Powerset([2, 6])  # 1*2, 3*2
        self.assertPowersetEqual(result, expected)

    def test_large_powerset_operations(self):
        """Test operations on larger powersets."""
        # Create larger powerset
        large_ps = Powerset(range(10))

        # Map to create even larger space
        mapped = large_ps.map(lambda x: x * 2)
        expected = Powerset(range(0, 20, 2))
        self.assertPowersetEqual(mapped, expected)

        # Bind to expand further
        expanded = large_ps.bind(lambda x: Powerset([x, x + 100]))
        self.assertEqual(len(expanded._inner_value), 20)  # Original 10 + 10 new values


if __name__ == "__main__":
    unittest.main()
