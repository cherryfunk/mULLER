import unittest

from muller.monad.non_empty_powerset import (
    NonEmptyPowerset,
    choice,
    from_list,
    join,
    singleton,
)


class TestNonEmptyPowersetMonad(unittest.TestCase):
    """Test cases for the NonEmptyPowerset monad implementation."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def assertNonEmptyPowersetEqual(
        self, ps1: NonEmptyPowerset, ps2: NonEmptyPowerset, msg=None
    ):
        """Helper method for comparing non-empty powersets."""
        self.assertEqual(ps1.value, ps2.value, msg=msg)

    def test_initialization(self):
        """Test NonEmptyPowerset monad initialization."""
        # Test with list
        ps = NonEmptyPowerset([1, 2, 3])
        self.assertEqual(ps.value, frozenset({1, 2, 3}))

        # Test with set
        ps_set = NonEmptyPowerset({4, 5, 6})
        self.assertEqual(ps_set.value, frozenset({4, 5, 6}))

        # Test with duplicates (should be removed)
        ps_dup = NonEmptyPowerset([1, 1, 2, 2, 3])
        self.assertEqual(ps_dup.value, frozenset({1, 2, 3}))

    def test_empty_initialization_raises_error(self):
        """Test that empty initialization raises ValueError."""
        with self.assertRaises(ValueError):
            NonEmptyPowerset([])

        with self.assertRaises(ValueError):
            NonEmptyPowerset(set())

        with self.assertRaises(ValueError):
            NonEmptyPowerset(frozenset())

    def test_unit_operation(self):
        """Test the unit operation."""
        value = "single"
        ps = NonEmptyPowerset.unit(value)

        # Should contain only the single value
        self.assertEqual(ps.value, frozenset({"single"}))
        self.assertEqual(len(ps.value), 1)

    def test_map_operation(self):
        """Test the map (functor) operation."""
        # Create non-empty powerset with numbers
        ps = NonEmptyPowerset([1, 2, 3])

        # Map with a function that doubles values
        mapped = ps.map(lambda x: x * 2)

        # Should contain doubled values
        expected = NonEmptyPowerset([2, 4, 6])
        self.assertNonEmptyPowersetEqual(mapped, expected)

    def test_map_preserves_non_emptiness(self):
        """Test that map preserves non-emptiness."""
        # Start with single element
        ps = NonEmptyPowerset([1])

        # Map should still be non-empty
        mapped = ps.map(lambda x: x * 10)
        self.assertEqual(len(mapped.value), 1)
        self.assertEqual(mapped.value, frozenset({10}))

    def test_map_with_duplicates(self):
        """Test map operation when function produces duplicate values."""
        # Create powerset where map will create duplicates
        ps = NonEmptyPowerset([1, 2, 3, 4])

        # Map with a function that makes some values the same
        mapped = ps.map(lambda x: x // 2)  # 1->0, 2->1, 3->1, 4->2

        # Should automatically deduplicate but remain non-empty
        expected = NonEmptyPowerset([0, 1, 2])
        self.assertNonEmptyPowersetEqual(mapped, expected)

    def test_bind_operation(self):
        """Test the bind (monadic composition) operation."""
        # Create initial non-empty powerset
        ps = NonEmptyPowerset([1, 2])

        # Bind with a function that creates new non-empty powersets
        def expand(x):
            return NonEmptyPowerset([x, x + 10])

        result = ps.bind(expand)

        # Should contain union of all results
        # 1 -> {1, 11}, 2 -> {2, 12}
        # Union: {1, 11, 2, 12}
        expected = NonEmptyPowerset([1, 11, 2, 12])
        self.assertNonEmptyPowersetEqual(result, expected)

    def test_bind_preserves_non_emptiness(self):
        """Test that bind preserves non-emptiness."""
        ps = NonEmptyPowerset([1])

        # Bind with function that always returns non-empty
        result = ps.bind(lambda x: NonEmptyPowerset([x * 2]))

        # Should still be non-empty
        self.assertGreater(len(result.value), 0)
        self.assertEqual(result.value, frozenset({2}))

    def test_monad_laws(self):
        """Test that the monad laws hold."""
        # Left identity: unit(a).bind(f) == f(a)
        a = 5

        def f(x):
            return NonEmptyPowerset([x, x + 1])

        left = NonEmptyPowerset.unit(a).bind(f)
        right = f(a)
        self.assertNonEmptyPowersetEqual(left, right)

        # Right identity: m.bind(unit) == m
        m = NonEmptyPowerset([1, 2, 3])
        bound = m.bind(NonEmptyPowerset.unit)
        self.assertNonEmptyPowersetEqual(m, bound)

        # Associativity: (m.bind(f)).bind(g) == m.bind(lambda x: f(x).bind(g))
        def g(x):
            return NonEmptyPowerset([x * 2, x * 3])

        left_assoc = m.bind(f).bind(g)
        right_assoc = m.bind(lambda x: f(x).bind(g))
        self.assertNonEmptyPowersetEqual(left_assoc, right_assoc)

    def test_functor_laws(self):
        """Test that the functor laws hold."""
        # Identity law: map(id) == id
        ps = NonEmptyPowerset([1, 2, 3])
        mapped_id = ps.map(lambda x: x)
        self.assertNonEmptyPowersetEqual(ps, mapped_id)

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

        self.assertNonEmptyPowersetEqual(left_side, right_side)

    def test_join_operation(self):
        """Test the join operation."""
        # Create nested non-empty powerset
        inner1 = NonEmptyPowerset([1, 2])
        inner2 = NonEmptyPowerset([3, 4])
        nested = NonEmptyPowerset([inner1, inner2])

        # Join should flatten
        flattened = join(nested)
        expected = NonEmptyPowerset([1, 2, 3, 4])
        self.assertNonEmptyPowersetEqual(flattened, expected)

    def test_join_preserves_non_emptiness(self):
        """Test that join preserves non-emptiness."""
        # Single nested element
        inner = NonEmptyPowerset([42])
        nested = NonEmptyPowerset([inner])

        flattened = join(nested)
        self.assertEqual(flattened.value, frozenset({42}))

    def test_singleton_function(self):
        """Test singleton convenience function."""
        ps = singleton("test")
        expected = NonEmptyPowerset(["test"])
        self.assertNonEmptyPowersetEqual(ps, expected)

    def test_from_list_function(self):
        """Test from_list convenience function."""
        ps = from_list([1, 2, 3, 2, 1])  # With duplicates
        expected = NonEmptyPowerset([1, 2, 3])
        self.assertNonEmptyPowersetEqual(ps, expected)

    def test_from_list_empty_raises_error(self):
        """Test that from_list with empty list raises error."""
        with self.assertRaises(ValueError):
            from_list([])

    def test_choice_function(self):
        """Test choice convenience function."""
        ps = choice("A", "B", "C")
        expected = NonEmptyPowerset(["A", "B", "C"])
        self.assertNonEmptyPowersetEqual(ps, expected)

        # Test with single choice
        ps_single = choice("only")
        expected_single = NonEmptyPowerset(["only"])
        self.assertNonEmptyPowersetEqual(ps_single, expected_single)

    def test_string_representation(self):
        """Test string representation of non-empty powersets."""
        ps = NonEmptyPowerset([3, 1, 2])
        repr_str = repr(ps)

        # Should contain "NonEmptyPowerset" and be sorted
        self.assertTrue("NonEmptyPowerset" in repr_str)
        # Should be sorted in the representation
        self.assertTrue(repr_str.index("1") < repr_str.index("2"))
        self.assertTrue(repr_str.index("2") < repr_str.index("3"))

    def test_complex_composition(self):
        """Test complex monadic compositions."""
        # Start with initial choices
        initial = NonEmptyPowerset([1, 2])

        # Chain multiple operations
        result = (
            initial.map(lambda x: x + 1)  # {2, 3}
            .bind(lambda x: NonEmptyPowerset([x, x * 2]))  # {2, 4, 3, 6}
            .map(lambda x: x - 1)
        )  # {1, 3, 2, 5}

        expected = NonEmptyPowerset([1, 3, 2, 5])
        self.assertNonEmptyPowersetEqual(result, expected)

    def test_non_deterministic_computation(self):
        """Test non-deterministic computation patterns."""
        # Simulate guessing a number
        guesses = NonEmptyPowerset([1, 2, 3, 4, 5])

        # Each guess leads to multiple possible outcomes
        def guess_outcome(guess):
            if guess <= 2:
                return NonEmptyPowerset(["too_low", "correct"])
            elif guess >= 4:
                return NonEmptyPowerset(["too_high", "correct"])
            else:
                return NonEmptyPowerset(["correct"])

        outcomes = guesses.bind(guess_outcome)
        expected = NonEmptyPowerset(["too_low", "correct", "too_high"])
        self.assertNonEmptyPowersetEqual(outcomes, expected)

    def test_guaranteed_results(self):
        """Test that operations always produce results (non-empty)."""
        ps = NonEmptyPowerset([1, 2, 3, 4, 5])

        # Any operation should maintain non-emptiness
        mapped = ps.map(lambda x: x * 100)
        self.assertGreater(len(mapped.value), 0)

        bound = ps.bind(lambda x: NonEmptyPowerset([x % 3]))
        self.assertGreater(len(bound.value), 0)

    def test_cartesian_product_simulation(self):
        """Test simulating cartesian product with monadic operations."""
        # Two independent choices
        first_choice = NonEmptyPowerset(["A", "B"])
        second_choice = NonEmptyPowerset([1, 2])

        # Combine them using bind
        combinations = first_choice.bind(
            lambda x: second_choice.bind(lambda y: NonEmptyPowerset([(x, y)]))
        )

        expected = NonEmptyPowerset([("A", 1), ("A", 2), ("B", 1), ("B", 2)])
        self.assertNonEmptyPowersetEqual(combinations, expected)

    def test_search_space_exploration(self):
        """Test search space exploration patterns."""
        # Start with possible moves
        moves = NonEmptyPowerset(["up", "down", "left", "right"])

        # Each move leads to new positions
        def next_positions(move):
            if move == "up":
                return NonEmptyPowerset(["position_1", "position_2"])
            elif move == "down":
                return NonEmptyPowerset(["position_3"])
            elif move == "left":
                return NonEmptyPowerset(["position_4", "position_5"])
            else:  # right
                return NonEmptyPowerset(["position_6"])

        reachable = moves.bind(next_positions)
        expected = NonEmptyPowerset(
            [
                "position_1",
                "position_2",
                "position_3",
                "position_4",
                "position_5",
                "position_6",
            ]
        )
        self.assertNonEmptyPowersetEqual(reachable, expected)

    def test_flatten_behavior(self):
        """Test flattening behavior of bind."""
        # Create nested structure
        ps = NonEmptyPowerset([1, 2])

        # Bind with function that returns powersets
        def duplicate_options(x):
            return NonEmptyPowerset([x, x, x + 10])  # Duplicates should be removed

        result = ps.bind(duplicate_options)

        # Should be flattened and deduplicated
        expected = NonEmptyPowerset([1, 11, 2, 12])
        self.assertNonEmptyPowersetEqual(result, expected)

    def test_minimal_size_preservation(self):
        """Test that minimal non-empty size is preserved."""
        # Start with single element
        minimal = NonEmptyPowerset([42])

        # Map to different single element
        mapped = minimal.map(lambda x: x * 2)
        self.assertEqual(len(mapped.value), 1)
        self.assertEqual(mapped.value, frozenset({84}))

        # Bind to different single element
        bound = minimal.bind(lambda x: NonEmptyPowerset([str(x)]))
        self.assertEqual(len(bound.value), 1)
        self.assertEqual(bound.value, frozenset({"42"}))

    def test_overlapping_results_in_bind(self):
        """Test bind operation when results overlap."""
        ps = NonEmptyPowerset([1, 2, 3])

        # Bind with function that can produce overlapping results
        def neighbors(x):
            return NonEmptyPowerset([x - 1, x, x + 1])

        result = ps.bind(neighbors)

        # 1 -> {0, 1, 2}, 2 -> {1, 2, 3}, 3 -> {2, 3, 4}
        # Union: {0, 1, 2, 3, 4}
        expected = NonEmptyPowerset([0, 1, 2, 3, 4])
        self.assertNonEmptyPowersetEqual(result, expected)

    def test_type_preservation(self):
        """Test that type information is preserved through operations."""
        # Start with strings
        ps_str = NonEmptyPowerset(["a", "b", "c"])

        # Map to integers
        ps_int = ps_str.map(ord)  # Convert to ASCII values
        self.assertIsInstance(list(ps_int.value)[0], int)

        # Bind back to strings
        ps_str2 = ps_int.bind(lambda x: NonEmptyPowerset([chr(x), chr(x + 1)]))
        self.assertIsInstance(list(ps_str2.value)[0], str)

    def test_large_computation_chains(self):
        """Test long chains of computations."""
        # Start with range
        ps = NonEmptyPowerset([1, 2, 3])

        # Long chain of operations
        result = (
            ps.map(lambda x: x + 1)
            .bind(lambda x: NonEmptyPowerset([x, x * 2]))
            .map(lambda x: x - 1)
            .bind(lambda x: NonEmptyPowerset([x]) if x > 0 else NonEmptyPowerset([1]))
            .map(lambda x: x * 10)
        )

        # Should still be non-empty and contain reasonable values
        self.assertGreater(len(result.value), 0)
        self.assertTrue(all(isinstance(x, int) for x in result.value))
        self.assertTrue(all(x > 0 for x in result.value))

    def test_deterministic_vs_non_deterministic(self):
        """Test difference between deterministic and non-deterministic computations."""
        # Deterministic: single value
        deterministic = NonEmptyPowerset([5])
        det_result = deterministic.bind(lambda x: NonEmptyPowerset([x * 2]))
        self.assertEqual(len(det_result.value), 1)

        # Non-deterministic: multiple values
        non_deterministic = NonEmptyPowerset([1, 2, 3])
        non_det_result = non_deterministic.bind(lambda x: NonEmptyPowerset([x * 2]))
        self.assertEqual(len(non_det_result.value), 3)

    def test_choice_explosion(self):
        """Test how choices can explode through bind operations."""
        # Start with 2 choices
        initial = NonEmptyPowerset([1, 2])

        # Each choice creates 3 new choices
        step1 = initial.bind(
            lambda x: NonEmptyPowerset([x * 10, x * 10 + 1, x * 10 + 2])
        )
        self.assertEqual(len(step1.value), 6)  # 2 * 3

        # Each of those creates 2 more choices
        step2 = step1.bind(lambda x: NonEmptyPowerset([x, x + 100]))
        self.assertEqual(len(step2.value), 12)  # 6 * 2


if __name__ == "__main__":
    unittest.main()
