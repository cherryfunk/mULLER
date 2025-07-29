import unittest
from typing import Callable

from muller.monad.identity import Identity


class TestIdentityMonad(unittest.TestCase):
    """Test cases for the Identity monad implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.tolerance_places = 6

    def assertAlmostEqualFloat(self, first: float, second: float, msg=None):
        """Helper method for comparing floats with tolerance."""
        self.assertAlmostEqual(first, second, places=self.tolerance_places, msg=msg)

    def test_initialization(self):
        """Test Identity monad initialization."""
        value = 42
        identity = Identity(value)
        
        # Should wrap the value
        self.assertEqual(identity.value, value)
        self.assertEqual(identity.get(), value)

    def test_insert_operation(self):
        """Test the insert (unit/return) operation."""
        value = "test"
        identity = Identity.insert(value)
        
        # Should create an Identity containing the value
        self.assertEqual(identity.get(), value)
        self.assertIsInstance(identity, Identity)

    def test_map_operation(self):
        """Test the map (functor) operation."""
        # Create Identity with integer
        identity = Identity(5)
        
        # Map with a function that doubles the value
        mapped = identity.map(lambda x: x * 2)
        
        # Should contain the transformed value
        self.assertEqual(mapped.get(), 10)
        self.assertIsInstance(mapped, Identity)

    def test_map_with_type_change(self):
        """Test map operation that changes the type."""
        # Start with integer
        identity = Identity(42)
        
        # Map to string
        mapped = identity.map(str)
        
        # Should contain string representation
        self.assertEqual(mapped.get(), "42")
        self.assertIsInstance(mapped, Identity)

    def test_bind_operation(self):
        """Test the bind (monadic composition) operation."""
        # Create Identity with value
        identity = Identity(3)
        
        # Bind with a function that creates new Identity
        def double_and_wrap(x):
            return Identity(x * 2)
        
        bound = identity.bind(double_and_wrap)
        
        # Should contain the result of the function
        self.assertEqual(bound.get(), 6)
        self.assertIsInstance(bound, Identity)

    def test_bind_with_type_change(self):
        """Test bind operation that changes the type."""
        # Start with integer
        identity = Identity(10)
        
        # Bind with function that returns string Identity
        def int_to_string_identity(x):
            return Identity(f"value_{x}")
        
        bound = identity.bind(int_to_string_identity)
        
        # Should contain the string
        self.assertEqual(bound.get(), "value_10")
        self.assertIsInstance(bound, Identity)

    def test_monad_laws(self):
        """Test that the monad laws hold."""
        # Left identity: unit(a).bind(f) == f(a)
        a = 7
        f = lambda x: Identity(x + 1)
        
        left = Identity.insert(a).bind(f)
        right = f(a)
        
        self.assertEqual(left.get(), right.get())
        
        # Right identity: m.bind(unit) == m
        m = Identity(15)
        bound = m.bind(Identity.insert)
        
        self.assertEqual(m.get(), bound.get())
        
        # Associativity: (m.bind(f)).bind(g) == m.bind(lambda x: f(x).bind(g))
        g = lambda x: Identity(x * 3)
        
        left_assoc = m.bind(f).bind(g)
        right_assoc = m.bind(lambda x: f(x).bind(g))
        
        self.assertEqual(left_assoc.get(), right_assoc.get())

    def test_functor_laws(self):
        """Test that the functor laws hold."""
        # Identity law: map(id) == id
        identity = Identity("test")
        mapped_id = identity.map(lambda x: x)
        
        self.assertEqual(identity.get(), mapped_id.get())
        
        # Composition law: map(f . g) == map(f) . map(g)
        f = lambda x: x * 2
        g = lambda x: x + 1
        
        identity_num = Identity(5)
        
        # map(f . g)
        composed_func = lambda x: f(g(x))
        left_side = identity_num.map(composed_func)
        
        # map(f) . map(g)
        right_side = identity_num.map(g).map(f)
        
        self.assertEqual(left_side.get(), right_side.get())

    def test_get_method(self):
        """Test the get method for extracting values."""
        # Test with different types
        identity_int = Identity(42)
        identity_str = Identity("hello")
        identity_list = Identity([1, 2, 3])
        identity_dict = Identity({"key": "value"})
        
        self.assertEqual(identity_int.get(), 42)
        self.assertEqual(identity_str.get(), "hello")
        self.assertEqual(identity_list.get(), [1, 2, 3])
        self.assertEqual(identity_dict.get(), {"key": "value"})

    def test_string_representation(self):
        """Test string representation of Identity."""
        identity = Identity("test_value")
        repr_str = repr(identity)
        
        # Should contain "Identity" and the wrapped value
        self.assertTrue("Identity" in repr_str)
        self.assertTrue("test_value" in repr_str)
        
        # Test with different types
        identity_num = Identity(123)
        repr_num = repr(identity_num)
        self.assertTrue("Identity" in repr_num)
        self.assertTrue("123" in repr_num)

    def test_complex_composition(self):
        """Test complex monadic compositions."""
        # Start with a value
        initial = Identity(2)
        
        # Chain multiple operations
        result = (initial
                 .map(lambda x: x + 1)        # 2 -> 3
                 .bind(lambda x: Identity(x * 2))  # 3 -> 6
                 .map(lambda x: x ** 2)       # 6 -> 36
                 .bind(lambda x: Identity(str(x))))  # 36 -> "36"
        
        # Should contain the final result
        self.assertEqual(result.get(), "36")
        self.assertIsInstance(result, Identity)

    def test_nested_identities(self):
        """Test working with nested Identity monads."""
        # Create Identity containing another Identity
        nested = Identity(Identity(5))
        
        # Get the outer value (which is an Identity)
        inner_identity = nested.get()
        self.assertIsInstance(inner_identity, Identity)
        self.assertEqual(inner_identity.get(), 5)
        
        # Bind to flatten the nesting
        flattened = nested.bind(lambda x: x)
        self.assertEqual(flattened.get(), 5)

    def test_identity_with_functions(self):
        """Test Identity monad containing functions."""
        # Create Identity containing a function
        func_identity = Identity(lambda x: x * 2)
        
        # Extract and use the function
        func = func_identity.get()
        result = func(5)
        self.assertEqual(result, 10)
        
        # Map over the function to create a new function
        mapped_func_identity = func_identity.map(
            lambda f: lambda x: f(x) + 1  # Compose with +1
        )
        
        new_func = mapped_func_identity.get()
        result2 = new_func(5)
        self.assertEqual(result2, 11)  # (5 * 2) + 1

    def test_identity_with_collections(self):
        """Test Identity monad with collections."""
        # Test with list
        list_identity = Identity([1, 2, 3])
        
        # Map to transform the list
        mapped_list = list_identity.map(lambda lst: [x * 2 for x in lst])
        self.assertEqual(mapped_list.get(), [2, 4, 6])
        
        # Test with dictionary
        dict_identity = Identity({"a": 1, "b": 2})
        
        # Map to transform values
        mapped_dict = dict_identity.map(
            lambda d: {k: v * 3 for k, v in d.items()}
        )
        self.assertEqual(mapped_dict.get(), {"a": 3, "b": 6})

    def test_error_preservation(self):
        """Test that Identity preserves computational context (including exceptions)."""
        # Identity doesn't handle exceptions, but let's test normal error flows
        identity = Identity(None)
        
        # Map should work even with None
        mapped = identity.map(lambda x: x if x is not None else "null")
        self.assertEqual(mapped.get(), "null")

    def test_sequential_computations(self):
        """Test sequential computations using Identity."""
        # Simulate a sequence of computations that could fail but don't
        def parse_int(s):
            return Identity(int(s))
        
        def validate_positive(n):
            return Identity(n) if n > 0 else Identity(None)
        
        def double_value(n):
            return Identity(n * 2) if n is not None else Identity(None)
        
        # Chain the computations
        result = (Identity("5")
                 .bind(parse_int)
                 .bind(validate_positive) 
                 .bind(double_value))
        
        self.assertEqual(result.get(), 10)

    def test_mathematical_operations(self):
        """Test mathematical operations within Identity."""
        # Test arithmetic operations
        x = Identity(10)
        y = Identity(3)
        
        # We need to extract values for operations since Identity doesn't implement arithmetic
        sum_result = Identity(x.get() + y.get())
        self.assertEqual(sum_result.get(), 13)
        
        # Or use bind to combine them
        def add_identities(x_id, y_id):
            return x_id.bind(lambda x: y_id.bind(lambda y: Identity(x + y)))
        
        combined = add_identities(x, y)
        self.assertEqual(combined.get(), 13)

    def test_comparison_operations(self):
        """Test comparison operations with Identity values."""
        id1 = Identity(5)
        id2 = Identity(5)
        id3 = Identity(10)
        
        # Identities are compared by their wrapped values when using .get()
        self.assertEqual(id1.get(), id2.get())
        self.assertNotEqual(id1.get(), id3.get())
        
        # Test comparison using bind
        def compare_greater(x_id, y_id):
            return x_id.bind(lambda x: y_id.bind(lambda y: Identity(x > y)))
        
        result = compare_greater(id3, id1)
        self.assertTrue(result.get())

    def test_type_safety(self):
        """Test type safety with Identity monad."""
        # Start with string
        str_identity = Identity("hello")
        
        # Map to get length (int)
        len_identity = str_identity.map(len)
        self.assertEqual(len_identity.get(), 5)
        self.assertIsInstance(len_identity.get(), int)
        
        # Map back to string
        str_len_identity = len_identity.map(str)
        self.assertEqual(str_len_identity.get(), "5")
        self.assertIsInstance(str_len_identity.get(), str)


if __name__ == '__main__':
    unittest.main()
