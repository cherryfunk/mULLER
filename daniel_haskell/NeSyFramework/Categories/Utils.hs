{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE InstanceSigs #-}

module NeSyFramework.Categories.Utils where

-- | Category tags
data DATA
data TENS

-- | The master class for category membership.
class In cat a

-- | Morphism validation infrastructure (The "Outside" check)
class Morphism cat f where
    verify :: f -> IO ()

-- Instance for standard morphisms (A -> B).
instance {-# OVERLAPPABLE #-} (In cat a, In cat b) => Morphism cat (a -> b) where
    verify :: (In cat a, In cat b) => (a -> b) -> IO ()
    verify _ = putStrLn "✓ Valid Tarski morphism."

-- Instance for Kleisli morphisms (A -> m B).
-- We check that A is in the category AND that the wrapped result (m b) is in the category.
instance {-# OVERLAPPING #-} (In cat a, In cat (m b), Monad m) => Morphism cat (a -> m b) where
    verify :: (In cat a, In cat (m b), Monad m) => (a -> m b) -> IO ()
    verify _ = putStrLn "✓ Valid Kleisli morphism (A -> TB)."
