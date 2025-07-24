{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}

import qualified Data.Set as Set

-- algebra of truth values
class Ord a => Aggr2SGrpBLat a where
  top, bot :: a
  neg :: a -> a
  conj, disj, implies :: a -> a -> a
  aggrE, aggrA :: Set.Set a -> a
  -- default implementations
  neg a = a `implies` bot
  a `implies` b = (neg a) `disj` b
  aggrE = Set.fold disj bot
  aggrA = Set.fold conj top

-- NeSy frameworks provide an algebra on T Omega
-- Beware that for a given t and omega, there can be only one instance
-- If needed, use several isomorphic copies of omega to
--   distinguish several instances
-- Another solution (which do not follow here) are identity types as in Hets
class (Monad t, Aggr2SGrpBLat (t omega)) => NeSyFramework t omega

-- Non-empty powerset instance
-- there is no standard non-empty set monad in Haskell
-- so we use the list monad instead. Omega is Bool
instance  Aggr2SGrpBLat [Bool] where
  top = [True]
  bot = [False]
  conj a b = [x && y | x<-a, y<-b]
  disj a b = [x || y | x<-a, y<-b]

instance NeSyFramework [] Bool 
  
