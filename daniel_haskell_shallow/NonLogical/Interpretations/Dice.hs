-- | Dice domain — Signature + Interpretation
module NonLogical.Interpretations.Dice where

import NonLogical.Monads.Dist (Dist (..))

--------------------------------------------------------------------------------
-- Σ: Non-Logical Vocabulary (sorts)
--------------------------------------------------------------------------------

-- | Sor
type DieResult = Int

-- | mFun: die :: Dist DieResult

--------------------------------------------------------------------------------
-- 𝓘: Interpretation and Syntctic Type Declarations
--------------------------------------------------------------------------------

-- | 𝓘(die) : mFun — uniform distribution over {1,...,6}
die :: Dist DieResult
die = Dist [(i, 1.0 / 6.0) | i <- [1 .. 6]]
