-- | Dice domain -- Signature + Interpretation
module A2_Interpretation.B3_NonLogical.Dice where

import A3_Semantics.B3_NonLogical.Monads.Dist (Dist (..))

------------------------------------------------------
-- Sigma: Non-Logical Vocabulary (sorts)
------------------------------------------------------

-- | Sor
type DieResult = Int

-- | mFun: die :: Dist DieResult

------------------------------------------------------
-- I: Interpretation and Syntctic Type Declarations
------------------------------------------------------

-- | I(die) : mFun -- uniform distribution over {1,...,6}
die :: Dist DieResult
die = Dist [(i, 1.0 / 6.0) | i <- [1 .. 6]]
