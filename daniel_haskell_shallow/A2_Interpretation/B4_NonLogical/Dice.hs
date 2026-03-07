-- | Dice domain -- Signature + Interpretation
module A2_Interpretation.B4_NonLogical.Dice where

import A1_Syntax.B4_NonLogical.Dice_Vocab (DieResult)
import A2_Interpretation.B1_Categorical.Monads.Dist (Dist (..))

------------------------------------------------------
-- I: Interpretation (Schema Instance + Function Definitions)
------------------------------------------------------

-- | I(die) : mFun -- uniform distribution over {1,...,6}
die :: Dist DieResult
die = Dist [(i, 1.0 / 6.0) | i <- [1 .. 6]]
