-- | Interpretation 𝓘_Σ of DiceSig in (DATA, Dist)
module NonLogical.Interpretations.Dice (die) where

import NonLogical.Monads.Dist (Dist (..))
import NonLogical.Signatures.DiceSig (DieResult)

-- | 𝓘(die) : mFun — uniform distribution over {1,...,6}
die :: Dist DieResult
die = Dist [(i, 1.0 / 6.0) | i <- [1 .. 6]]
