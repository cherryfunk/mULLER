module A1_Syntax.B2_Typological.TENS_Vocab where

import Torch (Tensor)

-- | The typological vocabulary for the tensor/vector category TENS.
--   Objects are tensor spaces (embedding spaces for vectorial interpretations).
--
--   In the vectorial interpretation (Def. parameterized-interpretation):
--     I^vec(S) : TensVocab   for each sort S in Sigma.
class TensVocab a

-- | A tensor space (R^n, R^{n x m}, etc.)
instance TensVocab Tensor

-- | Product of tensor spaces.
instance (TensVocab a, TensVocab b) => TensVocab (a, b)
