{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}

module A1_Syntax.B2_Typological.FDATA_Vocab where

import A1_Syntax.B2_Typological.DATA_Vocab (DataVocab)

-- | The cartesian closed, monad-closed extension of DATA.
--   FDATA = DATA + exponentials (function types) + monadic types.
--
--   Non-logical vocabulary interprets in DATA  (finite products only).
--   Logical vocabulary interprets in FDATA     (cartesian closed + monads).
class FdataVocab a

-- | Every DataVocab type is also an FdataVocab type.
instance (DataVocab a) => FdataVocab a

-- | Exponentials (function types): the cartesian closure.
instance (FdataVocab a, FdataVocab b) => FdataVocab (a -> b)

-- | Monadic closure: for any Monad m, if a is in FDATA then m a is in FDATA.
instance (Monad m, FdataVocab a) => FdataVocab (m a)
