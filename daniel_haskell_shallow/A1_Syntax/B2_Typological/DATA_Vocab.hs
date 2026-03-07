{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeSynonymInstances #-}

module A1_Syntax.B2_Typological.DATA_Vocab where

import Numeric.Natural (Natural)

-- | In our shallow embedding, the typological vocabulary $\Delta$
--   is formed directly by chosen Haskell types.
--   These types act as the purely syntactic datatype symbols.

-- | Base types of the typological vocabulary.
class DataVocab a

instance DataVocab Bool

instance DataVocab Natural

instance DataVocab Integer

instance DataVocab String

instance DataVocab Double

-- | Constructed types of the typological vocabulary.
instance DataVocab ()

instance (DataVocab a, DataVocab b) => DataVocab (a, b)

instance (DataVocab a) => DataVocab [a]
