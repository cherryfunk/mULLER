{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE NoStarIsType #-}

module A1_Syntax.B1_Categorical.CatVocab where

import Data.Functor.Identity (Identity)
import Data.Kind (Type)
import Data.Void (Void)

-- ============================================================
-- Categorical Vocabulary kappa (Syntax)
-- ============================================================
--
-- The categorical vocabulary kappa (Def. categorical-vocabulary)
-- consists of purely SYNTACTIC symbols. Their semantic
-- interpretation belongs to A2_Interpretation / A3_Semantics.

-- | CATEGORY SYMBOL: Type
--
--   'Type' (from Data.Kind) is the category symbol C.
--   It is a Haskell kind.
--
--   Its INTERPRETATION is the category Hask, whose:
--     Objects   = evaluation sets (inhabitants) of Haskell types
--     Morphisms = Haskell functions between them

-- | The category symbol C:
type C = Type

class CatVocab (a :: k)

-- | Functor symbols (Func):
instance CatVocab Identity -- id

instance CatVocab (->) -- Hom

instance CatVocab Void -- bot = vec0

instance CatVocab () -- top = vec1

instance CatVocab Either -- sqcup = oplus

instance CatVocab (,) -- sqcap = otimes

-- | Monad symbols:
instance CatVocab []

instance CatVocab Maybe
