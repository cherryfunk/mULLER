module NeSySystem.Signatures.Countable (countableSig) where

import Syntax (Signature (..))

-- | The Countable signature
-- Sorts: Nat, CoinSeq
-- Rels: >3, startsTT, isEven, isAnything
-- MFuncs: drawInt, drawStr, drawLazy, drawHeavy
countableSig :: Signature
countableSig =
  Signature
    { sortDecls = ["Nat", "CoinSeq"],
      funDecls = [],
      relDecls =
        [ (">3", ["Nat"]),
          ("startsTT", ["CoinSeq"]),
          ("isEven", ["Nat"]),
          ("isAnything", ["Nat"])
        ],
      mFunDecls =
        [ ("drawInt", [], "Nat"),
          ("drawStr", [], "CoinSeq"),
          ("drawLazy", [], "Nat"),
          ("drawHeavy", [], "Nat")
        ],
      mRelDecls = []
    }
