module NeSySystem.Signatures.Countable (countableSig) where

import Syntax (Signature (..))

-- | The Countable signature
-- Sorts: Nat, CoinSeq
-- Relations: >3, startsTT, isEven, isAnything
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
        ]
    }
