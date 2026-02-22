module NeSySystem.Signatures.Dice (diceSig) where

import Syntax (Signature (..))

-- | The Dice signature
-- One sort: DieResult
-- No function symbols
-- Relations: ==, even
diceSig :: Signature
diceSig =
  Signature
    { sortDecls = ["DieResult"],
      funDecls = [],
      relDecls =
        [ ("==", ["DieResult", "DieResult"]),
          ("even", ["DieResult"])
        ]
    }
